# 📜 Runhouse Detailed Overview

This is a strictly optional section for those who want a more comprehensive 
overview of Runhouse and the APIs. If you're just looking to get started, 
you can skip this section and jump to the [Quickstart](../x01_Quickstart/README.md).

## Table of Contents
1. 🏙 [High-level Architecture](#01-high-level-architecture)
1. Sends and Clusters 
   2. ssh / debugging
   3. Runhouse in Notebooks
1. Folders, Blobs, and Tables
2. 2. Logging in and setting configurations
   1. Getting set up on cloud providers / GPU quota
1. Saving and loading resources

## 01 🏙 High-level Architecture

Runhouse has four top-level objectives:
1. Allowing users to natively program across compute resources
2. Allowing users to command data between storage and compute
3. Making resources accessible across environments and users
4. Allowing resources to be shared among teams as living assets

It achieves the above by providing four pillar features:
1. **Compute** - The Send, Cluster, and Package APIs allow a seamless flow of code and
execution across local and remote compute. They blur the line 
between program execution and deployment, providing both a path of least resistence
for running a sub-routine on specific hardware, while unceremoniously turning that 
sub-routine into a reusable service. They also provide convenient dependency 
isolation and management, provider-agnostic provisioning and termination, and rich 
debugging and accessibility interfaces built-in.
2. **Data** - The Folder, Table, and Blob APIs provide a simple interface for storing, 
recalling, and moving data between the user's laptop, remote compute, cloud storage,
and specialized storage (e.g. data warehouses). They provide least-common-denominator
APIs across providers, allowing users to easily specify the actions they want to take on the
data without needed to dig into provider-specific APIs. We'd like to extend this to other
data concepts in the future, like kv-stores, time-series, vector and graph databases, etc.
3. **Accessibility** - Runhouse strives to provide a Google-Docs-like experience for 
portability and sharing of resources across users and environments. This is achieved by:
   1. The Resource Naming System (RNS) allows resources to be named, persisted, and recalled
   across environments. It consists of a lightweight metadata standard for each resource type
   which captures the information needed to load it in a new environment (e.g. Folder -> provider, 
   bucket, path, etc.), and a mechanism for saving and loading from either the working git repo or 
   a remote Runhouse key-value metadata store. The metadata store allows resources to be shared across 
   users and environments, while the git approach allows for local persistence and versioning or 
   sharing across OSS projects.
   2. The Secrets API provides a simple interface for storing and retrieving secrets
   to a allow a more seamless experience when accessing resources across environments. 
   It provides a simple interface for storing and retrieving secrets from a variety of 
   providers (e.g. AWS, Azure, GCP, Hugging Face, Github, etc.) as well as SSH Keys and
   custom secrets, and stores them in Hashicorp Vault.
4. **Management** - Runhouse provides tools for visibility and management of resources 
as long-living assets shared by teams or projects. Both resources and users can be 
organized into arbitrarily-nested groups to apply access permissions, default behaviors (e.g. 
default storage locations, compute providers, instance autotermination, etc.), project delineation,
or staging (e.g. dev vs. prod). The [management UI](api.run.house) provides an individual or 
admin view of all resources, secrets, groups, and sharing (this is only an MVP, and will be 
overhauled soon). Resource metadata is automatically versioned in RNS, allowing teams to maintain
single-sources of truth for assets with zero downtime to update or roll back, and trace exact 
lineage for any resource (assuming the underlying the resources are not being deleted). We provide
basic logging out of the box today, and are working on providing comprehensive logging, monitoring,
alerting.

## 02: 🖥 Compute: Sends, Clusters, and Packages

### 01: 🏘 Clusters

Clusters represent a set of machines which can be sent code or data, or a machine spec
that could be spun up in the event that we have some code or data to send to the machine.
Generally they are Ray clusters under the hood. There are two kinds of clusters today:

**1. BYO Cluster**
This is a machine or group of machines that you input the IP addresses of. You can
then interact with these machines through the other Runhouse APIs. This is useful
if you have an on-prem instance, or an account with Paperspace, Coreweave, or another
vertical provider, or simply want to spin up machines yourself through the cloud UI.
You can use the cluster factory constructor like so:

```python
gpu = rh.cluster(ips=['<ip of the cluster>'], 
                 ssh_creds={'ssh_user': '...', 'ssh_private_key':'<path_to_key>'},
                 name='rh-a10x')
```


**2. SkyPilot Clusters**

Runhouse can spin up and down boxes for you as needed using SkyPilot. When you
define a SkyPilot "cluster," you're primarily defining the configuration for us to spin
up the compute resources you need later. When someone then calls a send or similar, we'll 
spin the box back up for you.

SkyPilot also provides an excellent suite of CLI commands for basic instance 
management operations. Some important ones are:
* `sky status --refresh` - Get the status of the clusters *you launched from
this machine*. This will not pull the status for all the machines you've
launched from various environments. We plan to add this feature soon.
* `sky down --all` - This will take down (terminate, without persisting the
disk image) all clusters in the local SkyPilot context (the ones that show
when you run `sky status --refresh`). However, the best way to confirm that you don't
have any machines left running is always to check the cloud provider's UI.
* `sky down <cluster_name>` - This will take down a specific cluster.
* `ssh <cluster_name>` - This will ssh into the head node of the cluster.
SkyPilot cleverly adds the host information to your `~/.ssh/config` file, so
ssh will just work.
* `sky autostop -i <minutes, or -1> <cluster_name>` - This will set the
cluster to autostop after that many minutes of inactivity. By default this
number is 10 minutes, but you can set it to -1 to disable autostop entirely.
You can set your default autostop in `~/.rh/config.yaml`.

**Advanced Cluster Usage**

To start an ssh session into the cluster so you can poke around or debug:
```commandline
ssh rh-v100
```
Or in python:
```python
my_send.ssh()
```
We realize our logging situation is not ideal and we're working on improving it. For
now, the easiest way to view outputs and logs is by sshing into the cluster and running:
```commandline
screen -r
```
🚨 **Make sure not to control-C out of screen (use ctrl-A-D instead), or you'll kill the cluster's
gRPC server.** If you do, and as a general debugging practice, just run:
```python
cluster.restart_grpc_server()
```

If you prefer to work in notebooks, you can tunnel a JupyterLab server into your local browser:
```commandline
runhouse notebook my_cluster
```
or in python:
```python
my_send.notebook()
# or
my_cluster.notebook()
```

To run a shell command on the cluster:
```python
gpu.run(['git clone ...', 'pip install ...'])
```
This is useful for installing more complex dependencies.
`gpu.run_setup(...)` will make sure the command is only run once when the cluster is first created.

To run any Python on the cluster:
```python
gpu.run_python(['import torch', 'print(torch.__version__)'])
```
This is useful for debugging, or for running a script that you don't 
want to send to the cluster (e.g. because it has too many dependencies).

If you want to run an application on the cluster that requires a port to be open, e.g. Tensorboard, Gradio.
```python
gpu.ssh_tunnel(local_port=7860, remote_port=7860)
```

### 02: 🏹 Sends

**Advanced Send usage**

There are a number of ways to call a Send beyond just `__call__`.

`.remote` will call the function async (using Ray) and return a reference (Ray ObjectRef) to the object on the cluster. 
You can pass the ref into another send and it will be automatically dereferenced once on the cluster. 
This is a convenient way to avoid passing large objects back and forth to your laptop, or to 
run longer execution in notebooks without locking up the kernel.
```python
images_ref = generate_gpu.remote('A dog.', num_images=1, steps=50)
images = rh.get(images_ref)
# or
my_other_sent_function(images_ref)
```

`.enqueue` will queue up your function call on the cluster to make sure it doesn't run 
simultaneously with other calls, but will wait until the execution completes.

`map` and `starmap` are easy way to parallelize your function (again using Ray on the cluster).
```python
images = generate_gpu.map(['A dog.', 'A cat.', 'A biscuit.'], num_images=[1]*3, steps=[50]*3)
```
will run the function on each of the three prompts, and return a list of the results. Note that the
num_images and steps arguments are broadcasted to each prompt, so the first prompt will get 1 image.

```python
generate_gpu.starmap([('A dog.', 1), ('A cat.', 2), ('A biscuit.', 3)], steps=50)
```
is the same as map as above, but we can pass the arguments as a list of tuples, and we can pass the steps
argument as a single value, since it's the same for all three prompts.


### 03: 📓 Runhouse in Notebooks

It is easier to run in Colab if you create a Runhouse account so you can store your secrets
in Vault and load them into Colab with `rh.login()` (though not required, you can still drop them
into the Colab environment manually). See [Appendix 02](#02-saving-and-loading-secrets-from-vault) for more details.

**Note, if you simply prefer to work in notebooks but don't need a hosted notebook specifically,** 
you can simply call the following to tunnel a JupyterLab server into your local 
browser from your Runhouse cluster or send:
```commandline
runhouse notebook my_cluster
```
or in python:
```python
my_send.notebook()
# or
my_cluster.notebook()
```

#### Notes on notebooks

Notebooks are funny beasts. The code and variable inside them are not designed to be
reused to shuttled around. As such:
1) If you want to `rh.send` a function defined inside the notebook, it cannot contain variables or imports
from outside the function, and you should assign a `name` to the send. We will write the function
out to a separate `.py` file and import it from there, and the filename will be set to the `send.name`.
2) If you really want to use local variables or avoid writing out the function, you can set the
you can set `serialize_notebook_fn=True` in `rh.send()` to cloudpickle the function before sending it,
but we do not support saving and reloading these kinds of sends (cloudpickle does not support this 
kind of reuse and it will create issues).
3) It is nearly always better to try to write your code in a `.py` file somewhere and import it 
into the notebook, rather than define important functions in the notebook itself. You can also use the 
`%%writefile` magic to write your code into a file, and then import it back into the notebook. 

You can sync down your code or data from the cluster when you're done with:
```python
rh.folder(url='remote_directory', fs=rh.cluster('my_cluster').to('here', url='local_directory')
```

### 04: 📦 Packages 
(WIP)

## 03: Data: Folders, Tables, Blobs
(WIP)

## 04: Accessibility, Portability, and Sharing

### 01: The Resource Name System (RNS)
(WIP)

### 02: Secrets and Logging in

```python
import runhouse as rh
import sky
```

**Logging In**
In order to use Runhouse across environments, such as reusing a service from inside a Colab
or loading secrets or configs into a remote environment, you'll need to create an account and log in.
You don't need to create an account or log in if you only plan to use Runhouse's APIs in a single
environment, and don't plan to share resources with others.

Run this code wherever your cloud credentials are already saved, such as your laptop.
Follow the prompts to log in. If this is your first time logging in, you should probably upload your secrets,
but none of the other prompts will have any effect.
rh.login(upload_secrets=True, interactive=True)

You can do this interactively by running the following in your command line:
`runhouse login`

**Setting Config Options**
Runhouse stores user configs both locally in `~/.rh/config.yaml` and remotely in the Runhouse database.
This allows you to preserve your same config across environments. Some important configs to consider setting:

Whether to use spot instances (cheaper but can be reclaimed at any time) by default.
Note that this is False by default because you'll need to request spot quota from the
cloud providers to use spot instances. You can override this setting in the cluster factory constructor.
`rh.configs.set('use_spot', False)`

Clusters can start and stop dynamically to save money. If you set autostop = 10, the cluster will terminate after
10 minutes of inactivity. If you set autostop = -1, the cluster will stay up indefinitely. After the cluster
terminates, if you call a Send which is on that cluster, the Send will automatically start the cluster again.
You can also call `cluster.keep_warm(autostop=-1) to control this for an existing cluster.
`rh.configs.set('default_autostop', 30)`

You can set your default Cloud provider if you have multiple Cloud accounts set up locally. If you set it
to 'cheapest', SkyPilot will select the cheapest provider for your desired hardware (including spot pricing,
if enabled). You can set this to 'aws', 'gcp', or 'azure' too.
`rh.configs.set('default_provider', 'cheapest')`

Now that you've changed some configs, you probably want to save them to Runhouse to access them elsewhere.
`rh.configs.upload_defaults()`


**Viewing RPC logs**

We will provide more instruction about log persistence and steaming shortly, but in the meantime, if you need to see the
logs for the sends on a particular cluster, you can ssh into the cluster with `ssh <cluster name>` and
'screen -r' (*and use `control A+D` to exit. If you control-C you will stop the server*).
The server runs inside that screen instance, so logs are written to there.

**Restarting the RPC Server**

Sometimes the RPC server will crash, or you'll update a package that the server has already imported.
In those cases, you can try to restart just the server (~20 seconds) to save yourself the trouble of nuking
and reallocating the hardware itself (minutes). You can do this by `my_cluster.restart_grpc_server()`.

### 02: Saving and Loading Secrets from Vault

**Saving Secrets**

There are a few ways to save secrets to Runhouse to make
them available conveniently across environments.

If your secrets are saved into your local environment (e.g. `~/.aws/...`), 
the fastest way to save them is to run `runhouse login` in your command line 
(or `runhouse.login()` in a Python interpreter), which will prompt you for 
your Runhouse token and ask if you'd like to upload secrets. It will then 
extract secrets from your environment and upload them to Vault. Alternatively, 
you can run:
```
import runhouse as rh
rh.Secrets.extract_and_upload()
```

To add locally stored secrets for a specific provider (AWS, Azure, GCP):
```
rh.Secrets.put(provider="aws")
```

To add secrets for a custom provider or those not stored in local config files, use:

```
rh.Secrets.put(provider="snowflake", secret={"private_key": "************"})
```
These will not be automatically loaded into new environments via `runhouse login`,
but can be accessed in code via `rh.Secrets.get(provider="snowflake")`.

**Loading Secrets**

To load secrets into your environment, you can run `runhouse login` or `rh.login()` in
your command line or Python interpreter. This will prompt you for your Runhouse token
and download secrets from Vault. Alternatively, you can run:
```
import runhouse as rh
rh.Secrets.download_into_env()
# or
rh.Secrets.download_into_env(providers=["aws", "azure"])
```

To get secrets for a specific provider:
```
my_creds = rh.Secrets.get(provider="aws")
```
