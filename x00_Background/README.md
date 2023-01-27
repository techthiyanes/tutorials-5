# 📜

[//]: # (# TODO [DG])
2. Logging in and setting configurations
   1. Getting set up on cloud providers / GPU quota
1. Sends and Clusters 
   2. ssh / debugging
1. Folders, Blobs, and Tables
1. Runhouse in Notebooks
1. Saving and loading resources

## 01: Runhouse in Colab

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

### Notes on notebooks

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

## 02: Saving and Loading Secrets from Vault

### Saving Secrets

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

### Loading Secrets

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

## 03: Advanced Send and Cluster usage

### SSH / JupyterLab / Debugging

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

### Send call types

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
