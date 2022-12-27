# Before running this script, be sure to install the latest version of Runhouse:
# pip install git+https://github.com/run-house/runhouse.git@latest_patch

import runhouse as rh

# ----------- Logging In ------------
# In order to use Runhouse across environments, such as reusing a service from inside a Colab
# or loading secrets or configs into a remote environment, you'll need to create an account and log in.
# You don't need to create an account or log in if you only plan to use Runhouse's APIs in a single
# environment, and don't plan to share resources with others.

# Run this code wherever your cloud credentials are already saved, such as your laptop.
# Follow the prompts to log in. If this is your first time logging in, you should probably upload your secrets,
# but none of the other prompts will have any effect.
rh.login(upload_secrets=True)

# You can do this interactively by running the following in your command line:
# `runhouse login`

# ----------- Setting Config Options ------------
# Runhouse stores user configs both locally in `~/.rh/config.yaml` and remotely in the Runhouse database.
# This allows you to preserve your same config across environments. Some important configs to consider setting:

# Whether to use spot instances (cheaper but can be reclaimed at any time) by default.
# Note that this is False by default because you'll need to request spot quota from the
# cloud providers to use spot instances. You can override this setting in the cluster factory constructor.
rh.configs.set('use_spot', False)

# Where to save and load named resources by default. "RNS" is Runhouse's DNS-like Resource Naming System, and using it
# allows you to access your resources from any environment and share them with others. If you choose to store your
# resources locally, they'll be saved to the "rh/" directory inside your git root directory, which makes them accessible
# from the current environment and to your collaborators using the same git repo (once they pull down your changes).
# You can override this on a per-resource basis via the `save_to` and `load_from` arguments to resource constructors.
# Note that you much create a Runhouse account and log in before you can use RNS.
rh.configs.set('save_to', ['rns', 'local'])
rh.configs.set('load_from', ['rns', 'local'])

# You can view the resources you've stored in RNS in the Runhouse UI: https://api.run.house/dashboard/?option=created

# Clusters can start and stop dynamically to save money. If you set autostop = 10, the cluster will terminate after
# 10 minutes of inactivity. If you set autostop = -1, the cluster will stay up indefinitely. After the cluster
# terminates, if you call a Send which is on that cluster, the Send will automatically start the cluster again.
# You can also call `cluster.keep_warm(autostop=-1) to control this for an existing cluster.
rh.configs.set('default_autostop', -1)

# You can set your default Cloud provider if you have multiple Cloud accounts set up locally. If you set it
# to 'cheapest', SkyPilot will select the cheapest provider for your desired hardware (including spot pricing,
# if enabled). You can set this to 'aws', 'gcp', or 'azure' too.
rh.configs.set('default_provider', 'cheapest')

# Now that you've changed some configs, you probably want to save them to Runhouse to access them elsewhere.
rh.configs.upload_defaults()

# ----------- Checking your Cloud Credentials ------------
# To check whether you have cloud credentials properly set up in your environment, run:
# `sky check` in your command line.

# ----------- Viewing RPC logs ------------
# We will set up proper log persistence and steaming shortly, but in the meantime, if you need to see the
# logs for the sends on a particular cluster, you can ssh into the cluster with `ssh <cluster name>` and
# 'screen -r' (*and use `control A+D` to exit. If you control-C you will stop the server*).
# The server runs inside that screen instance, so logs are written to there.

# ----------- Restarting the RPC Server ------------
# Sometimes the RPC server will crash, or you'll update a package that the server has already imported.
# In those cases, you can try to restart just the server (~20 seconds) to save yourself the trouble of nuking
# and reallocating the hardware itself (minutes). You can do this by `my_cluster.restart_grpc_server()`.