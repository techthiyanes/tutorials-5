<h1 align="center">🏃‍♀️Runhouse 🏠 Tutorials</h1>
<p align="center">
  <a href="https://runhouse-docs.readthedocs-hosted.com/en/latest/"> 
    <img alt="Documentation" src="https://readthedocs.com/projects/runhouse-docs/badge/?version=latest">
  </a>
 
  <a href="https://join.slack.com/t/runhouse/shared_invite/zt-1j7pwsok1-vQy0Gesh55A2fPyyEVq8nQ"> 
    <img alt="Join Slack" src="https://img.shields.io/badge/Runhouse-Join%20Slack-fedcba?logo=slack">
  </a>
</p>



**tldr;** PyTorch lets you send any Python code or data `.to(device)`, so 
why can't you do `my_fn.to('a_gcp_a100')` or `my_table.to('parquet_in_s3')`? 
Runhouse allows just that: send code and data to any of your compute or 
data infra, all in Python, and continue to use them eagerly exactly as they were. 
Take a look at this [Stable Diffusion example](t01_Stable_Diffusion/t1_01_sd_generate.py) - 
it lets you run Stable Diffusion inferences on your own cloud GPU in minutes,
but there's no magic yaml, DSL, or "submitting for execution." 
And because it's not stateless, we can pin the model to GPU memory 
([tutorial 1-01a](t01_Stable_Diffusion/t1_01a_object_pinning.py)), and get ~2.5s/image 
inference time before any compilation. There's much more, like being able to access your 
code, compute, and data from anywhere with a Python interpreter and an internet connection, 
or sharing them with collaborators, so let's jump in.

These tutorials introduce you to 
the tools and usage patterns of Runhouse. We've devised them
to chart a fun path through our features, but you're 
welcome to hop around if you prefer.

1. [Runhouse Basics 🐣 - Fun with Stable Diffusion and DALL-E](t01_Stable_Diffusion/)
1. [A Runhouse Pipeline 👩‍🔧 - Fine-tuning BERT and Deploying](t02_BERT_fine_tuning/)
1. [Fancy Runhouse 🧑‍🎨 - Dreambooth or TIMM in <10 Minutes](t03_Existing_code/)
1. [Online Runhouse 👩‍💻 - DLRM Online Training and Deployment](t04_Online_learning/) (~Q1)

If you would be so kind, we would love if you could have a notes doc open
as you install and try Runhouse for the first time. Your first impressions, 
pain points, and highlights are very valuable to us.

### ⏰ If you only have 10 minutes:
* See our dreambooth tutorials ([training](https://github.com/run-house/tutorials/blob/main/t03_Existing_code/t3_01_dreambooth_train.py), 
[inference](https://github.com/run-house/tutorials/blob/main/t03_Existing_code/t3_01a_dreambooth_predict.py)), 
we think it's the easiest way anywhere to run dreambooth on your own cloud GPU 
(for managed dreambooth, check out [Modal Labs's dreambooth](https://modal.com/docs/guide/ex/dreambooth_app) or 
[StableBoost](http://stableboost.ai/)).
* See our BERT fine-tuning pipeline example, [here](./t02_BERT_fine_tuning).

## 🚨 Caution: This is an Unstable Alpha 🚨

Runhouse is heavily under development and unstable. We are quite 
a ways away from having our first stable release. We are sharing
it privately with a few select people to collect feedback, and
expect a lot of things to break off the bat.

## 🙋‍♂️ Getting Help

Please request to join our 
[slack workspace here](https://join.slack.com/t/runhouse/shared_invite/zt-1j7pwsok1-vQy0Gesh55A2fPyyEVq8nQ), 
or email us, or create an issue.

## 🕵️‍♀️ Where is the compute?

Runhouse is not managed compute or data. All of the compute and data in Runhouse
lives within your own infra and cloud provider accounts. As such, you'll need 
credentials with at least one of AWS, GCP, or Azure to try these tutorials,
as well as quota approval for GPU resources 
([See here](https://skypilot.readthedocs.io/en/latest/reference/quota.html) 
for more on this). If you're looking for a managed compute 
experience without a cloud account, we'd recommend our friends at 
[Modal Labs](https://modal.com/) or [Anyscale](https://anyscale.com/). At some point
we plan to support them as compute providers in Runhouse as well. Other sources of compute,
such as on-prem or Kubernetes, are also on the roadmap (likely through our friends at 
[SkyPilot](https://skypilot.readthedocs.io/)).

## 🔌 Installation

⚠️ On Apple M1 or M2 machines ⚠️, you will need to install grpcio with conda
before you install Runhouse - more specifically, before you install Ray. 
If you already have Ray installed, you can skip this.
[See here](https://docs.ray.io/en/master/ray-overview/installation.html#m1-mac-apple-silicon-support) 
for how to install grpc properly on Apple silicon. You'll only know if you did
this correctly if you run `ray.init()` in a Python interpreter. If you're 
having trouble with this, let us know.

Runhouse is not on Pypi, but we maintain a semi-stable branch in
Github. It can be installed with: 

`pip install git+https://github.com/run-house/runhouse.git@latest_patch`

As we apply patches we may update this version number. We will
notify you if we want you to upgrade your installation.

## 🔒 Creating an Account for Secrets and Resources

tldr; See this video (WIP) about what features creating an account enables.

Using Runhouse with only the OSS Python package is perfectly fine, and it
will use your cloud credentials saved into locations like `~/.aws/credentials`
or `~/.config/gcloud` by default. Right now we support AWS, GCP, Azure, and
Hugging Face credentials. However, you can unlock some very unique portability 
features by creating an account on [api.run.house](https://api.run.house) and 
saving your secrets, configs, and resources there. Think of the OSS-only 
experience as akin to Microsoft Office, while creating an account will
make your cloud resources sharable and accessible from anywhere like Google Docs. 
For example, if you store your secrets or resources in the Runhouse cloud, you can open a Google Colab, call 
`runhouse login`, and all of your secrets or resources will be available in 
the environment. 

**Note that your Runhouse account is not some managed or cloud
service; all of your compute and data resources are still in the cloud.** The
"resources" stored in Runhouse are strictly metadata that we've cleverly devised to 
allow this multiplayer sharing and portability.

Runhouse uses Hashicorp Vault (an industry standard) to store secrets, 
and provides a web service to allow you access your resources across 
multiple machines (more on this in tutorial 1). To create an account, 
visit [api.run.house](https://api.run.house),
or simply call `runhouse login` from the command line (or 
`rh.login()` from Python). This will link you to a page to 
login with your Google account and generate a token, which you can then
input into the command line or Python prompt. It will then offer for you
to upload your secrets, which will collect them from the local 
environment and store them in Vault. You only need to do this the first time
you log in or your secrets change. It will offer to upload your config as well,
which contains certain options like the default cloud provider or autostop 
time, but you can probably just ignore this for now. We provide reasonable 
defaults in Runhouse, such as selecting the cheapest cloud provider (for which
you have appropriate credentials) for the given hardware.

## ✈️ Checking and Managing your Clusters with SkyPilot

Runhouse uses [SkyPilot](https://skypilot.readthedocs.io/en/latest/) for 
much of the heavy lifting with the cloud providers. SkyPilot is a Python
library that provides a unified interface for launching and managing
cloud instances. We love it and you should give them a Github star 🤗.

To check that your cloud credentials are set up correctly, run `sky check`
in your command line. This will confirm which cloud providers are ready to
use, and will give detailed instructions if any setup is incomplete.

All Runhouse compute are SkyPilot clusters right now, so you should use 
their CLI to do basic management operations. Some important ones are:
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

## 👷‍♀️ Contributing

We welcome contributions! Please contact us if you're interested. There 
is so much to do.
