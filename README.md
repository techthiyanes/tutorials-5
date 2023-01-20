<h1 align="center">🏃‍♀️Runhouse 🏠 Tutorials</h1>


**tldr;** PyTorch lets you send any Python code or data `.to(device)`, so 
why can't you do `my_fn.to('a_gcp_a100')` or `my_table.to('parquet_in_s3')`? 
Runhouse allows just that: send code and data to any of your compute or 
data infra, all in Python, and continue to use them eagerly exactly as they were. 
Take a look at this [Stable Diffusion example](t01_Stable_Diffusion/p01_sd_generate.py) - 
it lets you run Stable Diffusion inferences on your own cloud GPU in minutes,
but there's no magic yaml, DSL, or "submitting for execution." 
And because it's not stateless, we can pin the model to GPU memory 
([tutorial 1-01a](t01_Stable_Diffusion/p02_faster_sd_generate.py)), and get super fast inference 
(~1.5s/image for SD-2 on an A100 before any compilation!). There's much more, like being able to access your 
code, compute, and data from anywhere with a Python interpreter and an internet connection, 
or sharing them with collaborators, so let's jump in.

These tutorials introduce you to 
the tools and usage patterns of Runhouse. We've devised them
to chart a fun path through our features, but you're 
welcome to hop around if you prefer.

1. [🐣 Runhouse Basics - Fun with Stable Diffusion and FLAN-T5](t01_Stable_Diffusion/)
1. [🧑‍🎨 Fancy Runhouse - Dreambooth in <10 Minutes](t02_Dreambooth/)
1. [👩‍🚀 Portability - DALL-E to SD img2img from Notebook to Inference Service](t03_DALLE_SD_pipeline/)
1. [👩‍👩‍👧‍👧 [WIP] Distributed - Large Model Training and Inference](t04_Distributed/)
1. [👩‍🔧 [WIP] Pipelines - Fine-tuning BERT and Deploying](t05_BERT_pipeline/)

[//]: # (1. [👩‍💻 [WIP] Online Learning - DLRM Online Training and Deployment]&#40;t04_Online_learning/&#41; &#40;~EOQ1&#41;)

[//]: # (1. [🤝 [WIP] Sharing - Collaborate on a model with your friends]&#40;t05_Sharing/&#41; &#40;~EOQ1&#41;)

If you would be so kind, we would love if you could have a notes doc open
as you install and try Runhouse for the first time. Your first impressions, 
pain points, and highlights are very valuable to us.

### 🛫 Installation & Setup

See [getting started](https://github.com/run-house/runhouse#-getting-started).

tldr;
```commandline
pip install git+https://github.com/run-house/runhouse.git@latest_patch
sky check
# Optionally, for portability (e.g. Colab):
runhouse login
```

### ⏰ If you only have 10 minutes:
* Take a look at the [Stable Diffusion example](t01_Stable_Diffusion/p01_sd_generate.py) 
  to understand how Runhouse allows you to interact with remote compute. 
  * The [FLAN-T5-XL](t01_Stable_Diffusion/p03_flan_t5_xl_generate.py) example then shows how we can easily reuse hardware and services. 
* See our dreambooth tutorials ([training](./t02_Dreambooth/p01_dreambooth_train.py), 
[inference](./t02_Dreambooth/p01a_dreambooth_predict.py)). 
We think they're the easiest way anywhere to run dreambooth on your own cloud GPU 
(for managed dreambooth, check out [Modal Labs's dreambooth](https://modal.com/docs/guide/ex/dreambooth_app) or 
[StableBoost](http://stableboost.ai/)).
* See how to launch a Gradio app to run CLIP Interrogator (generate Stable Diffusion prompts from images), 
[here](t02_Dreambooth/p02_gradio_clip_interrogator.py).
* See our BERT fine-tuning pipeline example, [here](./t05_BERT_pipeline).

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
lives within your own infra and cloud accounts. As such, you'll need 
credentials with at least one of AWS, GCP, or Azure (and Lambda Labs and IBM coming soon) 
to try these tutorials, as well as quota approval for GPU resources 
([See here](https://skypilot.readthedocs.io/en/latest/reference/quota.html) 
for more on this). If you're looking for a managed compute 
experience without a cloud account, we'd recommend our friends at 
[Modal Labs](https://modal.com/) or [Anyscale](https://anyscale.com/). At some point
we plan to support them as compute providers in Runhouse as well. Other sources of compute,
such as on-prem or Kubernetes, are also on the roadmap (likely through our friends at 
[SkyPilot](https://skypilot.readthedocs.io/)).

## 👷‍♀️ Contributing

We welcome contributions! Please contact us if you're interested. There 
is so much to do.
