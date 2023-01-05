# Runhouse Basics 🐣 - Fun with Stable Diffusion and DALL-E

_Watch time: ~10 minutes_

Runhouse is nothing more than an accessibility and sharing
layer into your own cloud compute and data resources. Let's
start with a simple example of how to use Runhouse to make an
easy and inexpensive way to play with Stable Diffusion.

## 00 [Quick setup](./t1_00_quick_setup.py)

This script will show you how to login and set a few default configurations. 
The code is meant to be read in Github, or run locally on your laptop with:
```commandline
python t1_00_quick_setup.py
```
Status: **Working.** \

## 01 [A Stable Diffusion service on a cloud V100 in 5 lines of code](./t1_01_sd_generate.py)

We'll start by using Runhouse to experiment with Stable 
Diffusion from your laptop, while the model actually runs on a V100
in the cloud. The code is meant to be read in Github, or run locally on your 
laptop with:
```commandline
python t1_01_sd_generate.py
```
Status: **Working.** \

## 01a [Pinning objects to GPU memory to improve performance](./t1_01a_object_pinning.py)

This tutorial shows how you can achieve high performance serving with 
Runhouse by pinning models to GPU memory, down to ~2.5s/image with Stable Diffusion 2 
(without compilation!). It's meant to be read in Github, or run locally on your laptop with:
```commandline
python t1_01a_object_pinning.py
```
Status: **Working.** \

## 02 [Calling your service from anywhere, and some fun with GPT-2](https://colab.research.google.com/github/run-house/tutorials/blob/main/t01_Stable_Diffusion/t1_02_Colab_Stable_Diffusion.ipynb)

We can call the Stable Diffusion function you ran on a V100 above from Colab
with no setup, installations, or changes. You can access it from anywhere with a python
interpreter and an internet connection.
This tutorial is meant to be [run from Colab.](https://colab.research.google.com/github/run-house/tutorials/blob/main/t01_Stable_Diffusion/t1_02_Colab_Stable_Diffusion.ipynb)
Status: **Working.**

## 03 [Running Karlo (DALL-E) on an A10G/A100](./t1_03_dalle_generate.py)

Status: **Working.**

## 04 [Experimenting with DALL-E->SD-Image-to-Image in Colab](https://colab.research.google.com/github/run-house/tutorials/blob/main/t01_Stable_Diffusion/t1_04_Colab_Dalle_to_SD_img2img.ipynb)

Status: **Working.**

## 05 [Trying Stable Diffusion Img2Img](t1_05_sd_img2img.py)

Status: **Working.**

## 06 [Deploying DALLE->SD as a multi-hop service (and multi-cloud!)](t1_06_dalle_sd_pipeline.py)

Status: **Working.**

## 07 [A personal text generation service with Flan-T5-XL](./t1_07_flan_t5_xl_generate.py)

Status: **Working.**

## 08 Sharing your service with friends

Status: WIP

# Appendices

## Saving and Loading Secrets from Vault

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