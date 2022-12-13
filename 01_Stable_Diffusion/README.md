# Getting Started 🐣 - Fun with Stable Diffusion and Dalle-Mini

_Watch time: ~10 minutes_

Runhouse is nothing more than an accessibility and sharing
layer into your own cloud compute and data resources. Let's
start with a simple example of how to use Runhouse to make an
easy and inexpensive way to play with Stable Diffusion.

## 01 A Stable Diffusion service on AWS in 5 lines of code

Video (WIP)

## 02 Calling your service from anywhere, and some fun with GPT-2

Video (WIP)

[This tutorial is meant to be run from Colab.](https://colab.research.google.com/github/run-house/tutorials/blob/main/01_Stable_Diffusion/p02_Colab_Stable_Diffusion.ipynb)

## 03 Putting Dalle-Mini on a TPU

Video (WIP)

## 04 Experimenting with Dalle-Mini->SD-Image-to-Image in Colab

Video (WIP)

## 05 Deploying DM->SD as a multi-hop service (and multi-cloud!)

Video (WIP)

## 06 Pinning objects to GPU memory to improve performance

Video (WIP)

## 07 Sharing your service with friends

Video (WIP)

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
rh.Secrets.extract_and_upload_secrets()
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