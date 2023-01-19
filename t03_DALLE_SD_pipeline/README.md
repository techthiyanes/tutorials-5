# 👩‍🚀 Portability - DALL-E to SD img2img from Notebook to Inference Service

## 00 [Quick setup](../t03_DALLE_SD_pipeline/p00_logging_in.py)

This step is optional, but creating a Runhouse account allows us to 
conveniently jump into a Colab with our credentials and resources at the ready.
See [Tutorial 1 Appendices 01 and 02](../t01_Stable_Diffusion/README.md#appendices) for more details
about logging in and running in notebooks.

This script will show you how to login and set a few default configurations. 
The code is meant to be read in Github, or run locally on your laptop with:
```commandline
python t1_00_quick_setup.py
```
Status: **Working.**

## 01 [Experimenting with DALL-E to SD img2img in a Notebook](https://colab.research.google.com/github/run-house/tutorials/blob/main/t03_DALLE_SD_pipeline/p01_Colab_Dalle_to_SD_img2img.ipynb)

I've heard that you get better images if you generate a seed image with DALL-E mini and then use that as the input to 
Stable Diffusion Img2Img. This is totally trivial to do with Runhouse, so let's try it out!

Kakaobrain just open-sourced Karlo, a full DALL-E reproduction, so we'll use that to generate our seed 
images. This tutorial will help us play around with this idea in a notebook setting
and get a sense of whether it's worth pursuing.

This notebook can be run locally with Jupyter or in 
[Colab]((https://colab.research.google.com/github/run-house/tutorials/blob/main/t03_DALLE_SD_pipeline/p01_Colab_Dalle_to_SD_img2img.ipynb)). 
If you're running locally, feel free to skip the login steps at the beginning of the notebook.

Status: **Working.**

## 02 [DALL-E to SD img2img Inference Service](../t03_DALLE_SD_pipeline/p02_dalle_to_sd_img2img.py)

Spoiler alert: it works! Let's turn this into a service that we can call from anywhere. This 
tutorial demonstrates how easy it is to take resources you created in a notebook and reuse them in
the real world, without having to copy and paste or refactor your code.

If you ran the previous tutorial in Colab, you needed to be logged in for the resources
to be available for this tutorial to use. If you were not logged in or didn't run in Colab, you can also
run the following scripts in the appendices to create the resources you need.
```commandline
python x01a_dalle_generate.py
python x01b_sd_img2img.py 
```
Status: **Working.**