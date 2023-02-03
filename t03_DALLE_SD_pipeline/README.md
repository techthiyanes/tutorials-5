# 👩‍🚀 Portability - DALL-E to SD img2img from Notebook to Inference Service

I've heard that you get better images if you generate a seed image with DALL-E
mini and then use that as the input to Stable Diffusion Img2Img. This is
totally trivial to do with Runhouse, so let's try it out!

## Table of Contents
- [Quick Setup](#00-quick-setup)
- [DALL-E to SD img2img in a Notebook](#01-experimenting-with-dall-e-to-sd-img2img-in-a-notebook)
- [DALL-E to SD img2img Inference Service](#02-dall-e-to-sd-img2img-inference-service)

## 00 Quick Setup

[//]: # (TODO Get rid of this and link out to getting started)

This step is optional, but creating a Runhouse account allows us to 
conveniently jump into a Colab with our credentials and resources at the ready.
See [Getting Started](../00_Getting_Started/) for more details
about logging in and running in notebooks.

This script will show you how to login and set a few default configurations. 
The code is meant to be read in Github, or run locally on your laptop with:
```commandline
python p00_logging_in.py
```

## 01 Experimenting with DALL-E to SD img2img in a Notebook

Kakaobrain just open-sourced Karlo, a full DALL-E reproduction, so we'll use that to generate our seed 
images. This tutorial will help us play around with this idea in a notebook setting.

[//]: # (TODO Make more visible)
This notebook can be run locally with Jupyter or in Colab. Please refer to the
[Colab](https://colab.research.google.com/github/run-house/tutorials/blob/main/t03_DALLE_SD_pipeline/p01_Colab_Dalle_to_SD_img2img.ipynb) for a code walkthrough. 
If you're running locally, feel free to skip the login steps at the beginning of the notebook.


## 02 DALL-E to SD img2img Inference Service

Spoiler alert: it works! Let's turn this into a service that we can call from anywhere. This 
tutorial demonstrates how easy it is to take resources you created in a notebook and reuse them in
the real world, without having to copy and paste or refactor your code.

>**Note**:
If you ran the previous tutorial in Colab, you needed to be logged in for the
resources to be available for this tutorial to use. If you were not logged in
or didn't run in Colab, you can also run the following scripts in the
appendices to create the resources you need.  
&emsp;`python x01a_dalle_generate.py`  
&emsp;`python x01b_sd_img2img.py `

The following function sets up Runhouse callables for the `sd_img2img_generate`
and `karlo_generate` from the Colab tutorial above, and returns the base and
diffused images that these functions generate.

```python
def dalle2sd_pipeline(prompt, num_dalle_images=1, num_sd_images=1, dalle_kwargs={}, sd_kwargs={}):
    sd_img2img_generate_gpu = rh.send(name='sd_img2img_generate')
    karlo_generate = rh.send(name='karlo_generate')
    base_images = karlo_generate(prompt, num_images=num_dalle_images, **dalle_kwargs)
    diffused_images = sd_img2img_generate_gpu(prompt, base_images, num_images=num_sd_images, **sd_kwargs)
    return base_images + diffused_images
```

To retrieve the images with a prompt:

```python
my_prompt = 'portrait of Harrison Ford eating a luscious Christmas ham'
rh_logo_sd_images = dalle2sd_pipeline(my_prompt, num_dalle_images=2, num_sd_images=2)
[image.show() for image in rh_logo_sd_images]
```

![](../assets/t03/p02_output.png)
