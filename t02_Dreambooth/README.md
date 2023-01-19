# 🧑‍🎨 Fancy Runhouse - Dreambooth in <10 Minutes

We want Runhouse to be aggressively zero-lift. Whatever code
structure, whatever execution environment (notebook, 
orchestrator, data app, CI/CD), you should be able to do something fun
and interesting with Runhouse in minutes, not months.

## 01 Dreambooth

Dreambooth is a popular app that lets you fine-tune Stable Diffusion on your
own images so you can reference your new concept in Stable Diffusion inferences.
Hugging Face published a [great tutorial](https://huggingface.co/blog/dreambooth),
but it's never easy to set up on your own hardware, so various Colabs are circulating
to help people get started. We can run way faster on our own GPU, and we don't even 
need to clone down the repo! This tutorial shows how to send a function to your 
hardware from just a github url pointing to the function.

It also shows you basics of the data side of Runhouse, by:
1) Creating an `rh.folder` with the training images and then sending it to the cluster with
`folder.to(my_gpu)`. 
2) Similarly, sending the folder containing the trained model to blob storage.

This is the tip of the iceberg, and there's much more about data on the way.

Status:
* Training: **Working.**
* Inference: **Working.**

## 02 [Running Huggging Face Spaces](./p02_gradio_clip_interrogator.py)

Writing prompts is hard. Luckily, CLIP Interrogator can take images and generate
Stable Diffusion prompts from them. There's a popular [Hugging Face Space for CLIP 
Interrogator](https://huggingface.co/spaces/pharma/CLIP-Interrogator), but it'd run 
faster on our own GPU. This tutorial shows you how easy it is to take any gradio app 
and send it to your GPU, tunneled into your browser.

Status: **Working.**

# Appendices

## 01: [Dreambooth in Colab](https://colab.research.google.com/github/run-house/tutorials/blob/main/t02_Dreambooth/x01_Colab_Dreambooth.ipynb)

If you prefer to read or run this tutorial in Colab, you can do so 
[here](https://colab.research.google.com/github/run-house/tutorials/blob/main/t02_Dreambooth/x01_Colab_Dreambooth.ipynb).
See [Tutorial 1 Appendices 01 and 02](../t01_Stable_Diffusion/README.md#appendices) for more details
about logging in and running in notebooks.

Status: **Working.**