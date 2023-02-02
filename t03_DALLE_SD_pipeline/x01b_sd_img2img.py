import runhouse as rh
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch


def sd_img2img_generate(prompt, base_images, num_images=1,
                        steps=50, strength=0.75, guidance_scale=7.5,
                        model_id="stabilityai/stable-diffusion-2-base"):
    # Here we're using Runhouse's object pinning to hold the model in GPU memory. See p01a for more details.
    # We're changing the name of the pinned model to avoid a collision if reusing the cluster from p01a.
    pipe = rh.get_pinned_object(model_id + 'im2img')
    if pipe is None:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id).to('cuda')
        rh.pin_to_memory(model_id + 'im2img', pipe)
    ret = []
    for image in base_images:
        ret = ret + pipe([prompt] * num_images, init_image=image.resize((512, 512)),
                         num_inference_steps=steps, strength=strength,
                         guidance_scale=guidance_scale).images
    return ret


if __name__ == "__main__":
    gpu = rh.cluster(name='rh-a10x').up_if_not()
    sd_img2img_generate_gpu = rh.send(fn=sd_img2img_generate, hardware=gpu, name='sd_img2img_generate').save()

    rh_prompt = 'A picture of a woman running above a picture of a house.'
    rh_base_image = Image.open('rh_logo.png').convert("RGB")

    # This takes ~3 mins to run the first time to download the model, and after that should only take ~1 sec per image.
    rh_logo_sd_images = sd_img2img_generate_gpu(rh_prompt, [rh_base_image],
                                                strength=.75, guidance_scale=7.5,
                                                num_images=4, steps=50)
    [image.show() for image in rh_logo_sd_images]

    # Now may be a good time to check on the memory utilization
    gpu.run(['nvidia-smi'])
