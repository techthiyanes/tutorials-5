import runhouse as rh
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch


def sd_img2img_generate(prompt, base_images, num_images=1,
                        steps=50, strength=0.75, guidance_scale=7.5,
                        model_id="stabilityai/stable-diffusion-2-base"):
    # Here we're using Runhouse's object pinning to hold the model in GPU memory. See p06 for more details.
    pipe = rh.get_pinned_object(model_id)
    if pipe is None:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id).to('cuda')
        rh.pin_to_memory(model_id, pipe)
    ret = []
    for image in base_images:
        ret = ret + pipe([prompt] * num_images, init_image=image.resize((512, 512)),
                         num_inference_steps=steps, strength=strength,
                         guidance_scale=guidance_scale).images
    return ret


if __name__ == "__main__":
    v100_gpu = rh.cluster(name='rh-v100', instance_type='V100:1', provider='cheapest')
    sd_img2img_generate_gpu = rh.send(fn=sd_img2img_generate, hardware=v100_gpu,
                                      reqs=['local:./', 'diffusers'],
                                      name='sd_img2img_generate')

    rh_prompt = 'A picture of a woman running above a picture of a house.'
    rh_base_image = Image.open('rh_logo.png').convert("RGB").resize((512, 512))
    rh_logo_sd_images = sd_img2img_generate_gpu(rh_prompt, [rh_base_image],
                                                strength=.75, guidance_scale=7.5,
                                                num_images=4, steps=50)
    [image.show() for image in rh_logo_sd_images]

    # Now may be a good time to check on the memory utilization
    v100_gpu.run(['nvidia-smi'])

    # dm_base_images = dm_generate(rh_prompt, num_images=4)
    # images = generate_gpu(rh_prompt, dm_base_images, num_images=4, steps=50)