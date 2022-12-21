import runhouse as rh
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch

from p03_dm_generate import dm_generate


def sd_img2img_generate(prompt, base_images, num_images=1,
                        steps=100, strength=0.75, guidance_scale=7.5,
                        model_id="stabilityai/stable-diffusion-2-base"):
    torch.cuda.empty_cache()
    torch.no_grad()
    sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id)
    sd_pipe = sd_pipe.to('cuda')
    ret = []
    for image in base_images:
        ret = ret + sd_pipe([prompt] * num_images, init_image=image.resize((512, 512)),
                            num_inference_steps=steps, strength=strength,
                            guidance_scale=guidance_scale).images
    return ret


if __name__ == "__main__":
    v100_gpu = rh.cluster(name='rh-v100', instance_type='V100:1', provider='cheapest', use_spot=False)
    sd_img2img_generate_gpu = rh.send(fn=sd_img2img_generate, hardware=v100_gpu,
                                      reqs=['./', 'diffusers'],
                                      load_secrets=True,
                                      load_from=['rns'], save_to=['rns'],
                                      name='sd_img2img_generate')

    rh_prompt = 'A picture of a woman running above a picture of a house.'
    rh_base_image = Image.open('rh_logo.png').convert("RGB").resize((512, 512))
    rh_logo_sd_images = sd_img2img_generate_gpu(rh_prompt, [rh_base_image],
                                                strength=.5, guidance_scale=7.5,
                                                num_images=4, steps=50)
    [image.show() for image in rh_logo_sd_images]

    # dm_base_images = dm_generate(rh_prompt, num_images=4)
    # images = generate_gpu(rh_prompt, dm_base_images, num_images=4, steps=50)
