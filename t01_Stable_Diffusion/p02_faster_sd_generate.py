import runhouse as rh
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch


def sd_generate_pinned(prompt, num_images=1, steps=100, guidance_scale=7.5,
                       model_id='stabilityai/stable-diffusion-2-base',
                       dtype=torch.float16, revision="fp16"):
    pipe = rh.get_pinned_object(model_id)
    if pipe is None:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, revision=revision).to("cuda")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)  # Apparently works better for dreambooth
        rh.pin_to_memory(model_id, pipe)
    return pipe(prompt, num_images_per_prompt=num_images,
                num_inference_steps=steps, guidance_scale=guidance_scale).images


if __name__ == "__main__":
    gpu = rh.cluster(name='rh-a10x', instance_type='A100:1')  # On GCP and Azure
    # gpu = rh.cluster(name='rh-a10x', instance_type='g5.2xlarge', provider='aws')  # On AWS
    generate_gpu = rh.send(fn=sd_generate_pinned, hardware=gpu, name='sd_generate').save()
    my_prompt = 'A hot dog made of matcha powder.'
    images = generate_gpu(my_prompt, num_images=4, steps=50)
    [image.show() for image in images]

    # You can find more techniques for speeding up Stable Diffusion here:
    # https://huggingface.co/docs/diffusers/optimization/fp16
