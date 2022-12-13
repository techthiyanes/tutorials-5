import runhouse as rh
from diffusers import StableDiffusionPipeline
import torch
import ray

def sd_generate_with_simple_pinning(prompt, num_images=1, steps=100,
                                    guidance_scale=7.5, model_id='stabilityai/stable-diffusion-2-base'):
    pipe = rh.get_pinned_object(model_id)
    if pipe is None:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16").to("cuda")
        rh.pin_to_memory(model_id, pipe)
    return pipe([prompt] * num_images, num_inference_steps=steps, guidance_scale=guidance_scale).images

def sd_generate_with_multinode_pinning(prompt, num_images=1, steps=100,
                                       guidance_scale=7.5, model_id='stabilityai/stable-diffusion-2-base'):
    pipe_ref = rh.get_pinned_object(model_id+'ray_ref')
    if pipe_ref is None:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16").to("cuda")
        rh.pin_to_memory(model_id+'ray_ref', ray.put(pipe))
    else:
        pipe = ray.get(pipe_ref)
    return pipe([prompt] * num_images, num_inference_steps=steps, guidance_scale=guidance_scale).images

if __name__ == "__main__":
    gpu = rh.cluster(name='rh-v100', instance_type='V100:1', provider='cheapest', use_spot=True)
    generate_gpu = rh.send(fn=sd_generate_with_simple_pinning, hardware=gpu,
                           reqs=['./', 'diffusers'], load_secrets=True)
    rh_prompt = 'A hot dog made of matcha powder.'
    images = generate_gpu(rh_prompt, num_images=4, steps=50)
    [image.show() for image in images]

    # You can find more techniques for speeding up Stable Diffusion here:
    # https://huggingface.co/docs/diffusers/optimization/fp16
