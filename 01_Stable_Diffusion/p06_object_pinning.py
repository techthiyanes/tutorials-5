import runhouse as rh
from diffusers import StableDiffusionPipeline
import torch
import ray

def sd_generate_with_simple_pinning(prompt, num_images=1, steps=100,
                                    guidance_scale=7.5, model_id="runwayml/stable-diffusion-v1-5"):
    pipe = rh.get_pinned_object('sd_pipeline')
    if pipe is None:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16").to("cuda")
        rh.pin_to_memory('sd_pipeline', pipe)
    return pipe([prompt] * num_images, num_inference_steps=steps, guidance_scale=guidance_scale).images

def sd_generate_with_multinode_pinning(prompt, num_images=1, steps=100,
                                       guidance_scale=7.5, model_id="runwayml/stable-diffusion-v1-5"):
    pipe_ref = rh.get_pinned_object('sd_pipeline_ray_ref')
    if pipe_ref is None:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16").to("cuda")
        rh.pin_to_memory('sd_pipeline_ray_ref', ray.put(pipe))
    else:
        pipe = ray.get(pipe_ref)
    return pipe([prompt] * num_images, num_inference_steps=steps, guidance_scale=guidance_scale).images

if __name__ == "__main__":
    gpu = rh.cluster(name='rh-v100', instance_type='V100:1', provider='gcp', use_spot=True)
    generate_gpu = rh.send(fn=sd_generate_with_simple_pinning, hardware=gpu,
                           reqs=['reqs:default', 'diffusers'], create=True).send_secrets()

    rh_prompt = 'A digital illustration of a woman running on the roof of a house.'
    images = generate_gpu(rh_prompt, num_images=1, steps=50)
    [image.show() for image in images]
