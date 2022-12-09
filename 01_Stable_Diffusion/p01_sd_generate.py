import runhouse as rh
from diffusers import StableDiffusionPipeline
import torch

# This function simply takes a prompt (and a few settings), loads the stable diffusion model, feeds the
# prompt into the model, and returns the resulting images. Note that it takes a while to run the first time,
# as it needs to download and locally save the model.
def sd_generate(prompt, num_images=1,
                steps=100, guidance_scale=7.5,
                model_id='stabilityai/stable-diffusion-2-base'):
    # Note, you may need to visit https://huggingface.co/stabilityai/stable-diffusion-2-base to opt-into using the model
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision='fp16').to('cuda')
    return pipe([prompt] * num_images, num_inference_steps=steps, guidance_scale=guidance_scale).images

if __name__ == "__main__":
    # Let's define a cluster that we want to send our function to.
    gpu = rh.cluster(name='rh-v100', instance_type='V100:1', provider='cheapest', use_spot=True)
    # Note that we could also just pass 'rh-v100' to the send function to use a builtin hardware configuration. You
    # can see the other builtins by running `rh.ls('^')`, and inspect the parameters by running
    # `rh.load('^rh-8-cpu', instantiate=False)`.
    generate_gpu = rh.send(fn=sd_generate, hardware=gpu,
                           reqs=['./', 'diffusers'], load_secrets=True,
                           name='sd_generate', save_to=['rns'])
    rh_prompt = 'A digital illustration of a woman running on the roof of a house.'
    images = generate_gpu(rh_prompt, num_images=4, steps=50)
    [image.show() for image in images]
