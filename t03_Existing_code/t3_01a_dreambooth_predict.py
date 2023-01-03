import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from accelerate import Accelerator
import runhouse as rh

# We could just use this as the generate function, but the new function below adds support for checkpoints
# from t01_Stable_Diffusion.t1_01a_object_pinning import sd_generate_with_simple_pinning

def sd_generate_pinned(prompt, num_images=1, steps=100,
                       guidance_scale=7.5, model_id='stabilityai/stable-diffusion-2-base',
                       dtype=torch.float16, revision="fp16", checkpoint=None):
    pin = model_id + '/' + checkpoint if checkpoint else model_id
    pipe = rh.get_pinned_object(pin)
    if pipe is None:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, revision=revision).to("cuda")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        if checkpoint:
            accelerator = Accelerator()
            unet, text_encoder = accelerator.prepare(pipe.unet, pipe.text_encoder)
            accelerator.load_state(model_id + '/' + checkpoint)
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                unet=accelerator.unwrap_model(unet),
                text_encoder=accelerator.unwrap_model(text_encoder),
                torch_dtype=dtype
            ).to('cuda')
        rh.pin_to_memory(pin, pipe)
    return pipe([prompt] * num_images, num_inference_steps=steps, guidance_scale=guidance_scale).images


if __name__ == "__main__":
    a100 = rh.cluster(name='rh-a100')
    # a100.flush_pins()
    generate_dreambooth = rh.send(fn=sd_generate_pinned, hardware=a100)
    my_prompt = "sks person riding a rhinoceros, wearing safari attire in the African savanna"
    model_path = 'dreambooth/output'
    images = generate_dreambooth(my_prompt,
                                 model_id=model_path,
                                 num_images=4, guidance_scale=7.5,
                                 steps=100)
    [image.show() for image in images]
