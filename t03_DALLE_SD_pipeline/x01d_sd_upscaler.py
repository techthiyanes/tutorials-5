import runhouse as rh
from diffusers import StableDiffusionUpscalePipeline
import torch


def sd_upscaler_generate(prompt, base_images, num_images=1,
                         steps=50, guidance_scale=7.5, attention_slicing=None,
                         model_id="stabilityai/stable-diffusion-x4-upscaler"):
    # Here we're using Runhouse's object pinning to hold the model in GPU memory. See p01a for more details.
    pipe = rh.get_pinned_object(model_id + 'upscaler')
    if pipe is None:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        pipe = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16).to('cuda')
        if attention_slicing is not None:
            pipe.enable_attention_slicing(attention_slicing)
        # pipe.enable_xformers_memory_efficient_attention = True
        rh.pin_to_memory(model_id + 'upscaler', pipe)
    torch.no_grad()
    ret = []
    for image in base_images:
        ret = ret + pipe(prompt,
                         image=image,
                         num_images_per_prompt=num_images,
                         num_inference_steps=steps,
                         guidance_scale=guidance_scale).images
    return ret

if __name__ == "__main__":
    dalle_generate = rh.function(name='karlo_generate')
    my_prompt = 'Shrek giving a TED talk to an audience of Minions.'
    base_images = dalle_generate(my_prompt)
    base_images[0].show()

    # v100_gpu = rh.cluster(name='rh-v100', instance_type='V100:1', provider='cheapest')
    sd_upscaler_generate_gpu = rh.function(fn=sd_upscaler_generate, system=dalle_generate.system,
                                       env=['local:./'], name='sd_upscaler_generate')

    sd_upscaled_images = sd_upscaler_generate_gpu(my_prompt, base_images, guidance_scale=7.5,
                                                  num_images=2, steps=50)
    [image.show() for image in sd_upscaled_images]
