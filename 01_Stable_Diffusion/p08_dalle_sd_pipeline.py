import runhouse as rh
from PIL import Image


def dalle2sd_pipeline(prompt, num_dalle_images=1, num_sd_images=1, dalle_kwargs={}, sd_kwargs={}):
    sd_img2img_generate_gpu = rh.send(name='sd_img2img_generate', load_from=['local'])
    karlo_generate = rh.send(name='karlo_generate', load_from=['local'])
    base_images = karlo_generate(prompt, num_images=num_dalle_images, **dalle_kwargs)
    diffused_images = sd_img2img_generate_gpu(prompt, base_images, num_images=num_sd_images, **sd_kwargs)
    return base_images + diffused_images


if __name__ == "__main__":
    my_prompt = 'portrait of Harrison Ford eating a luscious Christmas ham'
    rh_logo_sd_images = dalle2sd_pipeline(my_prompt, num_dalle_images=2, num_sd_images=2)
    [image.show() for image in rh_logo_sd_images]

    # It would be trivial to add upsampling here too!
    # https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler
