import runhouse as rh

from t01_Stable_Diffusion.p02_faster_sd_generate import sd_generate_pinned

if __name__ == "__main__":
    gpu = rh.cluster(name='rh-a10x').up_if_not()
    generate_dreambooth = rh.send(fn=sd_generate_pinned, hardware=gpu)
    my_prompt = "sks person riding a goat through a field of purple flowers"
    model_path = 'dreambooth/output'
    images = generate_dreambooth(my_prompt,
                                 model_id=model_path,
                                 num_images=4, guidance_scale=7.5,
                                 steps=100)
    [image.show() for image in images]
