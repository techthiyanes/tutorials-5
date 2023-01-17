import runhouse as rh

from t01_Stable_Diffusion.t1_01a_object_pinning import sd_generate_pinned

if __name__ == "__main__":
    a100 = rh.cluster(name='rh-a100')
    # a100.flush_pins()
    generate_dreambooth = rh.send(fn=sd_generate_pinned, hardware=a100)
    my_prompt = "sks person riding a goat through a field of purple flowers"
    model_path = 'dreambooth/output'
    images = generate_dreambooth(my_prompt,
                                 model_id=model_path,
                                 num_images=4, guidance_scale=7.5,
                                 steps=100)
    [image.show() for image in images]
