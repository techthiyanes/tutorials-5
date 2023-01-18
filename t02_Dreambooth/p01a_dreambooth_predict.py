import runhouse as rh

from t01_Stable_Diffusion.p02_faster_sd_generate import sd_generate_pinned

if __name__ == "__main__":
    gpu = rh.cluster(name='rh-a10x', instance_type='A100:1')  # On GCP and Azure
    # gpu = rh.cluster(name='rh-a10x', instance_type='g5.2xlarge', provider='aws')  # On AWS

    generate_dreambooth = rh.send(fn=sd_generate_pinned, hardware=gpu)
    my_prompt = "sks person riding a goat through a field of purple flowers"
    model_path = 'dreambooth/output'
    images = generate_dreambooth(my_prompt,
                                 model_id=model_path,
                                 num_images=4, guidance_scale=7.5,
                                 steps=100)
    [image.show() for image in images]
