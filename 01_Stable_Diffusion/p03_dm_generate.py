import runhouse as rh
from min_dalle import MinDalle
import torch
from PIL import Image
import random


def dm_generate(prompt, num_images_sqrt=1, supercondition_factor=32, is_mega=True, seed=50, top_k=64):
    torch.cuda.empty_cache()
    torch.no_grad()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    dalle = MinDalle(device='cuda', is_mega=is_mega, is_reusable=True, dtype=torch.float16)
    images = dalle.generate_images(prompt, seed=seed, grid_size=num_images_sqrt,
                                   temperature=1, top_k=top_k, supercondition_factor=supercondition_factor)
    del dalle
    images = images.to(torch.uint8).to('cpu').numpy()
    return [Image.fromarray(images[i]) for i in range(num_images_sqrt**2)]


if __name__ == "__main__":
    # This is a bit too small, need to switch to A10G
    gpu = rh.cluster(name='rh-v100', instance_type='V100:1', provider='cheapest', use_spot=False)
    generate__dm_gpu = rh.send(fn=dm_generate, hardware=gpu,
                               reqs=['./', 'min-dalle'], load_secrets=False)
    rh_prompt = 'A hot dog made of matcha powder.'
    images = generate__dm_gpu(rh_prompt, num_images_sqrt=1, seed=random.randint(0, 1000))
    [image.show() for image in images]