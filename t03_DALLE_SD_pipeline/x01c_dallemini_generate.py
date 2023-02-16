import runhouse as rh
from min_dalle import MinDalle
import torch
from PIL import Image
import random


def dm_generate(prompt, num_images_sqrt=1, supercondition_factor=32, is_mega=True, seed=50, top_k=64):
    dalle = rh.get_pinned_object('dalle-mini')
    if dalle is None:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        dalle = MinDalle(device='cuda', is_mega=is_mega, is_reusable=True, dtype=torch.float16)
        rh.pin_to_memory('dalle-mini', dalle)
    images = dalle.generate_images(prompt, seed=seed, grid_size=num_images_sqrt,
                                   temperature=1, top_k=top_k, supercondition_factor=supercondition_factor)
    images = images.to(torch.uint8).to('cpu').numpy()
    return [Image.fromarray(images[i]) for i in range(num_images_sqrt**2)]


if __name__ == "__main__":
    gpu = rh.cluster(name='rh-a10x') if rh.exists('rh-a10x') else rh.cluster(name='rh-a10x', instance_type='A100:1')
    generate_dm_gpu = rh.function(fn=dm_generate, system=gpu, reqs=['./', 'min-dalle'], name='dm_generate').save()

    my_prompt = 'A hot dog made of matcha powder.'
    images = generate_dm_gpu(my_prompt, num_images_sqrt=1, seed=random.randint(0, 1000))
    [image.show() for image in images]