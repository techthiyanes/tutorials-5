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
    # Single A10G is only available on AWS, but you can use an A100 on GCP or Azure instead.
    # See this helpful guide to cloud GPUs for more details: https://www.paperspace.com/gpu-cloud-comparison
    gpu = rh.cluster(name='rh-a10g', instance_type='A10G:1')
    generate_dm_gpu = rh.send(fn=dm_generate, hardware=gpu,
                              reqs=['./', 'min-dalle'])

    # We need to install PyTorch for CUDA 11.6 on A10G or A100, you can comment this out after the first run.
    gpu.run(['pip3 install torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu116'])
    # If you're running into CUDA errors and just installed the torch version above, you may need to
    # restart the gRPC server to freshly import the package.
    # gpu.restart_grpc_server(resync_rh=True)

    my_prompt = 'A hot dog made of matcha powder.'
    images = generate_dm_gpu(my_prompt, num_images_sqrt=1, seed=random.randint(0, 1000))
    [image.show() for image in images]