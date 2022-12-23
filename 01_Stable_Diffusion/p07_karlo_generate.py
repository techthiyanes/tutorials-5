import runhouse as rh
from diffusers import UnCLIPPipeline
import torch


def unclip_generate(prompt,
                    model_id='kakaobrain/karlo-v1-alpha',
                    num_images=1,
                    **model_kwargs):
    pipe = rh.get_pinned_object(model_id)
    if pipe is None:
        pipe = UnCLIPPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to('cuda')
        rh.pin_to_memory(model_id, pipe)
    return pipe([prompt], num_images_per_prompt=num_images, **model_kwargs).images


if __name__ == "__main__":
    # This is a bit too small, need to switch to A10G
    gpu = rh.cluster(name='rh-a10g', instance_type='A10G:1', provider='cheapest')
    generate_karlo_gpu = rh.send(fn=unclip_generate,
                                 hardware=gpu,
                                 reqs=['local:./', 'diffusers', 'transformers', 'accelerate', 'safetensors'],
                                 name='karlo_generate')
    # We need to install PyTorch for CUDA 11.6 on A10G or A100, you can comment this out after the first run.
    gpu.run(['pip3 install torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu116'])
    # If you're running into CUDA errors and just installed the torch version above, you may need to
    # restart the gRPC server to freshly import the package.
    # gpu.restart_grpc_server()

    # The model takes a long time to download and send to GPU the first time you run, but after that it only takes
    # 4 seconds per image.
    my_prompt = 'beautiful fantasy painting of Tom Hanks as Samurai in sakura field'
    images = generate_karlo_gpu(my_prompt, num_images=4)
    [image.show() for image in images]