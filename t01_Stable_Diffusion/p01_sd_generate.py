import runhouse as rh
from diffusers import StableDiffusionPipeline
import torch


def sd_generate(prompt, num_images=1, steps=100, guidance_scale=7.5, model_id='stabilityai/stable-diffusion-2-base'):
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision='fp16').to('cuda')
    return pipe([prompt] * num_images, num_inference_steps=steps, guidance_scale=guidance_scale).images


if __name__ == "__main__":
    # For GCP and Azure:
    gpu = rh.cluster(name='rh-a10x', instance_type='A100:1')
    # For AWS (single A100s not available, and need a g5.2xlarge rather than a base A10G because it has more CPU RAM):
    # gpu = rh.cluster(name='rh-a10x', instance_type='g5.2xlarge', provider='aws')
    # By default, we (via SkyPilot) use the cheapest cloud provider that has your specified hardware available and has
    # credentials saved in your environment. Run `sky check` to see which providers you're set up to use. You can set
    # a specific provider by passing `provider='gcp'`, or a specific instance type like `instance_type='p3.2xlarge'`,
    # or `use_spot=True` to use spot instances. See this helpful guide to cloud GPUs for more details:
    # https://www.paperspace.com/gpu-cloud-comparison

    generate_gpu = rh.send(fn=sd_generate, hardware=gpu, reqs=['./'])
    # generate_gpu is a Python callable just like our original function. We can call it and get back results just
    # like we would locally, as long as the inputs and outputs are serializable with cloudpickle and <2GB (a soft
    # limit we've set in grpc but can change if needed). This step will spin up the cluster if not already up.
    # Passing `reqs=['./']` will sync over this git repo and install its dependencies, but you can pass many other
    # kinds of dependencies as you'll see in later tutorials.

    images = generate_gpu('A digital illustration of a woman running on the roof of a house.', num_images=2, steps=50)
    # The first time we call this function it will need to download the model onto the cluster, which can take a minute.
    # We can reuse generate_gpu as much as we please, like a normal Python function.
    [image.show() for image in images]

    generate_gpu.save(name='sd_generate')
    # Runhouse is all about making your work accessible from anywhere. We'll reuse this function in the next tutorial,
    # so let's save and name it. If you've created an account and logged in, it will be saved to our store. If not,
    # it will be saved locally to the `rh/` directory of this git repo.

    gpu.keep_warm()
    # By default this GPU will terminate after 30 minutes, but let's keep it up so we can reuse it for many tutorials.
    # You can also do `gpu.keep_warm(autostop_mins=10)` to stop the cluster after 10 minutes of inactivity.

    # gpu.teardown()
    # If you like, you can terminate the cluster in your cloud provider's UI, or through Runhouse as follows:
    # You can also run `sky down my_gpu` or `sky down --all` from the command line.
