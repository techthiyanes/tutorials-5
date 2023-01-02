import runhouse as rh
from diffusers import StableDiffusionPipeline
import torch

# This function simply takes a prompt (and a few settings), loads the stable diffusion model, feeds the
# prompt into the model, and returns the resulting images. Note that it takes a while to run the first time,
# as it needs to download and locally save the model.
def sd_generate(prompt, num_images=1,
                steps=100, guidance_scale=7.5,
                model_id='stabilityai/stable-diffusion-2-base'):
    # You may need to visit https://huggingface.co/stabilityai/stable-diffusion-2-base to opt-into using the model,
    # and then provide your token here to download it.
    # You can paste your token into the `use_auth_token=` argument of `from_pretrained` below, or you can save it to
    # Runhouse (see Appendix A of the README) and it will be automatically loaded onto the cluster via
    # `load_secrets=True` below.
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision='fp16').to('cuda')
    return pipe([prompt] * num_images, num_inference_steps=steps, guidance_scale=guidance_scale).images

if __name__ == "__main__":
    # Let's define a cluster that we want to send our function to.
    # Note that we could also just pass '^rh-v100' to the send function to use a builtin hardware configuration. You
    # can see the other builtins by running `rh.ls('^')`, and inspect the parameters by running
    # `rh.load('^rh-8-cpu', instantiate=False)`.
    # We're using SkyPilot to select the cheapest cloud provider that has a V100 available and has
    # credentials saved in your environment (e.g. ~/.aws/...). You can run `sky check` in your terminal to see
    # which cloud providers you're set up to use. You can set a specific provider by passing `provider='gcp'`,
    # or a specific instance type like `instance_type='p3.2xlarge'`.
    gpu = rh.cluster(name='rh-v100', instance_type='V100:1', provider='cheapest')

    # Now we'll send our function to our v100. Note that this step can take several minutes the first time you run it,
    # while we wait for hardware to be provisioned and install the dependencies. Once the cluster is already up, you
    # can send new functions to it or change the function above and resend in seconds. So in practice, the first time
    # you run this file it can take several minutes, and after that it's very fast, mostly just the remote execution
    # of the function itself.
    generate_gpu = rh.send(fn=sd_generate, hardware=gpu,
                           reqs=['./', 'torch==1.12.0', 'diffusers'],
                           name='sd_generate', save_to=['rns'])

    # generate_gpu is a Python callable just like our original function. We can call it and get back results just
    # like we would locally, as long as the inputs and outputs are serializable with cloudpickle and <2GB (a soft
    # limit we've set in grpc but can change if needed).
    # The first time we call the below it will need to download the model onto the cluster, which can take a minute.
    my_prompt = 'A digital illustration of a woman running on the roof of a house.'
    images = generate_gpu(my_prompt, num_images=2, steps=50)
    [image.show() for image in images]
    # You can find other kwargs into the model here:
    # https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion#diffusers.StableDiffusionPipeline.__call__

    # We can reuse generate_gpu as much as we please, and after the first execution it will run much faster
    # (~8 seconds / image) because the model is already on the cluster (unless we change the model_id argument).
    # In tutorial p06 you'll see how you can get this down to ~2.5 seconds per image.


    # Some other neat things we can do with the "Send" (serverless endpoint):

    # generate_gpu.ssh()
    # will start an ssh session into the cluster so you can poke around or debug. You can also run
    # `ssh rh-v100` to ssh into the cluster from the command line.

    # generate_gpu.notebook()
    # will install JupyterLab on the cluster, start a new JupyterLab server, and tunnel you into it (as
    # long as you have a browser on your local machine). This might make more sense in an interactive
    # Python interpreter.

    # images = generate_gpu.map(['A dog.', 'A cat.', 'A biscuit.'], num_images=[1]*3, steps=[50]*3)
    # will run the function on each of the three prompts, and return a list of the results. Note that the
    # num_images and steps arguments are broadcasted to each prompt, so the first prompt will get 1 image.

    # generate_gpu.starmap([('A dog.', 1), ('A cat.', 2), ('A biscuit.', 3)], steps=50)
    # is the same as map as above, but we can pass the arguments as a list of tuples, and we can pass the steps
    # argument as a single value, since it's the same for all three prompts.


    # And other neat things we can do with the "Cluster":
    # (all of the below can equivalently be called on generate_gpu.hardware)

    # gpu.run(['git clone ...', 'pip install ...'])
    # Run any shell command on the cluster. This is useful for installing more complex dependencies.

    # gpu.run_python(['import torch', 'print(torch.__version__)'])
    # Run any Python code on the cluster. This is useful for debugging, or for running a script that you don't
    # want to send to the cluster (e.g. because it has too many dependencies).

    # gpu.ssh_tunnel(local_port=7860, remote_port=7860)
    # If you want to run an application on the cluster that requires a port to be open, e.g. Gradio.

    # gpu.keep_warm()
    # If you want to keep the cluster up and running indefinitely. You can also use this to set an autostop
    # time, e.g. `generate_gpu.hardware.keep_warm(autostop_mins=10)` to stop the cluster after 10 minutes of
    # inactivity.

    # gpu.down()
    # Terminates the cluster, and deletes the data on it. You can also run `sky down rh-v100`
    # or `sky down --all` from the command line.


