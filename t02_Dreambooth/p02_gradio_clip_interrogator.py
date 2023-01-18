import gradio as gr
import runhouse as rh

def launch_gradio_space(name):
    gr.Interface.load(name).launch()

# Based on https://huggingface.co/spaces/pharma/CLIP-Interrogator/

if __name__ == "__main__":
    gpu = rh.cluster(name='rh-a10x', instance_type='A100:1')  # On GCP and Azure
    # gpu = rh.cluster(name='rh-a10x', instance_type='g5.2xlarge', provider='aws')  # On AWS
    my_space = rh.send(fn=launch_gradio_space, hardware=gpu,
                       reqs=['./', 'gradio',
                             'fairscale', 'ftfy', 'huggingface-hub',
                             'Pillow', 'timm', 'open_clip_torch', 'clip-interrogator==0.3.1',
                             ])
    gpu.ssh_tunnel(local_port=7860, remote_port=7860)
    gpu.keep_warm()
    try:
        print('Space will now be available at http://localhost:7860')
        print('The first time you run this it needs to download the model, which can take ~10 minutes.')
        my_space.enqueue('spaces/pharma/CLIP-Interrogator')
        # To stop the space, just terminate this script.
    except KeyboardInterrupt as e:
        gpu.restart_grpc_server(resync_rh=False)
