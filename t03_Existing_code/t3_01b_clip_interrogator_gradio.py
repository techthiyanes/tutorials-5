import gradio as gr
import runhouse as rh

def launch_gradio_space(name):
    gr.Interface.load(name).launch()

if __name__ == "__main__":
    gpu = rh.cluster(name='rh-a100', instance_type='A100:1', provider='cheapest', use_spot=False)
    my_space = rh.send(fn=launch_gradio_space, hardware=gpu,
                       reqs=['./', 'gradio',
                             'torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu116',  # Need CUDA 11.6 for A100
                             'torchvision --upgrade --extra-index-url https://download.pytorch.org/whl/cu116,',
                             'fairscale', 'ftfy', 'huggingface-hub',
                             'Pillow', 'timm', 'transformers', 'open_clip_torch', 'clip-interrogator==0.3.1',
                             ])
    my_space.hardware.ssh_tunnel(local_port=7860, remote_port=7860)
    my_space.hardware.keep_warm()
    try:
        my_space.enqueue('spaces/pharma/CLIP-Interrogator')
        # Space will now be available at http://localhost:7860
        # To stop the space, just terminate this script.
    except KeyboardInterrupt as e:
        gpu.restart_grpc_server(resync_rh=False)

