import gradio as gr
import runhouse as rh

def launch_gradio_space(name):
    gr.Interface.load(name).launch()

if __name__ == "__main__":
    my_space = rh.send(fn=launch_gradio_space, hardware='^rh-v100', reqs=['./', 'gradio'], load_secrets=True)
    my_space.hardware.ssh_tunnel(local_port=7860, remote_port=7860)
    my_space.hardware.keep_warm()
    my_space('spaces/multimodalart/dreambooth-training')

    v100 = rh.cluster(name='^rh-v100', autostop_mins=-1)
    if not v100.is_up():
        v100.reup_cluster()
    v100.send_secrets()
    v100.install_packages(['./', 'diffusers'])
    v100.ssh_tunnel(local_port=7860, remote_port=7860)
    v100.run(['git clone https://huggingface.co/spaces/multimodalart/dreambooth-training',
              'python -m dreambooth-training/app.py'])