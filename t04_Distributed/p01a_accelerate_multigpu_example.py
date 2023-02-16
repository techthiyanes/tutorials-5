import runhouse as rh
import argparse
from accelerate.utils import PrepareForLaunch, patch_environment
import torch

def launch_training(training_function, *args):
    num_processes = torch.cuda.device_count()
    print(f'Device count: {num_processes}')
    with patch_environment(world_size=num_processes, master_addr="127.0.01", master_port="29500",
                           mixed_precision=args[1].mixed_precision):
        launcher = PrepareForLaunch(training_function, distributed_type="MULTI_GPU")
        torch.multiprocessing.start_processes(launcher, args=args, nprocs=num_processes, start_method="spawn")


if __name__ == "__main__":
    gpu = rh.cluster(name='rh-4-v100', instance_type='V100:4', provider='cheapest', use_spot=False)
    # gpu.restart_grpc_server(resync_rh=True)
    train_gpu = rh.function(
        fn='https://github.com/huggingface/accelerate/blob/v0.15.0/examples/nlp_example.py:training_function',
        system=gpu,
        reqs=['./', 'pip:./accelerate', 'torch==1.12.0', 'evaluate', 'transformers',
              'datasets==2.3.2', 'scipy', 'scikit-learn', 'tqdm', 'tensorboard'],
        name='train_bert_glue')

    launch_training = rh.function(fn=launch_training).to(gpu)

    train_args = argparse.Namespace(cpu=False, mixed_precision='fp16')
    hps = {"lr": 2e-5, "num_epochs": 3, "seed": 42, "batch_size": 16}
    launch_training(train_gpu, hps, train_args)

    # Alternatively, we can just run as instructed in the README (but only because there's already a wrapper CLI):
    # gpu.run(['accelerate launch --multi_gpu accelerate/examples/nlp_example.py'])