import argparse
import runhouse as rh

# ----------------- README ----------------- #
# Here we simply demonstrate how to use Runhouse to run existing code from Github.

if __name__ == "__main__":
    gpu = rh.cluster(name='rh-a10x') if rh.exists('rh-a10x') else rh.cluster(name='rh-a10x', instance_type='A100:1')
    train_gpu = rh.function(
        fn='https://github.com/huggingface/accelerate/blob/v0.15.0/examples/nlp_example.py:training_function',
        system=gpu,
        env=['pip:./accelerate', 'transformers', 'datasets', 'evaluate',
              'tqdm', 'scipy', 'scikit-learn', 'tensorboard', 'torch'],
        name='train_bert_glue')

    train_args = argparse.Namespace(cpu=False, mixed_precision='bf16')
    hps = {"lr": 2e-5, "num_epochs": 3, "seed": 42, "batch_size": 16}
    train_gpu(hps, train_args, stream_logs=True)

    # Alternatively, we can just run as instructed in the README (but then we leave Python):
    # gpu.run(['python ./nlp_example.py --fp16'])
