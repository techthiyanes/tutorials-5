from transformers import AutoModelForSequenceClassification, get_scheduler
import ray.cloudpickle as pickle
import torch
import runhouse as rh
from tqdm.auto import tqdm  # progress bar

# Based on https://huggingface.co/docs/transformers/training#train-in-native-pytorch

def fine_tune_model(model, optimizer, preprocessed_table, num_epochs=3, batch_size=8):
    # Set data format to pytorch tensors
    preprocessed_table.stream_format = 'torch'
    print(preprocessed_table._folder.data_config)
    device = torch.device("cuda")
    model.to(device)

    num_training_steps = num_epochs * len(preprocessed_table)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    progress_bar = tqdm(range(num_training_steps))
    model.train()

    # https://huggingface.co/course/chapter8/2?fw=pt
    for epoch in range(num_epochs):
        for batch in preprocessed_table.stream(batch_size=batch_size, as_dict=True):
            # TODO [JL] - Use a smaller torch type (IntTensor doesn't work)
            batch = {k: v.type(torch.LongTensor).to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    # Save as anonymous blob to local file system on the cluster
    return rh.blob(data=pickle.dumps(model)).save()


def get_model(num_labels, model_id='bert-base-cased'):
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)
    return model


def get_optimizer(model, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    return optimizer


if __name__ == "__main__":
    gpu = rh.cluster(name='rh-a10x', instance_type='A100:1')  # On GCP and Azure
    # gpu = rh.cluster(name='rh-a10x', instance_type='g5.2xlarge', provider='aws')  # On AWS

    preprocessed_yelp = rh.table(name="preprocessed-tokenized-dataset")

    ft_model = rh.send(fn=fine_tune_model,
                       hardware=gpu,
                       load_secrets=True,
                       name='finetune_ddp_1gpu').save()

    # The load_secrets argument above will load the secrets onto the cluster from your Runhouse account (api.run.house),
    # and will only work if you've already uploaded secrets to runhouse (e.g. during `runhouse login`). You need your
    # SkyPilot ssh keys on the gpu cluster because we're streaming in the table directly from the 32-cpu cluster.

    # If you'd like to run this tutorial without an account or saved secrets, you can uncomment this line:
    # ft_model.send_secrets(providers=['sky'])

    # Send get_model and get_optimizer to the cluster so we can call .remote and instantiate them on the cluster
    model_on_gpu = rh.send(fn=get_model, hardware=gpu, dryrun=True)
    optimizer_on_gpu = rh.send(fn=get_optimizer, hardware=gpu, dryrun=True)

    # Receive an object ref for the model and optimizer
    bert_model = model_on_gpu.remote(num_labels=5, model_id='bert-base-cased')
    adam_optimizer = optimizer_on_gpu.remote(model=bert_model, lr=5e-5)

    trained_model = ft_model(bert_model,
                             adam_optimizer,
                             preprocessed_yelp,
                             num_epochs=3).from_cluster(gpu)

    # Save model in s3 bucket, and the metadata in Runhouse RNS
    print("trained model:\n", trained_model)
    trained_model.to('s3').save(name='yelp_fine_tuned_bert')
