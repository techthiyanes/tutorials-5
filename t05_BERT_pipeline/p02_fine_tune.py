from transformers import AutoModelForSequenceClassification, get_scheduler
from accelerate import Accelerator
import ray.cloudpickle as pickle
import torch
from torch.utils.data import DataLoader
import runhouse as rh
from tqdm.auto import tqdm  # progress bar


def fine_tune_model(preprocessed_table, model, optimizer, num_epochs=3, batch_size=8):
    # https://huggingface.co/docs/transformers/accelerate
    accelerator = Accelerator()

    # Set data format to pytorch tensors
    preprocessed_table.set_format('torch')

    train_dataloader = DataLoader(preprocessed_table, batch_size=batch_size)
    train_dataloader, model, optimizer = accelerator.prepare(train_dataloader, model, optimizer)

    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    progress_bar = tqdm(range(num_training_steps))
    model.train()

    # https://huggingface.co/course/chapter8/2?fw=pt
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    return rh.blob(data=pickle.dumps(model))


def get_model(num_labels, model_id='bert-base-cased'):
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)
    return model


def get_optimizer(model, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    return optimizer


if __name__ == "__main__":
    # rh.set_folder('~/bert/sentiment_analysis', create=True)

    gpu = rh.cluster(name='rh-a10x', instance_type='A100:1')  # On GCP and Azure
    # gpu = rh.cluster(name='rh-a10x', instance_type='g5.2xlarge', provider='aws')  # On AWS

    ft_model = rh.send(fn=fine_tune_model,
                       hardware=gpu,
                       name='finetune_ddp_1gpu',
                       reqs=['torch==1.12.0'],
                       load_secrets=True).save()

    # The load_secrets argument above will load the secrets onto the cluster from your Runhouse account (api.run.house),
    # and will only work if you've already uploaded secrets to runhouse (e.g. during `runhouse login`).

    # If you'd like to run this tutorial without an account or saved secrets, you can uncomment this line:
    # ft_model.send_secrets(providers=['sky'])

    # Send model and optimizer to the cluster to be initialized
    model_on_gpu = rh.send(fn=get_model, hardware=gpu, dryrun=True)
    optimizer_on_gpu = rh.send(fn=get_optimizer, hardware=gpu, dryrun=True)

    # Receive an object ref for the model and optimizer
    bert_model = model_on_gpu.remote(num_labels=5, model_id='bert-base-cased')
    adam_optimizer = optimizer_on_gpu.remote(model=bert_model, lr=5e-5)

    preprocessed_table = rh.table(name="preprocessed-tokenized-dataset")

    trained_model = ft_model(preprocessed_table,
                             bert_model,
                             adam_optimizer,
                             num_epochs=3).from_cluster(gpu)

    trained_model.save(name='yelp_fine_tuned_bert')
