from transformers import AutoModelForSequenceClassification, get_scheduler
from accelerate import Accelerator
import ray.cloudpickle as pickle
import torch
from torch.utils.data import DataLoader
import runhouse as rh
from tqdm.auto import tqdm  # progress bar


def fine_tune_model(preprocessed_table, model, optimizer, num_epochs=3, batch_size=8):
    accelerator = Accelerator()
    train_dataloader = DataLoader(preprocessed_table, batch_size=batch_size)
    train_dataloader, model, optimizer = accelerator.prepare(train_dataloader, model, optimizer)

    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    progress_bar = tqdm(range(num_training_steps))

    model.train()
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


def get_model_and_optimizer(num_labels, lr, model_id='bert-base-cased'):
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    return model, optimizer


if __name__ == "__main__":
    # rh.set_folder('~/bert/sentiment_analysis', create=True)

    gpus = rh.cluster(name='v100', instance_type='V100:1', provider='cheapest', use_spot=False)
    # gpus.restart_grpc_server(resync_rh=True)

    ft_model = rh.send(fn=fine_tune_model,
                       hardware=gpus,
                       name='finetune_ddp_1gpu',
                       reqs=['torch==1.12.0'],
                       load_secrets=True,
                       )
    # The load_secrets argument above will load the secrets onto the cluster from your Runhouse account (api.run.house),
    # and will only work if you've already uploaded secrets to runhouse (e.g. during `runhouse login`).
    # If you'd like to run this tutorial without an account or saved secrets, you can uncomment this line:
    # ft_model.send_secrets(secrets=['sky'])

    get_model_and_optimizer_on_cluster = rh.send(fn=get_model_and_optimizer, hardware=gpus, dryrun=True)
    bert_model, adam_optimizer = get_model_and_optimizer_on_cluster.remote(model_id='bert-base-cased',
                                                                           num_labels=5, lr=5e-5)

    preprocessed_table = rh.table(name="yelp_bert_preprocessed")

    trained_model = ft_model(preprocessed_table,
                             bert_model,
                             adam_optimizer,
                             num_epochs=3).from_cluster(gpus)

    trained_model.save(name='yelp_fine_tuned_bert')
