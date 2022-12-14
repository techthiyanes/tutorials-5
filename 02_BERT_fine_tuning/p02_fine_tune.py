from transformers import AutoModelForSequenceClassification, get_scheduler
from accelerate import Accelerator
import ray.cloudpickle as pickle
import torch
from torch.utils.data import DataLoader
import runhouse as rh
from tqdm.auto import tqdm  # progress bar


def fine_tune_model(preprocessed_data, model, optimizer, num_epochs=3, batch_size=8):
    accelerator = Accelerator()

    train_dataloader = DataLoader(preprocessed_data['train'], shuffle=True, batch_size=batch_size)
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
    rh.set_folder('~/bert/sentiment_analysis', create=True)
    preprocessed_table = rh.table(name="yelp_bert_preprocessed")
    bert_model, adam_optimizer = get_model_and_optimizer(model_id='bert-base-cased', num_labels=5, lr=5e-5)

    gpus = rh.cluster(name='4-v100s', instance_type='V100:4', provider='cheapest', use_spot=False)
    ft_model = rh.Send(fn=fine_tune_model, hardware=gpus, name='finetune_ddp_4gpu')
    trained_model = ft_model(preprocessed_table,
                             bert_model,
                             adam_optimizer,
                             epochs=3).from_cluster(gpus)
    trained_model.save(name='yelp_fine_tuned_bert', save_to=['rns', 'local'])
