from transformers import AutoModelForSequenceClassification, get_scheduler
from accelerate import Accelerator
import torch
import runhouse as rh
from tqdm.auto import tqdm  # progress bar

def fine_tune_bert(preprocessed_data_name, epochs, model_out_name):
    accelerator = Accelerator()
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    preprocessed_data = rh.Table(name=preprocessed_data_name)
    train_dataloader = preprocessed_data['train'].stream(shuffle=True, batch_size=8)
    eval_dataloader = preprocessed_data['test'].stream(batch_size=8)

    train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer
    )

    num_training_steps = epochs * len(train_dataloader)
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

        metric = load_metric("accuracy")
        model.eval()
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        metric.compute()
        print(f'Epoch {epoch} accuracy: {metric}')

    return rh.current_folder().put({model_out_name: pickle.dumps(model)})

if __name__ == "__main__":
    my_8_gpus = rh.cluster(provider='aws',
                           name='my_8_gpus')
    bert_ft = rh.Send(fn=fine_tune_bert,
                      hardware='my_8_gpus',
                      name='BERT_finetune_8gpu')
    trained_model = bert_ft(preprocessed_data_name='yelp_bert_preprocessed',
                            epochs=3,
                            model_out_name='yelp_fine_tuned_bert')
