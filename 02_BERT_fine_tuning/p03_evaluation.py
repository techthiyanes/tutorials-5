import pickle

from datasets import load_metric
from accelerate import Accelerator
import torch
import runhouse as rh


def evaluate_model(preprocessed_data, model):
    accelerator = Accelerator()
    eval_dataloader = DataLoader(preprocessed_data['test'], shuffle=False, batch_size=32)
    eval_dataloader, model = accelerator.prepare(eval_dataloader, model)

    metric = load_metric("accuracy")
    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    metric.compute()
    return metric


if __name__ == "__main__":
    rh.set_folder('~/bert/sentiment_analysis', create=True)
    preprocessed_table = rh.table(name="yelp_bert_preprocessed")
    trained_model = pickle.loads(rh.blob(name='yelp_fine_tuned_bert', load_from=['rns', 'local']).data)

    model_eval = rh.Send(fn=evaluate_model,
                         hardware='4-v100s',
                         name='evaluate_model')
    test_accuracy = model_eval(preprocessed_table, trained_model)
    print('Test accuracy:', test_accuracy)
