import ray.cloudpickle as pickle

from datasets import load_metric
from accelerate import Accelerator
import torch
import runhouse as rh
from torch.utils.data import DataLoader


def evaluate_model(preprocessed_dataset, model):
    model = pickle.loads(model.data)
    accelerator = Accelerator()

    preprocessed_data = preprocessed_dataset.fetch()

    eval_dataloader = DataLoader(preprocessed_data, shuffle=False, batch_size=32)
    eval_dataloader, model = accelerator.prepare(eval_dataloader, model)

    metric = load_metric("accuracy")
    model.eval()

    for batch in eval_dataloader:
        batch_labels = batch.pop("labels")
        batch = {k: torch.stack(v) if isinstance(v, list) else v for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch_labels)

    metric.compute()
    return metric


if __name__ == "__main__":
    # rh.set_folder('~/bert/sentiment_analysis', create=True)

    v100 = rh.cluster('^rh-4-v100').up_if_not()

    # Load model blob object (generated and saved in P02) - we'll fetch the data on the cluster
    trained_model = rh.blob(name='yelp_fine_tuned_bert', dryrun=True)

    model_eval = rh.function(fn=evaluate_model,
                             system=v100,
                             name='evaluate_model',
                             reqs=['scikit-learn'])

    preprocessed_dataset = rh.table(name="preprocessed-tokenized-dataset", dryrun=True)

    test_accuracy = model_eval(preprocessed_dataset, trained_model)
    print('Test accuracy:', test_accuracy)
