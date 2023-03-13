import logging
from datasets import load_metric
from accelerate import Accelerator
import torch
import runhouse as rh
from torch.utils.data import DataLoader
import ray.cloudpickle as pickle


def evaluate_model(preprocessed_data, trained_model, batch_size=32):
    model = pickle.loads(trained_model.data)
    accelerator = Accelerator()

    # Load the data itself on the cluster
    eval_dataloader = DataLoader(preprocessed_data, shuffle=False, batch_size=batch_size)
    eval_dataloader, model = accelerator.prepare(eval_dataloader, model)

    metric = load_metric("accuracy")
    model.eval()

    for batch in eval_dataloader:
        try:
            labels = batch.pop("labels")
            batch = {k: torch.stack(v).reshape([batch_size, len(v)]) for k, v in batch.items()}

            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=labels)

        except Exception as e:
            logging.error(f'Failed to predict batch: {e}')

    accuracy = metric.compute()
    return accuracy


if __name__ == "__main__":
    v100 = rh.cluster('^rh-4-v100', instance_type='V100:4').up_if_not()

    # Load model we created in P02 (note: we'll load the blob itself on the cluster later)
    trained_model = rh.Blob.from_name(name='yelp_fine_tuned_bert')

    model_eval = rh.function(fn=evaluate_model,
                             system=v100,
                             name='evaluate_model',
                             reqs=['scikit-learn', 's3fs'])

    # Load the dataset we created in P01
    preprocessed_dataset = rh.Table.from_name(name="preprocessed-tokenized-dataset")
    preprocessed_data = preprocessed_dataset.fetch()

    test_accuracy = model_eval(preprocessed_data, trained_model)
    print('Test accuracy:', test_accuracy)
