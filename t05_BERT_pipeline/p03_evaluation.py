import runhouse as rh
from datasets import load_metric
import torch
import ray.cloudpickle as pickle
from tqdm.auto import tqdm


def evaluate_model(model, preprocessed_test_set, batch_size=32):
    model = pickle.loads(model.data)
    preprocessed_test_set.stream_format = 'torch'
    device = torch.device("cuda")
    model.to(device)

    metric = load_metric("accuracy")
    progress_bar = tqdm(range(len(preprocessed_test_set)))
    print("Evaluating model.")
    model.eval()

    for batch in preprocessed_test_set.stream(batch_size=batch_size, as_dict=True):
        batch = {k: v.to(device).long() for k, v in batch.items()}
        labels = batch.pop("labels")

        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=labels)
        progress_bar.update(batch_size)

    accuracy = metric.compute()
    return accuracy


if __name__ == "__main__":
    gpu = rh.cluster(name='rh-a10x') if rh.exists('rh-a10x') else rh.cluster(name='rh-a10x', instance_type='A100:1')

    model_eval = rh.function(fn=evaluate_model,
                             system=gpu,
                             name='evaluate_model',
                             reqs=['scikit-learn', 's3fs'])

    # Load model we created in P02 (note: we'll unpickle the file on the cluster later)
    trained_model = rh.Blob.from_name(name='yelp_fine_tuned_bert')
    preprocessed_yelp_test = rh.Table.from_name(name="preprocessed-yelp-test")

    test_accuracy = model_eval(trained_model, preprocessed_yelp_test, batch_size=64, stream_logs=True)
    print('Test accuracy:', test_accuracy)
