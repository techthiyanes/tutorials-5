import os
import runhouse as rh
from transformers import AutoTokenizer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')


def tokenize_function(examples):
    return tokenizer(examples, padding="max_length", truncation=True)


def tokenize_dataset(hf_dataset):
    if isinstance(hf_dataset, str) and rh.exists(hf_dataset):
        hf_dataset = rh.table(name=hf_dataset).convert_to('hf_dataset')

    tokenized_datasets = hf_dataset.map(tokenize_function,
                                        input_columns=['text'],
                                        num_proc=os.cpu_count(),
                                        batched=True)

    # https://github.com/huggingface/transformers/issues/12631
    # Remove the text column because the model does not accept raw text as an input
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])

    # Rename the label column to labels because the model expects the argument to be named labels
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # We'll return the table object here so the user of this service can save it to whatever datastore they
    # prefer, under whichever Runhouse name they prefer. We need to call save() to write it down to the local
    # filesystem on the cluster, as we're only returning a reference to the user rather than the full dataset.
    return rh.table(data=tokenized_datasets).save()


if __name__ == "__main__":
    cpu = rh.cluster("^rh-32-cpu").up_if_not()

    preproc = rh.function(fn=tokenize_dataset,
                          system=cpu,
                          reqs=['local:./', 'datasets', 'transformers'],
                          name="BERT_preproc_32cpu")

    # Not being saved, just a helper here to load the dataset on the cluster instead of locally
    # (and then sending it up).
    remote_load_dataset = rh.function(fn=load_dataset,
                                      system=preproc.system,
                                      dryrun=True)

    # Notice how we call this function with `.remote` - this calls the function async, leaves the result on the
    # cluster, and gives us back a reference (Ray ObjectRef) to the object that we can then pass to other functions
    # on the cluster, and they'll auto-resolve to our object.
    yelp_dataset_ref = remote_load_dataset.remote("yelp_review_full", split='train[:1%]')

    # converts the table's file references to sftp file references without copying it
    preprocessed_yelp = preproc(yelp_dataset_ref)

    batches = preprocessed_yelp.stream(batch_size=100)
    for batch in batches:
        print(batch)
        break

    preprocessed_yelp.save(name="preprocessed-tokenized-dataset", overwrite=True)
