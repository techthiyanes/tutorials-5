import os
import runhouse as rh
from transformers import AutoTokenizer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def tokenize_dataset(hf_dataset):
    if isinstance(hf_dataset, str) and rh.exists(hf_dataset):
        hf_dataset = rh.table(name=hf_dataset).convert_to('hf_dataset')

    tokenized_datasets = hf_dataset.map(tokenize_function,
                                        # input_columns=['text'],
                                        # num_proc=os.cpu_count(),
                                        batched=True)

    # https://github.com/huggingface/transformers/issues/12631
    # Remove the text column because the model does not accept raw text as an input
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])

    # Rename the label column to labels because the model expects the argument to be named labels
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # Set the format of the dataset to return PyTorch tensors instead of lists
    tokenized_datasets.set_format("torch")

    return rh.table(data=tokenized_datasets).save()


if __name__ == "__main__":
    # rh.set_folder('~/bert', create=True)

    preproc = rh.send(fn=tokenize_dataset,
                      hardware="^rh-32-cpu",
                      name="BERT_preproc_32cpu")
    preproc.hardware.restart_grpc_server()

    # TODO [DG] this folder is not setting properly
    # rh.set_folder('./sentiment_analysis', create=True)

    # Not being saved, just a helper here
    remote_load_dataset = rh.send(fn=load_dataset,
                                  hardware=preproc.hardware,
                                  dryrun=True)

    yelp_dataset_ref = remote_load_dataset.remote("yelp_review_full", split='train[:1%]')
    # from_cluster converts the table's file references to sftp file references without copying it
    preprocessed_yelp = preproc(yelp_dataset_ref).from_cluster(preproc.hardware)

    batches = preprocessed_yelp.stream(batch_size=100)
    for idx, batch in enumerate(batches):
        # convert each batch into a huggingface dataset
        batch_dataset = preprocessed_yelp.to_dataset(batch)
        print(f"Preprocessed batch:\n {batch_dataset}")
        break

    preprocessed_yelp.save(name="preprocessed-tokenized-dataset")
