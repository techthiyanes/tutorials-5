import os
import runhouse as rh
from transformers import AutoTokenizer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')


def tokenize_function(examples):
    return tokenizer(examples, padding="max_length", truncation=True)


def tokenize_dataset(hf_dataset):
    if isinstance(hf_dataset, str) and rh.exists(hf_dataset, load_from=['rns', 'local']):
        hf_dataset = rh.table(name=hf_dataset).convert_to('hf_dataset')

    preprocessed = hf_dataset.map(tokenize_function,
                                  input_columns=['text'],
                                  batched=True,
                                  num_proc=os.cpu_count())
    return rh.table(data=preprocessed,
                    name='preprocessed-tokenized-dataset')


if __name__ == "__main__":
    rh.set_folder('~/bert', create=True)
    preproc = rh.send(fn=tokenize_dataset,
                      hardware="^rh-32-cpu",
                      name="BERT_preproc_32cpu",
                      save_to=['local'])

    rh.set_folder('./sentiment_analysis', create=True)
    # Not being saved, just a helper here
    remote_load_dataset = rh.send(fn=load_dataset,
                                  hardware=preproc.hardware,
                                  dryrun=True)

    yelp_dataset_ref = remote_load_dataset.remote("yelp_review_full", split='train[:1%]')
    # from_cluster converts the table's file references to sftp file references without copying it
    preprocessed_yelp = preproc(yelp_dataset_ref).from_cluster(preproc.hardware)

    # Stream in the table as a pyarrow table, and convert each batch to a Huggingface dataset
    batches = preprocessed_yelp.stream(batch_size=1000)
    for idx, batch in enumerate(batches):
        # convert each batch into a huggingface dataset
        batch_dataset = preprocessed_yelp.to_dataset(batch)
        print(batch_dataset)
        break

    preprocessed_yelp.save(name="yelp_bert_preprocessed",
                           save_to=['rns', 'local'])
