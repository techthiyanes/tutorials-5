import os
import runhouse as rh
from transformers import AutoTokenizer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

def tokenize_function(examples):
    return tokenizer(examples, padding="max_length", truncation=True)

def tokenize_dataset(hf_dataset_name):
    # Check if data is already saved in this Runhouse folder before redownloading and preprocessing data
    # if rh.exists(name=out_table_name, type='table'):
    #     return rh.Table(name=out_table_name)

    hf_dataset = load_dataset(hf_dataset_name)

    preprocessed = hf_dataset['test'].map(tokenize_function,
                                  input_columns=['text'],
                                  batched=True,
                                  num_proc=os.cpu_count())
    # return preprocessed['text'][0]
    return rh.table(data=preprocessed)

if __name__ == "__main__":
    rh.set_folder('/donnyg/bert', create=True)
    preproc = rh.send(fn=tokenize_dataset,
                      hardware="^rh-32-cpu",
                      name="BERT_preproc_32cpu"
                      )

    rh.set_folder('./sentiment_analysis', create=True)
    preproc_table_name = "yelp_bert_preprocessed"
    if not rh.exists(name=preproc_table_name, resource_type='table', load_from=['rns']):
        preprocessed_yelp = preproc("yelp_review_full")
        preprocessed_yelp.save(name=preproc_table_name, save_to=['rns'])