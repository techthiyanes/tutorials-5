import pickle

import runhouse as rh

from p02_fine_tune import get_model, get_optimizer
from p04_inference import create_prediction_service


def simple_bert_fine_tuning_service(dataset,
                                    epochs=3,
                                    model_id='bert-base-cased'):
    # Preprocess
    preproc_table_name = dataset.name + "_preprocessed"
    if not rh.exists(name=preproc_table_name, resource_type='table'):
        preproc = rh.function(name="BERT_preproc_32cpu")
        preprocessed_table = preproc(dataset)
        preprocessed_table.save(name=preproc_table_name)
    else:
        preprocessed_table = rh.table(name=preproc_table_name)

    # Train
    model_name = dataset.name + '_ft_bert'
    if not rh.exists(name=model_name, resource_type='blob'):
        bert_model = get_model(num_labels=5)
        adam_optimizer = get_optimizer(model=model_id, lr=5e-5)
        ft_model = rh.function(name='finetune_ddp_4gpu')
        trained_model = ft_model(preprocessed_table,
                                 bert_model,
                                 adam_optimizer,
                                 epochs=epochs)
        trained_model.save(name=model_name)
    else:
        trained_model = pickle.loads(rh.blob(name=model_name).data)

    # Evaluate
    model_eval = rh.function(name='evaluate_model')
    test_accuracy = model_eval(preprocessed_table, trained_model)
    assert test_accuracy > 0.8, "Model accuracy is too low for production!"

    # Deploy
    rh.function(name='deploy_model', fn=pipelined_bert_fine_tuning_service)
    predict_fn = create_prediction_service(model_name)
    predict_service = rh.function(fn=predict_fn,
                                  system="^rh-1-cpu",
                                  name=dataset.name + "_bert_ft_service")

    return predict_service


def pipelined_bert_fine_tuning_service():
    """Create an end to pipeline containing:
     preprocessing: create tokenized dataset on 32 CPU cluster
     fine tuning: Fine tune the model on an A10G cluster. Model generated on the cluster will be written to s3.
     model eval: Evaluate the model accuracy on a 4-V100s.
     deployment: Deploy the model service on a single CPU.
     inference: Call the sentiment service with some sample data.

     Notice how easily we can create this pipeline natively in pipeline, without the need to translate it into some
     DAG based DSL. We have the ability to easily call into services running on heterogenuous hardware simply by creating
     them with Runhouse.
    """
    # TODO
    pass


if __name__ == "__main__":
    yelp_table = rh.table(name="yelp_reviews", path='yelp_review_full')
    bert_sa_service = simple_bert_fine_tuning_service(yelp_table, epochs=3)

    prompt = "I could eat hot dogs at Larry's every day."
    print(f'Review: {prompt}; sentiment score: {bert_sa_service(prompt)}')
