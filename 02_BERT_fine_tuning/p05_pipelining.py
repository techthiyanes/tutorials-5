import pickle

import runhouse as rh

from p02_fine_tune import get_model_and_optimizer
from p04_inference import create_prediction_service


def simple_bert_fine_tuning_service(dataset,
                                    epochs=3,
                                    model_id='bert-base-cased'):
    # Preprocess
    preproc_table_name = dataset.name + "_preprocessed"
    if not rh.exists(name=preproc_table_name, resource_type='table', load_from=['rns', 'local']):
        preproc = rh.send(name="BERT_preproc_32cpu")
        preprocessed_table = preproc(dataset).from_cluster(preproc.hardware)
        preprocessed_table.save(name=preproc_table_name, save_to=['rns', 'local'])
    else:
        preprocessed_table = rh.table(name=preproc_table_name, load_from=['rns', 'local'])

    # Train
    model_name = dataset.name + '_ft_bert'
    if not rh.exists(name=model_name, resource_type='blob', load_from=['rns', 'local']):
        bert_model, adam_optimizer = get_model_and_optimizer(model_id=model_id, num_labels=5, lr=5e-5)
        ft_model = rh.send(name='finetune_ddp_4gpu')
        trained_model = ft_model(preprocessed_table,
                                 bert_model,
                                 adam_optimizer,
                                 epochs=epochs).from_cluster(gpus)
        trained_model.save(name=model_name, save_to=['rns', 'local'])
    else:
        trained_model = pickle.loads(rh.blob(name=model_name, load_from=['rns', 'local']).data)

    # Evaluate
    model_eval = rh.send(name='evaluate_model')
    test_accuracy = model_eval(preprocessed_table, trained_model)
    assert test_accuracy > 0.8, "Model accuracy is too low for production!"

    # Deploy
    rh.send(name='deploy_model', fn=pipelined_bert_fine_tuning_service)
    predict_fn = create_prediction_service(model_name)
    predict_service = rh.send(fn=predict_fn,
                              hardware="^rh-1-cpu",
                              name=dataset.name + "_bert_ft_service",
                              save_to=['rns', 'local'])

    return predict_service


def pipelined_bert_fine_tuning_service(dataset,
                                       epochs):
    # TODO
    pass


if __name__ == "__main__":
    rh.set_folder('/donnyg/bert/sentiment_analysis', create=True)
    yelp_table = rh.table(name="yelp_reviews", data_source='hf', url='yelp_review_full',
                          load_from=['rns', 'local'], save_to=['rns', 'local'])
    bert_sa_service = simple_bert_fine_tuning_service(yelp_table, epochs=3)
    prompt = "I could eat hot dogs at Larry's every day."
    print(f'Review: {prompt}; sentiment score: {bert_sa_service(prompt)}')
