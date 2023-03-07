import pickle

import runhouse as rh

from p01_preprocess import tokenize_function


def create_prediction_service(model_name):
    def predict_from_text(samples, refresh_model=False):
        sa_model = rh.get_pinned_object(model_name)
        if sa_model is None or refresh_model:
            sa_model = pickle.loads(rh.blob(name=model_name).data).to("cuda")
            rh.pin_to_memory(model_name, sa_model)

        tokens = tokenize_function(samples)
        return sa_model(tokens)

    return predict_from_text


if __name__ == "__main__":
    # rh.set_folder('~/bert/sentiment_analysis', create=True)
    predict_fn = create_prediction_service('yelp_fine_tuned_bert')

    bert_sa_service = rh.function(fn=predict_fn,
                                  system="^rh-1-cpu",
                                  name="prediction_service")
    new_examples = [
        'This place is sick!',
        'The service left much to be desired.',
        'The appetizers were hit or miss.',
        'The elote ribs are a must.',
    ]
    sentiment_scores = bert_sa_service(new_examples)
    print('Test samples and sentiment scores:')
    [print(f'{sample}: {score}') for (sample, score) in zip(new_examples, sentiment_scores)]

    # TODO show sharing.
    # Collaborators or other environments can use the following to get the microservice callable,
    # without any additional installations.
    # BERT_sa_service = rh.function(name="<your username>/bert/sentiment_analysis/prediction_service")
    # Note that I can manage who has access to my Functions, and all my other Runhouse resources, via
    # a single access control plane. I can share them with individual Runhouse accounts, my team,
    # my company, or the general public.
