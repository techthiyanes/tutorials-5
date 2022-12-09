import runhouse as rh

def bert_yelp_sentiment_analyzer(samples, cache_name=None):
    sa_model = rh.blob(name='yelp_fine_tuned_bert').data
    cached_scores = []
    if cache_name:
        cache = rh.KVstore(name=cache_name)
        cached_scores = cache.get(samples)  # Returns None if not found
        samples = [t for (res, t) in zip(cached_scores, samples) if res is not None]
        if len(samples) == 0:
            return cached_scores
    tokens = tokenize_function(samples)
    model_scores = sa_model(tokens)
    if cache_name:
        # TODO make sure logically works with Nones in list
        cache.set(zip(samples, model_scores))
        return [cached if cached is not None else model_scores.pop(0) for cached in cached_scores]
    return model_scores

if __name__ == "__main__":
    bert_sa_service = rh.Send(bert_yelp_sentiment_analyzer,
                              hardware="rh_1_cpu",
                              name="BERT_sa_service")
    new_examples = [
        'This place is sick!',
        'The service left much to be desired.',
        'The appetizers were hit or miss.',
        'The elote ribs are a must.',
    ]
    sentiment_scores = bert_sa_service(new_examples, cache_name="bert_yelp_encoder_cache")
    print('Test samples and sentiment scores:')
    [print(f'{sample}: {score}') for (sample, score) in zip(new_examples, sentiment_scores)]

    # We can keep this Send warm, with specified replicas and regions
    bert_sa_service.keep_warm(min_replicas=10, max_replicas=20)
    bert_sa_service.keep_warm(regions=['us-east-1', 'us-west-2'], min_replicas=[10, 5], max_replicas=[20, 10])

    # Collaborators or other environments can use the following to get the microservice callable,
    # without any additional installations.
    BERT_sa_service = rh.Send(name="donnyg/BERT_sa_service")
    # Note that I can manage who has access to my Sends, and all my other Runhouse resources, via
    # a single access control plane. I can share them with individual Runhouse accounts, my team,
    # my company, or the general public.

    # We can also call the service via an HTTP endpoint
    BERT_sa_service.http_url()

    # And we can open a generated docsite for the API too
    BERT_sa_service.api_docs()

    # Note sends can contain other sends, and we can turn the script above into a BERT fine-tuning
    # Send to do training-as-a-service, HPO, and more.

    # Many topics are not covered above, such as scheduling this code as a recurring pipeline, how this
    # code looks when resources are already shared among a team or the public, CI/CD, streaming data
    # between Sends, other resource types, versioning and A/B testing, online learning, complex
    # distribution, hyperparameter sweeps, early stopping, large-project code organization, and much more.