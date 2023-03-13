from p01_preprocess import tokenize_function
import requests
import ray
from ray import serve
import ray.cloudpickle as pickle
import runhouse as rh

"""
Steps to create an inference service:
(1) Initialize Ray and connect Python to Ray cluster
(2) Build an HTTP server for each actor / class using the @serve.deployment decorator.
(3) For each actor, create a __call__ method to handle incoming requests (this will be run when the service is called).
(4) Start the model serving process via serve.start()
(5) Using actor.deploy(), deploy the actor as an HTTP server onto the head node.
"""


@serve.deployment(route_prefix="/sentiment_scores")  # accept REST requests at the route specified
class ModelInference:
    def __init__(self, model_name):
        """Load the model by specified name."""
        self.model = pickle.loads(rh.blob(name=model_name, dryrun=True).data)

    async def __call__(self, request):
        """Receive the request object, process it, call the model with the samples provided, and return result."""
        payload: bytes = await request.body()
        samples = pickle.loads(payload)
        tokens = tokenize_function(samples)
        result = self.model(tokens)
        return pickle.dumps(result)


if __name__ == "__main__":
    # Create a Ray cluster
    ray.init(address="auto")

    serve.start(address="auto", http_options={"host": "0.0.0.0"})

    # deploys the ModelInference actor at the associated REST endpoint we specified in the decorator
    ModelInference.deploy(model_name='yelp_fine_tuned_bert')

    new_examples = [
        'This place is sick!',
        'The service left much to be desired.',
        'The appetizers were hit or miss.',
        'The elote ribs are a must.',
    ]

    resp = requests.post(url="http://127.0.0.1:8000/sentiment_scores", data=pickle.dumps(new_examples))

    sentiment_scores = pickle.loads(resp.content)

    print('Test samples and sentiment scores:')
    [print(f'{sample}: {score}') for (sample, score) in zip(new_examples, sentiment_scores)]
