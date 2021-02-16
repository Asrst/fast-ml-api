from locust import HttpUser, TaskSet, task, between, tag
from sklearn import datasets

"""
To run locust:
locust -f ./tests/load_test.py
access locust ui @ 127.0.0.1:8089
"""


class SklearnPredict(TaskSet):
    @task
    def predict(self):
        X, y = datasets.make_classification(n_samples=1000, n_features=10)
        request_body = {"inputs": X.tolist()}
        self.client.post('/sklearn-gnb/predict', json=request_body)

    # @tag('healthcheck')
    # @task
    # def health_check(self):
    #     self.client.get('/')


class SklearnFastAPILoadTest(HttpUser):
    tasks = [SklearnPredict]
    host = 'http://127.0.0.1:8000'
    stop_timeout = 200
    # wait_time = between(1, 2)
