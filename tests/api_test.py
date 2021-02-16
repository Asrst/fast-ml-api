import os
os.sys.path.append('.')
from fastapi.testclient import TestClient
from project_api.main import app
from sklearn import datasets


def test_prediction_success():
    api_route = '/sklearn-gnb/predict'
    X, y = datasets.make_classification(n_samples=1000, n_features=10)
    request_body = {"inputs": X.tolist()}

    with TestClient(app) as client:
        response = client.post(api_route, json=request_body)
        response_json = response.json()
        # print(response_json)
        assert response.status_code == 200
        assert 'predictions' in response_json
        print('api_test success...')


if __name__ == '__main__':
    test_prediction_success()
