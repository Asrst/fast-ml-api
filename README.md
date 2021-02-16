#### Fast-ML-API

[![Gitpod Ready-to-Code](https://img.shields.io/badge/Gitpod-ready--to--code-blue?logo=gitpod)](https://gitpod.io/#https://github.com/Asrst/fast-ml-api)

- ML model inference serving template using Fast API and Docker

#### Other Features to consider
- Using `ujson` or `rjson`.
- Token Auth.
- Exception Handling.
- Model Versioning with APIRoute.
- Message Broker to support Request Batching.

# Instructions

### 1. Clone the repository OR Use Docker
#### manually
```
cd fast-ml-api
bash run.sh
python tests/api_test.py
uvicorn project_api.main:app
```
(OR)

#### build the docker-image & run the container
```
bash deploy.sh
```

### 2. Go to docs to get available apis
http://127.0.0.1:8000/docs


### 3. Stop the server
```
ctrl + c
```