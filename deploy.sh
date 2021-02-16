docker build -t fast-api-build .
docker run -d -p 80:80 --name fast_ml_api fast-ml-api-build
