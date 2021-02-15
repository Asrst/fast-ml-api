docker build -t fast-ml-api-build .
docker run -d -p 80:80 --name fast-ml-api fast-ml-api-build
