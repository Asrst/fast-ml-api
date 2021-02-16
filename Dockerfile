FROM python:3.8
LABEL maintainer="https://github.com/Asrst"

ADD requirements.txt
RUN pip install -r requirements.txt

EXPOSE 80

COPY ./project_api /project_api

COPY ./model_store /model_store

WORKDIR "/project_api"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
