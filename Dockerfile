FROM python:3.7
LABEL maintainer="https://github.com/Asrst"

ADD requirements.txt
RUN pip install -r requirements.txt

EXPOSE 80

COPY ./app /app

COPY ./model_store /model_store

WORKDIR "/app"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
