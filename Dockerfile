FROM python:3.9.5-slim-buster

RUN apt update -y
WORKDIR /app

COPY . /app
RUN apt-get update && pip install -r requirements.txt

CMD ["python3", "app.py"]