FROM python:3.7-slim-buster

RUN apt-get update && apt-get install -y --no-install-recommends gcc

COPY requirements.txt .

RUN pip install -r requirements.txt

# COPY ../bin .

# CMD python bin/main.py
CMD "/bin/bash"
