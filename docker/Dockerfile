FROM python:3.7-slim-buster

RUN apt-get update && apt-get install -y --no-install-recommends \
     gcc \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6

COPY requirements.txt .

RUN pip install -r requirements.txt

CMD "/bin/bash"
# CMD ["cd", "gui", "&&", "python", "bin/main.py"]
