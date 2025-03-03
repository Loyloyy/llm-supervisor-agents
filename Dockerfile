FROM nvcr.io/nvidia/pytorch:24.01-py3
LABEL authors="Aloysius_Tan"

# Allow statements and log messages to immediately appear in Knative logs
ENV PYTHONUNBUFFERED=True
ARG DEBIAN_FRONTEND=noninteractive


# SYSTEM
RUN apt-get update --yes --quiet && apt-get install --yes --quiet --no-install-recommends \
    software-properties-common \
    build-essential apt-utils \
    wget curl vim git ca-certificates kmod libcairo2-dev pkg-config 

# PYTHON 3.11
RUN apt-get update --yes --quiet
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3.11-lib2to3 \
    python3.11-gdbm \
    python3.11-tk \
    pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 999
RUN update-alternatives --config python3 #&& ln -s /usr/bin/python3 /usr/bin/python

ENV APP_HOME=/app
WORKDIR $APP_HOME

COPY requirements.txt /app/requirements.txt
COPY ./images /app/images
COPY ./src/app.py /app/app.py
COPY ./src/tools.py /app/tools.py
COPY ./src/utils.py /app/utils.py

RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir

EXPOSE 7860

CMD ["python", "app.py"]