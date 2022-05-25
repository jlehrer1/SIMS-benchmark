FROM pytorch/pytorch

WORKDIR /src

RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN apt-get --allow-releaseinfo-change update && \
    apt-get install -y --no-install-recommends \
        curl \
        sudo \
        vim 

RUN curl -L https://bit.ly/glances | /bin/bash

RUN pip install scvi-tools wandb 
RUN pip install pytorch_lightning
RUN pip install boto3 

COPY . .