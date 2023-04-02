#!/bin/bash

docker build -t rwkv .
docker rm -f rwkv-ui
docker run -d --name rwkv-ui --restart always --gpus all -v $PWD/cache:/root/.cache -p 0.0.0.0:7860:7860 rwkv
