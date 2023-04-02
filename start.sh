#!/bin/bash

docker build -t rwkv .
docker rm -f raven
# docker run --rm -it --name raven \
docker run -d --name raven --restart always \
	--gpus all \
	-v $PWD/cache:/root/.cache \
	-v $PWD/flagging:/flagging \
	-p 0.0.0.0:7860:7860 rwkv

# docker logs -f raven
