#!/bin/bash

torchserve --start --model-store=/home/model-server/model-store --ts-config=/home/model-server/config.properties --models=all

tail -f /dev/null
