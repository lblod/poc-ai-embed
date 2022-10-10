FROM pytorch/torchserve:0.6.0-cpu

COPY serve_pretrained.py .
COPY ./models /home/model-server/models
COPY ./config.properties /home/model-server/config.properties
COPY ./pts-entrypoint.sh /home/model-server/pts-entrypoint.sh

RUN torch-model-archiver --model-name=EmbedBert \
    -r "/home/model-server/models/requirements.txt"\
    --version=1.0 \
    --serialized-file=/home/model-server/models/fake_model.bin \
    --handler=serve_pretrained.py  \
    --runtime=python3 \
    --extra-files="/home/model-server/models/config.json" \
    --export-path=/home/model-server/model-store

ENV TRANSFORMERS_OFFLINE='1'
CMD [ "/bin/bash", "/home/model-server/pts-entrypoint.sh" ]
