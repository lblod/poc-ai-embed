# poc-ai-embed
This service is a pytorch serve api.

## Getting started
In order to run this code, you will either have to build it localy or use our build container. Keep in mind you will have to
supply the container with our model (you can copy it from a google storage bucket).

### Copy our model form the google storage bucket (public read access)
An easy way to download files from a google cloud bucket, is by installing the gsutil client. You can find more information
on how to install this client here.

Once the client is installed, you can execute the following command to pull the models to your current folder
```
gsutil -m cp -r gs://abb-textgen-models/RobertaModel_PDF_V1 .
```

### Starting the docker container
First you pull the container (can be skipped --> will be pulled either if not present when executing the run command)
```
docker pull lblod/poc-ai-embed
```

```
docker run -it --rm  -v <folder_container_model>:/models/ -p 8080:8080 lblod/poc-ai-embed
```

## How it works
For the inner workings of the code, you could check out the [serve_pretrained.py](https://github.com/lblod/poc-ai-embed/blob/master/serve_pretrained.py) file. The documentation in there should
suffice in explaining the concept. In case you would want to test the api, you can look at the [example_request.py](https://github.com/lblod/poc-ai-embed/blob/master/example_request.py) script.


Embedding API by ML2Grow
