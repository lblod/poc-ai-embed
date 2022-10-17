import json
import logging
from abc import ABC

import torch
from torch import nn
from transformers import RobertaModel, RobertaTokenizer
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class TransformersClassifierHandler(BaseHandler, ABC):
    """
    Transformers text classifier handler class. This handler takes a text (string) and
    as input and returns the classification text based on the serialized transformers' checkpoint.
    """

    def __init__(self):
        super(TransformersClassifierHandler, self).__init__()

        self.torch_softmax = None
        self.metrics = None
        self.manifest = None
        self.tokenizer = None
        self.model = None

        self._batch_size = 0
        self.initialized = False
    def preprocess(self, data: dict):
        """
        Very basic preprocessing code that splits the data into list for text id's and  list for texts.
        :param data:Dict with text
        :return:list for text id's and  list for texts
        """
        logger.info(f"Received data: {data}")
        data = data[0]["body"]["texts"]
        sentences = [d["text"] for d in data]
        ids = [d["text_id"] for d in data]
        return ids, sentences


    def initialize(self, ctx):
        self.manifest = ctx.manifest
        self.metrics = ctx.metrics
        self.torch_softmax = nn.Softmax(dim=1)

        properties = ctx.system_properties
        self._batch_size = properties["batch_size"]
        model_name = "/models/RobertaModel_PDF_V1"

        # Read model serialize/pt file
        self.model = RobertaModel.from_pretrained(model_name)
        self.model.eval()
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)

        self.initialized = True

    def inference(self, inputs: list) -> dict:
        """
        Predict the class of a  list of texts using a trained transformer model.
        As langauge models only embed 512 words at a time, the embedding for the whole text is calculated by
        embedding 512 chunks and taking the average.
        :param inputs: a list of text str
        :return: a list of tensors containing the embeddings
        """

        # NOTE: This makes the assumption that your model expects text to be tokenized
        # with "input_ids" and "token_type_ids" - which is true for some popular transformer models, e.g. bert.
        # If your transformer model expects different tokenization, adapt this code to suit
        # its expected input format.
        def reprocess_encodings(x: list, max_length: int = 512):
            """
            Helper function to split a list of text tokens into slices of 512 and make corresponding attention masks

            :param x:a list of ints representing text tokens
            :param max_length: int length of slices to split the
            token list into.
            :return: dict containing a tensor with input ids and a tensor with attention mask both
            sliced into 512 slices
            """
            resulting_input, input_idss, attention_masks = [], [], []

            for i in range(0, len(x), max_length):
                arr = x[i:i + max_length]

                if len(arr) == 512:
                    input_ids = torch.tensor(arr, dtype=torch.long)
                    attention_mask = torch.ones(512, dtype=torch.long)

                else:  # No chunks of size 512
                    # Create zero padding tensor
                    input_ids, attention_mask = torch.zeros(512, dtype=torch.long), torch.zeros(512, dtype=torch.long)

                    # Update zero padding tensor with actual data
                    len_arr = len(arr)
                    input_ids[:len_arr] = torch.tensor(arr, dtype=torch.long)
                    attention_mask[:len_arr] = torch.ones(len_arr, dtype=torch.long)

                input_idss.append(input_ids)
                attention_masks.append(attention_mask)

            # Batching them together
            input_ids_processed = torch.stack(input_idss)
            attention_ids_processed = torch.stack(attention_masks)

            return {"input_ids": input_ids_processed, "attention_mask": attention_ids_processed}

        # Clean texts up
        texts = [" ".join(text.split()) for text in inputs]

        # I am aware that this is not actually fully batched, but that's not really relevant here.
        with torch.no_grad():
            pooled_embeddings = []
            for text in texts:
                # Tokenize data
                result = reprocess_encodings(self.tokenizer.encode_plus(text, None, add_special_tokens=False)["input_ids"])
                # Embed that and add to list
                pooled_embeddings.append(self.model(**result)["pooler_output"])

        return pooled_embeddings

    def postprocess(self, ids, embeddings):
        """
        Take the average embedding of all embedding chunks of a text
        Zips the Ids and embeddings back up and returns the result in a dict that is serializable

        :param ids: list of int text_ids
        :param embeddings: list of Tensors with embeddings
        :return:Dict text ids and embedding as a list of floats
        """

        processed_embeddings = []
        for embedding in embeddings:
            # Calculate the mean embedding of all embedding chunks
            new_embedding = embedding.squeeze(0) if embedding.shape[0] == 1 else torch.mean(embedding, 0)

            # turn tensor list of float and add to processed list
            processed_embeddings.append(new_embedding.squeeze(0).tolist())

        return {
            "texts": [{"text_id": int(item[0]), "embedding": item[1]} for item in list(zip(ids, processed_embeddings))]}


_service = TransformersClassifierHandler()


def handle(in_data, context):
    try:

        if not _service.initialized:
            _service.initialize(context)

        if in_data is None:
            return None

        ids, texts = _service.preprocess(in_data)
        embeddings = _service.inference(texts)
        json_return = _service.postprocess(ids, embeddings)

        return [json.dumps(json_return)]
    except Exception as e:
        raise e
