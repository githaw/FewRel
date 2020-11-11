# -*- encoding: utf8 -*-

"""
__author__ = Jocky Hawk
__copyright__ = Copyright 2020
__version__ = 0.1
__status = Dev
"""

import os.path
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import scipy
import torch  # PyTorch
from transformers import BertConfig, BertModel, BertTokenizer


def get_tokens(text: str, tokenizer: BertTokenizer, config: BertConfig) -> List[str]:
    tokens = tokenizer.tokenize(text)

    max_length = config.max_position_embeddings
    tokens = tokens[: max_length - 1]

    tokens = [tokenizer.cls_token] + tokens

    return tokens


if __name__ == "__main__":

    # https://huggingface.co/transformers/pretrained_models.html
    # model_name = "bert-base-uncased"
    # model_name = "./chinese_electra_small_discriminator_pytorch"
    model_name = "google/electra-small-discriminator"

    # Need to use the same tokenizer that was used to train the model so that it breaks
    # up words into tokens the same way.
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # This model is huge!!!!!!!!
    model = BertModel.from_pretrained(model_name)

    # Parameters used by the pre-trained model
    config = BertConfig.from_pretrained(model_name)

    text = "I want to the store."
    # text = "我想去小卖部。"
    tokens = get_tokens(text, tokenizer, config)
    print(tokens)

    token_ids: List[int] = tokenizer.convert_tokens_to_ids(tokens)
    print(token_ids)

    token_ids_tensor = torch.tensor(token_ids)
    print(token_ids_tensor.shape, token_ids_tensor)

    token_ids_tensor = torch.unsqueeze(token_ids_tensor, 0)
    print(token_ids_tensor.shape, token_ids_tensor)

    last_hidden_state, pooler_output = model(token_ids_tensor)

    # pooler output is the last layer hidden state of the first token.
    # Since this uses attention, it takes the whole sequence into account.
    vector = pooler_output

    print(type(vector), vector, vector.shape)
