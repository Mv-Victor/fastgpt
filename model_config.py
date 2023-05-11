#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   config.py    
@Contact :   1181348296@qq.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/4/20 14:19   HZD      1.0         None
'''
import torch.cuda
import torch.backends
import langchain.document_loaders
# loader path
SYS_DOC_LOADER_PATH = "langchain.document_loaders."

EMBEDING_MODEL_DICT = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
}

# Embedding running device
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# supported LLM models
LLM_MODEL_DICT = {
    "chatglm-6b-int4-qe": "THUDM/chatglm-6b-int4-qe",
    "chatglm-6b-int4": "THUDM/chatglm-6b-int4",
    "chatglm-6b": "THUDM/chatglm-6b",
}

# LLM model name
LLM_MODEL = "chatglm-6b"

# embedding model name
EMB_MODEL = "ernie-base"

# LLM running device
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

