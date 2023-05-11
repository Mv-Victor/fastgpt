#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   local_qa_emb_model.py    
@Contact :   1181348296@qq.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/4/19 21:33   HZD      1.0         None
'''
from langchain import LLMChain, PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import *
import os
from typing import List
import sentence_transformers

from chatglm_llm import ChatGLM
from utils import class_from_path
from loguru import logger
from model_config import EMBEDDING_DEVICE,EMBEDING_MODEL_DICT,SYS_DOC_LOADER_PATH,LLM_MODEL_DICT,LLM_DEVICE,LLM_MODEL,EMB_MODEL


# Embedding running device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class LocalQAEmb:
    llm: object = None
    embeddings: object = None

    def init_cfg(self,
                 embedding_model: str = EMB_MODEL,
                 llm_model: str = LLM_MODEL,
                 llm_device: str = LLM_DEVICE,
                 llm_config_str: str = '{}',
                 ):
        '''
        初始化模型
        :param embedding_model: 三种emb模型一种
        :return:
        '''
        self.llm = ChatGLM(config_str=llm_config_str)
        self.llm.load_model(model_name_or_path=LLM_MODEL_DICT[llm_model],
                            llm_device=llm_device)
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDING_MODEL_DICT[embedding_model], )
        self.embeddings.client = sentence_transformers.SentenceTransformer(self.embeddings.model_name,
                                                                           device=EMBEDDING_DEVICE)

    def get_query_emb(self, query):
        '''
        获取query的emb
        :param query:
        :return:
        '''
        vectors = self.embeddings.embed_query(query)
        return vectors

    def get_doc_emb(self,
                    filepath: str or List[str],
                    loaderType: str or List[str],):
        '''
        获取文档的emb
        :param filepath: 文档路径
        :param loaderType: langchain.document_loaders定义的加载文档类型
        :return: 1024维度*doc_num的向量
        '''
        if isinstance(filepath, str):
            if not os.path.exists(filepath):
                logger.error("文档路径不存在")
                return
            elif os.path.isfile(filepath):
                file = os.path.split(filepath)[-1]
                try:
                    loader = class_from_path(SYS_DOC_LOADER_PATH + loaderType)(filepath, mode="elements")
                    docs = loader.load()
                    logger.info(f"{file} 已成功加载")
                except Exception as e:
                    logger.error(f"{file} 未能成功加载")
                    logger.error(e)
                    logger.error(traceback.format_exc())
                    return
            elif os.path.isdir(filepath):
                docs = []
                for file in os.listdir(filepath):
                    fullfilepath = os.path.join(filepath, file)
                    try:
                        loader = class_from_path(SYS_DOC_LOADER_PATH + loaderType)(fullfilepath, mode="elements")
                        docs += loader.load()
                        logger.info(f"{file} 已成功加载")
                    except Exception as e:
                        logger.error(f"{file} 未能成功加载")
                        logger.error(e)
                        logger.error(traceback.format_exc())
                        return
        else:
            docs = []
            for file in filepath:
                try:
                    loader = class_from_path(SYS_DOC_LOADER_PATH + loaderType)(file, mode="elements")
                    docs += loader.load()
                    logger.info(f"{file} 已成功加载")
                except:
                    logger.error(f"{file} 未能成功加载")
                    logger.error(e)
                    logger.error(traceback.format_exc())
                    return

        chunks = [doc.page_content for doc in docs]
        vectors = self.embeddings.embed_documents(chunks)
        return chunks, vectors

    def get_knowledge_based_answer(self,
                                   query,
                                   chat_history=[],
                                   context = [],
                                   stream = False):
        '''
        根据prompt调好的query输出回答
        :param query:
        :param chat_history: 历史问答记录
        :param context: 最近邻搜索出的上下文信息
        :param stream: 是否流式返回
        :return:
        '''
        self.llm.history = chat_history

        template = """
        参考上下文回答问题
        上下文: {context}
        问题: {question}"""
        prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=' '.join(context))

        if stream:
            query = prompt.format()
            for result in self.llm.stream_generate_prompt(query):
                logger.info(f"streaming query:{query} \n answer:{result}")
                yield result

        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        result = llm_chain.run(query)
        logger.info(f"query:{query} \n answer:{result}")
        return result
