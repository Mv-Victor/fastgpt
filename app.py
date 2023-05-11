#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Contact :   1181348296@qq.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/4/16 23:50   HZD      1.0         None
'''
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import traceback
from flask import Flask, jsonify, request, make_response
from local_qa_emb_model import LocalQAEmb
from loguru import logger

from model_config import LLM_MODEL

logger.add("log/fastgpt_log_{time}.log", rotation="500MB", encoding="utf-8", enqueue=True, retention="10 days")
app = Flask(__name__)

@app.route('/v1/doc_embeddings', methods=['POST'])
def get_doc_emb():
    '''
    TODO: 增加对oss的支持
    :return:
    '''
    request_body = request.get_json()
    doc_path = request_body.get('doc_path', './static/FastGPT.html')
    loader_type = request_body.get('loader_type', 'UnstructuredHTMLLoader')
    emb_model = request_body.get('model', 'ernie-base')
    local_doc_qa = LocalQAEmb()
    try:
        local_doc_qa.init_cfg(emb_model)
        logger.info("""文档向量模型已成功加载""")
        chunks, vec = local_doc_qa.get_doc_emb(doc_path, loader_type)
        res = jsonify({'embedding':vec, 'chunks':chunks})
        logger.info("""文档向量返回:%s"""%str(res))
        return res
    except Exception as e:
        logger.error("""文档向量模型加载失败""")
        logger.error(e)
        logger.error(traceback.format_exc())
        return make_response(jsonify({"data":"文档向量模型加载失败"}), 504)

@app.route('/v1/embeddings', methods=['POST'])
def get_query_emb():
    request_body = request.get_json()
    emb_model = request_body.get('model', 'ernie-base')
    query = request_body.get('input')
    local_doc_qa = LocalQAEmb()
    try:
        local_doc_qa.init_cfg(emb_model)
        logger.info("""查询模型已成功加载""")
        vec = local_doc_qa.get_query_emb(query)
        res = jsonify({'embedding':vec})
        logger.info("""查询向量返回:%s"""%str(res))
        return res
    except Exception as e:
        logger.error("""查询模型加载失败""")
        logger.error(e)
        logger.error(traceback.format_exc())
        return make_response(jsonify({"data":"查询模型加载失败"}), 505)

@app.route('/v1/chat/completions', methods=['POST'])
def get_llm_req():
    '''
    TODO: 增加对stream返回的支持
    :return:
    '''
    request_body = request.get_json()
    query = request_body.get('message')
    history = request_body.get('history', [])
    context = request_body.get('context', [])
    llm_model = request_body.get('model', LLM_MODEL)
    llm_config_str = request_body.get('config_str', '{}')
    local_doc_qa = LocalQAEmb()
    try:
        local_doc_qa.init_cfg(llm_model=llm_model, llm_config_str=llm_config_str)
        logger.info("""LLM模型已成功加载""")
        resp = local_doc_qa.get_knowledge_based_answer(query, history, context)
        res = jsonify({'message':resp})
        logger.info("""LLM回答返回:%s"""%str(res))
        return res
    except Exception as e:
        logger.error("""LLM模型加载失败""")
        logger.error(e)
        logger.error(traceback.format_exc())
        return make_response(jsonify({"data":"LLM模型加载失败"}), 506)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5004, debug=True)