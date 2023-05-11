#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ASGI_app.py    
@Contact :   1181348296@qq.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/5/5 15:49   HZD      1.0         None
'''
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
from fastapi import FastAPI, Request, status
from fastapi.responses import StreamingResponse,JSONResponse
from local_qa_emb_model import LocalQAEmb
import uvicorn, json
from loguru import logger

from model_config import LLM_MODEL

logger.add("log/fastgpt_log_{time}.log", rotation="500MB", encoding="utf-8", enqueue=True, retention="10 days")
app = FastAPI()

@app.post('/v1/doc_embeddings')
async def get_doc_emb(request: Request):
    '''
    TODO: 增加对oss的支持
    :return:
    '''
    request_body = await request.json()
    request_body = json.dumps(request_body)
    request_body = json.loads(request_body)
    doc_path = request_body.get('doc_path', './static/FastGPT.html')
    loader_type = request_body.get('loader_type', 'UnstructuredHTMLLoader')
    emb_model = request_body.get('model', 'ernie-base')
    local_doc_qa = LocalQAEmb()
    try:
        local_doc_qa.init_cfg(emb_model)
        logger.info("""文档向量模型已成功加载""")
        chunks, vec = local_doc_qa.get_doc_emb(doc_path, loader_type)
        res = {'embedding':vec, 'chunks':chunks}
        logger.info("""文档向量返回:%s"""%str(res))
        return res
    except Exception as e:
        logger.error("""文档向量模型加载失败""")
        logger.error(e)
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"data":"文档向量模型加载失败"})

@app.post('/v1/embeddings')
async def get_query_emb(request: Request):
    request_body = await request.json()
    request_body = json.dumps(request_body)
    request_body = json.loads(request_body)
    emb_model = request_body.get('model', 'ernie-base')
    query = request_body.get('input')
    local_doc_qa = LocalQAEmb()
    try:
        local_doc_qa.init_cfg(emb_model)
        logger.info("""查询模型已成功加载""")
        vec = local_doc_qa.get_query_emb(query)
        res = {'embedding':vec}
        logger.info("""查询向量返回:%s"""%str(res))
        return res
    except Exception as e:
        logger.error("""查询模型加载失败""")
        logger.error(e)
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"data":"查询模型加载失败"})

@app.post('/v1/chat/completions')
async def get_llm_req(request: Request):
    '''
    TODO: 增加对stream返回的支持
    :return:
    '''
    request_body = await request.json()
    request_body = json.dumps(request_body)
    request_body = json.loads(request_body)
    query = request_body.get('message')
    stream = request_body.get('stream', False)
    history = request_body.get('history', [])
    context = request_body.get('context', [])
    llm_model = request_body.get('model', LLM_MODEL)
    llm_config_str = request_body.get('config_str', '{}')
    local_doc_qa = LocalQAEmb()
    try:
        local_doc_qa.init_cfg(llm_model=llm_model, llm_config_str=llm_config_str)
        logger.info("""LLM模型已成功加载""")
        if stream:
            return StreamingResponse(local_doc_qa.get_knowledge_based_answer(query, history, context), media_type="text/event-stream")
        resp = local_doc_qa.get_knowledge_based_answer(query, history, context)
        res = {'message':resp}
        logger.info("""LLM回答返回:%s"""%str(res))
        return res
    except Exception as e:
        logger.error("""LLM模型加载失败""")
        logger.error(e)
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"data":"LLM模型加载失败"})

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5005, workers=1)