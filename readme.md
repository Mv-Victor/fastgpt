# 模型私有化部署fastgpt

## 介绍

💡 受 imClumsyPanda 的项目 [langchain-ChatGLM](https://github.com/imClumsyPanda/langchain-ChatGLM) 启发，采用flask、uvicorn建立了基于开源模型实现的文档与查询embedding转换库

Embedding 选用的是 [GanymedeNil/text2vec-large-chinese](https://huggingface.co/GanymedeNil/text2vec-large-chinese/tree/main)

## 使用方式

### 硬件需求
- Embedding 模型硬件需求

  本项目中默认选用的 Embedding 模型 [GanymedeNil/text2vec-large-chinese](https://huggingface.co/GanymedeNil/text2vec-large-chinese/tree/main) 约占用显存 3GB，也可修改为在 CPU 中运行。v0版本大小18.7GB，相比于v1版本的4.74GB增加了Ernie模型等依赖，v1版本直接访问接口即可自动下载模型。

### api调用示例
#### 获取文档向量
```
localhost:port/v1/doc_embeddings
method: 'POST'
```
##### requestParams:  
**'doc_path'**, default='./static/FastGPT.html', type=str, 含义是文件路径  
**'loader_type'**, default='UnstructuredHTMLLoader', type=str， 含义是loader类型，参考[这里](https://python.langchain.com/en/latest/modules/indexes/document_loaders.html)  
**'model'**, default='ernie-base', type=str， 含义是嵌入模型，目前支持"ernie-tiny"、"ernie-base"、"text2vec"

##### return:
{'doc_vec':vec, 'chunks':chunks}
doc_vec含义是文档向量  
chunks含义是向量的语义，和doc_vec同dim0大小

#### 获取查询向量
```
localhost:port/v1/embeddings
method: 'POST'
```
##### requestParams:
**'message'**, type=str, 含义是查询语句  
**''**, default='ernie-base', type=str， 含义是嵌入模型，目前支持"ernie-tiny"、"ernie-base"、"text2vec"

##### return:
{'message_vec':vec, 'message':message}  
message_vec含义是查询向量  
message

#### 获取llm回答
```
localhost:/v1/chat/completions
method: 'POST'
```
##### requestParams:
**'message'**, type=str, 含义是prompt更新后的问题  
**'history'**, type=list, default=[], 含义是历史对话信息  
**'context'**, type=list, default=[], 含义是相似度匹配出的上下文  
**'stream'**, type=bool, default=False, 含义是是否采用流式返回，如为True则返回 SSE 数据流  
**'model'**, default='chatglm-6b-int4', type=str, 含义是llm模型, 目前支持"chatglm-6b-int4-qe"、"chatglm-6b-int4"、"chatglm-6b"  
**'config_str'**, default='{}', type=str, 含义是模型配置信息, 目前支持max_token: int、temperature: float、top_p: float、history_len: int

##### return:
{'answer':resp}

命令行访问示例
```shell
curl -X POST "http://127.0.0.1:5001/v1/chat/completions" -H 'Content-Type: application/json' -d '{"message": "什么是aha？","history":[["Aha是一个校园企业，专注竞赛分享，你记住了吗","我记住了"]]}'

curl -X POST "localhost:5001/v1/embeddings" -H 'Content-Type: application/json' -d '{"input": "aha_pocket"}' -o /home/user/fastgpt/test_gpt.txt

curl -X POST "localhost:5001/v1/doc_embeddings" -H 'Content-Type: application/json' -d '{"doc_path": "./static/FastGPT.html","loader_type":"UnstructuredHTMLLoader","model":"ernie-base"}' -o /home/user/fastgpt/test_gpt.txt
```

# 腾讯云镜像仓库拉取
docker pull ccr.ccs.tencentyun.com/aha_zjut/fastgpt:v0

