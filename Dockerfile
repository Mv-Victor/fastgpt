FROM python:3.8
COPY . /fastgpt_sgi/
WORKDIR /fastgpt_sgi/
RUN pip install -r /fastgpt_sgi/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
#CMD ["python", "/fastgpt_sgi/ASGI_app.py"]
CMD ["uvicorn", "ASGI_app:app", "--port=5005", "--workers=2"]
# CMD ["sh", "-c", "python /fastgpt_sgi/ASGI_app.py & python /fastgpt_sgi/app.py --no-reload"]
# docker build -t vc/fastgpt:v1 .
# docker run -it --gpus=all --network host --name fastgpt_sgi -v /home/huangzhidong/fastgpt/log:/fastgpt_sgi/log -d vc/fastgpt:v1
