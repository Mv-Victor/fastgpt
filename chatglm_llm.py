import json
from langchain.llms.base import LLM
from typing import List
from langchain.llms.utils import enforce_stop_tokens
from transformers import AutoTokenizer, AutoModel
import torch
from model_config import LLM_DEVICE

from typing import Dict, Optional

DEVICE = LLM_DEVICE
DEVICE_ID = "1" if torch.cuda.is_available() else None
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus

    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 transformer.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
    device_map = {'transformer.word_embeddings': 0,
                  'transformer.final_layernorm': 0, 'lm_head': 0}

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'transformer.layers.{i}'] = gpu_target
        used += 1

    return device_map

def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text

class ChatGLM(LLM):
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    history = []
    tokenizer: object = None
    model: object = None
    history_len: int = 10

    def __init__(self, config_str='{}'):
        super().__init__()
        config = json.loads(config_str)
        self.max_token:int = config['max_token'] if 'max_token' in config else self.max_token
        self.temperature: float = config['temperature'] if 'temperature' in config else self.temperature
        self.top_p: float = config['top_p'] if 'top_p' in config else self.top_p
        self.history_len: int = config['history_len'] if 'history_len' in config else self.history_len

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    def stream_generate_prompt(self, prompt: str):
        for response, _ in self.model.stream_chat(
                self.tokenizer,
                prompt,
                history=self.history[-self.history_len:] if self.history_len>0 else [],
                max_length=self.max_token,
                temperature=self.temperature,
        ):
            yield parse_text(response)

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None) -> str:
        response, _ = self.model.chat(
            self.tokenizer,
            prompt,
            history=self.history[-self.history_len:] if self.history_len>0 else [],
            max_length=self.max_token,
            temperature=self.temperature,
        )
        torch_gc()
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        return parse_text(response)

    def load_model(self,
                   model_name_or_path: str = "THUDM/chatglm-6b",
                   llm_device=LLM_DEVICE,
                   device_map: Optional[Dict[str, int]] = None,
                   **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        if torch.cuda.is_available() and llm_device.lower().startswith("cuda"):
            # 根据当前设备GPU数量决定是否进行多卡部署
            num_gpus = torch.cuda.device_count()
            if num_gpus < 2 and device_map is None:
                self.model = (
                    AutoModel.from_pretrained(
                        model_name_or_path,
                        trust_remote_code=True,
                        **kwargs)
                    .half()
                    .cuda()
                    # AutoModel.from_pretrained(model_name_or_path, load_in_8bit=True, trust_remote_code=True,
                    #                           torch_dtype=torch.float16, device_map="auto",)
                )
            else:
                from accelerate import dispatch_model

                model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, **kwargs).half()
                # model = AutoModel.from_pretrained(model_name_or_path, load_in_8bit=True, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto",)
                # 可传入device_map自定义每张卡的部署情况
                if device_map is None:
                    device_map = auto_configure_device_map(num_gpus)

                self.model = dispatch_model(model, device_map=device_map)
        else:
            self.model = (
                AutoModel.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=True)
                .float()
                .to(llm_device)
            )
        self.model = self.model.eval()
