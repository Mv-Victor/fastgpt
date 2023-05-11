#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py    
@Contact :   1181348296@qq.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/4/16 22:27   HZD      1.0         None
'''
import importlib
from typing import Callable

def class_from_path(path: str) -> Callable:
    module_name, class_name = path.rsplit(".", 1)
    class_object = getattr(importlib.import_module(module_name), class_name)
    return class_object