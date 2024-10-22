import pandas as pd
import numpy as np
import re
from gpparse import GPParse
from .functions_brandnew import _function_map
class GPParse:

    def __init__(self):
        pass

    @classmethod
    def parse(cls, expression, factor_data):

        # 判断是因子还是公式
        if expression.count('(') == 0:
            if expression.count(',') == 0:
                param_list = factor_data[expression.strip()].values
            else:
                param_list = expression.split(',')
                param_list = [factor_data[s.strip()].values for s in param_list]
            return param_list
        # 对表达式中的公式和因子进行分解
        else:
            start_index = expression.find('(')
            end_index = expression.rfind(')')
            function = expression[: start_index]
            param = expression[start_index + 1: end_index]

            # 对参数的表达式进行解析
            index_list = [m.start() for m in re.finditer(',', param)]
            index_list.append(len(param))
            param_list = []
            start_index = 0

            for index in index_list:
                p = param[start_index: index]
                if p.count('(') == p.count(')'):
                    param_list.append(p.strip())
                    start_index = index + 1

            # if len(param_list) ==  _function_map[function].arity:
            #     print('function: %s的参数为：' % function)
            #     print(param_list)

            if len(param_list) == 1:
                return _function_map[function](cls.parse(param_list[0], factor_data))
            else:
                return _function_map[function](*[cls.parse(p, factor_data) for p in param_list])
