import pandas as pd
import polars as pl
from matplotlib import pyplot as plt

from examples.sympy_define import *
from expr_codegen.expr import string_to_exprs
from expr_codegen.tool import ExprTool

# 防止sympy_define导入被IDE删除
_ = Eq

# ======================================
# 数据准备，请先运行同目录下的`prepare_data.py`
df_input = pl.read_parquet('data.parquet')
df_input = pd.read_parquet('data.parquet')
df_output = None


def main():
    # 表达式设置
    exprs_src = """
    MA_10=ts_mean(CLOSE, 10)
    MA_40=ts_mean(ts_mean(CLOSE, 5), 40)
    """
    exprs_src = string_to_exprs(exprs_src, globals())

    # 生成代码
    tool = ExprTool(date='date', asset='asset')
    codes, G = tool.all(exprs_src, style='pandas', template_file='template.py.j2', fast=True)

    # 打印代码
    print(codes)

    # 执行代码
    exec(codes, globals())

    # 写在def中时，exec中的df就取不到了，达到了保护数据的目的
    # UnboundLocalError: cannot access local variable 'df' where it is not associated with a value
    # print(df.columns)
    print(df_input.columns)
    print(df_output.columns)

    # df = df_output.to_pandas()
    df = df_output
    df = df.set_index(['asset', 'date'])

    for s in ['s_100', 's_200']:
        stock = df.loc[s]
        stock[['CLOSE', 'MA_10', 'MA_40']].plot()
    plt.show()


if __name__ == "__main__":
    main()
