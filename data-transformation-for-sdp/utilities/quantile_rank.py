import numpy as np
import pandas as pd


def calculate_quantiles(column_data, num_quantiles=10):
    """计算单列的分位数点"""
    quantiles = [np.percentile(column_data, k * 10) for k in range(1, num_quantiles)]
    return quantiles


def rank_transform_column(column_data, quantiles):
    """将单列数据转换为等级"""
    ranked_data = np.zeros_like(column_data)
    ranked_data[column_data <= quantiles[0]] = 1  # Q1

    for k in range(1, len(quantiles)):
        mask = (column_data > quantiles[k - 1]) & (column_data <= quantiles[k])
        ranked_data[mask] = k + 1

    ranked_data[column_data > quantiles[-1]] = 10  # > Q9
    return ranked_data


def rank_transform_dataframe(df, reference_df=None):
    """
    对DataFrame的每一列进行十分位数等级转换

    参数:
    df: 要转换的DataFrame
    reference_df: 可选，用于计算分位数的参考DataFrame(如训练集+目标集)
                 如果不提供，则使用df自身计算分位数

    返回:
    转换后的DataFrame，值变为1-10的等级
    """
    if reference_df is not None:
        # 确保两个DataFrame的列相同
        assert set(df.columns) == set(reference_df.columns), "列名不匹配"
        reference_data = reference_df
    else:
        reference_data = df

    ranked_df = pd.DataFrame(index=df.index)

    for col in df.columns:
        # 计算该列的分位数
        col_quantiles = calculate_quantiles(reference_data[col].values)

        # 转换该列数据
        ranked_df[col] = rank_transform_column(df[col].values, col_quantiles)

    return ranked_df.astype(int)  # 确保返回整数类型