import pandas as pd


def remove_constant_columns(df):
    """
    删除DataFrame中的恒定值列并返回被删除列的索引
    参数:
        df: pandas DataFrame
    返回:
        元组(删除恒定列后的DataFrame, 被删除列的索引列表)
    """
    # 找出所有恒定列（只有一个唯一值或所有值相同的列）
    constant_columns = [col for col in df.columns if df[col].nunique() == 1]

    # 获取被删除列的索引
    dropped_indices = [df.columns.get_loc(col) for col in constant_columns]

    if constant_columns:
        return df.drop(columns=constant_columns), dropped_indices
    else:
        return df, []