"""
Function: Normalizing data is very useful in deep learning or other machine learning algorithms. This script provides
          two method which the first one is to normalize pandas.DataFrame, return normalized data, mean values and
          std values.
          The second one is to recover raw data by normalized data, mean values and std values.
          正规化是一种非常有用的方法，它经常被用在算法的数据预处理中，这个脚本提供了两个方法，第一个是将pandas.DataFrame变成正规化的数据，
          并返回均值和方差，第二个方法是根据正规化后的数据、均值和方差来恢复原始数据。
          在预测任务中，我们经常先要正规化输入数据，并将原始结果恢复。
Author: Du Fei
Create Time: 2020/7/18 16:19
"""

import numpy as np
import pandas as pd


def recover_data_from_normalized(_normalized_df, _mean_df, _std_df, _axis=1):
    """
    Recover data by normalized results, mean value and std value
    :param _normalized_df: normalized data
    :param _mean_df: mean value
    :param _std_df: std value
    :param _axis: normalized data along index/row (0) or column (1)
    :return:
    """

    # if axis=0, then the shape of _mean_df and _std_df is [n_cols,1], thus, we should transpose _df
    if _axis == 0:
        _mean_df = _mean_df.T
        _std_df = _std_df.T

    return _normalized_df * _std_df.values + _mean_df.values


def normalize_df(_df, _axis=1):
    """
    Normalize pandas.DataFrame
    :param _df: raw input DataFrame
    :param _axis: normalized data along index/row (0) or column (1)
    :return:
    """

    _mean_df = _df.mean(axis=_axis).to_frame()
    _std_df = _df.std(axis=_axis, ddof=1).to_frame()

    # if axis=0, then the shape of _mean_df and _std_df is [n_cols,1], thus, we should transpose _df
    if _axis == 0:
        res_df = ((_df.T - _mean_df.values) / _std_df.values).T
    else:
        res_df = (_df - _mean_df.values) / _std_df.values

    return res_df, _mean_df, _std_df


def generate_data(_rows, _cols):
    """
    This method will generate sample data
    :param _rows: number of rows
    :param _cols: number of index
    :return numpy array
    """
    res_array = np.zeros((_rows, _cols))
    for row_index in range(_rows):
        for col_index in range(_cols):
            res_array[row_index][col_index] = row_index + col_index + 1

    return res_array


if __name__ == '__main__':
    axis = 0

    input_df = pd.DataFrame(generate_data(5, 6))
    print("----------input df-------------------")
    print(input_df)

    normalized_df, mean_df, std_df = normalize_df(input_df, _axis=axis)
    print("----------normalized df-------------------")
    print(normalized_df)

    raw_df = recover_data_from_normalized(normalized_df, mean_df, std_df, _axis=axis)
    print("----------raw df-------------------")
    print(raw_df)
