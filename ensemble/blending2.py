#!/usr/bin/env python
# -*-coding:utf-8-*-

'''
融合：加权平均
权值：0.4yuna + 0.1yuna_normal + 0.2keng + 0.2keng_normal + 0.1lake
前期处理：将两个模型结果先按12小时调均值再整体调均值，然后才融合
成绩：B榜(13961)
'''

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from scipy.stats import mode
import csv
import matplotlib.dates
import matplotlib.pyplot as plt
from datetime import *
import urllib, urllib.parse, urllib.request
import json, random

# 导入数据
def importDf(url, sep=' ', na_values='-1', header='infer', index_col=None, colNames=None):
    df = pd.read_csv(url, sep=sep, na_values='-1', header=header, index_col=index_col, names=colNames)
    return df

# 导出预测结果
def exportResult(df, fileName, header=True, index=False, sep=' '):
    df.to_csv('./%s' % fileName, sep=sep, header=header, index=index)

def main():
    # 数据导入
    dfs = []
    df = importDf('../xgb1/xgboost2B.txt', index_col=0)
    df.columns = ['yuna']
    dfs.append(df)
    df = importDf('../xgb1/xgboost2B_15normal.txt', index_col=0)
    df.columns = ['yuna_normal']
    dfs.append(df)
    df = importDf('../keng/fusai_b_xgb_5_15_adjust.txt', index_col=0)
    df.columns = ['keng']
    dfs.append(df)
    df = importDf('../keng/fusai_b_xgb_5_14_normal_adjust.txt', index_col=0)
    df.columns = ['keng_normal']
    dfs.append(df)
    df = importDf('../lake/lgb2B_with_normal_hashour_after_lake_5_15_9_all.txt', index_col=0)
    df.columns = ['lake_normal']
    dfs.append(df)
    df = pd.concat(dfs, axis=1)

    # 模型训练
    modelName = 'blending2B'
    df['predicted_score'] = 0.4*df['yuna'] + 0.1*df['yuna_normal'] + 0.2*df['keng'] + 0.2*df['keng_normal'] + 0.1*df['lake_normal']
    print(df.head())

    # 结果导出
    df.reset_index(inplace=True)
    exportResult(df, "%s_preview.csv" % modelName)
    exportResult(df[['instance_id','predicted_score']], "%s.txt" % modelName)

if __name__ == '__main__':
    main()
