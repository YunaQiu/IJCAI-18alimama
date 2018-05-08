#!/usr/bin/env python
# -*-coding:utf-8-*-

'''
目标均值：
0    0.051762
1    0.044351
2    0.039073
3    0.038068
4    0.040123
5    0.044436
6    0.043755
7    0.048183
8    0.047861
9    0.045140
10   0.043612
11   0.043149
12   0.042641
13   0.040599
14   0.039930
15   0.040100
16   0.039957
17   0.040177
18   0.039958
19   0.039547
20   0.039542
21   0.036749
22   0.035445
23   0.034270
'''

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from scipy.stats import mode
import math
import csv
import matplotlib.dates
import matplotlib.pyplot as plt
from datetime import *
import urllib, urllib.parse, urllib.request
import json, random

#定义调整函数
def resultAdjustment(result_df, t, showMean=False):
    result_df_temp = result_df.copy()
    result_df_temp['x'] = result_df_temp.predicted_score.map(lambda x: -(math.log(((1 - x) / x), math.e)))
    result_df_temp['adjust_result'] = result_df_temp.x.map(lambda x: 1 / (1 + math.exp(-(x + t))))
    if showMean:
    	print(result_df_temp['adjust_result'].mean())
    return result_df_temp['adjust_result']

# 导出预测结果
def exportResult(df, fileName, header=True, index=False, sep=' '):
    df.to_csv('./%s' % fileName, sep=sep, header=header, index=index)

def main():
	df = pd.read_csv('xgboost2A_hashour.txt', sep=' ')
	print(df.info())
	df2 = df.copy()
	df2.loc[df2.hour==12, 'predicted_score'] = resultAdjustment(df2[df2.hour==12], -0.17146)
	df2.loc[df2.hour==13, 'predicted_score'] = resultAdjustment(df2[df2.hour==13], -0.21287)
	df2.loc[df2.hour==14, 'predicted_score'] = resultAdjustment(df2[df2.hour==14], -0.21203)
	df2.loc[df2.hour==15, 'predicted_score'] = resultAdjustment(df2[df2.hour==15], -0.20677)
	df2.loc[df2.hour==16, 'predicted_score'] = resultAdjustment(df2[df2.hour==16], -0.2235)
	df2.loc[df2.hour==17, 'predicted_score'] = resultAdjustment(df2[df2.hour==17], -0.2248)
	df2.loc[df2.hour==18, 'predicted_score'] = resultAdjustment(df2[df2.hour==18], -0.23838)
	df2.loc[df2.hour==19, 'predicted_score'] = resultAdjustment(df2[df2.hour==19], -0.2590)
	df2.loc[df2.hour==20, 'predicted_score'] = resultAdjustment(df2[df2.hour==20], -0.27083)
	df2.loc[df2.hour==21, 'predicted_score'] = resultAdjustment(df2[df2.hour==21], -0.356)
	df2.loc[df2.hour==22, 'predicted_score'] = resultAdjustment(df2[df2.hour==22], -0.4127)
	df2.loc[df2.hour==23, 'predicted_score'] = resultAdjustment(df2[df2.hour==23], -0.45808)
	print(pd.pivot_table(df2, index=['hour'], values='predicted_score', aggfunc=np.mean))
	print(np.mean(df2.predicted_score))
	exportResult(df2[['instance_id','predicted_score']], "%s.txt" % 'xgboost2A')

if __name__ == '__main__':
	main()
