#!/usr/bin/env python
# -*-coding:utf-8-*-

'''
模型： ffm
模型参数：-s 4
        -l 0.00002
        -r 0.02
        -t 70（最大迭代次数）
        -k field数目减1
数值特征：商品销量等级，商品收藏等级，
        店铺历史交易率
        用户历史交易率
        商品历史交易率，商品历史点击占同类比率
        用户距上次点击该商品时长，用户过去一小时浏览该商品次数占同类比例
        用户距上次浏览该类别时长，用户历史浏览该类别占用户历史浏览记录的比例
        用户历史浏览该价位占用户历史浏览记录的比例
稀疏特征：商品类别1（类别2替换类别1），商品城市编码，商品价格等级，商品id
        用户id，用户性别类型，用户年龄等级，用户星级
        店铺id，店铺评论量等级，店铺星级
        小时数，广告展示页面编号
特殊处理: 训练模型时先划分80%训练集和20%的验证集，迭代固定次数后，取验证集误差最小的迭代次数N，然后再将所有数据作为训练集迭代N次
        正式模型时重复训练3次，得到3个模型再将结果平均
结果： A榜（0.08192）

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
import json
import re
import subprocess

from sklearn.preprocessing import *
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, StratifiedKFold
from sklearn.externals import joblib

# 导入数据
def importDf(url, sep=' ', header='infer', index_col=None, colNames=None):
    df = pd.read_csv(url, sep=sep, header=header, index_col=index_col, names=colNames)
    return df

# 缩放字段至0-1
def scalerFea(df, cols):
    df.dropna(inplace=True, subset=cols)
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols].values)
    return df,scaler

# 填补缺失特征
def fillFea(df, feaList, value=0):
    for x in [x for x in feaList if x not in df.columns]:
        df[x] = value
    return df

# 计算单特征与标签的F值
def getFeaScore(X, y, feaNames):
    resultDf = pd.DataFrame(index=feaNames)
    selecter = SelectKBest(f_classif, 'all').fit(X, y)
    resultDf['scores'] = selecter.scores_
    resultDf['p_values'] = selecter.pvalues_
    return resultDf

# 矩估计法计算贝叶斯平滑参数
def countBetaParamByMME(inputArr):
    EX = inputArr.mean()
    EX2 = (inputArr ** 2).mean()
    alpha = (EX*(EX-EX2)) / (EX2 - EX**2)
    beta = alpha * (1/EX - 1)
    return alpha,beta

# 对numpy数组进行贝叶斯平滑处理
def biasSmooth(aArr, bArr, method='MME', alpha=None, beta=None):
    ratioArr = aArr / bArr
    if method=='MME':
        alpha,beta = countBetaParamByMME(ratioArr[ratioArr==ratioArr])
        # print(alpha,beta)
    resultArr = (aArr+alpha) / (bArr+alpha+beta)

    return resultArr

# 转化数据集字段格式，并去重
def formatDf(df):
    df = df.applymap(lambda x: np.nan if (x==-1)|(x=='-1') else x)
    df.drop_duplicates(inplace=True)
    df['context_timestamp'] = pd.to_datetime(df.context_timestamp, unit='s')
    return df

# 拆分多维度拼接的字段
def splitMultiFea(df):
    tempDf = df.drop_duplicates(subset=['item_id'])[['item_id','item_category_list','item_property_list']]
    tempDf['item_category_list'] = tempDf[tempDf.item_category_list.notnull()]['item_category_list'].map(lambda x: x.split(';'))
    tempDf['item_category0'] = tempDf[tempDf.item_category_list.notnull()]['item_category_list'].map(lambda x: x[0])
    tempDf['item_category1'] = tempDf[tempDf.item_category_list.notnull()]['item_category_list'].map(lambda x: x[1] if len(x)>1 else np.nan)
    tempDf['item_category2'] = tempDf[tempDf.item_category_list.notnull()]['item_category_list'].map(lambda x: x[2] if len(x)>2 else np.nan)
    tempDf.loc[tempDf.item_category2.notnull(), 'item_category1'] = tempDf.loc[tempDf.item_category2.notnull(), 'item_category2']
    tempDf['item_property_list'] = tempDf[tempDf.item_property_list.notnull()]['item_property_list'].map(lambda x: x.split(';'))
    df = df.drop(['item_category_list','item_property_list'], axis=1).merge(tempDf, how='left', on='item_id')
    df['predict_category_property'] = df[df.predict_category_property.notnull()]['predict_category_property'].map(
        lambda x: {kv.split(':')[0]:(kv.split(':')[1].split(',') if kv.split(':')[1]!='-1' else []) for kv in x.split(';')})
    return df

# 新增多维度拼接的id特征
def combineKey(df):
    df['user_item'] = df['user_id'].astype('str') + '_' + df['item_id'].astype('str')
    df['user_shop'] = df['user_id'].astype('str') + '_' + df['shop_id'].astype('str')
    return df

# 添加时间特征
def addTimeFea(df):
    df['hour'] = df.context_timestamp.dt.hour
    df['hour2'] = df['hour'] // 2
    df['day'] = df.context_timestamp.dt.day
    df['date'] = pd.to_datetime(df.context_timestamp.dt.date)
    return df

# 填充商品特征缺失值
def itemFeaFillna(df, hisDf=None):
    if isinstance(hisDf, pd.DataFrame):
        originDf = pd.concat([hisDf,df],ignore_index=True)
    else:
        originDf = df.copy()
    tempDf = pd.pivot_table(originDf.drop_duplicates(subset=['item_id']), index=['shop_id'], values=['item_sales_level'], aggfunc=np.median)
    df.loc[df.item_sales_level.isnull(), 'item_sales_level'] = df.loc[df.item_sales_level.isnull(), 'shop_id'].map(lambda x: tempDf.loc[x,'item_sales_level'])

    tempDf = pd.pivot_table(originDf.drop_duplicates(subset=['item_id']), index=['item_collected_level'], values=['item_sales_level'], aggfunc=np.mean)
    df.loc[df.item_sales_level.isnull(), 'item_sales_level'] = df.loc[df.item_sales_level.isnull(), 'item_collected_level'].map(lambda x: tempDf.loc[x,'item_sales_level'])

    df['item_sales_level'] = np.around(df['item_sales_level'])
    return df

# 填充店铺特征缺失值
def shopFeaFillna(df, hisDf=None):
    if isinstance(hisDf, pd.DataFrame):
        originDf = pd.concat([hisDf,df],ignore_index=True)
    else:
        originDf = df.copy()
    df['shop_review_positive_rate'].fillna(originDf['shop_review_positive_rate'].mean(), inplace=True)
    df['shop_score_service'].fillna(originDf['shop_score_service'].mean(), inplace=True)
    df['shop_score_delivery'].fillna(originDf['shop_score_delivery'].mean(), inplace=True)
    df['shop_score_description'].fillna(originDf['shop_score_description'].mean(), inplace=True)
    return df

# 添加商品类别统计特征
def addCateFea(df, hisDf=None):
    if isinstance(hisDf, pd.DataFrame):
        originDf = pd.concat([hisDf,df],ignore_index=True)
    else:
        originDf = df.copy()
    tempDf = pd.pivot_table(originDf, index=['item_category1','date'], values=['is_trade'], aggfunc=[len, np.sum])
    tempDf.columns = ['show','trade']
    tempDf.reset_index(inplace=True)
    tempDf['same'] = tempDf['item_category1'].shift(1)
    tempDf['same'] = tempDf['same']==tempDf['item_category1']
    showList = []
    tradeList = []
    showTemp = tradeTemp = 0
    for same,c,t in tempDf[['same','show','trade']].values:
        showList.append(showTemp if same else 0)
        tradeList.append(tradeTemp if same else 0)
        showTemp = showTemp+c if same else c
        tradeTemp = tradeTemp+t if same else t
    tempDf['cate_his_show'] = showList
    tempDf['cate_his_trade'] = tradeList
    tempDf['cate_his_trade_ratio'] = biasSmooth(tempDf.cate_his_trade.values, tempDf.cate_his_show.values)
    df = df.merge(tempDf[['item_category1','date','cate_his_show','cate_his_trade','cate_his_trade_ratio']], how='left', on=['item_category1','date'])
    return df

# 添加商品历史浏览量及购买量特征
def addShopFea(df, hisDf=None):
    if isinstance(hisDf, pd.DataFrame):
        originDf = pd.concat([hisDf,df],ignore_index=True)
    else:
        originDf = df.copy()
    tempDf = pd.pivot_table(originDf, index=['shop_id','date'], values='is_trade', aggfunc=[len, sum])
    tempDf.columns = ['show','trade']
    tempDf.reset_index(inplace=True)
    tempDf['same_shop'] = tempDf['shop_id'].shift(1)
    tempDf['same_shop'] = tempDf['shop_id'] == tempDf['same_shop']
    showList = []
    tradeList = []
    showTemp = tradeTemp = 0
    for same,c,t in tempDf[['same_shop','show','trade']].values:
        showList.append(showTemp if same else 0)
        tradeList.append(tradeTemp if same else 0)
        showTemp = showTemp+c if same else c
        tradeTemp = tradeTemp+t if same else t
    tempDf['shop_his_show'] = showList
    tempDf['shop_his_trade'] = tradeList
    tempDf['shop_his_trade_ratio'] = biasSmooth(tempDf.shop_his_trade.values, tempDf.shop_his_show.values)
    df = df.merge(tempDf[['shop_id','date','shop_his_show','shop_his_trade','shop_his_trade_ratio']], how='left', on=['shop_id','date'])
    df['shop_his_trade_ratio'].fillna(0, inplace=True)
    df['shop_his_show_ratio'] = biasSmooth(df.shop_his_show.values, df.cate_his_show.values)
    df['shop_catetrade_ratio_delta'] = df['shop_his_trade_ratio'] - df['cate_his_trade_ratio']

    shopDf = originDf.drop_duplicates(['item_category1','shop_id'])
    tempDf = pd.pivot_table(shopDf, index=['item_category1'], values=['shop_review_positive_rate','shop_score_service','shop_score_delivery','shop_score_description'], aggfunc=np.mean)
    tempDf.columns = ['shop_review_positive_mean','shop_score_delivery_mean','shop_score_description_mean','shop_score_service_mean']
    tempDf.reset_index(inplace=True)
    df = df.merge(tempDf, how='left', on='item_category1')
    df['shop_review_positive_delta'] = df['shop_review_positive_rate'] - df['shop_review_positive_mean']
    df['shop_score_service_delta'] = df['shop_score_service'] - df['shop_score_service_mean']
    df['shop_score_delivery_delta'] = df['shop_score_delivery'] - df['shop_score_delivery_mean']
    df['shop_score_description_delta'] = df['shop_score_description'] - df['shop_score_description_mean']
    return df

# 添加商品历史浏览量及购买量特征
def addItemFea(df, hisDf=None):
    if isinstance(hisDf, pd.DataFrame):
        originDf = pd.concat([hisDf,df],ignore_index=True)
    else:
        originDf = df.copy()
    tempDf = pd.pivot_table(originDf, index=['item_id','date'], values='is_trade', aggfunc=[len, sum])
    tempDf.columns = ['show','trade']
    tempDf.reset_index(inplace=True)
    tempDf['same_item'] = tempDf['item_id'].shift(1)
    tempDf['same_item'] = tempDf['item_id'] == tempDf['same_item']
    showList = []
    tradeList = []
    showTemp = tradeTemp = 0
    for same,c,t in tempDf[['same_item','show','trade']].values:
        showList.append(showTemp if same else 0)
        tradeList.append(tradeTemp if same else 0)
        showTemp = showTemp+c if same else c
        tradeTemp = tradeTemp+t if same else t
    tempDf['item_his_show'] = showList
    tempDf['item_his_trade'] = tradeList
    tempDf['item_his_trade_ratio'] = biasSmooth(tempDf.item_his_trade.values, tempDf.item_his_show.values)
    df = df.merge(tempDf[['item_id','date','item_his_show','item_his_trade','item_his_trade_ratio']], how='left', on=['item_id','date'])
    df['item_his_trade_ratio'].fillna(0, inplace=True)
    df['item_his_show_ratio'] = biasSmooth(df.item_his_show.values, df.cate_his_show.values)
    df['item_catetrade_ratio_delta'] = df['item_his_trade_ratio'] - df['cate_his_trade_ratio']

    itemDf = originDf.drop_duplicates(['item_id'])
    tempDf = pd.pivot_table(itemDf, index=['item_category1'], values=['item_price_level','item_sales_level'], aggfunc=np.mean)
    tempDf.columns = ['cate_price_mean','cate_sales_mean']
    tempDf.reset_index(inplace=True)
    df = df.merge(tempDf, how='left', on='item_category1')
    df['item_cateprice_delta'] = df['item_price_level'] - df['cate_price_mean']
    df['item_catesales_delta'] = df['item_sales_level'] - df['cate_sales_mean']
    return df

# 添加用户维度特征
def addUserFea(df, hisDf=None):
    if isinstance(hisDf, pd.DataFrame):
        originDf = pd.concat([hisDf,df],ignore_index=True)
    else:
        originDf = df.copy()
    tempDf = pd.pivot_table(originDf, index=['user_id','date'], values=['is_trade'], aggfunc=[len,np.sum])
    tempDf.columns = ['show', 'trade']
    tempDf.reset_index(inplace=True)
    tempDf['last_user'] = tempDf['user_id'].shift(1)
    tempDf['last_user'] = tempDf['user_id']==tempDf['last_user']
    showList,tradeList = [[] for i in range(2)]
    showTemp = tradeTemp = 0
    for same,show,trade in tempDf[['last_user','show', 'trade']].values:
        showList.append(showTemp if same else 0)
        showTemp = showTemp+show if same else show
        tradeList.append(tradeTemp if same else 0)
        tradeTemp = tradeTemp+trade if same else trade
    tempDf['user_his_show'] = showList
    tempDf['user_his_trade'] = tradeList
    tempDf['user_his_trade_ratio'] = biasSmooth(tempDf.user_his_trade.values, tempDf.user_his_show.values)
    df = df.merge(tempDf[['user_id','date','user_his_show', 'user_his_trade','user_his_trade_ratio']], how='left', on=['user_id','date'])
    df['user_his_trade_ratio'].fillna(0, inplace=True)

    tempDf = pd.pivot_table(originDf, index=['user_id','context_timestamp'], values=['is_trade'], aggfunc=len)
    tempDf.columns = ['show']
    tempDf.reset_index(inplace=True)
    tempDf['last_user_id'] = tempDf['user_id'].shift(1)
    tempDf['last_user_id'] = tempDf['last_user_id']==tempDf['user_id']
    tempDf['last_show_time'] = tempDf['context_timestamp'].shift(1)
    tempDf.loc[~tempDf.last_user_id, 'last_show_time'] = np.nan
    tempDf['user_last_show_timedelta'] = tempDf['context_timestamp'] - tempDf['last_show_time']
    tempDf['user_last_show_timedelta'] = tempDf['user_last_show_timedelta'].dt.seconds
    tempDf['user_last_show_timedelta'].fillna(999999, inplace=True)
    hourShowList = []
    hourShowTemp = {}
    for same, dt, show in tempDf[['last_user_id','context_timestamp','show']].values:
        if same:
            [hourShowTemp.pop(k) for k in list(hourShowTemp) if k<dt-timedelta(hours=1)]
            hourShowList.append(np.sum(list(hourShowTemp.values())))
            hourShowTemp[dt] = show
        else:
            hourShowList.append(0)
            hourShowTemp = {dt:show}
    tempDf['user_lasthour_show'] = hourShowList
    df = df.merge(tempDf[['user_id','context_timestamp','user_last_show_timedelta','user_lasthour_show']], how='left', on=['user_id','context_timestamp'])
    return df

# 添加用户与类目关联维度的特征
def addUserCateFea(df, hisDf=None):
    if isinstance(hisDf, pd.DataFrame):
        originDf = pd.concat([hisDf,df],ignore_index=True)
    else:
        originDf = df.copy()
    tempDf = pd.pivot_table(originDf, index=['user_id','item_category1','context_timestamp'], values=['is_trade'], aggfunc=len)
    tempDf.columns = ['show']
    tempDf.reset_index(inplace=True)
    tempDf[['last_user','last_cate','last_time']] = tempDf[['user_id','item_category1','context_timestamp']].shift(1)
    tempDf['same'] = (tempDf.user_id==tempDf.last_user) & (tempDf.item_category1==tempDf.last_cate)
    tempDf.loc[~tempDf.same, 'last_time'] = np.nan
    tempDf['uc_last_show_timedelta'] = tempDf['context_timestamp'] - tempDf['last_time']
    tempDf['uc_last_show_timedelta'] = tempDf['uc_last_show_timedelta'].dt.seconds
    tempDf['uc_last_show_timedelta'].fillna(999999, inplace=True)
    hourShowList = []
    hourShowTemp = {}
    for same, dt, show in tempDf[['same','context_timestamp','show']].values:
        if same:
            [hourShowTemp.pop(k) for k in list(hourShowTemp) if k<dt-timedelta(hours=1)]
            hourShowList.append(np.sum(list(hourShowTemp.values())))
            hourShowTemp[dt] = show
        else:
            hourShowList.append(0)
            hourShowTemp = {dt:show}
    tempDf['uc_lasthour_show'] = hourShowList
    df = df.merge(tempDf[['user_id','item_category1','context_timestamp','uc_last_show_timedelta','uc_lasthour_show']], how='left', on=['user_id','item_category1','context_timestamp'])

    tempDf = pd.pivot_table(originDf, index=['user_id','item_category1','date'], values=['is_trade'], aggfunc=[len,np.sum])
    tempDf.columns = ['show', 'trade']
    tempDf.reset_index(inplace=True)
    tempDf[['last_user','last_cate']] = tempDf[['user_id','item_category1']].shift(1)
    tempDf['same'] = (tempDf.last_user==tempDf.user_id) & (tempDf.last_cate==tempDf.item_category1)
    showList,tradeList = ([] for i in range(2))
    showTemp = tradeTemp = 0
    for same,show,trade in tempDf[['same','show','trade']].values:
        showList.append(showTemp if same else 0)
        tradeList.append(tradeTemp if same else 0)
        showTemp = showTemp+show if same else show
        tradeTemp = tradeTemp+trade if same else trade
    tempDf['uc_his_show'] = showList
    tempDf['uc_his_trade'] = tradeList
    tempDf['uc_his_trade_ratio'] = biasSmooth(tempDf.uc_his_trade.values, tempDf.uc_his_show.values)
    df = df.merge(tempDf[['user_id','item_category1','date','uc_his_show', 'uc_his_trade','uc_his_trade_ratio']], how='left', on=['user_id','item_category1','date'])
    df.fillna({k:0 for k in ['uc_lastdate_show','uc_lastdate_trade','uc_his_trade_ratio']}, inplace=True)
    df['uc_his_show_ratio'] = biasSmooth(df.uc_his_show.values, df.user_his_show.values)
    return df

# 添加用户商品关联维度的特征
def addUserItemFea(df, hisDf=None):
    if isinstance(hisDf, pd.DataFrame):
        originDf = pd.concat([hisDf,df],ignore_index=True)
    else:
        originDf = df.copy()
    tempDf = pd.pivot_table(originDf, index=['user_item','context_timestamp'], values=['is_trade'], aggfunc=len)
    tempDf.columns = ['show']
    tempDf.reset_index(inplace=True)
    tempDf['last_user_item'] = tempDf['user_item'].shift(1)
    tempDf['last_user_item'] = tempDf['last_user_item']==tempDf['user_item']
    tempDf['last_show_time'] = tempDf['context_timestamp'].shift(1)
    tempDf.loc[~tempDf.last_user_item, 'last_show_time'] = np.nan
    tempDf['ui_last_show_timedelta'] = tempDf['context_timestamp'] - tempDf['last_show_time']
    tempDf['ui_last_show_timedelta'] = tempDf['ui_last_show_timedelta'].dt.seconds
    tempDf['ui_last_show_timedelta'].fillna(999999, inplace=True)
    hourShowList = []
    hourShowTemp = {}
    for same, dt, show in tempDf[['last_user_item','context_timestamp','show']].values:
        if same:
            [hourShowTemp.pop(k) for k in list(hourShowTemp) if k<dt-timedelta(hours=1)]
            hourShowList.append(np.sum(list(hourShowTemp.values())))
            hourShowTemp[dt] = show
        else:
            hourShowList.append(0)
            hourShowTemp = {dt:show}
    tempDf['ui_lasthour_show'] = hourShowList
    df = df.merge(tempDf[['user_item','context_timestamp','ui_last_show_timedelta','ui_lasthour_show']], how='left', on=['user_item','context_timestamp'])
    df['ui_lasthour_show_ratio'] = biasSmooth(df.ui_lasthour_show.values, df.user_lasthour_show.values)

    tempDf = pd.pivot_table(originDf, index=['user_item','date'], values=['is_trade'], aggfunc=[len,np.sum])
    tempDf.columns = ['ui_lastdate_show', 'ui_lastdate_trade']
    tempDf.reset_index(inplace=True)
    tempDf['date'] = tempDf['date'] + timedelta(days=1)
    df = df.merge(tempDf[['user_item','date','ui_lastdate_show', 'ui_lastdate_trade']], how='left', on=['user_item','date'])
    df.fillna({k:0 for k in ['ui_lastdate_show', 'ui_lastdate_trade']}, inplace=True)
    return df

# 统计用户该价格段商品的统计特征
def addUserPriceFea(df, hisDf=None):
    if isinstance(hisDf, pd.DataFrame):
        originDf = pd.concat([hisDf,df], ignore_index=True)
    else:
        originDf = df.copy()
    tempDf = pd.pivot_table(originDf, index=['user_id','item_price_level','date'], values='is_trade', aggfunc=[len,np.sum])
    tempDf.columns = ['show','trade']
    tempDf.reset_index(inplace=True)
    tempDf[['last_user','last_price']] = tempDf[['user_id','item_price_level']].shift(1)
    tempDf['same'] = (tempDf['user_id']==tempDf['last_user']) & (tempDf['item_price_level']==tempDf['last_price'])
    showList,tradeList = ([] for i in range(2))
    showTemp = tradeTemp = 0
    for same,show,trade in tempDf[['same','show','trade']].values:
        showList.append(showTemp if same else 0)
        tradeList.append(tradeTemp if same else 0)
        showTemp = showTemp+show if same else show
        tradeTemp = tradeTemp+trade if same else trade
    tempDf['up_his_show'] = showList
    tempDf['up_his_trade'] = tradeList
    df = df.merge(tempDf[['user_id','item_price_level','date','up_his_show','up_his_trade']], how='left', on=['user_id','item_price_level','date'])
    df['up_his_show_ratio'] = biasSmooth(df.up_his_show.values, df.user_his_show.values)
    df.fillna({k:0 for k in ['up_his_show','up_his_trade','up_his_show_ratio']}, inplace=True)
    return df

# 将级别转换成权重
def rankToWeight(df, colNames):
    for col in colNames:
        df[col] = np.max(df[col])+1 - df[col]
    print(df[colNames].head())
    return df

# 特征方法汇总
def feaFactory(df, hisDf=None):
    startTime = datetime.now()
    df = formatDf(df)
    df = splitMultiFea(df)
    df = combineKey(df)
    df = addTimeFea(df)
    df = addCateFea(df, hisDf)
    df = addUserFea(df, hisDf)
    df = addShopFea(df, hisDf)
    df = addItemFea(df, hisDf)
    df = itemFeaFillna(df, hisDf)
    df = shopFeaFillna(df, hisDf)
    # df = rankToWeight(df, ['user_last_show_timedelta','context_page_id'])
    df = addUserCateFea(df, hisDf)
    df = addUserItemFea(df, hisDf)
    df = addUserPriceFea(df, hisDf)
    print('finished feaFactory: ', datetime.now() - startTime)
    return df

# 划分训练集和测试集
def trainTestSplit(df, splitDate=pd.to_datetime('2018-09-23'), trainPeriod=3, testPeriod=1):
    trainIdx = df[(df.context_timestamp<splitDate)&(df.context_timestamp>=splitDate-timedelta(days=trainPeriod))].index
    testIdx = df[(df.context_timestamp>=splitDate)&(df.context_timestamp<splitDate+timedelta(days=testPeriod))].index
    return (trainIdx, testIdx)

# 导出预测结果
def exportResult(df, fileName, header=True, index=False, sep=' '):
    df.to_csv('./%s' % fileName, sep=sep, header=header, index=index)

# 获取stacking下一层数据集
def getOof(clf, trainX, trainY, testX, nFold=5):
    oofTrain = np.zeros(trainX.shape[0])
    oofTest = np.zeros(testX.shape[0])
    oofTestSkf = np.zeros((testX.shape[0], nFold))
    kf = KFold(n_splits=nFold, shuffle=True)
    for i, (trainIdx, testIdx) in enumerate(kf.split(trainX)):
        kfTrainX = trainX[trainIdx]
        kfTrainY = trainY[trainIdx]
        kfTestX = trainX[testIdx]
        clf.trainAutoIter(kfTrainX, kfTrainY, verbose=False)
        oofTrain[testIdx] = clf.predict(kfTestX)
        oofTestSkf[:,i] = clf.predict(testX)
    oofTest[:] = oofTestSkf.mean(axis=1)
    return oofTrain, oofTest

class LibFFM():
    def __init__(self, df, numFea, strFea, labelName, multiFea=[]):
        ffmDf, scaler = scalerFea(df.fillna({k:0 for k in numFea}), numFea)
        onehotEncoders,feaNo = self.getOnehotFeaDict(ffmDf, strFea, startCount=len(numFea))
        if len(multiFea)>0:
            mulChoiceEncoders, feaNo = self.getMulChoiceFeaDict(ffmDf, multiFea, startCount=feaNo)
        else:
            mulChoiceEncoders = None
        self.label = labelName
        self.strFea = strFea
        self.numFea = numFea
        self.scaler = scaler
        self.onehotEncoders = onehotEncoders
        self.mulChoiceEncoders = mulChoiceEncoders
        self.fieldNum = len(numFea) + len(strFea) + len(multiFea)
        self.feaNum = feaNo

    # 获取onehot字段的特征值编码字典
    def getOnehotFeaDict(self, df, fea, startCount=0):
        feaCount = startCount
        onehotEncoders = {k:dict() for k in fea}
        for f in fea:
            values = set(df[f].dropna().values)
            onehotEncoders[f] = {v:i+feaCount for i,v in enumerate(values)}
            feaCount += len(values)
        return onehotEncoders,feaCount

    # 获取onehot字段的特征值编码字典
    def getMulChoiceFeaDict(self, df, fea, startCount=0):
        feaCount = startCount
        mulChoiceEncoders = {k:dict() for k in fea}
        for f in fea:
            values = []
            [values.extend(x) for x in df[f].dropna().values]
            values = set(values)
            mulChoiceEncoders[f] = {v:i+feaCount for i,v in enumerate(values)}
            feaCount += len(values)
        return mulChoiceEncoders,feaCount

    # 获取ffm数据集
    def getFFMDf(self, df):
        strFea = self.strFea
        numFea = self.numFea
        if self.label not in df.columns:
            ffmDf = df[numFea + strFea]
        else:
            ffmDf = df[[self.label] + numFea + strFea]

        ffmDf.fillna({k:0 for k in numFea}, inplace=True)
        ffmDf.loc[:,numFea] = self.scaler.transform(ffmDf[numFea].values)

        for i,fea in enumerate(numFea):
            ffmDf.loc[:,fea] = list(map(lambda x: '%d:%d:%f'%(i,i,x) if x==x else None, ffmDf[fea]))
        offset = len(numFea)
        for i,fea in enumerate(strFea):
            fieldId = i+offset
            onehotEncoder = self.onehotEncoders[fea]
            ffmDf.loc[:,fea] = list(map(lambda x: '%d:%d:%d'%(fieldId,onehotEncoder[x],1) if x==x else None, ffmDf[fea]))
        return ffmDf

    # 以ffm格式导出数据集
    def exportFFMDf(self, df, filePath):
        df.to_csv(filePath, sep=' ', header=False, index=False)

    # 导入ffm结果数据集
    def importFFMResult(self, url):
        result = pd.read_csv(url, header=None, index_col=None)
        return result[0].values

    # 调用ffm程序训练模型
    def train(self, train, test=None, thread=4, l2=0.00002, eta=0.02, iter_num=100, factor=None, modelName='ffm_model_temp', trainFileName='ffm_train_temp.ffm', testFileName='ffm_valid_temp.ffm', autoStop=True):
        factor = self.fieldNum-1 if factor==None else factor
        hasTest = isinstance(test, np.ndarray)
        autoStop = hasTest&autoStop
        self.exportFFMDf(pd.DataFrame(train), trainFileName)
        self.exportFFMDf(pd.DataFrame(test), testFileName) if hasTest else None
        cmd = ['libffm/ffm-train','-s', str(thread),'-l',str(l2),'-r',str(eta),'-t',str(iter_num),'-k',str(factor)]
        if hasTest:
            cmd.extend(['-p',testFileName])
            if autoStop:
                cmd.extend(['--auto-stop'])
        cmd.extend([trainFileName,'%s.model'%modelName])
        result = subprocess.check_output(cmd)
        print(result.decode())
        if autoStop:
            iterNum, trainLoss, testLoss = re.findall(r"\n\s+(\d+)\s+([\d.]+)\s+([\d.]+)\nAuto-stop", result.decode(), re.S)[0]
            return int(iterNum), float(trainLoss), float(testLoss)
        elif hasTest:
            iterNum, trainLoss, testLoss = re.findall(r"\n\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+\n$", result.decode(), re.S)[0]
            return float(trainLoss), float(testLoss)
        else:
            iterNum, trainLoss = re.findall(r"\n\s+(\d+)\s+([\d.]+)\s+[\d.]+\n$", result.decode(), re.S)[0]
            return int(iterNum), float(trainLoss)

    # 调用ffm程序训练模型
    def trainAutoIter(self, X, y, thread=4, l2=0.00002, eta=0.02, iter_num=70, factor=None, modelName='ffm_model_temp', trainFileName='ffm_train_temp.ffm', testFileName='ffm_valid_temp.ffm', verbose=True):
        factor = self.fieldNum-1 if factor==None else factor
        train,test,_,_ = train_test_split(np.column_stack((y, X)), y, test_size=0.2, stratify=y)
        self.exportFFMDf(pd.DataFrame(train), trainFileName)
        self.exportFFMDf(pd.DataFrame(test), testFileName)
        cmd = ['libffm/ffm-train','-s', str(thread),'-l',str(l2),'-r',str(eta),'-t',str(iter_num),'-k',str(factor),'-p',testFileName]
        cmd.extend([trainFileName,'%s.model'%modelName])
        # startTime = datetime.now()
        result = subprocess.check_output(cmd)
        # print('pretraining time:', datetime.now() - startTime)
        lines = re.findall(r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+$", result.decode(), re.M)
        resultDf = pd.DataFrame(np.array(lines).astype(float), columns=['iter_num','train_loss','test_loss'])
        resultDf['rank'] = resultDf['test_loss'].rank(method='first')
        if verbose:
            print(resultDf)
        idx = resultDf[resultDf['rank']==1].index[0]
        iterNum = resultDf.loc[idx,'iter_num']

        self.exportFFMDf(pd.DataFrame(np.column_stack((y, X))), trainFileName)
        cmd = ['libffm/ffm-train','-s', str(thread),'-l',str(l2),'-r',str(eta),'-t',str(iterNum),'-k',str(factor)]
        cmd.extend([trainFileName,'%s.model'%modelName])
        startTime = datetime.now()
        result = subprocess.check_output(cmd)
        # print('training time:', datetime.now() - startTime)
        return iterNum

    # 调用ffm程序预测测试集
    def predict(self, testDf, modelName='ffm_model_temp', testFileName='ffm_test_temp.ffm', outputName='ffm_predict_temp.txt'):
        self.exportFFMDf(testDf, testFileName)
        cmd = ['libffm/ffm-predict', testFileName, '%s.model'%modelName, outputName]
        subprocess.check_output(cmd)
        result = self.importFFMResult(outputName)
        return result

    # 获取stacking下一层数据集
    def getOof(self, trainDf, testDf, nFold=5):
        oofTrain = np.zeros(trainDf.shape[0])
        oofTest = np.zeros(testDf.shape[0])
        oofTestSkf = np.zeros((testDf.shape[0], nFold))
        kf = KFold(n_splits=nFold, shuffle=True)
        for i, (trainIdx, testIdx) in enumerate(kf.split(trainDf.values)):
            kfTrainDf = trainDf.iloc[trainIdx,:]
            kfTestDf = trainDf.iloc[testIdx,:]
            self.trainAutoIter(kfTrainDf[numFea+strFea], kfTrainDf['is_trade'])
            oofTrain[testIdx] = self.predict(kfTestDf)
            oofTestSkf[:,i] = self.predict(testDf)
        oofTest[:] = oofTestSkf.mean(axis=1)
        return oofTrain, oofTest

if __name__ == '__main__':
    # 导入数据
    df = importDf('../data/round1_ijcai_18_train_20180301.txt')
    predictDf = importDf('../data/round1_ijcai_18_test_a_20180301.txt')

    # 特征处理
    df = feaFactory(df)
    predictDf = feaFactory(predictDf, df)

    # 特征筛选
    numFea = [
        'item_sales_level','item_collected_level',
        # 'shop_review_positive_rate','shop_score_service','shop_score_delivery','shop_score_description',
        'shop_his_trade_ratio',#'shop_his_show','shop_his_trade','shop_score_service_delta','shop_score_delivery_delta','shop_score_description_delta','shop_review_positive_delta',
        'user_his_trade_ratio',#'user_last_show_timedelta','user_his_show',
        'item_his_trade_ratio','item_his_show_ratio',#'item_catesales_delta','item_cateprice_delta','item_his_show','item_his_trade',
        'ui_last_show_timedelta','ui_lasthour_show_ratio',
        'uc_last_show_timedelta','uc_his_trade_ratio',
        'up_his_show_ratio',
    ]
    strFea = [
        'item_category1','item_city_id','item_price_level','item_id',#'item_brand_id','item_pv_level',
        'user_id','user_gender_id','user_age_level','user_star_level',
        'shop_id','shop_review_num_level','shop_star_level',
        'hour','context_page_id',
    ]
    print(df[numFea+strFea].info())
    # exit()

    # 将数据集转成FFM模型数据集
    startTime = datetime.now()
    libFFM = LibFFM(pd.concat([df, predictDf],ignore_index=True), numFea, strFea, labelName='is_trade')
    print('field count:', libFFM.fieldNum)
    print('feature count:', libFFM.feaNum)
    totalFFMDf, predictFFMDf = [libFFM.getFFMDf(x) for x in [df, predictDf]]
    print('transform time:', datetime.now()-startTime)

    # 测试模型效果
    tempDf = pd.DataFrame(index=['iter','loss'])
    for dt in pd.date_range(start='2018-09-21', end='2018-09-23', freq='D'):
        trainIdx, testIdx = trainTestSplit(df, dt, trainPeriod=7)
        trainDf = df.loc[trainIdx]
        testDf = df.loc[testIdx]
        trainFFMDf = totalFFMDf.loc[trainIdx]
        testFFMDf = totalFFMDf.loc[testIdx]
        iterNum = libFFM.trainAutoIter(trainFFMDf[numFea+strFea].values, trainFFMDf['is_trade'].values)
        tempDf[dt] = [iterNum, metrics.log_loss(testDf['is_trade'].values, libFFM.predict(testFFMDf))]
    print(tempDf)
    exit()

    # 正式模型训练及预测
    modelName = "ffm1A"
    tempDf = pd.DataFrame(index=predictDf.index)
    repeatTimes = 3
    for i in range(repeatTimes):
        iterNum = libFFM.trainAutoIter(totalFFMDf[numFea+strFea].values, totalFFMDf['is_trade'].values)
        print('iter num:',iterNum)
        tempDf[i] = libFFM.predict(predictFFMDf)   
    predictDf['predicted_score'] = tempDf.mean(axis=1)
    print(predictDf.info())
    print('predicted_score average:',predictDf['predicted_score'].mean())
    exportResult(predictDf[['instance_id','predicted_score']], "%s.txt" % modelName)
    exit()

    # 生成stacking数据集
    df['predicted_score'] = np.nan
    predictDf['predicted_score'] = np.nan
    # df.loc[:,'predicted_score'], predictDf.loc[:,'predicted_score'] = getOof(libFFM, totalFFMDf[numFea+strFea].values, totalFFMDf['is_trade'].values, predictFFMDf.values)
    df.loc[:,'predicted_score'], predictDf.loc[:,'predicted_score'] = libFFM.getOof(totalFFMDf, predictFFMDf)
    exportResult(df[['instance_id','predicted_score']], "%s_oof_train.csv" % modelName)
    exportResult(predictDf[['instance_id','predicted_score']], "%s_oof_test.csv" % modelName)
    # exportResult(predictDf[['instance_id','predicted_score']], "%s.txt" % modelName)
