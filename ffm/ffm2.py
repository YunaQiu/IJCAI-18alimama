#!/usr/bin/env python
# -*-coding:utf-8-*-

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

# 分箱函数
def feaBinning(arr, bins, start=0, fillna=None):
    result = arr.copy()
    if fillna!=None:
        result[np.isnan(result)] = fillna
    for i,b in enumerate(bins):
        if i==0:
            idx = np.where(arr<b)
        else:
            idx = np.where((arr<b)&(arr>=bins[i-1]))
        result[idx] = start+i
    idx = np.where(arr>=bins[-1])
    result[idx] = start+len(bins)
    return result

# 对数组集合进行合并操作
def listAdd(l):
    result = []
    [result.extend(x) for x in l]
    return result

# 转化数据集字段格式，并去重
def formatDf(df):
    df = df.applymap(lambda x: np.nan if (x==-1)|(x=='-1') else x)
    df['context_timestamp'] = df['context_timestamp'].map(lambda x: datetime.fromtimestamp(x))
    return df

# 拆分多维度拼接的字段
def splitMultiFea(df):
    tempDf = df.drop_duplicates(subset=['item_id'])[['item_id','item_category_list','item_property_list']]
    tempDf['item_category_list'] = tempDf[tempDf.item_category_list.notnull()]['item_category_list'].map(lambda x: x.split(';'))
    tempDf['item_category0'] = tempDf[tempDf.item_category_list.notnull()]['item_category_list'].map(lambda x: x[0])
    tempDf['item_category1'] = tempDf[tempDf.item_category_list.notnull()]['item_category_list'].map(lambda x: x[1] if len(x)>1 else np.nan)
    tempDf['item_category2'] = tempDf[tempDf.item_category_list.notnull()]['item_category_list'].map(lambda x: x[2] if len(x)>2 else np.nan)
    # tempDf.loc[tempDf.item_category2.notnull(), 'item_category1'] = tempDf.loc[tempDf.item_category2.notnull(), 'item_category2']
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

# 对商品特征进行预处理
def cleanItemFea(df):
    df.loc[df.item_sales_level<3, 'item_sales_level'] = 3
    df.loc[df.item_sales_level>16, 'item_sales_level'] = 16
    df.loc[df.item_sales_level.isnull(), 'item_sales_level'] = 3
    df.loc[df.item_price_level<4, 'item_price_level'] = 3
    df.loc[df.item_price_level>9, 'item_price_level'] = 9
    df.loc[df.item_collected_level<6, 'item_collected_level'] = 6
    df.loc[df.item_pv_level<10, 'item_pv_level'] = 10
    return df

# 对用户特征进行预处理
def cleanShopFea(df):
    df.loc[df.shop_star_level<5009, 'shop_star_level'] = 5008
    df.loc[df.shop_star_level>5018, 'shop_star_level'] = 5018
    df.loc[df.shop_review_num_level<10, 'shop_review_num_level'] = 9
    df.loc[df.shop_review_num_level>21, 'shop_review_num_level'] = 21
    return df

# 对用户特征进行预处理
def cleanUserFea(df):
    df.loc[df.user_gender_id.isnull(), 'user_gender_id'] = -1
    df.loc[df.user_age_level>1006, 'user_age_level'] = 1006
    df.loc[df.user_star_level>3009, 'user_star_level'] = 3009
    df.loc[df.user_star_level.isnull(), 'user_star_level'] = 3000
    return df

# 添加时间特征
def addTimeFea(df, **params):
    df['hour'] = df.context_timestamp.dt.hour
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

# 按日期统计过去几天的点击数，交易数
def statDateTrade(df, index, statDates=None, skipDates=1):
    tempDf = pd.pivot_table(df, index=index, columns='date', values='is_trade', aggfunc=[len,np.sum])
    if statDates==None:
        for i,dt in enumerate(tempDf.columns.levels[-1][skipDates:]):
            tempDf.loc[:,pd.IndexSlice['show',dt]] = tempDf['len'].iloc[:,:i+skipDates].sum(axis=1)
            tempDf.loc[:,pd.IndexSlice['trade',dt]] = tempDf['sum'].iloc[:,:i+skipDates].sum(axis=1)
    else:
        for i,dt in enumerate(tempDf.columns.levels[-1][statDates:]):
            tempDf.loc[:,pd.IndexSlice['show',dt]] = tempDf['len'].iloc[:,i:i+statDates].sum(axis=1)
            tempDf.loc[:,pd.IndexSlice['trade',dt]] = tempDf['sum'].iloc[:,i:i+statDates].sum(axis=1)
    tempDf = tempDf.stack()
    # tempDf['trade_ratio'] = biasSmooth(tempDf['trade'].values, tempDf['show'].values)
    return tempDf[['show','trade']]

# 按日期统计过去几天的数目，总和
def statDateLenSum(df, index, values, statDates=None, skipDates=1):
    tempDf = pd.pivot_table(df, index=index, columns='date', values=values, aggfunc=[len,np.sum])
    if statDates==None:
        for i,dt in enumerate(tempDf.columns.levels[-1][skipDates:]):
            tempDf.loc[:,pd.IndexSlice['addup_len',dt]] = tempDf['len'].iloc[:,:i+skipDates].sum(axis=1)
            tempDf.loc[:,pd.IndexSlice['addup_sum',dt]] = tempDf['sum'].iloc[:,:i+skipDates].sum(axis=1)
    else:
        for i,dt in enumerate(tempDf.columns.levels[-1][statDates:]):
            tempDf.loc[:,pd.IndexSlice['addup_len',dt]] = tempDf['len'].iloc[:,i:i+statDates].sum(axis=1)
            tempDf.loc[:,pd.IndexSlice['addup_sum',dt]] = tempDf['sum'].iloc[:,i:i+statDates].sum(axis=1)
    tempDf = tempDf.stack()
    return tempDf[['addup_len','addup_sum']]

# 按维度统计用户年龄特征
def statUserAge(df, originDf, index, prefix):
    tempDf = pd.pivot_table(originDf, index=index, values=['user_age_level'], aggfunc=[np.mean, np.std])
    tempDf.columns = [colname%prefix for colname in ['%s_age_mean','%s_age_std']]
    df = df.merge(tempDf, how='left', left_on=index, right_index=True)
    df['%s_age_delta'%prefix] = df['user_age_level'] - df['%s_age_mean'%prefix]
    return df

# 按维度统计用户性别特征
def statUserGender(df, originDf, index, prefix):
    tempDf = pd.pivot_table(originDf, index=index, values='is_trade', columns=['user_gender_id'], aggfunc=[len])
    startTime = datetime.now()
    sumCol = tempDf.sum(axis=1)
    aver = tempDf['len'].sum() / sumCol.sum()
    for x in tempDf.columns.levels[-1]:
        tempDf.loc[:,pd.IndexSlice['ratio',x]] = biasSmooth(tempDf['len'][x].values, sumCol.values)
        tempDf.loc[:,pd.IndexSlice['delta',x]] = tempDf.loc[:,pd.IndexSlice['ratio',x]] - aver.loc[x]
    del tempDf['len']
    tempDf = tempDf.stack()
    cols = ['%s_gender_%s'%(prefix,x) for x in ['ratio_delta','ratio']]
    tempDf.columns = cols
    df = df.merge(tempDf, how='left', left_on=[index,'user_gender_id'], right_index=True)
    return df

# 统计商品特征
def statItemFea(df, originDf, index, prefix):
    itemDf = originDf.drop_duplicates(['item_id'])
    tempDf = pd.pivot_table(itemDf, index=index, values=['item_sales_level','item_collected_level','item_pv_level','item_price_level'], aggfunc={'item_sales_level':[len,np.sum,np.mean], 'item_collected_level':[np.sum,np.mean], 'item_pv_level':[np.sum,np.mean], 'item_price_level':[np.sum,np.mean]})
    tempDf.columns = [col%prefix for col in ['%s_collected_mean','%s_collected_sum','%s_price_mean','%s_price_sum','%s_pv_mean','%s_pv_sum','%s_item_count','%s_sales_mean','%s_sales_sum']]
    df = df.merge(tempDf, how='left', left_on=index,right_index=True)
    return df

# 添加商品类别统计特征
def addCateFea(df, originDf=None, **params):
    df['cate_age_delta_level'] = df['cate_age_delta'] // 1
    # df.loc[df.cate_age_delta_level<-4,'cate_age_delta_level'] = -4
    # df.loc[df.cate_age_delta_level>3,'cate_age_delta_level'] = 3
    return df

# 添加商品历史浏览量及购买量特征
def addShopFea(df, originDf=None, **params):
    df['shop_review_positive_rate_level'] = df['shop_review_positive_rate'] // 0.01
    df['shop_score_service_level'] = df['shop_score_service'] // 0.01
    df['shop_score_delivery_level'] = df['shop_score_delivery'] // 0.01
    df['shop_score_description_level'] = df['shop_score_description'] // 0.01
    # df.loc[df.shop_review_positive_rate_level<94, 'shop_review_positive_rate_level'] = 94
    # df.loc[df.shop_score_service_level<93, 'shop_score_service_level'] = 93
    # df.loc[df.shop_score_delivery_level<93, 'shop_score_delivery_level'] = 93
    # df.loc[df.shop_score_description_level<92, 'shop_score_description_level'] = 92

    df['shop_lastdate_trade_level'] = df['shop_lastdate_trade'] // 2
    # df.loc[df.shop_lastdate_trade_level>8,'shop_lastdate_trade_level'] = 9
    df['shop_lastdate_trade_level'].fillna(-1, inplace=True)
    df['shop_lastdate_trade_ratio_level'] = df['shop_lastdate_trade_ratio'] // 0.005
    # df.loc[df.shop_lastdate_trade_ratio_level>7,'shop_lastdate_trade_ratio_level'] = 8
    df['shop_lastdate_trade_ratio_level'].fillna(-1, inplace=True)
    df['shop_lastdate_show_delta_level'] = df['shop_lastdate_show_delta'] // 1000
    # df.loc[df.shop_lastdate_show_delta_level>7,'shop_lastdate_show_delta_level'] = 8
    df['shop_lastdate_show_delta_level'].fillna(-1, inplace=True)

    df['shop_age_delta_level'] = df['shop_age_delta'] // 1
    # df.loc[df.shop_age_delta_level>3,'shop_age_delta_level'] = 3
    # df.loc[df.shop_age_delta_level<-2,'shop_age_delta_level'] = -3

    df['shop_sales_delta_level'] = df['shop_sales_delta'] // 1
    # df.loc[df.shop_sales_delta_level>6,'shop_sales_delta_level'] = 7
    # df.loc[df.shop_sales_delta_level<-6,'shop_sales_delta_level'] = -6
    df['shop_collected_delta_level'] = df['shop_collected_delta'] // 1
    # df.loc[df.shop_collected_delta_level>6,'shop_collected_delta_level'] = 7
    # df.loc[df.shop_collected_delta_level<-5,'shop_collected_delta_level'] = -6
    df['shop_pv_delta_level'] = df['shop_pv_delta'] // 1
    # df.loc[df.shop_pv_delta_level>6,'shop_pv_delta_level'] = 7
    # df.loc[df.shop_pv_delta_level<-4,'shop_pv_delta_level'] = -5
    df['shop_item_count_level'] = df['shop_item_count'] // 5
    # df.loc[df.shop_item_count_level>7,'shop_item_count_level'] = 8

    df['shop_review_num_delta_level'] = df['shop_review_num_delta'] // 2
    # df.loc[df.shop_review_num_delta_level>4,'shop_review_num_delta_level'] = 4
    # df.loc[df.shop_review_num_delta_level<-3,'shop_review_num_delta_level'] = -3
    df['shop_star_level_delta_level'] = df['shop_star_level_delta'] // 2
    # df.loc[df.shop_star_level_delta_level>3,'shop_star_level_delta_level'] = 3
    # df.loc[df.shop_star_level_delta_level<-2,'shop_star_level_delta_level'] = -2
    return df

# 添加商品历史浏览量及购买量特征
def addItemFea(df, originDf=None, **params):
    # df['item_prop_num_level'] = df['item_prop_num'] // 5
    df['item_prop_num_level'] = df['item_prop_num'] // 5
    # df.loc[df.item_prop_num_level<1,'item_prop_num_level'] = 1
    # df.loc[df.item_prop_num_level>15,'item_prop_num_level'] = 16
    df['item_his_trade_level'] = df['item_his_trade'] // 5
    # df['item_his_trade_level'] = df['item_his_trade'] // 10
    # df.loc[df.item_his_trade_level>5,'item_his_trade_level'] = 6
    df['item_lastdate_trade_level'] = df['item_lastdate_trade'] // 1
    # df['item_lastdate_trade_level'] = df['item_lastdate_trade'] // 0.005
    # df.loc[df.item_lastdate_trade_level>8,'item_lastdate_trade_level'] = 9
    df['item_lastdate_trade_level'].fillna(-9999, inplace=True)
    df['item_lastdate_trade_ratio_level'] = df['item_lastdate_trade_ratio'] // 0.001
    # df['item_lastdate_trade_ratio_level'] = df['item_lastdate_trade_ratio'] // 0.005
    # df.loc[df.item_lastdate_trade_ratio_level>12,'item_lastdate_trade_ratio_level'] = 13
    df['item_lastdate_trade_ratio_level'].fillna(-9999, inplace=True)

    df['item_lastdate_trade_delta_level'] = df['item_lastdate_trade_delta'] // 1
    # df.loc[df.item_lastdate_trade_delta_level>3,'item_lastdate_trade_delta_level'] = 4
    df['item_lastdate_trade_delta_level'].fillna(-9999, inplace=True)
    df['item_lastdate_trade_ratio_delta_level'] = df['item_lastdate_trade_ratio_delta'] // 0.001
    # df['item_lastdate_trade_ratio_delta_level'] = df['item_lastdate_trade_ratio_delta'] // 0.01
    # df.loc[df.item_lastdate_trade_ratio_delta_level>3,'item_lastdate_trade_ratio_delta_level'] = 4
    # df.loc[df.item_lastdate_trade_ratio_delta_level<-2,'item_lastdate_trade_ratio_delta_level'] = -2
    df['item_lastdate_trade_ratio_delta_level'].fillna(-9999, inplace=True)

    df['item_sales_delta_level'] = df['item_sales_delta'] // 1
    # df.loc[df.item_sales_delta_level<-6,'item_sales_delta_level'] = -6
    # df.loc[df.item_sales_delta_level>10,'item_sales_delta_level'] = 10
    df['item_collected_delta_level'] = df['item_collected_delta'] // 1
    # df.loc[df.item_collected_delta_level<-8,'item_collected_delta_level'] = -8
    df['item_pv_delta_level'] = df['item_pv_delta'] // 1
    # df.loc[df.item_pv_delta_level<-3,'item_pv_delta_level'] = -4
    df['item_age_delta_level'] = df['item_age_delta'] // 1
    # df.loc[df.item_age_delta_level>3,'item_age_delta_level'] = 3
    # df.loc[df.item_age_delta_level<-3,'item_age_delta_level'] = -3
    df['item_age_delta_level'].fillna(-3, inplace=True)
    return df

# 添加用户维度特征
def addUserFea(df, originDf=None, **params):
    df = df.fillna({k:0 for k in ['user_lastdate_show','user_lastdate_trade']})
    df.loc[(df.user_last_show_timedelta==999999)&(df.user_next_show_timedelta<999999),'user_near_timedelta'] = -99999
    df.loc[(df.user_last_show_timedelta<999999)&(df.user_next_show_timedelta==999999),'user_near_timedelta'] = 99999
    df.loc[(df.user_last_show_timedelta==999999)&(df.user_next_show_timedelta==999999),'user_near_timedelta'] = np.nan
    df.loc[(df.user_lasthour_show==0)&(df.user_nexthour_show==0),'user_nearhour_show_delta'] = np.nan

    df['user_last_show_timedelta_level'] = df['user_last_show_timedelta'] // 100
    # df.loc[df.user_last_show_timedelta_level>10,'user_last_show_timedelta_level'] = 11
    df['user_next_show_timedelta_level'] = df['user_next_show_timedelta'] // 100
    # df.loc[df.user_next_show_timedelta_level>10,'user_next_show_timedelta_level'] = 11
    df['user_his_show_level'] = df['user_his_show'].values
    # df.loc[df.user_his_show_level>23,'user_his_show_level'] = 24
    df['user_lasthour_show_level'] = df['user_lasthour_show'].values
    # df.loc[df.user_lasthour_show_level>10,'user_lasthour_show_level'] = 11
    df['user_nexthour_show_level'] = df['user_nexthour_show'].values
    # df.loc[df.user_nexthour_show_level>10,'user_nexthour_show_level'] = 11
    df['user_lastdate_show_level'] = df['user_lastdate_show'].values
    # df.loc[df.user_lastdate_show_level>13,'user_lastdate_show_level'] = 14
    df['user_lastdate_trade_ratio_level'] = df['user_lastdate_trade_ratio'] // 0.0005
    # df.loc[df.user_lastdate_trade_ratio_level>20,'user_lastdate_trade_ratio_level'] = 21
    df['user_his_trade_ratio_level'] = df['user_his_trade_ratio'] // 0.0005
    # df.loc[df.user_his_trade_ratio_level>10,'user_his_trade_ratio_level'] = 11
    df['user_near_timedelta_level'] = df['user_near_timedelta'] // 600
    # df.loc[df.user_his_trade_ratio_level<0,'user_his_trade_ratio_level'] = -1
    # df.loc[df.user_his_trade_ratio_level>0,'user_his_trade_ratio_level'] = 1
    df.loc[df.user_his_trade_ratio_level.isnull(),'user_his_trade_ratio_level'] = 9999
    df['user_nearhour_show_delta_level'] = df['user_nearhour_show_delta'] // 600
    # df.loc[df.user_nearhour_show_delta_level>7,'user_nearhour_show_delta_level'] = 8
    # df.loc[df.user_nearhour_show_delta_level<-8,'user_nearhour_show_delta_level'] = -9
    df.loc[df.user_nearhour_show_delta_level.isnull(),'user_nearhour_show_delta_level'] = 9999
    return df

# 添加广告商品与查询词的相关性特征
def addContextFea(df, **params):
    df['predict_cate_num_level'] = df['predict_cate_num'].values
    df['predict_prop_num_level'] = df['predict_prop_num'].values
    df['prop_intersect_num_level'] = df['prop_intersect_num'].values
    df['prop_jaccard_level'] = df['prop_jaccard'] // 0.01
    # df.loc[df.predict_cate_num_level>10, 'predict_cate_num_level'] = 10
    # df.loc[df.predict_prop_num_level>5, 'predict_prop_num_level'] = 5
    # df.loc[df.prop_intersect_num_level>4, 'prop_intersect_num_level'] = 4
    # df.loc[df.prop_jaccard_level>14, 'prop_jaccard_level'] = 15
    return df

# 添加品牌相关特征
def addBrandFea(df, originDf=None, **params):
    tempDf = df.drop_duplicates(['item_id'])
    tempDf = pd.pivot_table(tempDf, index=['shop_id','item_brand_id'], values=['item_id'], aggfunc=[len])
    tempDf.columns = ['sb_item_count']
    tempDf.reset_index(inplace=True)
    tempDf2 = tempDf.shop_id.value_counts().to_frame()
    tempDf2.columns = ['shop_brand_count']
    df = df.merge(tempDf, how='left', on=['shop_id','item_brand_id'])
    df = df.merge(tempDf2, how='left', left_on=['shop_id'], right_index=True)
    df['shop_brand_item_ratio'] = biasSmooth(df['sb_item_count'].values, df['shop_item_count'].values)
    df['shop_brand_count_ratio'] = biasSmooth(df['shop_brand_count'].values, df['shop_item_count'].values)

    df['brand_his_trade_ratio_level'] = df['brand_his_trade_ratio'] // 0.01
    # df.loc[df.brand_his_trade_ratio_level>6,'brand_his_trade_ratio_level'] = 6
    # df.loc[df.brand_his_trade_ratio_level.isnull(),'brand_his_trade_ratio_level'] = -1
    df['brand_lastdate_trade_ratio_level'] = df['brand_lastdate_trade_ratio'] // 0.01
    # df.loc[df.brand_lastdate_trade_ratio_level>5,'brand_lastdate_trade_ratio_level'] = 5
    # df.loc[df.brand_lastdate_trade_ratio_level.isnull(),'brand_lastdate_trade_ratio_level'] = -1
    df['brand_collected_delta_level'] = df['brand_collected_delta'] // 1
    # df.loc[df.brand_collected_delta_level>6,'brand_collected_delta_level'] = 6
    # df.loc[df.brand_collected_delta_level<-5,'brand_collected_delta_level'] = -5
    df.loc[df.brand_collected_delta_level.isnull(),'brand_collected_delta_level'] = -9999
    df['brand_price_delta_level'] = df['brand_price_delta'] // 1
    # df.loc[df.brand_price_delta_level<-2,'brand_price_delta_level'] = -2
    # df.loc[df.brand_price_delta_level>1,'brand_price_delta_level'] = 1
    df.loc[df.brand_price_delta_level.isnull(),'brand_price_delta_level'] = -9999
    df['brand_pv_delta_level'] = df['brand_pv_delta'] // 1
    # df.loc[df.brand_pv_delta_level<-3,'brand_pv_delta_level'] = -3
    # df.loc[df.brand_pv_delta_level>6,'brand_pv_delta_level'] = 6
    df.loc[df.brand_pv_delta_level.isnull(),'brand_pv_delta_level'] = -9999
    df['brand_sales_delta_level'] = df['brand_sales_delta'] // 1
    # df.loc[df.brand_sales_delta_level<-5,'brand_sales_delta_level'] = -5
    # df.loc[df.brand_sales_delta_level>6,'brand_sales_delta_level'] = 6
    df.loc[df.brand_sales_delta_level.isnull(),'brand_sales_delta_level'] = -9999
    df['brand_age_delta_level'] = df['brand_age_delta'] // 1
    # df.loc[df.brand_age_delta_level<-3,'brand_age_delta_level'] = -3
    # df.loc[df.brand_age_delta_level>2,'brand_age_delta_level'] = 2
    df.loc[df.brand_age_delta_level.isnull(),'brand_age_delta_level'] = -9999
    df['shop_brand_item_ratio_level'] = df['shop_brand_item_ratio'] // 0.1
    # df.loc[df.shop_brand_item_ratio_level<3,'shop_brand_item_ratio_level'] = 3
    df.loc[df.shop_brand_item_ratio_level.isnull(),'shop_brand_item_ratio_level'] = -1
    df['shop_brand_count_ratio_level'] = df['shop_brand_count_ratio'] // 0.1
    # df.loc[df.shop_brand_count_ratio_level>3,'shop_brand_count_ratio_level'] = 3
    df.loc[df.shop_brand_count_ratio_level.isnull(),'shop_brand_count_ratio_level'] = -1
    return df

# 添加用户与类目关联维度的特征
def addUserCateFea(df, hisDf=None, **params):
    df.loc[(df.uc_last_show_timedelta==999999)&(df.uc_next_show_timedelta<999999),'uc_near_timedelta'] = -99999
    df.loc[(df.uc_last_show_timedelta<999999)&(df.uc_next_show_timedelta==999999),'uc_near_timedelta'] = 99999
    df.loc[(df.uc_last_show_timedelta==999999)&(df.uc_next_show_timedelta==999999),'uc_near_timedelta'] = np.nan
    df.loc[(df.uc_lasthour_show==0)&(df.uc_nexthour_show==0), 'uc_nearhour_show_delta'] = np.nan

    df['uc_his_trade_ratio_level'] = df['uc_his_trade_ratio'] // 0.0001
    # df.loc[df.uc_his_trade_ratio_level>250,'uc_his_trade_ratio_level'] = 251
    # df.loc[df.uc_his_trade_ratio_level<3,'uc_his_trade_ratio_level'] = 2
    df['uc_next_show_timedelta_level'] = df['uc_his_trade_ratio'] // 0.0001
    # df.loc[(df.uc_next_show_timedelta_level>5)&(df.uc_next_show_timedelta_level<999999),'uc_next_show_timedelta_level'] = 6
    df['uc_price_mean_level'] = df['uc_price_mean'] // 1
    # df.loc[df.uc_price_mean_level<4,'uc_price_mean_level'] = 4
    # df.loc[df.uc_price_mean_level>9,'uc_price_mean_level'] = 9
    df['uc_price_delta_level'] = df['uc_price_delta'] // 1
    # df.loc[df.uc_price_delta_level<-2,'uc_price_delta_level'] = -2
    # df.loc[df.uc_price_delta_level>2,'uc_price_delta_level'] = 2
    df['uc_near_timedelta_level'] = df['uc_near_timedelta'] // 100
    # df.loc[df.uc_near_timedelta_level>10,'uc_near_timedelta_level'] = 10
    # df.loc[df.uc_near_timedelta_level<-10,'uc_near_timedelta_level'] = -10
    # df.loc[(df.uc_near_timedelta_level>1)&(df.uc_near_timedelta_level<10),'uc_near_timedelta_level'] = 5
    # df.loc[(df.uc_near_timedelta_level<-1)&(df.uc_near_timedelta_level>-10),'uc_near_timedelta_level'] = -5
    df.loc[df.uc_near_timedelta_level.isnull(),'uc_near_timedelta_level'] = 999999
    df['uc_nearhour_show_delta_level'] = df['uc_nearhour_show_delta'].values
    # df.loc[df.uc_nearhour_show_delta_level>4, 'uc_nearhour_show_delta_level'] = 5
    # df.loc[df.uc_nearhour_show_delta_level<-4, 'uc_nearhour_show_delta_level'] = -5
    df.loc[df.uc_nearhour_show_delta_level.isnull(),'uc_nearhour_show_delta_level'] = 999999
    return df

# 添加用户商品关联维度的特征
def addUserItemFea(df, hisDf=None, **params):
    df['ui_last_show_timedelta_level'] = df['ui_last_show_timedelta'] // 300
    # df.loc[(df.ui_last_show_timedelta_level>3)&(df.ui_last_show_timedelta_level<99999),'ui_last_show_timedelta_level'] = 3
    df['ui_lasthour_show_level'] = df['ui_lasthour_show'].values
    # df.loc[df.ui_lasthour_show_level>2,'ui_lasthour_show_level'] = 2
    df['ui_lasthour_show_ratio_level'] = df['ui_lasthour_show_ratio'] // 0.1
    # df.loc[df.ui_lasthour_show_ratio_level>4,'ui_lasthour_show_ratio_level'] = 5
    df['ui_lastdate_show_level'] = df['ui_lastdate_show'].values
    # df.loc[df.ui_lastdate_show_level>2,'ui_lastdate_show_level'] = 2
    return df

# 统计用户该价格段商品的统计特征
def addUserPriceFea(df, hisDf=None, **params):
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

# 统计商品与各类别用户群交叉特征
def addCateCrossFea(df, **params):
    df['ca_his_trade_ratio'] = df['ca_his_trade_ratio'] // 0.005
    df.loc[df.ca_his_trade_ratio>8,'ca_his_trade_ratio'] = 8
    df.loc[df.ca_his_trade_ratio.isnull(),'ca_his_trade_ratio'] = -999999
    df['cg_his_trade_ratio'] = df['cg_his_trade_ratio'] // 0.005
    df.loc[df.cg_his_trade_ratio>7,'cg_his_trade_ratio'] = 7
    return df

# 统计商品与各类别用户群交叉特征
def addItemCrossFea(df, **params):
    df['ia_his_trade_ratio_level'] = df['ia_his_trade_ratio'] // 0.005
    df.loc[df.ia_his_trade_ratio_level>9,'ia_his_trade_ratio_level'] = 10
    df.loc[df.ia_his_trade_ratio_level.isnull(),'ia_his_trade_ratio_level'] = -999999
    df['ia_his_trade_delta_level'] = df['ia_his_trade_delta'] // 0.01
    df.loc[(df.ia_his_trade_delta_level<-4),'ia_his_trade_delta_level'] = -4
    df.loc[(df.ia_his_trade_delta_level>5),'ia_his_trade_delta_level'] = 5
    df.loc[df.ia_his_trade_delta_level.isnull(),'ia_his_trade_delta_level'] = -999999
    df['ig_his_trade_ratio_level'] = df['ig_his_trade_ratio'] // 0.005
    df.loc[df.ig_his_trade_ratio_level>14,'ig_his_trade_ratio_level'] = 14
    df['ig_his_trade_delta_level'] = df['ig_his_trade_delta'] // 0.01
    df.loc[df.ig_his_trade_delta_level>3,'ig_his_trade_delta_level'] = 4
    df.loc[df.ig_his_trade_delta_level<-4,'ig_his_trade_delta_level'] = -4
    return df

# 数据清洗
def dataCleaning(df):
    startTime = datetime.now()
    df['context_timestamp'] = df['timestamp'].map(lambda x: datetime.fromtimestamp(x))
    tempDf = df.drop_duplicates(subset=['item_id'])[['item_id','item_property_list_str']]
    tempDf['item_property_list'] = tempDf[tempDf.item_property_list_str.notnull()]['item_property_list_str'].map(lambda x: x.split(';'))
    df = df.merge(tempDf, how='left', on='item_id')
    df['predict_category_property'] = df[df.predict_category_property_str.notnull()]['predict_category_property_str'].map(
        lambda x: {kv.split(':')[0]:(kv.split(':')[1].split(',') if kv.split(':')[1]!='-1' else []) for kv in x.split(';')})
    df['date'] = pd.to_datetime(df['date_str'])
    print('cleaning time:', datetime.now()-startTime)
    return df

# 特征方法汇总
def feaFactory(df, hisDf=None, **args):
    startTime = datetime.now()
    # df = addTimeFea(df, **params)
    # df = itemFeaFillna(df, hisDf)
    # df = shopFeaFillna(df, hisDf)
    df = addCateFea(df, hisDf)
    df = addUserFea(df, hisDf)
    df = addShopFea(df, hisDf)
    df = addItemFea(df, hisDf)
    df = addBrandFea(df, hisDf)
    df = addContextFea(df)
    df = addUserCateFea(df, hisDf)
    df = addUserItemFea(df, hisDf)
    # df = addUserPriceFea(df, hisDf)
    df = addCateCrossFea(df)
    df = addItemCrossFea(df)
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
    def __init__(self, df, numFea, strFea, labelName, multiFea=[], colSep=' ', iterNum=None):
        if len(numFea)>0:
            ffmDf, scaler = scalerFea(df.fillna({k:0 for k in numFea}), numFea)
        else:
            ffmDf = df.copy()
            scaler = None
        onehotEncoders,feaNo = self.getOnehotFeaDict(ffmDf, strFea, startCount=len(numFea))
        mulChoiceEncoders, feaNo = self.getMulChoiceFeaDict(ffmDf, multiFea, startCount=feaNo)
        self.label = labelName
        self.strFea = strFea
        self.numFea = numFea
        self.multiFea = multiFea
        self.scaler = scaler
        self.onehotEncoders = onehotEncoders
        self.mulChoiceEncoders = mulChoiceEncoders
        self.fieldNum = len(numFea) + len(strFea) + len(multiFea)
        self.feaNum = feaNo
        self.colSep = colSep
        self.iterNum = iterNum

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
            values = listAdd(df[f].dropna().values)
            values = set(values)
            mulChoiceEncoders[f] = {v:i+feaCount for i,v in enumerate(values)}
            feaCount += len(values)
        return mulChoiceEncoders,feaCount

    # 获取ffm数据集
    def getFFMDf(self, df):
        strFea = self.strFea
        numFea = self.numFea
        multiFea = self.multiFea
        if self.label not in df.columns:
            ffmDf = df[numFea + strFea + multiFea]
        else:
            ffmDf = df[[self.label] + numFea + strFea + multiFea]

        if len(numFea)>0:
            ffmDf.fillna({k:0 for k in numFea}, inplace=True)
            ffmDf.loc[:,numFea] = self.scaler.transform(ffmDf[numFea].values)

        startTime = datetime.now()
        ffmDf['constant'] = '%d:%d:1'%(self.fieldNum,self.feaNum)
        fieldOffset = 1
        for i,fea in enumerate(numFea):
            fieldId = i + fieldOffset
            ffmDf.loc[:,fea] = list(map(lambda x: '%d:%d:%f'%(fieldId,fieldId,x) if x==x else None, ffmDf[fea]))
        fieldOffset = len(numFea)
        for i,fea in enumerate(strFea):
            fieldId = i+fieldOffset
            onehotEncoder = self.onehotEncoders[fea]
            ffmDf.loc[:,fea] = list(map(lambda x: '%d:%d:%d'%(fieldId,onehotEncoder[x],1) if x==x else None, ffmDf[fea]))
        fieldOffset += len(strFea)
        for i,fea in enumerate(multiFea):
            fieldId = i+fieldOffset
            mulChoiceEncoder = self.mulChoiceEncoders[fea]
            ffmDf.loc[:,fea] = list(map(lambda x: self.colSep.join(['%d:%d:%d'%(fieldId,mulChoiceEncoder[i],1) for i in x]) if x==x else None, ffmDf[fea]))
        # print(ffmDf[df.predict_category_property].head())
        return ffmDf

    # 以ffm格式导出数据集
    def exportFFMDf(self, df, filePath):
        outputSeries = df.apply(lambda x: self.colSep.join(x[x.notnull()].astype(str).values), axis=1)
        outputSeries.to_csv(filePath, header=False, index=False)

    # 导入ffm结果数据集
    def importFFMResult(self, url):
        result = pd.read_csv(url, header=None, index_col=None)
        return result[0].values

    # 调用ffm程序训练模型
    def train(self, trainX, trainy, testX=None, testy=None, thread=4, l2=0.00002, eta=0.02, iter_num=None, factor=None, modelName='ffm_model_temp', trainFileName='ffm_train_temp.ffm', testFileName='ffm_valid_temp.ffm', autoStop=False, verbose=True):
        factor = self.fieldNum if factor==None else factor
        iter_num = iter_num if iter_num!=None else (self.iterNum if self.iterNum!=None else 50)
        hasTest = isinstance(testy, np.ndarray)
        autoStop = hasTest&autoStop
        self.exportFFMDf(pd.DataFrame(np.column_stack((trainy, trainX))), trainFileName)
        self.exportFFMDf(pd.DataFrame(np.column_stack((testy, testX))), testFileName) if hasTest else None

        cmd = ['libffm/ffm-train','-s', str(thread),'-l',str(l2),'-r',str(eta),'-t',str(iter_num),'-k',str(factor)]
        if hasTest:
            cmd.extend(['-p',testFileName])
            if autoStop:
                cmd.extend(['--auto-stop'])
        cmd.extend([trainFileName,'%s.model'%modelName])
        result = subprocess.check_output(cmd)
        if verbose:
            print(result.decode())

        if autoStop:
            iterNum, trainLoss, testLoss = re.findall(r"\n\s+(\d+)\s+([\d.]+)\s+([\d.]+)\nAuto-stop", result.decode(), re.S)[0]
            return int(iterNum), float(trainLoss), float(testLoss)
        elif hasTest:
            iterNum, trainLoss, testLoss = re.findall(r"\n\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+\n$", result.decode(), re.S)[0]
            return float(trainLoss), float(testLoss)
        else:
            iterNum, trainLoss = re.findall(r"\n\s+(\d+)\s+([\d.]+)\s+[\d.]+\n$", result.decode(), re.S)[0]
            return float(trainLoss)

    # 调用ffm程序训练模型
    def trainAutoIter(self, X, y, thread=7, l2=0.00002, eta=0.02, iter_num=180, factor=None, modelName='ffm_model_temp', trainFileName='ffm_train_temp.ffm', testFileName='ffm_valid_temp.ffm', verbose=True):
        factor = self.fieldNum if factor==None else factor
        train,test,_,_ = train_test_split(np.column_stack((y, X)), y, test_size=0.2, stratify=y)
        self.exportFFMDf(pd.DataFrame(train), trainFileName)
        self.exportFFMDf(pd.DataFrame(test), testFileName)
        cmd = ['libffm/ffm-train','-s', str(thread),'-l',str(l2),'-r',str(eta),'-t',str(iter_num),'-k',str(factor),'-p',testFileName]
        cmd.extend([trainFileName,'%s.model'%modelName])
        if verbose:
            print('cmd: ', ' '.join(cmd))
        result = subprocess.check_output(cmd)
        lines = re.findall(r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+$", result.decode(), re.M)
        resultDf = pd.DataFrame(np.array(lines).astype(float), columns=['iter_num','train_loss','test_loss'])
        resultDf['rank'] = resultDf['test_loss'].rank(method='first')
        if verbose:
            print(resultDf.iloc[:60],resultDf.iloc[60:120],resultDf.iloc[120:])
        idx = resultDf[resultDf['rank']==1].index[0]
        iterNum = resultDf.loc[idx,'iter_num']

        self.exportFFMDf(pd.DataFrame(np.column_stack((y, X))), trainFileName)
        cmd = ['libffm/ffm-train','-s', str(thread),'-l',str(l2),'-r',str(eta),'-t',str(iterNum),'-k',str(factor)]
        cmd.extend([trainFileName,'%s.model'%modelName])
        result = subprocess.check_output(cmd)
        return iterNum

    # 调用ffm程序预测测试集
    def predict(self, X, modelName='ffm_model_temp', testFileName='ffm_test_temp.ffm', outputName='ffm_predict_temp.txt'):
        self.exportFFMDf(pd.DataFrame(X), testFileName)
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
            self.train(kfTrainDf[fea], kfTrainDf['is_trade'])
            # self.train(kfTrainDf[fea].values, kfTrainDf['is_trade'].values, iter_num=iterNum)
            oofTrain[testIdx] = self.predict(kfTestDf[fea].values)
            oofTestSkf[:,i] = self.predict(testDf[fea].values)
        oofTest[:] = oofTestSkf.mean(axis=1)
        return oofTrain, oofTest

if __name__ == '__main__':
    # 准备数据
    startTime = datetime.now()
    df = importDf('../data/train_fea_special_sample.csv')
    # df = importDf('../data/train_fea_special.csv')
    df['dataset'] = 0
    predictDf = importDf('../data/test_fea_sample.csv')
    # predictDf = importDf('../data/test_fea.csv')
    predictDf['dataset'] = -1
    originDf = pd.concat([df,predictDf], ignore_index=True)
    originDf = originDf.sample(frac=0.05)
    print('prepare dataset time:', datetime.now()-startTime)

    # 特征处理
    startTime = datetime.now()
    originDf = dataCleaning(originDf)
    originDf = feaFactory(originDf)
    df = originDf[(originDf['dataset']>=0)]
    predictDf = originDf[originDf['dataset']==-1]

    # 特征筛选
    numFea = [
        'hour_trade_ratio',
        # 'item_his_trade_ratio',
        # 'shop_his_trade_ratio',
    ]
    strFea = [
        'item_id','item_sales_level','item_price_level','item_collected_level','item_pv_level','item_city_id','item_category1','item_category2','item_brand_id',
        'user_id','user_gender_id','user_age_level','user_occupation_id','user_star_level',
        'context_page_id',#'hour',#'hour2',
c
        'predict_cate_num_level','cate_intersect_num','predict_prop_num_level','prop_intersect_num_level',#'prop_jaccard_level',
        # 'user_last_show_timedelta_level','user_next_show_timedelta_level','user_lasthour_show_level','user_nexthour_show_level','user_lastdate_show_level',#'user_lastdate_trade_ratio_level','user_his_trade_ratio_level','user_near_timedelta_level','user_nearhour_show_delta_level',
        # 'cate_age_delta_level',
        'item_prop_num_level','item_his_trade_level','item_lastdate_trade_level','item_lastdate_trade_ratio_level','item_lastdate_trade_delta_level','item_lastdate_trade_ratio_delta_level','item_sales_delta_level','item_collected_delta_level','item_pv_delta_level','item_age_delta_level',
        # 'shop_lastdate_trade_level','shop_lastdate_trade_ratio_level','shop_lastdate_show_delta_level','shop_age_delta_level','shop_sales_delta_level','shop_collected_delta_level','shop_pv_delta_level','shop_item_count_level','shop_review_num_delta_level','shop_star_level_delta_level',
        # 'brand_his_trade_ratio_level','brand_lastdate_trade_ratio_level','brand_collected_delta_level','brand_price_delta_level','brand_pv_delta_level','brand_sales_delta_level','brand_age_delta_level','shop_brand_item_ratio_level','shop_brand_count_ratio_level',
        #
        # 'uc_his_trade_ratio_level','uc_next_show_timedelta_level','uc_near_timedelta_level','uc_nearhour_show_delta_level','uc_price_mean_level','uc_price_delta_level',
        # 'ui_last_show_timedelta_level','ui_lasthour_show_level','ui_lasthour_show_ratio_level','ui_lastdate_show_level',
        # 'ca_his_trade_ratio','cg_his_trade_ratio',
        # 'ia_his_trade_ratio_level','ia_his_trade_delta_level','ig_his_trade_ratio_level','ig_his_trade_delta_level'
    ]
    multiFea = [
        # 'item_property_list',
        # 'predict_property',
    ]
    fea = numFea + strFea + multiFea
    print(df[fea].info())
    # exit()

    # 将数据集转成FFM模型数据集
    startTime = datetime.now()
    libFFM = LibFFM(originDf, numFea, strFea, multiFea=multiFea, labelName='is_trade')
    print('field count:', libFFM.fieldNum)
    print('feature count:', libFFM.feaNum)
    totalFFMDf, predictFFMDf = [libFFM.getFFMDf(x) for x in [df, predictDf]]
    print('transform time:', datetime.now()-startTime)

    # 正式模型
    modelName = "lr2A"
    startTime = datetime.now()
    iterNum = libFFM.trainAutoIter(totalFFMDf[fea].values, totalFFMDf['is_trade'].values)
    libFFM.iterNum = iterNum
    # iterNum = libFFM.train(totalFFMDf[fea].values, totalFFMDf['is_trade'].values)
    print('training time: ', datetime.now()-startTime)
    # exit()

    # 开始预测
    predictDf.loc[:,'predicted_score'] = libFFM.predict(predictFFMDf[fea].values)
    print("预测结果：\n",predictDf[['instance_id','predicted_score']].head())
    print('预测均值：', predictDf['predicted_score'].mean())
    # exportResult(predictDf[['instance_id','predicted_score']], "%s.txt" % modelName)
    # exportResult(predictDf[['instance_id','predicted_score','hour']], "%s_hashour.txt" % modelName)

    # 生成stacking数据集
    df['predicted_score'] = np.nan
    predictDf['predicted_score'] = np.nan
    df.loc[:,'predicted_score'], predictDf.loc[:,'predicted_score'] = getOof(libFFM, totalFFMDf[fea].values, totalFFMDf['is_trade'].values, predictFFMDf[fea].values)
    # df.loc[:,'predicted_score'], predictDf.loc[:,'predicted_score'] = getOof(clf, dfX, dfy, predictX, stratify=True)#, weight=df['weight'].values
    print('oof training time: ', datetime.now()-startTime)
    cost = metrics.log_loss(df['is_trade'].values, df['predicted_score'].values)
    print('train loss: ', cost)
    print('7th train loss', metrics.log_loss(df.loc[df.date=='2018-09-07','is_trade'].values, df.loc[df.date=='2018-09-07','predicted_score'].values))
    print('train predict: \n',df[['instance_id','predicted_score']].head())
    print('normal dates predict aver:', df.loc[df.context_timestamp<'2018-09-05','predicted_score'].mean())
    print('7th train predict aver:', df.loc[df.date=='2018-09-07','predicted_score'].mean())
    print('test predict: \n',predictDf[['instance_id','predicted_score']].head())
    print('test predict aver:', predictDf['predicted_score'].mean())
    exit()
    exportResult(df[['instance_id','predicted_score']], "%s_oof_train.csv" % modelName)
    exportResult(predictDf[['instance_id','predicted_score']], "%s_oof_test.csv" % modelName)
