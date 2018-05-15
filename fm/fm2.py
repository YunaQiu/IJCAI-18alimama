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
import json, random

from sklearn.preprocessing import *
from sklearn import linear_model
from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, StratifiedKFold
from sklearn.externals import joblib

# 导入数据
def importDf(url, sep=' ', na_values='-1', header='infer', index_col=None, colNames=None):
    df = pd.read_csv(url, sep=sep, na_values='-1', header=header, index_col=index_col, names=colNames)
    return df

# 添加one-hot编码并保留原字段
def addOneHot(df, colName):
    if isinstance(colName, str):
        colName = [colName]
    colTemp = df[colName]
    df = pd.get_dummies(df, columns=colName)
    df = pd.concat([df, colTemp], axis=1)
    return df

# 缩放字段至0-1
def scalerFea(df, cols):
    df.dropna(inplace=True, subset=cols)
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols].values)
    return df,scaler

# 对数组集合进行合并操作
def listAdd(l):
    result = []
    [result.extend(x) for x in l]
    return result

# 对不同标签进行抽样处理
def getSubsample(labelList, ratio=0.8, repeat=False, params=None):
    if not isinstance(params, dict):
        if isinstance(ratio, (float, int)):
            params = {k:{'ratio':ratio, 'repeat':repeat} for k in set(labelList)}
        else:
            params={k:{'ratio':ratio[k], 'repeat':repeat} for k in ratio.keys()}
    resultIdx = []
    for label in params.keys():
        param = params[label]
        tempList = np.where(labelList==label)[0]
        sampleSize = np.ceil(len(tempList)*params[label]['ratio']).astype(int)
        if (~param['repeat'])&(param['ratio']<=1):
            resultIdx.extend(random.sample(tempList.tolist(),sampleSize))
        else:
            resultIdx.extend(tempList[np.random.randint(len(tempList),size=sampleSize)])
    return resultIdx

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
    resultArr = (aArr+alpha) / (bArr+alpha+beta)
    return resultArr

# 转化数据集字段格式，并去重
def formatDf(df):
    df['timestamp'] = df['context_timestamp'].values
    df['context_timestamp'] = df['context_timestamp'].map(lambda x: datetime.fromtimestamp(x))
    return df

# 拆分多维度拼接的字段
def splitMultiFea(df):
    tempDf = df.drop_duplicates(subset=['item_id'])[['item_id','item_category_list','item_property_list']]
    tempDf['item_category_list_str'] = tempDf['item_category_list'].values
    tempDf['item_property_list_str'] = tempDf['item_property_list'].values
    tempDf['item_category_list'] = tempDf[tempDf.item_category_list.notnull()]['item_category_list'].map(lambda x: x.split(';'))
    tempDf['item_category0'] = tempDf[tempDf.item_category_list.notnull()]['item_category_list'].map(lambda x: x[0])
    tempDf['item_category1'] = tempDf[tempDf.item_category_list.notnull()]['item_category_list'].map(lambda x: x[1] if len(x)>1 else np.nan)
    tempDf['item_category2'] = tempDf[tempDf.item_category_list.notnull()]['item_category_list'].map(lambda x: x[2] if len(x)>2 else np.nan)
    tempDf['item_property_list'] = tempDf[tempDf.item_property_list.notnull()]['item_property_list'].map(lambda x: x.split(';'))
    df = df.drop(['item_category_list','item_property_list'], axis=1).merge(tempDf, how='left', on='item_id')
    df['predict_category_property_str'] = df['predict_category_property'].values
    df['predict_category_property'] = df[df.predict_category_property.notnull()]['predict_category_property'].map(
        lambda x: {kv.split(':')[0]:(kv.split(':')[1].split(',') if kv.split(':')[1]!='-1' else []) for kv in x.split(';')})
    return df

# 新增多维度拼接的id特征
def combineKey(df):
    df['user_item'] = df['user_id'].astype('str') + '_' + df['item_id'].astype('str')
    return df

# 对商品特征进行预处理
def cleanItemFea(df):
    df.loc[df.item_sales_level.isnull(), 'item_sales_level'] = 0
    df.loc[df.item_price_level<1, 'item_price_level'] = 1
    df.loc[df.item_price_level>10, 'item_price_level'] = 10
    df.loc[df.item_collected_level>17, 'item_collected_level'] = 17
    df.loc[df.item_pv_level<6, 'item_pv_level'] = 6
    return df

# 对用户特征进行预处理
def cleanShopFea(df):
    df.loc[df.shop_star_level<5002, 'shop_star_level'] = 5002
    df.loc[df.shop_review_num_level<4, 'shop_review_num_level'] = 3
    df.loc[df.shop_review_num_level>23, 'shop_review_num_level'] = 24
    return df

# 对用户特征进行预处理
def cleanUserFea(df):
    df.loc[df.user_gender_id.isnull(), 'user_gender_id'] = -1
    df.loc[df.user_star_level>3009, 'user_star_level'] = 3009
    # df.loc[df.user_star_level.isnull(), 'user_star_level'] = 3000
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

# 添加时间特征
def addTimeFea(df, **params):
    df['hour'] = df.context_timestamp.dt.hour
    df['day'] = df.context_timestamp.dt.day
    df['date'] = pd.to_datetime(df.context_timestamp.dt.date)
    df['special_date_dist'] = (date(2018,9,7) - df['date']).dt.days

    # 计算当前小时的转化率，测试集转化率采用普通日期均值加上与特殊日期的差值
    tempDf = pd.pivot_table(df, index=['hour','date'], values='is_trade', aggfunc=[len,np.sum])
    tempDf.columns = ['click','trade']
    tempDf['ratio'] = biasSmooth(tempDf['trade'].values,tempDf['click'].values)
    tempDf.drop(['click','trade'], axis=1, inplace=True)
    tempDf = tempDf.unstack()
    tempDf = tempDf['ratio']
    tempDf['mean'] = tempDf.iloc[:,:4].apply(lambda x: x.mean(), axis=1)
    normalAm = tempDf.loc[(tempDf.index>=9)&(tempDf.index<12)]['mean'].mean()
    specialAm = tempDf.loc[(tempDf.index>=9)&(tempDf.index<12)][pd.to_datetime('2018-09-07')].mean()
    delta = specialAm - normalAm
    tempDf.loc[tempDf.index>11, pd.to_datetime('2018-09-07')] = tempDf.loc[tempDf.index>11, 'mean'] + delta
    tempDf = tempDf.iloc[:,:-1].stack().to_frame()
    tempDf.columns = ['hour_trade_ratio']
    tempDf.reset_index(inplace=True)
    tempDf['date'] = pd.to_datetime(tempDf['date'])
    df = df.merge(tempDf, how='left', on=['hour','date'])
    return df

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
    df.loc[df.cate_age_delta_level<-4,'cate_age_delta_level'] = -4
    df.loc[df.cate_age_delta_level>3,'cate_age_delta_level'] = 3
    return df

# 添加商品历史浏览量及购买量特征
def addShopFea(df, originDf=None, **params):
    df['shop_review_positive_rate_level'] = df['shop_review_positive_rate'] // 0.01
    df['shop_score_service_level'] = df['shop_score_service'] // 0.01
    df['shop_score_delivery_level'] = df['shop_score_delivery'] // 0.01
    df['shop_score_description_level'] = df['shop_score_description'] // 0.01
    df.loc[df.shop_review_positive_rate_level<94, 'shop_review_positive_rate_level'] = 94
    df.loc[df.shop_score_service_level<93, 'shop_score_service_level'] = 93
    df.loc[df.shop_score_delivery_level<93, 'shop_score_delivery_level'] = 93
    df.loc[df.shop_score_description_level<92, 'shop_score_description_level'] = 92

    df['shop_lastdate_trade_level'] = df['shop_lastdate_trade'] // 2
    df.loc[df.shop_lastdate_trade_level>8,'shop_lastdate_trade_level'] = 9
    df['shop_lastdate_trade_level'].fillna(-1, inplace=True)
    df['shop_lastdate_trade_ratio_level'] = df['shop_lastdate_trade_ratio'] // 0.005
    df.loc[df.shop_lastdate_trade_ratio_level>7,'shop_lastdate_trade_ratio_level'] = 8
    df['shop_lastdate_trade_ratio_level'].fillna(-1, inplace=True)
    df['shop_lastdate_show_delta_level'] = df['shop_lastdate_show_delta'] // 1000
    df.loc[df.shop_lastdate_show_delta_level>7,'shop_lastdate_show_delta_level'] = 8
    df['shop_lastdate_show_delta_level'].fillna(-1, inplace=True)

    df['shop_age_delta_level'] = df['shop_age_delta'] // 1
    df.loc[df.shop_age_delta_level>3,'shop_age_delta_level'] = 3
    df.loc[df.shop_age_delta_level<-2,'shop_age_delta_level'] = -3

    df['shop_sales_delta_level'] = df['shop_sales_delta'] // 1
    df.loc[df.shop_sales_delta_level>6,'shop_sales_delta_level'] = 7
    df.loc[df.shop_sales_delta_level<-6,'shop_sales_delta_level'] = -6
    df['shop_collected_delta_level'] = df['shop_collected_delta'] // 1
    df.loc[df.shop_collected_delta_level>6,'shop_collected_delta_level'] = 7
    df.loc[df.shop_collected_delta_level<-5,'shop_collected_delta_level'] = -6
    df['shop_pv_delta_level'] = df['shop_pv_delta'] // 1
    df.loc[df.shop_pv_delta_level>6,'shop_pv_delta_level'] = 7
    df.loc[df.shop_pv_delta_level<-4,'shop_pv_delta_level'] = -5
    df['shop_item_count_level'] = df['shop_item_count'] // 5
    df.loc[df.shop_item_count_level>7,'shop_item_count_level'] = 8

    df['shop_review_num_delta_level'] = df['shop_review_num_delta'] // 2
    df.loc[df.shop_review_num_delta_level>4,'shop_review_num_delta_level'] = 4
    df.loc[df.shop_review_num_delta_level<-3,'shop_review_num_delta_level'] = -3
    df['shop_star_level_delta_level'] = df['shop_star_level_delta'] // 2
    df.loc[df.shop_star_level_delta_level>3,'shop_star_level_delta_level'] = 3
    df.loc[df.shop_star_level_delta_level<-2,'shop_star_level_delta_level'] = -2
    return df

# 添加商品历史浏览量及购买量特征
def addItemFea(df, originDf=None, **params):
    df['item_prop_num_level'] = df['item_prop_num'] // 5
    df.loc[df.item_prop_num_level<1,'item_prop_num_level'] = 1
    df.loc[df.item_prop_num_level>15,'item_prop_num_level'] = 16
    df['item_his_trade_level'] = df['item_his_trade'] // 10
    df.loc[df.item_his_trade_level>5,'item_his_trade_level'] = 6
    df['item_lastdate_trade_level'] = df['item_lastdate_trade'] // 0.005
    df.loc[df.item_lastdate_trade_level>8,'item_lastdate_trade_level'] = 9
    df['item_lastdate_trade_level'].fillna(-9999, inplace=True)
    df['item_lastdate_trade_ratio_level'] = df['item_lastdate_trade_ratio'] // 0.005
    df.loc[df.item_lastdate_trade_ratio_level>12,'item_lastdate_trade_ratio_level'] = 13
    df['item_lastdate_trade_ratio_level'].fillna(-9999, inplace=True)

    df['item_lastdate_trade_delta_level'] = df['item_lastdate_trade_delta'] // 1
    df.loc[df.item_lastdate_trade_delta_level>3,'item_lastdate_trade_delta_level'] = 4
    df['item_lastdate_trade_delta_level'].fillna(-9999, inplace=True)
    df['item_lastdate_trade_ratio_delta_level'] = df['item_lastdate_trade_ratio_delta'] // 0.01
    df.loc[df.item_lastdate_trade_ratio_delta_level>3,'item_lastdate_trade_ratio_delta_level'] = 4
    df.loc[df.item_lastdate_trade_ratio_delta_level<-2,'item_lastdate_trade_ratio_delta_level'] = -2
    df['item_lastdate_trade_ratio_delta_level'].fillna(-9999, inplace=True)

    df['item_sales_delta_level'] = df['item_sales_delta'] // 1
    df.loc[df.item_sales_delta_level<-6,'item_sales_delta_level'] = -6
    df.loc[df.item_sales_delta_level>10,'item_sales_delta_level'] = 10
    df['item_collected_delta_level'] = df['item_collected_delta'] // 1
    df.loc[df.item_collected_delta_level<-8,'item_collected_delta_level'] = -8
    df['item_pv_delta_level'] = df['item_pv_delta'] // 1
    df.loc[df.item_pv_delta_level<-3,'item_pv_delta_level'] = -4
    df['item_age_delta_level'] = df['item_age_delta'] // 1
    df.loc[df.item_age_delta_level>3,'item_age_delta_level'] = 3
    df.loc[df.item_age_delta_level<-3,'item_age_delta_level'] = -3
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
    df.loc[df.user_last_show_timedelta_level>10,'user_last_show_timedelta_level'] = 11
    df['user_next_show_timedelta_level'] = df['user_next_show_timedelta'] // 100
    df.loc[df.user_next_show_timedelta_level>10,'user_next_show_timedelta_level'] = 11
    df['user_his_show_level'] = df['user_his_show'].values
    df.loc[df.user_his_show_level>23,'user_his_show_level'] = 24
    df['user_lasthour_show_level'] = df['user_lasthour_show'].values
    df.loc[df.user_lasthour_show_level>10,'user_lasthour_show_level'] = 11
    df['user_nexthour_show_level'] = df['user_nexthour_show'].values
    df.loc[df.user_nexthour_show_level>10,'user_nexthour_show_level'] = 11
    df['user_lastdate_show_level'] = df['user_lastdate_show'].values
    df.loc[df.user_lastdate_show_level>13,'user_lastdate_show_level'] = 14
    df['user_lastdate_trade_ratio_level'] = df['user_lastdate_trade_ratio'] // 0.0005
    df.loc[df.user_lastdate_trade_ratio_level>20,'user_lastdate_trade_ratio_level'] = 21
    df['user_his_trade_ratio_level'] = df['user_his_trade_ratio'] // 0.0005
    df.loc[df.user_his_trade_ratio_level>10,'user_his_trade_ratio_level'] = 11
    df['user_near_timedelta_level'] = df['user_near_timedelta'] // 600
    df.loc[df.user_his_trade_ratio_level<0,'user_his_trade_ratio_level'] = -1
    df.loc[df.user_his_trade_ratio_level>0,'user_his_trade_ratio_level'] = 1
    df.loc[df.user_his_trade_ratio_level.isnull(),'user_his_trade_ratio_level'] = 9999
    df['user_nearhour_show_delta_level'] = df['user_nearhour_show_delta'] // 600
    df.loc[df.user_nearhour_show_delta_level>7,'user_nearhour_show_delta_level'] = 8
    df.loc[df.user_nearhour_show_delta_level<-8,'user_nearhour_show_delta_level'] = -9
    df.loc[df.user_nearhour_show_delta_level.isnull(),'user_nearhour_show_delta_level'] = 9999
    return df

# 添加广告商品与查询词的相关性特征
def addContextFea(df, **params):
    df['predict_cate_num_level'] = df['predict_cate_num'].values
    df['predict_prop_num_level'] = df['predict_prop_num'].values
    df['prop_intersect_num_level'] = df['prop_intersect_num'].values
    df['prop_jaccard_level'] = df['prop_jaccard'] // 0.01
    df.loc[df.predict_cate_num_level>10, 'predict_cate_num_level'] = 10
    df.loc[df.predict_prop_num_level>5, 'predict_prop_num_level'] = 5
    df.loc[df.prop_intersect_num_level>4, 'prop_intersect_num_level'] = 4
    df.loc[df.prop_jaccard_level>14, 'prop_jaccard_level'] = 15
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
    df.loc[df.brand_his_trade_ratio_level>6,'brand_his_trade_ratio_level'] = 6
    df.loc[df.brand_his_trade_ratio_level.isnull(),'brand_his_trade_ratio_level'] = -1
    df['brand_lastdate_trade_ratio_level'] = df['brand_lastdate_trade_ratio'] // 0.01
    df.loc[df.brand_lastdate_trade_ratio_level>5,'brand_lastdate_trade_ratio_level'] = 5
    df.loc[df.brand_lastdate_trade_ratio_level.isnull(),'brand_lastdate_trade_ratio_level'] = -1
    df['brand_collected_delta_level'] = df['brand_collected_delta'] // 1
    df.loc[df.brand_collected_delta_level>6,'brand_collected_delta_level'] = 6
    df.loc[df.brand_collected_delta_level<-5,'brand_collected_delta_level'] = -5
    df.loc[df.brand_collected_delta_level.isnull(),'brand_collected_delta_level'] = -9999
    df['brand_price_delta_level'] = df['brand_price_delta'] // 1
    df.loc[df.brand_price_delta_level<-2,'brand_price_delta_level'] = -2
    df.loc[df.brand_price_delta_level>1,'brand_price_delta_level'] = 1
    df.loc[df.brand_price_delta_level.isnull(),'brand_price_delta_level'] = -9999
    df['brand_pv_delta_level'] = df['brand_pv_delta'] // 1
    df.loc[df.brand_pv_delta_level<-3,'brand_pv_delta_level'] = -3
    df.loc[df.brand_pv_delta_level>6,'brand_pv_delta_level'] = 6
    df.loc[df.brand_pv_delta_level.isnull(),'brand_pv_delta_level'] = -9999
    df['brand_sales_delta_level'] = df['brand_sales_delta'] // 1
    df.loc[df.brand_sales_delta_level<-5,'brand_sales_delta_level'] = -5
    df.loc[df.brand_sales_delta_level>6,'brand_sales_delta_level'] = 6
    df.loc[df.brand_sales_delta_level.isnull(),'brand_sales_delta_level'] = -9999
    df['brand_age_delta_level'] = df['brand_age_delta'] // 1
    df.loc[df.brand_age_delta_level<-3,'brand_age_delta_level'] = -3
    df.loc[df.brand_age_delta_level>2,'brand_age_delta_level'] = 2
    df.loc[df.brand_age_delta_level.isnull(),'brand_age_delta_level'] = -9999
    df['shop_brand_item_ratio_level'] = df['shop_brand_item_ratio'] // 0.1
    df.loc[df.shop_brand_item_ratio_level<3,'shop_brand_item_ratio_level'] = 3
    df.loc[df.shop_brand_item_ratio_level.isnull(),'shop_brand_item_ratio_level'] = -1
    df['shop_brand_count_ratio_level'] = df['shop_brand_count_ratio'] // 0.1
    df.loc[df.shop_brand_count_ratio_level>3,'shop_brand_count_ratio_level'] = 3
    df.loc[df.shop_brand_count_ratio_level.isnull(),'shop_brand_count_ratio_level'] = -1
    return df

# 添加用户与类目关联维度的特征
def addUserCateFea(df, originDf=None, **params):
    df.loc[(df.uc_last_show_timedelta==999999)&(df.uc_next_show_timedelta<999999),'uc_near_timedelta'] = -99999
    df.loc[(df.uc_last_show_timedelta<999999)&(df.uc_next_show_timedelta==999999),'uc_near_timedelta'] = 99999
    df.loc[(df.uc_last_show_timedelta==999999)&(df.uc_next_show_timedelta==999999),'uc_near_timedelta'] = np.nan
    df.loc[(df.uc_lasthour_show==0)&(df.uc_nexthour_show==0), 'uc_nearhour_show_delta'] = np.nan

    df['uc_his_trade_ratio_level'] = df['uc_his_trade_ratio'] // 0.0001
    df.loc[df.uc_his_trade_ratio_level>250,'uc_his_trade_ratio_level'] = 251
    df.loc[df.uc_his_trade_ratio_level<3,'uc_his_trade_ratio_level'] = 2
    df['uc_next_show_timedelta_level'] = df['uc_his_trade_ratio'] // 0.0001
    df.loc[(df.uc_next_show_timedelta_level>5)&(df.uc_next_show_timedelta_level<999999),'uc_next_show_timedelta_level'] = 6
    df['uc_price_mean_level'] = df['uc_price_mean'] // 1
    df.loc[df.uc_price_mean_level<4,'uc_price_mean_level'] = 4
    df.loc[df.uc_price_mean_level>9,'uc_price_mean_level'] = 9
    df['uc_price_delta_level'] = df['uc_price_delta'] // 1
    df.loc[df.uc_price_delta_level<-2,'uc_price_delta_level'] = -2
    df.loc[df.uc_price_delta_level>2,'uc_price_delta_level'] = 2
    df['uc_near_timedelta_level'] = df['uc_near_timedelta'] // 100
    df.loc[df.uc_near_timedelta_level>10,'uc_near_timedelta_level'] = 10
    df.loc[df.uc_near_timedelta_level<-10,'uc_near_timedelta_level'] = -10
    df.loc[(df.uc_near_timedelta_level>1)&(df.uc_near_timedelta_level<10),'uc_near_timedelta_level'] = 5
    df.loc[(df.uc_near_timedelta_level<-1)&(df.uc_near_timedelta_level>-10),'uc_near_timedelta_level'] = -5
    df.loc[df.uc_near_timedelta_level.isnull(),'uc_near_timedelta_level'] = 999999
    df['uc_nearhour_show_delta_level'] = df['uc_nearhour_show_delta'].values
    df.loc[df.uc_nearhour_show_delta_level>4, 'uc_nearhour_show_delta_level'] = 5
    df.loc[df.uc_nearhour_show_delta_level<-4, 'uc_nearhour_show_delta_level'] = -5
    df.loc[df.uc_nearhour_show_delta_level.isnull(),'uc_nearhour_show_delta_level'] = 999999
    return df

# 添加用户商品关联维度的特征
def addUserItemFea(df, originDf=None, **params):
    df['ui_last_show_timedelta_level'] = df['ui_last_show_timedelta'] // 300
    df.loc[(df.ui_last_show_timedelta_level>3)&(df.ui_last_show_timedelta_level<99999),'ui_last_show_timedelta_level'] = 3
    df['ui_lasthour_show_level'] = df['ui_lasthour_show'].values
    df.loc[df.ui_lasthour_show_level>2,'ui_lasthour_show_level'] = 2
    df['ui_lasthour_show_ratio_level'] = df['ui_lasthour_show_ratio'] // 0.1
    df.loc[df.ui_lasthour_show_ratio_level>4,'ui_lasthour_show_ratio_level'] = 5
    df['ui_lastdate_show_level'] = df['ui_lastdate_show'].values
    df.loc[df.ui_lastdate_show_level>2,'ui_lastdate_show_level'] = 2
    return df

# 统计用户该价格段商品的统计特征
def addUserShopFea(df, originDf=None, **params):
    if not isinstance(originDf, pd.DataFrame):
        originDf = df.copy()
        originDf['is_trade'].fillna(0,inplace=True)
    tempDf = statDateTrade(originDf, ['user_id','shop_id'], **params['statDateTrade'])
    tempDf.columns = ['us_his_show','us_his_trade']
    tempDf['us_his_trade_ratio'] = biasSmooth(tempDf['us_his_trade'].values, tempDf['us_his_show'].values)
    df = df.merge(tempDf, how='left', left_on=['user_id','shop_id','date'], right_index=True)
    return df

# 统计用户该价格段商品的统计特征
def addUserPriceFea(df, originDf=None, **params):
    if not isinstance(originDf, pd.DataFrame):
        originDf = df.copy()
        originDf['is_trade'].fillna(0,inplace=True)
    tempDf = statDateTrade(originDf, ['user_id','item_price_level'], **params['statDateTrade'])
    tempDf.columns = ['up_his_show','up_his_trade']
    tempDf['up_his_trade_ratio'] = biasSmooth(tempDf['up_his_trade'].values, tempDf['up_his_show'].values)
    df = df.merge(tempDf, how='left', left_on=['user_id','item_price_level','date'], right_index=True)
    for x in set(df.item_price_level.dropna().values):
        idx = df[df.item_price_level==x].index
        df.loc[idx, 'up_his_show_ratio'] = biasSmooth(df.loc[idx, 'up_his_show'].values, df.loc[idx,'user_his_show'].values)
    return df

# 统计用户该价格段商品的统计特征
def addUserHourFea(df, originDf=None, **params):
    if not isinstance(originDf, pd.DataFrame):
        originDf = df.copy()
        originDf['is_trade'].fillna(0,inplace=True)
    tempDf = statDateTrade(originDf, ['user_id','hour'], **params['statDateTrade'])
    tempDf.columns = ['uh_his_show','uh_his_trade']
    tempDf['uh_his_trade_ratio'] = biasSmooth(tempDf['uh_his_trade'].values, tempDf['uh_his_show'].values)
    df = df.merge(tempDf, how='left', left_on=['user_id','hour','date'], right_index=True)
    for x in set(df.hour.dropna().values):
        idx = df[df.hour==x].index
        df.loc[idx, 'uh_his_show_ratio'] = biasSmooth(df.loc[idx, 'uh_his_show'].values, df.loc[idx,'user_his_show'].values)
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

# 统计品牌与各类别用户群交叉特征
def addBrandCrossFea(df, **params):
    tempDf = statDateTrade(df, ['item_brand_id','user_age_level'], **params['statDateTrade'])
    tempDf.columns = ['ba_his_show','ba_his_trade']
    tempDf['ba_his_trade_ratio'] = biasSmooth(tempDf['ba_his_trade'].values, tempDf['ba_his_show'].values)
    df = df.merge(tempDf, how='left', left_on=['item_brand_id','user_age_level','date'], right_index=True)

    tempDf = statDateTrade(df, ['item_brand_id','user_gender_id'], **params['statDateTrade'])
    tempDf.columns = ['bg_his_show','bg_his_trade']
    tempDf['bg_his_trade_ratio'] = biasSmooth(tempDf['bg_his_trade'].values, tempDf['bg_his_show'].values)
    df = df.merge(tempDf, how='left', left_on=['item_brand_id','user_gender_id','date'], right_index=True)
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
def feaFactory(df, originDf=None, **args):
    params = {
        'statDateTrade': {
            'statDates': None,
            'skipDates': 1,
        }
    }
    for k1,arg in args.items():
        for k2,v in arg.items():
            params[k1][k2] = v
    # df = addTimeFea(df, **params)
    df = addCateFea(df, originDf, **params)
    df = addUserFea(df, originDf, **params)
    df = addShopFea(df, originDf, **params)
    df = addItemFea(df, originDf, **params)
    df = addContextFea(df, **params)
    df = addBrandFea(df,originDf, **params)
    df = addUserCateFea(df, originDf, **params)
    df = addUserItemFea(df, originDf, **params)
    # df = addUserShopFea(df, originDf, **params)
    # df = addUserPriceFea(df, originDf, **params)
    # df = addUserHourFea(df, originDf, **params)
    df = addCateCrossFea(df, **params)
    # df = addBrandCrossFea(df, **params)
    df = addItemCrossFea(df, **params)
    print('finished feaFactory: ', datetime.now() - startTime)
    return df

# 字典矩阵转化
class DictDataset():
    def __init__(self, df, numFea=[], strFea=[]):
        self.numFea = numFea
        self.strFea = strFea
        print('fea counts: ', len(numFea) + len(strFea))

        dfV = self.getDictList(df)
        dv = DictVectorizer().fit(dfV)
        self.dv = dv

    def getDictList(self, df):
        numFea = self.numFea
        strFea = self.strFea

        df.loc[:,numFea] = df.loc[:,numFea].astype(float)
        df.loc[:,strFea] = df.loc[:,strFea].astype(str)
        df.loc[:,strFea] = df.loc[:,strFea].applymap(lambda x: np.nan if x=='nan' else x)
        dfV = df.to_dict('records')
        for i,x in enumerate(dfV):
            dfV[i] = {k:v for k,v in x.items() if v==v}
        return dfV

    def transform(self, df):
        dfV = self.getDictList(df)
        return self.dv.transform(dfV)

# 训练模型
def trainModel(X, y, verbose=True, num_factors=100):
    fm = pylibfm.FM(num_factors=num_factors, num_iter=10, verbose=verbose, task="classification", initial_learning_rate=0.001, learning_rate_schedule="invscaling")
    fm.fit(X,y)
    return fm

# 导出预测结果
def exportResult(df, fileName, header=True, index=False, sep=' '):
    df.to_csv('./%s' % fileName, sep=sep, header=header, index=index)

# 获取stacking下一层数据集
def getOof(clf, trainX, trainY, testX, nFold=5, stratify=False, weight=None):
    oofTrain = np.zeros(trainX.shape[0])
    oofTest = np.zeros(testX.shape[0])
    oofTestSkf = np.zeros((testX.shape[0], nFold))
    if stratify:
        kf = StratifiedKFold(n_splits=nFold, shuffle=True)
    else:
        kf = KFold(n_splits=nFold, shuffle=True)
    for i, (trainIdx, testIdx) in enumerate(kf.split(trainX, trainY)):
        kfTrainX = trainX[trainIdx]
        kfTrainY = trainY[trainIdx]
        kfTestX = trainX[testIdx]
        if weight != None:
            kfWeight = weight[trainIdx]
        else:
            kfWeight = None
        clf.fit(kfTrainX, kfTrainY)
        oofTrain[testIdx] = clf.predict(kfTestX)
        oofTestSkf[:,i] = clf.predict(testX)
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
        # 'user_his_show',
        # 'user_his_trade_ratio',
        'item_his_trade_ratio',
        'shop_his_trade_ratio',
        # 'cate_his_trade_ratio','cate2_his_trade_ratio',
        # 'uc_his_trade_ratio',
        # 'ca_his_trade_ratio','cg_his_trade_ratio',
        # 'ia_his_trade_ratio',
        # 'ig_his_trade_ratio',
    ]
    strFea = [
        'item_id','item_sales_level','item_price_level','item_collected_level','item_pv_level','item_city_id','item_category1','item_category2','item_brand_id',
        'user_id','user_gender_id','user_age_level','user_occupation_id','user_star_level',
        'context_page_id','hour',#'hour2',
        'shop_id','shop_star_level','shop_review_num_level','shop_review_positive_rate_level','shop_score_service_level','shop_score_delivery_level','shop_score_description_level',

        'predict_cate_num_level','cate_intersect_num','predict_prop_num_level','prop_intersect_num_level',#'prop_jaccard_level',
        'user_last_show_timedelta_level','user_next_show_timedelta_level','user_lasthour_show_level','user_nexthour_show_level','user_lastdate_show_level','user_lastdate_trade_ratio_level','user_his_trade_ratio_level','user_near_timedelta_level','user_nearhour_show_delta_level',
        'cate_age_delta_level',
        'item_prop_num_level','item_his_trade_level','item_lastdate_trade_level','item_lastdate_trade_ratio_level','item_lastdate_trade_delta_level','item_lastdate_trade_ratio_delta_level','item_sales_delta_level','item_collected_delta_level','item_pv_delta_level','item_age_delta_level',
        'shop_lastdate_trade_level','shop_lastdate_trade_ratio_level','shop_lastdate_show_delta_level','shop_age_delta_level','shop_sales_delta_level','shop_collected_delta_level','shop_pv_delta_level','shop_item_count_level','shop_review_num_delta_level','shop_star_level_delta_level',
        'brand_his_trade_ratio_level','brand_lastdate_trade_ratio_level','brand_collected_delta_level','brand_price_delta_level','brand_pv_delta_level','brand_sales_delta_level','brand_age_delta_level','shop_brand_item_ratio_level','shop_brand_count_ratio_level',

        'uc_his_trade_ratio_level','uc_next_show_timedelta_level','uc_near_timedelta_level','uc_nearhour_show_delta_level','uc_price_mean_level','uc_price_delta_level',
        'ui_last_show_timedelta_level','ui_lasthour_show_level','ui_lasthour_show_ratio_level','ui_lastdate_show_level',
        'ca_his_trade_ratio','cg_his_trade_ratio',
        'ia_his_trade_ratio_level','ia_his_trade_delta_level','ig_his_trade_ratio_level','ig_his_trade_delta_level',
    ]
    fea = numFea + strFea
    print(df[fea].info())
    print(predictDf[fea].info())
    # exit()

    # 正式模型
    modelName = "lr2A"
    startTime = datetime.now()
    dv = DictDataset(originDf[fea], numFea, strFea)
    dfX,predictX = [dv.transform(x[fea]) for x in [df,predictDf]]
    dfy = df['is_trade'].values
    clf = trainModel(dfX, dfy, num_factors=len(fea))
    joblib.dump(clf, './%s.pkl' % (modelName), compress=3)
    print('training time: ', datetime.now()-startTime)

    # 开始预测
    predictDf.loc[:,'predicted_score'] = clf.predict(predictX)
    print("预测结果：\n",predictDf[['instance_id','predicted_score']].head())
    print('预测均值：', predictDf['predicted_score'].mean())
    # exportResult(predictDf[['instance_id','predicted_score']], "%s.txt" % modelName)
    # exportResult(predictDf[['instance_id','predicted_score','hour']], "%s_hashour.txt" % modelName)

    # 生成stacking数据集
    df['predicted_score'] = np.nan
    predictDf['predicted_score'] = np.nan
    df.loc[:,'predicted_score'], predictDf.loc[:,'predicted_score'] = getOof(clf, dfX, dfy, predictX, stratify=True)#, weight=df['weight'].values
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
