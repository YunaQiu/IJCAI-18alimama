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
import xgboost as xgb
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
    df['minute'] = df.context_timestamp.dt.minute
    df['day'] = df.context_timestamp.dt.day
    df['date'] = pd.to_datetime(df.context_timestamp.dt.date)
    df['date_str'] = df['date'].astype(str)
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
    if not isinstance(originDf, pd.DataFrame):
        originDf = df.copy()
        originDf['is_trade'].fillna(0,inplace=True)
    tempDf = statDateTrade(originDf, 'item_category1', **params['statDateTrade'])
    tempDf.columns = ['cate_his_show','cate_his_trade']
    tempDf['cate_his_trade_ratio'] = biasSmooth(tempDf['cate_his_trade'].values, tempDf['cate_his_show'].values)
    df = df.merge(tempDf, how='left', left_on=['item_category1','date'], right_index=True)

    tempDf = statDateTrade(originDf, 'item_category2', **params['statDateTrade'])
    tempDf.columns = ['cate2_his_show','cate2_his_trade']
    tempDf['cate2_his_trade_ratio'] = biasSmooth(tempDf['cate2_his_trade'].values, tempDf['cate2_his_show'].values)
    df = df.merge(tempDf, how='left', left_on=['item_category2','date'], right_index=True)

    tempDf = pd.pivot_table(df, index=['item_category1','date'], values='is_trade', aggfunc=[len,np.sum])
    tempDf.columns = ['cate_lastdate_show','cate_lastdate_trade']
    tempDf['cate_lastdate_trade_ratio'] = biasSmooth(tempDf['cate_lastdate_trade'].values, tempDf['cate_lastdate_show'].values)
    tempDf.reset_index(inplace=True)
    tempDf['date'] = tempDf['date'] + timedelta(days=1)
    df = df.merge(tempDf, how='left', on=['item_category1','date'])

    df = statItemFea(df, originDf, 'item_category1', 'cate')
    df = statUserAge(df, originDf, 'item_category1', 'cate')
    return df

# 添加商品历史浏览量及购买量特征
def addShopFea(df, originDf=None, **params):
    if not isinstance(originDf, pd.DataFrame):
        originDf = df.copy()
        originDf['is_trade'].fillna(0,inplace=True)
    tempDf = statDateTrade(df, 'shop_id', **params['statDateTrade'])
    tempDf.columns = ['shop_his_show','shop_his_trade']
    tempDf.reset_index(inplace=True)
    tempDf['shop_his_trade_ratio'] = biasSmooth(tempDf['shop_his_trade'].values, tempDf['shop_his_show'].values)
    df = df.merge(tempDf, how='left', on=['shop_id','date'])

    tempDf = pd.pivot_table(df, index=['shop_id','date'], values='is_trade', aggfunc=[len,np.sum])
    tempDf.columns = ['shop_lastdate_show','shop_lastdate_trade']
    tempDf['shop_lastdate_trade_ratio'] = biasSmooth(tempDf['shop_lastdate_trade'].values, tempDf['shop_lastdate_show'].values)
    tempDf.reset_index(inplace=True)
    tempDf['date'] = tempDf['date'] + timedelta(days=1)
    df = df.merge(tempDf, how='left', on=['shop_id','date'])

    df = statItemFea(df, df, 'shop_id', 'shop')
    df['shop_sales_delta'] = df['shop_sales_mean'] - df['cate_sales_mean']
    df['shop_collected_delta'] = df['shop_collected_mean'] - df['cate_collected_mean']
    df['shop_price_delta'] = df['shop_price_mean'] - df['cate_price_mean']
    df['shop_pv_delta'] = df['shop_pv_mean'] - df['cate_pv_mean']

    tempDf = pd.pivot_table(df.drop_duplicates(['shop_id','date']), index=['item_category1','date'], values=['shop_lastdate_show','shop_lastdate_trade'], aggfunc=np.mean)
    tempDf.columns = ['shop_lastdate_show_mean','shop_lastdate_trade_mean']
    df = df.merge(tempDf, how='left', left_on=['item_category1','date'], right_index=True)
    df['shop_lastdate_show_delta'] = df['shop_lastdate_show'] - df['shop_lastdate_show_mean']
    df['shop_lastdate_trade_delta'] = df['shop_lastdate_trade'] - df['shop_lastdate_trade_mean']

    shopDf = df.drop_duplicates(['item_category1','shop_id'], keep='last')
    tempDf = pd.pivot_table(shopDf, index=['item_category1'], values=['shop_review_positive_rate','shop_score_service','shop_score_delivery','shop_score_description','shop_star_level','shop_review_num_level','shop_item_count'], aggfunc=np.mean)
    tempDf.columns = ['shop_item_count_mean','shop_review_num_mean','shop_review_positive_mean','shop_score_delivery_mean','shop_score_description_mean','shop_score_service_mean','shop_star_level_mean']
    tempDf.reset_index(inplace=True)
    df = df.merge(tempDf, how='left', on='item_category1')
    df['shop_review_num_delta'] = df['shop_review_num_level'] - df['shop_review_num_mean']
    df['shop_review_positive_delta'] = df['shop_review_positive_rate'] - df['shop_review_positive_mean']
    df['shop_score_service_delta'] = df['shop_score_service'] - df['shop_score_service_mean']
    df['shop_score_delivery_delta'] = df['shop_score_delivery'] - df['shop_score_delivery_mean']
    df['shop_score_description_delta'] = df['shop_score_description'] - df['shop_score_description_mean']
    df['shop_star_level_delta'] = df['shop_star_level'] - df['shop_star_level_mean']
    df['shop_item_count_delta'] = df['shop_item_count'] - df['shop_item_count_mean']

    df = statUserAge(df, df, 'shop_id', 'shop')
    df = statUserGender(df, df, 'shop_id', 'shop')

    tempDf = df.drop_duplicates(['item_id'])
    tempDf = pd.pivot_table(tempDf, index=['shop_id','item_brand_id'], values=['item_id'], aggfunc=[len])
    tempDf.columns = ['shop_brand_item_count']
    tempDf.reset_index(inplace=True)
    tempDf2 = tempDf.shop_id.value_counts().to_frame()
    tempDf2.columns = ['shop_brand_count']
    df = df.merge(tempDf, how='left', on=['shop_id','item_brand_id'])
    df = df.merge(tempDf2, how='left', left_on=['shop_id'], right_index=True)
    df['shop_brand_item_ratio'] = biasSmooth(df['shop_brand_item_count'].values, df['shop_item_count'].values)
    df['shop_brand_count_ratio'] = biasSmooth(df['shop_brand_count'].values, df['shop_item_count'].values)
    df['shop_brand_special_degree'] = biasSmooth(df['shop_item_count'].values,(df['shop_item_count']+df['shop_brand_count']).values)
    return df

# 添加商品历史浏览量及购买量特征
def addItemFea(df, originDf=None, **params):
    if not isinstance(originDf, pd.DataFrame):
        originDf = df.copy()
        originDf['is_trade'].fillna(0,inplace=True)
    tempDf = statDateTrade(originDf, 'item_id', **params['statDateTrade'])
    tempDf.columns = ['item_his_show','item_his_trade']
    tempDf['item_his_trade_ratio'] = biasSmooth(tempDf['item_his_trade'].values, tempDf['item_his_show'].values)
    df = df.merge(tempDf, how='left', left_on=['item_id','date'], right_index=True)
    df['item_prop_num'] = df['item_property_list'].dropna().map(lambda x: len(x))

    tempDf = pd.pivot_table(df, index=['item_id','date'], values='is_trade', aggfunc=[len,np.sum])
    tempDf.columns = ['item_lastdate_show','item_lastdate_trade']
    tempDf['item_lastdate_trade_ratio'] = biasSmooth(tempDf['item_lastdate_trade'].values, tempDf['item_lastdate_show'].values)
    tempDf.reset_index(inplace=True)
    tempDf['date'] = tempDf['date'] + timedelta(days=1)
    df = df.merge(tempDf, how='left', on=['item_id','date'])

    df['item_sales_delta'] = df['item_sales_level'] - df['cate_sales_mean']
    df['item_collected_delta'] = df['item_collected_level'] - df['cate_collected_mean']
    df['item_price_delta'] = df['item_price_level'] - df['cate_price_mean']
    df['item_pv_delta'] = df['item_pv_level'] - df['cate_pv_mean']

    tempDf = pd.pivot_table(df.drop_duplicates(['item_id','date']), index=['item_category1','date'], values=['item_lastdate_show','item_lastdate_trade'], aggfunc=np.mean)
    tempDf.columns = ['item_lastdate_show_mean','item_lastdate_trade_mean']
    df = df.merge(tempDf, how='left', left_on=['item_category1','date'], right_index=True)
    df['item_lastdate_show_delta'] = df['item_lastdate_show'] - df['item_lastdate_show_mean']
    df['item_lastdate_trade_delta'] = df['item_lastdate_trade'] - df['item_lastdate_trade_mean']
    df['item_lastdate_trade_ratio_delta'] = df['item_lastdate_trade_ratio'] - df['cate_lastdate_trade_ratio']

    df = statUserAge(df, originDf, 'item_id', 'item')
    df = statUserGender(df, originDf, 'item_id', 'item')
    return df

# 添加用户维度特征
def addUserFea(df, originDf=None, **params):
    if not isinstance(originDf, pd.DataFrame):
        originDf = df.copy()
        originDf['is_trade'].fillna(0,inplace=True)
    startTime = datetime.now()
    tempDf = statDateTrade(originDf, 'user_id', **params['statDateTrade'])
    tempDf.columns = ['user_his_show','user_his_trade']
    tempDf['user_his_trade_ratio'] = biasSmooth(tempDf['user_his_trade'].values, tempDf['user_his_show'].values)
    df = df.merge(tempDf, how='left', left_on=['user_id','date'], right_index=True)

    tempDf = pd.pivot_table(originDf, index=['user_id','context_timestamp'], values=['is_trade'], aggfunc=len)
    tempDf.columns = ['show']
    tempDf.reset_index(inplace=True)
    tempDf['last_user_id'] = tempDf['user_id'].shift(1)
    tempDf['last_user_id'] = tempDf['last_user_id']==tempDf['user_id']
    tempDf['last_show_time'] = tempDf['context_timestamp'].shift(1)
    tempDf.loc[~tempDf.last_user_id, 'last_show_time'] = np.nan
    tempDf['user_last_show_timedelta'] = tempDf['context_timestamp'] - tempDf['last_show_time']
    tempDf['user_last_show_timedelta'] = tempDf['user_last_show_timedelta'].dt.seconds
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

    # 穿越特征
    tempDf = pd.pivot_table(df, index=['user_id','context_timestamp'], values=['is_trade'], aggfunc=[len])
    tempDf.sort_index(ascending=False, inplace=True)
    tempDf.columns = ['show']
    tempDf.reset_index(inplace=True)
    tempDf['next_user'] = tempDf['user_id'].shift(1)
    tempDf['next_time'] = tempDf['context_timestamp'].shift(1)
    tempDf['same_next'] = (tempDf.user_id==tempDf.next_user)
    tempDf.loc[~tempDf.same_next, 'next_time'] = np.nan
    tempDf['user_next_show_timedelta'] = tempDf['next_time'] - tempDf['context_timestamp']
    tempDf['user_next_show_timedelta'] = tempDf['user_next_show_timedelta'].dt.seconds
    hourShowList = []
    hourShowTemp = {}
    for same, dt, show in tempDf[['same_next','context_timestamp','show']].values:
        if same:
            [hourShowTemp.pop(k) for k in list(hourShowTemp) if k>dt+timedelta(hours=1)]
            hourShowList.append(np.sum(list(hourShowTemp.values())))
            hourShowTemp[dt] = show
        else:
            hourShowList.append(0)
            hourShowTemp = {dt:show}
    tempDf['user_nexthour_show'] = hourShowList
    df = df.merge(tempDf[['user_id','context_timestamp','user_next_show_timedelta','user_nexthour_show']], how='left', on=['user_id','context_timestamp'])
    df['user_near_timedelta'] = df['user_next_show_timedelta'] - df['user_last_show_timedelta']
    df.loc[(df.user_near_timedelta.isnull())&(df.user_next_show_timedelta.notnull()), ['user_near_timedelta']] = -999999
    df.loc[(df.user_near_timedelta.isnull())&(df.user_last_show_timedelta.notnull()), ['user_near_timedelta']] = 999999
    df['user_nearhour_show_delta'] = df['user_nexthour_show'] - df['user_lasthour_show']
    df.loc[(df.user_lasthour_show==0)&(df.user_nexthour_show==0), ['user_nearhour_show_delta']] = np.nan
    df.fillna({k:999999 for k in ['user_last_show_timedelta', 'user_next_show_timedelta']}, inplace=True)

    tempDf = pd.pivot_table(df, index=['user_id','date'], values=['is_trade'], aggfunc=[len,np.sum])
    tempDf.columns = ['user_lastdate_show','user_lastdate_trade']
    tempDf.reset_index(inplace=True)
    tempDf['date'] = tempDf['date'] + timedelta(days=1)
    df = df.merge(tempDf, how='left', on=['user_id','date'])
    df.fillna({k:0 for k in ['user_lastdate_show', 'user_lastdate_trade','user_lasthour_show']}, inplace=True)
    df['user_lastdate_trade_ratio'] = biasSmooth(df['user_lastdate_trade'].values, df['user_lastdate_show'].values)
    return df

# 添加广告商品与查询词的相关性特征
def addContextFea(df, **params):
    df['predict_category'] = df['predict_category_property'].dropna().map(lambda x: list(x.keys()))
    df['predict_cate_num'] = df['predict_category'].dropna().map(lambda x: len(x))
    idx = df[df.predict_category_property.notnull()].index
    df.loc[idx,'cate_intersect_num'] = list(map(lambda x: len(np.intersect1d(x[0],x[1])), df.loc[idx, ['item_category_list','predict_category']].values))
    df['predict_property'] = [set() for i in range(len(df))]
    idx = df[(df.item_category2.notnull())&(df.predict_category_property.notnull())].index
    df.loc[idx,'predict_property'] = list(map(lambda x: x[2]|set(x[1][x[0]]) if (x[0] in x[1].keys()) else x[2], df.loc[idx,['item_category2','predict_category_property','predict_property']].values))
    idx = df[(df.item_category1.notnull())&(df.predict_category_property.notnull())].index
    df.loc[idx,'predict_property'] = list(map(lambda x: x[2]|set(x[1][x[0]]) if (x[0] in x[1].keys()) else x[2], df.loc[idx,['item_category1','predict_category_property','predict_property']].values))
    df['predict_property'] = df['predict_property'].map(lambda x: np.nan if len(x)==0 else list(x))
    df['predict_prop_num'] = df[df.predict_property.notnull()]['predict_property'].map(lambda x: len(x))
    idx = df[(df.predict_property.notnull())&(df.item_property_list.notnull())].index
    df.loc[idx, 'prop_intersect_num'] = list(map(lambda x: len(np.intersect1d(x[0],x[1])), df.loc[idx, ['item_property_list','predict_property']].values))
    df.loc[idx,'prop_union_num'] = list(map(lambda x: len(np.union1d(x[0],x[1])), df.loc[idx, ['item_property_list','predict_property']].values))
    df['prop_jaccard'] = df['prop_intersect_num'] / df['prop_union_num']
    df['prop_predict_ratio'] = df['prop_intersect_num'] / df['predict_prop_num']
    df['prop_item_ratio'] = df['prop_intersect_num'] / df['item_prop_num']
    df['prop_jaccard_bias'] = biasSmooth(df['prop_intersect_num'].values, df['prop_union_num'].values)
    df['prop_predict_ratio_bias'] = biasSmooth(df['prop_intersect_num'].values, df['predict_prop_num'].values)
    df['prop_item_ratio_bias'] = biasSmooth(df['prop_intersect_num'].values, df['item_prop_num'].values)
    # df.fillna({k:-1 for k in ['predict_prop_num','prop_intersect_num','prop_union_num','prop_jaccard','prop_predict_ratio','prop_item_ratio']}, inplace=True)
    return df

# 添加品牌相关特征
def addBrandFea(df, originDf=None, **params):
    if not isinstance(originDf, pd.DataFrame):
        originDf = df.copy()
        originDf['is_trade'].fillna(0,inplace=True)
    brandDf = df.drop_duplicates(['item_id']).item_brand_id.value_counts()
    dropBrand = brandDf[brandDf<2].index
    df.loc[df.item_brand_id.isin(dropBrand),'item_brand_id'] = np.nan
    tempDf = statDateTrade(originDf, 'item_brand_id', **params['statDateTrade'])
    tempDf.columns = ['brand_his_show','brand_his_trade']
    tempDf['brand_his_trade_ratio'] = biasSmooth(tempDf['brand_his_trade'].values, tempDf['brand_his_show'].values)
    df = df.merge(tempDf, how='left', left_on=['item_brand_id','date'], right_index=True)

    tempDf = pd.pivot_table(df, index=['item_brand_id','date'], values='is_trade', aggfunc=[len,np.sum])
    tempDf.columns = ['brand_lastdate_show','brand_lastdate_trade']
    tempDf['brand_lastdate_trade_ratio'] = biasSmooth(tempDf['brand_lastdate_trade'].values, tempDf['brand_lastdate_show'].values)
    tempDf.reset_index(inplace=True)
    tempDf['date'] = tempDf['date'] + timedelta(days=1)
    df = df.merge(tempDf, how='left', on=['item_brand_id','date'])

    df = statItemFea(df, originDf, 'item_brand_id', 'brand')
    df['brand_sales_delta'] = df['brand_sales_mean'] - df['cate_sales_mean']
    df['brand_collected_delta'] = df['brand_collected_mean'] - df['cate_collected_mean']
    df['brand_price_delta'] = df['brand_price_mean'] - df['cate_price_mean']
    df['brand_pv_delta'] = df['brand_pv_mean'] - df['cate_pv_mean']

    df = statUserAge(df, originDf, 'item_brand_id', 'brand')
    df = statUserGender(df, originDf, 'item_brand_id', 'brand')
    return df

# 添加用户与类目关联维度的特征
def addUserCateFea(df, originDf=None, **params):
    if not isinstance(originDf, pd.DataFrame):
        originDf = df.copy()
        originDf['is_trade'].fillna(0,inplace=True)
    tempDf = pd.pivot_table(originDf, index=['user_id','item_category1','context_timestamp'], values=['item_price_level'], aggfunc=[len,np.mean])
    tempDf.columns = ['show','price']
    tempDf.reset_index(inplace=True)
    tempDf['last_user'] = tempDf['user_id'].shift(1)
    tempDf['last_cate'] = tempDf['item_category1'].shift(1)
    tempDf['last_time'] = tempDf['context_timestamp'].shift(1)
    tempDf['same'] = (tempDf.user_id==tempDf.last_user) & (tempDf.item_category1==tempDf.last_cate)
    tempDf.loc[~tempDf.same, 'last_time'] = np.nan
    tempDf['uc_last_show_timedelta'] = tempDf['context_timestamp'] - tempDf['last_time']
    tempDf['uc_last_show_timedelta'] = tempDf['uc_last_show_timedelta'].dt.seconds
    hourShowList = []
    hourShowTemp = {}
    priceList = []
    tempPrice = []
    for same, dt, show, price in tempDf[['same','context_timestamp','show','price']].values:
        if same:
            [hourShowTemp.pop(k) for k in list(hourShowTemp) if k<dt-timedelta(hours=1)]
            hourShowList.append(np.sum(list(hourShowTemp.values())))
            hourShowTemp[dt] = show
            priceList.append(tempPrice)
            tempPrice = tempPrice+[price]
        else:
            hourShowList.append(0)
            hourShowTemp = {dt:show}
            priceList.append(np.nan)
            tempPrice = [price]
    tempDf['uc_lasthour_show'] = hourShowList
    tempDf['uc_price_mean'] = priceList
    tempDf['uc_price_mean'] = tempDf.loc[tempDf.uc_price_mean.notnull(),'uc_price_mean'].map(lambda x: np.mean(x))
    df = df.merge(tempDf[['user_id','item_category1','context_timestamp','uc_last_show_timedelta','uc_lasthour_show','uc_price_mean']], how='left', on=['user_id','item_category1','context_timestamp'])
    df['uc_lasthour_show'].fillna(0, inplace=True)
    df['uc_lasthour_show_ratio'] = biasSmooth(df.uc_lasthour_show.values, df.user_lasthour_show.values)
    df['uc_price_delta'] = df['item_price_level'] - df['uc_price_mean']

    # 穿越特征
    tempDf = pd.pivot_table(df, index=['user_id','item_category1','context_timestamp'], values=['item_price_level'], aggfunc=[len])
    tempDf.sort_index(ascending=False, inplace=True)
    tempDf.columns = ['show']
    tempDf.reset_index(inplace=True)
    tempDf['next_user'] = tempDf['user_id'].shift(1)
    tempDf['next_cate'] = tempDf['item_category1'].shift(1)
    tempDf['next_time'] = tempDf['context_timestamp'].shift(1)
    tempDf['same_next'] = (tempDf.user_id==tempDf.next_user) & (tempDf.item_category1==tempDf.next_cate)
    tempDf.loc[~tempDf.same_next, 'next_time'] = np.nan
    tempDf['uc_next_show_timedelta'] = tempDf['next_time'] - tempDf['context_timestamp']
    tempDf['uc_next_show_timedelta'] = tempDf['uc_next_show_timedelta'].dt.seconds
    hourShowList = []
    hourShowTemp = {}
    for same, dt, show in tempDf[['same_next','context_timestamp','show']].values:
        if same:
            [hourShowTemp.pop(k) for k in list(hourShowTemp) if k>dt+timedelta(hours=1)]
            hourShowList.append(np.sum(list(hourShowTemp.values())))
            hourShowTemp[dt] = show
        else:
            hourShowList.append(0)
            hourShowTemp = {dt:show}
    tempDf['uc_nexthour_show'] = hourShowList
    df = df.merge(tempDf[['user_id','item_category1','context_timestamp','uc_next_show_timedelta','uc_nexthour_show']], how='left', on=['user_id','item_category1','context_timestamp'])
    df['uc_near_timedelta'] = df['uc_next_show_timedelta'] - df['uc_last_show_timedelta']
    df.loc[(df.uc_near_timedelta.isnull())&(df.uc_next_show_timedelta.notnull()), ['uc_near_timedelta']] = -999999
    df.loc[(df.uc_near_timedelta.isnull())&(df.uc_last_show_timedelta.notnull()), ['uc_near_timedelta']] = 999999
    df['uc_nearhour_show_delta'] = df['uc_nexthour_show'] - df['uc_lasthour_show']
    df.loc[(df.uc_lasthour_show==0)&(df.uc_nexthour_show==0), ['uc_nearhour_show_delta']] = np.nan
    df.fillna({k:999999 for k in ['uc_last_show_timedelta', 'uc_next_show_timedelta']}, inplace=True)

    tempDf = statDateTrade(originDf, ['user_id','item_category1'], **params['statDateTrade'])
    tempDf.columns = ['uc_his_show','uc_his_trade']
    tempDf['uc_his_trade_ratio'] = biasSmooth(tempDf['uc_his_trade'].values, tempDf['uc_his_show'].values)
    df = df.merge(tempDf, how='left', left_on=['user_id','item_category1','date'], right_index=True)
    return df

# 添加用户商品关联维度的特征
def addUserItemFea(df, originDf=None, **params):
    if not isinstance(originDf, pd.DataFrame):
        originDf = df.copy()
        originDf['is_trade'].fillna(0,inplace=True)
    tempDf = pd.pivot_table(originDf, index=['user_item','context_timestamp'], values=['is_trade'], aggfunc=len)
    tempDf.columns = ['show']
    tempDf.reset_index(inplace=True)
    tempDf['same'] = tempDf['user_item'].shift(1)
    tempDf['same'] = tempDf['same']==tempDf['user_item']
    tempDf['last_show_time'] = tempDf['context_timestamp'].shift(1)
    tempDf.loc[~tempDf.same, 'last_show_time'] = np.nan
    tempDf['ui_last_show_timedelta'] = tempDf['context_timestamp'] - tempDf['last_show_time']
    tempDf['ui_last_show_timedelta'] = tempDf['ui_last_show_timedelta'].dt.seconds
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
    tempDf['ui_lasthour_show'] = hourShowList
    df = df.merge(tempDf[['user_item','context_timestamp','ui_last_show_timedelta','ui_lasthour_show']], how='left', on=['user_item','context_timestamp'])
    df['ui_lasthour_show_ratio'] = biasSmooth(df.ui_lasthour_show.values, df.uc_lasthour_show.values)

    # 穿越特征
    tempDf = pd.pivot_table(originDf, index=['user_item','context_timestamp'], values=['is_trade'], aggfunc=len)
    tempDf.sort_index(ascending=False, inplace=True)
    tempDf.columns = ['show']
    tempDf.reset_index(inplace=True)
    tempDf['same'] = tempDf['user_item'].shift(1)
    tempDf['same'] = tempDf['same']==tempDf['user_item']
    tempDf['next_show_time'] = tempDf['context_timestamp'].shift(1)
    tempDf.loc[~tempDf.same, 'next_show_time'] = np.nan
    tempDf['ui_next_show_timedelta'] = tempDf['next_show_time'] - tempDf['context_timestamp']
    tempDf['ui_next_show_timedelta'] = tempDf['ui_next_show_timedelta'].dt.seconds
    hourShowList = []
    hourShowTemp = {}
    for same, dt, show in tempDf[['same','context_timestamp','show']].values:
        if same:
            [hourShowTemp.pop(k) for k in list(hourShowTemp) if k>dt+timedelta(hours=1)]
            hourShowList.append(np.sum(list(hourShowTemp.values())))
            hourShowTemp[dt] = show
        else:
            hourShowList.append(0)
            hourShowTemp = {dt:show}
    tempDf['ui_nexthour_show'] = hourShowList
    df = df.merge(tempDf[['user_item','context_timestamp','ui_next_show_timedelta','ui_nexthour_show']], how='left', on=['user_item','context_timestamp'])
    df['ui_nexthour_show_ratio'] = biasSmooth(df.ui_nexthour_show.values, df.uc_nexthour_show.values)

    df['ui_near_timedelta'] = df['ui_next_show_timedelta'] - df['ui_last_show_timedelta']
    df.loc[(df.ui_near_timedelta.isnull())&(df.ui_next_show_timedelta.notnull()), ['ui_near_timedelta']] = -999999
    df.loc[(df.ui_near_timedelta.isnull())&(df.ui_last_show_timedelta.notnull()), ['ui_near_timedelta']] = 999999
    df['ui_nearhour_show_delta'] = df['ui_nexthour_show'] - df['ui_lasthour_show']
    df.loc[(df.ui_lasthour_show==0)&(df.ui_nexthour_show==0), ['ui_nearhour_show_delta']] = np.nan
    df.fillna({k:999999 for k in ['ui_last_show_timedelta', 'ui_next_show_timedelta']}, inplace=True)

    tempDf = pd.pivot_table(originDf, index=['user_item','date'], values=['is_trade'], aggfunc=[len,np.sum])
    tempDf.columns = ['ui_lastdate_show', 'ui_lastdate_trade']
    tempDf.reset_index(inplace=True)
    tempDf['date'] = tempDf['date'] + timedelta(days=1)
    df = df.merge(tempDf[['user_item','date','ui_lastdate_show', 'ui_lastdate_trade']], how='left', on=['user_item','date'])
    df.fillna({k:0 for k in ['ui_lastdate_show', 'ui_lastdate_trade','ui_lasthour_show']}, inplace=True)

    tempDf = statDateTrade(originDf, ['user_id','item_id'], **params['statDateTrade'])
    tempDf.columns = ['ui_his_show','ui_his_trade']
    tempDf['ui_his_trade_ratio'] = biasSmooth(tempDf['ui_his_trade'].values, tempDf['ui_his_show'].values)
    df = df.merge(tempDf, how='left', left_on=['user_id','item_id','date'], right_index=True)
    return df

# 统计用户该店铺的统计特征
def addUserShopFea(df, originDf=None, **params):
    if not isinstance(originDf, pd.DataFrame):
        originDf = df.copy()
        originDf['is_trade'].fillna(0,inplace=True)
    tempDf = statDateTrade(originDf, ['user_id','shop_id'], **params['statDateTrade'])
    tempDf.columns = ['us_his_show','us_his_trade']
    tempDf['us_his_trade_ratio'] = biasSmooth(tempDf['us_his_trade'].values, tempDf['us_his_show'].values)
    df = df.merge(tempDf, how='left', left_on=['user_id','shop_id','date'], right_index=True)

    tempDf = pd.pivot_table(originDf, index=['user_id','shop_id','context_timestamp'], values=['is_trade'], aggfunc=[len])
    tempDf.columns = ['show']
    tempDf.reset_index(inplace=True)
    tempDf['last_user'] = tempDf['user_id'].shift(1)
    tempDf['last_shop'] = tempDf['shop_id'].shift(1)
    tempDf['last_time'] = tempDf['context_timestamp'].shift(1)
    tempDf['same'] = (tempDf.user_id==tempDf.last_user) & (tempDf.shop_id==tempDf.last_shop)
    tempDf.loc[~tempDf.same, 'last_time'] = np.nan
    tempDf['us_last_show_timedelta'] = tempDf['context_timestamp'] - tempDf['last_time']
    tempDf['us_last_show_timedelta'] = tempDf['us_last_show_timedelta'].dt.seconds
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
    tempDf['us_lasthour_show'] = hourShowList
    df = df.merge(tempDf[['user_id','shop_id','context_timestamp','us_last_show_timedelta','us_lasthour_show']], how='left', on=['user_id','shop_id','context_timestamp'])
    df['us_lasthour_show_ratio'] = biasSmooth(df.us_lasthour_show.values, df.user_lasthour_show.values)

    # 穿越特征
    tempDf = pd.pivot_table(df, index=['user_id','shop_id','context_timestamp'], values=['item_price_level'], aggfunc=[len])
    tempDf.sort_index(ascending=False, inplace=True)
    tempDf.columns = ['show']
    tempDf.reset_index(inplace=True)
    tempDf['next_user'] = tempDf['user_id'].shift(1)
    tempDf['next_shop'] = tempDf['shop_id'].shift(1)
    tempDf['next_time'] = tempDf['context_timestamp'].shift(1)
    tempDf['same_next'] = (tempDf.user_id==tempDf.next_user) & (tempDf.shop_id==tempDf.next_shop)
    tempDf.loc[~tempDf.same_next, 'next_time'] = np.nan
    tempDf['us_next_show_timedelta'] = tempDf['next_time'] - tempDf['context_timestamp']
    tempDf['us_next_show_timedelta'] = tempDf['us_next_show_timedelta'].dt.seconds
    hourShowList = []
    hourShowTemp = {}
    for same, dt, show in tempDf[['same_next','context_timestamp','show']].values:
        if same:
            [hourShowTemp.pop(k) for k in list(hourShowTemp) if k>dt+timedelta(hours=1)]
            hourShowList.append(np.sum(list(hourShowTemp.values())))
            hourShowTemp[dt] = show
        else:
            hourShowList.append(0)
            hourShowTemp = {dt:show}
    tempDf['us_nexthour_show'] = hourShowList
    df = df.merge(tempDf[['user_id','shop_id','context_timestamp','us_next_show_timedelta','us_nexthour_show']], how='left', on=['user_id','shop_id','context_timestamp'])
    df['us_near_timedelta'] = df['us_next_show_timedelta'] - df['us_last_show_timedelta']
    df.loc[(df.us_near_timedelta.isnull())&(df.us_next_show_timedelta.notnull()), ['us_near_timedelta']] = -999999
    df.loc[(df.us_near_timedelta.isnull())&(df.us_last_show_timedelta.notnull()), ['us_near_timedelta']] = 999999
    df['us_nearhour_show_delta'] = df['us_nexthour_show'] - df['us_lasthour_show']
    df.loc[(df.us_lasthour_show==0)&(df.us_nexthour_show==0), ['us_nearhour_show_delta']] = np.nan
    df.fillna({k:999999 for k in ['us_last_show_timedelta', 'us_next_show_timedelta']}, inplace=True)

    tempDf = pd.pivot_table(originDf, index=['user_id','shop_id','date'], values=['is_trade'], aggfunc=[len,np.sum])
    tempDf.columns = ['us_lastdate_show', 'us_lastdate_trade']
    tempDf.reset_index(inplace=True)
    tempDf['date'] = tempDf['date'] + timedelta(days=1)
    df = df.merge(tempDf[['user_id','shop_id','date','us_lastdate_show', 'us_lastdate_trade']], how='left', on=['user_id','shop_id','date'])
    df.fillna({k:0 for k in ['us_lastdate_show', 'us_lastdate_trade']}, inplace=True)
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
    tempDf = statDateTrade(df, ['item_category1','user_age_level'], **params['statDateTrade'])
    tempDf.columns = ['ca_his_show','ca_his_trade']
    tempDf['ca_his_trade_ratio'] = biasSmooth(tempDf['ca_his_trade'].values, tempDf['ca_his_show'].values)
    df = df.merge(tempDf, how='left', left_on=['item_category1','user_age_level','date'], right_index=True)
    for x in set(df.user_age_level.dropna().values):
        idx = df[df.user_age_level==x].index
        df.loc[idx, 'ca_his_show_ratio'] = biasSmooth(df.loc[idx, 'ca_his_show'].values, df.loc[idx,'cate_his_show'].values)

    tempDf = statDateTrade(df, ['item_category1','user_gender_id'], **params['statDateTrade'])
    tempDf.columns = ['cg_his_show','cg_his_trade']
    tempDf['cg_his_trade_ratio'] = biasSmooth(tempDf['cg_his_trade'].values, tempDf['cg_his_show'].values)
    df = df.merge(tempDf, how='left', left_on=['item_category1','user_gender_id','date'], right_index=True)
    for x in set(df.user_age_level.dropna().values):
        idx = df[df.user_age_level==x].index
        df.loc[idx, 'cg_his_show_ratio'] = biasSmooth(df.loc[idx, 'cg_his_show'].values, df.loc[idx,'cate_his_show'].values)
    return df

# 统计商品与各类别用户群交叉特征
def addItemCrossFea(df, **params):
    tempDf = statDateTrade(df, ['item_id','user_age_level'], **params['statDateTrade'])
    tempDf.columns = ['ia_his_show','ia_his_trade']
    tempDf['ia_his_trade_ratio'] = biasSmooth(tempDf['ia_his_trade'].values, tempDf['ia_his_show'].values)
    df = df.merge(tempDf, how='left', left_on=['item_id','user_age_level','date'], right_index=True)
    for x in set(df.user_age_level.dropna().values):
        idx = df[df.user_age_level==x].index
        df.loc[idx, 'ia_his_show_ratio'] = biasSmooth(df.loc[idx, 'ia_his_show'].values, df.loc[idx,'item_his_show'].values)
    df['ia_his_show_delta'] = df['ia_his_show_ratio'] - df['ca_his_show_ratio']
    df['ia_his_trade_delta'] = df['ia_his_trade_ratio'] - df['ca_his_trade_ratio']

    tempDf = statDateTrade(df, ['item_id','user_gender_id'], **params['statDateTrade'])
    tempDf.columns = ['ig_his_show','ig_his_trade']
    tempDf['ig_his_trade_ratio'] = biasSmooth(tempDf['ig_his_trade'].values, tempDf['ig_his_show'].values)
    df = df.merge(tempDf, how='left', left_on=['item_id','user_gender_id','date'], right_index=True)
    for x in set(df.user_gender_id.dropna().values):
        idx = df[df.user_gender_id==x].index
        df.loc[idx, 'ig_his_show_ratio'] = biasSmooth(df.loc[idx, 'ig_his_show'].values, df.loc[idx,'item_his_show'].values)
    df['ig_his_show_delta'] = df['ig_his_show_ratio'] - df['cg_his_show_ratio']
    df['ig_his_trade_delta'] = df['ig_his_trade_ratio'] - df['cg_his_trade_ratio']
    return df

# 统计品牌与各类别用户群交叉特征
def addBrandCrossFea(df, **params):
    tempDf = statDateTrade(df, ['item_brand_id','user_age_level'], **params['statDateTrade'])
    tempDf.columns = ['ba_his_show','ba_his_trade']
    tempDf['ba_his_trade_ratio'] = biasSmooth(tempDf['ba_his_trade'].values, tempDf['ba_his_show'].values)
    df = df.merge(tempDf, how='left', left_on=['item_brand_id','user_age_level','date'], right_index=True)
    for x in set(df.user_age_level.dropna().values):
        idx = df[df.user_age_level==x].index
        df.loc[idx, 'ba_his_show_ratio'] = biasSmooth(df.loc[idx, 'ba_his_show'].values, df.loc[idx,'brand_his_show'].values)
    df['ba_his_show_delta'] = df['ba_his_show_ratio'] - df['ca_his_show_ratio']
    df['ba_his_trade_delta'] = df['ba_his_trade_ratio'] - df['ca_his_trade_ratio']

    tempDf = statDateTrade(df, ['item_brand_id','user_gender_id'], **params['statDateTrade'])
    tempDf.columns = ['bg_his_show','bg_his_trade']
    tempDf['bg_his_trade_ratio'] = biasSmooth(tempDf['bg_his_trade'].values, tempDf['bg_his_show'].values)
    df = df.merge(tempDf, how='left', left_on=['item_brand_id','user_gender_id','date'], right_index=True)
    for x in set(df.user_gender_id.dropna().values):
        idx = df[df.user_gender_id==x].index
        df.loc[idx, 'bg_his_show_ratio'] = biasSmooth(df.loc[idx, 'bg_his_show'].values, df.loc[idx,'brand_his_show'].values)
    df['bg_his_show_delta'] = df['bg_his_show_ratio'] - df['cg_his_show_ratio']
    df['bg_his_trade_delta'] = df['bg_his_trade_ratio'] - df['cg_his_trade_ratio']
    return df

# 数据清洗
def dataCleaning(df):
    startTime = datetime.now()
    df = formatDf(df)
    df = splitMultiFea(df)
    df = combineKey(df)
    df = cleanItemFea(df)
    df = cleanUserFea(df)
    df = cleanShopFea(df)
    print('cleaning time:', datetime.now()-startTime)
    return df

# 特征方法汇总
def feaFactory(df, originDf=None, **args):
    startTime = datetime.now()
    params = {
        'statDateTrade': {
            'statDates': None,
            'skipDates': 1,
        }
    }
    for k1,arg in args.items():
        for k2,v in arg.items():
            params[k1][k2] = v
    df = addTimeFea(df, **params)
    print('finished time fea: ', datetime.now() - startTime)
    df = addCateFea(df, originDf, **params)
    print('finished cate fea: ', datetime.now() - startTime)
    df = addUserFea(df, originDf, **params)
    print('finished user fea: ', datetime.now() - startTime)
    df = addShopFea(df, originDf, **params)
    print('finished shop fea: ', datetime.now() - startTime)
    df = addItemFea(df, originDf, **params)
    print('finished item fea: ', datetime.now() - startTime)
    df = addContextFea(df, **params)
    print('finished context fea: ', datetime.now() - startTime)
    df = addBrandFea(df,originDf, **params)
    print('finished brand fea: ', datetime.now() - startTime)
    df = addUserCateFea(df, originDf, **params)
    print('finished uc fea: ', datetime.now() - startTime)
    df = addUserItemFea(df, originDf, **params)
    print('finished ui fea: ', datetime.now() - startTime)
    df = addUserShopFea(df, originDf, **params)
    print('finished us fea: ', datetime.now() - startTime)
    df = addUserPriceFea(df, originDf, **params)
    print('finished up fea: ', datetime.now() - startTime)
    # df = addUserHourFea(df, originDf, **params)
    df = addCateCrossFea(df, **params)
    print('finished catecross fea: ', datetime.now() - startTime)
    df = addItemCrossFea(df, **params)
    print('finished itemcross fea: ', datetime.now() - startTime)
    df = addBrandCrossFea(df, **params)
    print('finished brandcross fea: ', datetime.now() - startTime)
    print('finished feaFactory: ', datetime.now() - startTime)
    return df

# 导出预测结果
def exportResult(df, fileName, header=True, index=False, sep=' '):
    df.to_csv('./%s' % fileName, sep=sep, header=header, index=index)


def main():
    # 准备数据
    startTime = datetime.now()
    # df = importDf('./data/round2_train_sample.txt')
    df = importDf('./data/round2_train.txt')
    df['dataset'] = 0
    # dfA = importDf('./data/round2_test_a_sample.txt')
    dfA = importDf('./data/round2_ijcai_18_test_a_20180425.txt')
    dfA['dataset'] = -1
    # predictDf = importDf('./data/round2_test_b_sample.txt')
    predictDf = importDf('./data/round2_ijcai_18_test_b_20180510.txt')
    predictDf['dataset'] = -2
    originDf = pd.concat([df,dfA,predictDf], ignore_index=True)
    # originDf = originDf.sample(frac=0.1)
    print('prepare dataset time:', datetime.now()-startTime)

    # 特征处理
    originDf = dataCleaning(originDf)
    originDf = feaFactory(originDf)
    df = originDf[(originDf.date=='2018-09-07')&(originDf['dataset']>=0)]
    df2 = originDf[(originDf.date.isin(['2018-09-01','2018-09-02','2018-09-03']))&(originDf['dataset']>=0)]
    dfA = originDf[originDf['dataset']==-1]
    predictDf = originDf[originDf['dataset']==-2]

    print(df.iloc[:,:80].info())
    print(df.iloc[:,80:160].info())
    print(df.iloc[:,160:].info())

    fea2 = set(df.columns.values)
    fea2 = fea2 - set(['context_timestamp','dataset','predict_category_property','item_category_list','item_property_list','date','predict_category','predict_property'])
    fea2 = list(fea2)
    print(df.loc[:,fea2].info())
    print(df2.loc[:,fea2].info())
    print(dfA.loc[:,fea2].info())
    print(predictDf.loc[:,fea2].info())
    exit()
    exportResult(df[fea2], "./data/train_fea_special.csv")
    print('export special finished')
    exportResult(df2[fea2], "./data/train_fea_normal.csv")
    print('export normal finished')
    exportResult(dfA[fea2], "./data/test_a_fea.csv")
    print('export testa finished')
    exportResult(predictDf[fea2], "./data/test_b_fea.csv")
    print('export testb finished')

if __name__ == '__main__':
    main()
