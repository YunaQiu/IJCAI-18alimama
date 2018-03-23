#!/usr/bin/env python
# -*-coding:utf-8-*-

'''
模型： xgboostCV
模型参数：'objective': 'binary:logistic',
        'eval_metric':'logloss',
        'silent': True,
        'eta': 0.1,
        'max_depth': 4,
        'gamma': 0.01,
        'subsample':0.7,
        'colsample_bytree': 0.75,
        'min_child_weight': 8,
        'max_delta_step': 2,
        'num_boost_round': 1500
        'early_stopping_rounds': 10
        'nfold': 3
特征： 商品一级类目，商品二级类目，商品城市id，商品销量等级，商品收藏量等级，商品价格等级
      用户性别编号/用户年龄段/用户星级等级
      小时数，2倍小时数，展示页码编号
      店铺好评率/店铺服务评分/店铺物流评分/店铺描述评分
      用户历史点击数（当天以前），用户历史交易数（当天以前），用户历史转化率（当天以前），用户距离上次点击时长（秒），用户过去一小时点击数
      商品历史点击数（当天以前），商品历史交易数（当天以前），商品历史转化率（当天以前）
      用户距离上次浏览该商品的时长，用户过去一小时浏览该商品的次数，用户前一天浏览该商品的次数，用户前一天购买该商品的次数
结果： A榜（0.08168）

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

from sklearn.preprocessing import *
import xgboost as xgb
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.externals import joblib

# 导入数据
def importDf(url, sep=' ', header='infer', index_col=None, colNames=None):
    df = pd.read_csv(url, sep=sep, header=header, index_col=index_col, names=colNames)
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

# 转化数据集字段格式，并去重
def formatDf(df):
    strCols = ['instance_id','item_id','user_id','context_id','shop_id','item_brand_id','item_city_id','user_gender_id','user_occupation_id']
    df[strCols] = df[strCols].astype('str')
    df = df.applymap(lambda x: np.nan if (x==-1)|(x=='-1') else x)
    df.drop_duplicates(inplace=True)
    df['context_timestamp'] = pd.to_datetime(df.context_timestamp, unit='s')
    return df

# 计算单特征与标签的F值
def getFeaScore(X, y, feaNames):
    resultDf = pd.DataFrame(index=feaNames)
    selecter = SelectKBest(f_classif, 'all').fit(X, y)
    resultDf['scores'] = selecter.scores_
    resultDf['p_values'] = selecter.pvalues_
    return resultDf

# 拆分多维度拼接的字段
def splitMultiFea(df):
    tempDf = df.drop_duplicates(subset=['item_id'])[['item_id','item_category_list','item_property_list']]
    tempDf['item_category_list'] = tempDf[tempDf.item_category_list.notnull()]['item_category_list'].map(lambda x: x.split(';'))
    tempDf['item_category0'] = tempDf[tempDf.item_category_list.notnull()]['item_category_list'].map(lambda x: x[0])
    tempDf['item_category1'] = tempDf[tempDf.item_category_list.notnull()]['item_category_list'].map(lambda x: x[1] if len(x)>1 else np.nan)
    tempDf['item_category2'] = tempDf[tempDf.item_category_list.notnull()]['item_category_list'].map(lambda x: x[2] if len(x)>2 else np.nan)
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

# 添加商品历史浏览量及购买量特征
def addItemFea(df, hisDf=None):
    if isinstance(hisDf, pd.DataFrame):
        originDf = pd.concat([hisDf,df],ignore_index=True)
    else:
        originDf = df.copy()
    tempDf = pd.pivot_table(originDf, index=['item_id','date'], values='is_trade', aggfunc=[len, sum])
    tempDf.columns = ['click','trade']
    tempDf.reset_index(inplace=True)
    tempDf['same_item'] = tempDf['item_id'].shift(1)
    tempDf['same_item'] = tempDf['item_id'] == tempDf['same_item']
    clickList = []
    tradeList = []
    clickTemp = tradeTemp = 0
    for same,c,t in tempDf[['same_item','click','trade']].values:
        clickList.append(clickTemp if same else 0)
        tradeList.append(tradeTemp if same else 0)
        clickTemp = clickTemp+c if same else c
        tradeTemp = tradeTemp+t if same else t
    tempDf['item_his_click'] = clickList
    tempDf['item_his_trade'] = tradeList
    tempDf['item_his_trade_ratio'] = tempDf['item_his_trade'] / tempDf['item_his_click']
    df = df.merge(tempDf[['item_id','date','item_his_click','item_his_trade','item_his_trade_ratio']], how='left', on=['item_id','date'])
    df['item_his_trade_ratio'].fillna(0, inplace=True)

    tempDf = pd.pivot_table(originDf, index=['item_id','context_timestamp'], values=['is_trade'], aggfunc=len)
    tempDf.columns = ['click']
    tempDf.reset_index(inplace=True)
    tempDf['last_item_id'] = tempDf['item_id'].shift(1)
    tempDf['last_item_id'] = tempDf['last_item_id']==tempDf['item_id']
    tempDf['last_click_time'] = tempDf['context_timestamp'].shift(1)
    tempDf.loc[~tempDf.last_item_id, 'last_click_time'] = np.nan
    tempDf['item_last_click_timedelta'] = tempDf['context_timestamp'] - tempDf['last_click_time']
    tempDf['item_last_click_timedelta'] = tempDf['item_last_click_timedelta'].dt.seconds
    tempDf['item_last_click_timedelta'].fillna(999999, inplace=True)
    hourClickList = []
    hourClickTemp = {}
    for same, dt, click in tempDf[['last_item_id','context_timestamp','click']].values:
        if same:
            [hourClickTemp.pop(k) for k in list(hourClickTemp) if k<dt-timedelta(hours=1)]
            hourClickList.append(np.sum(list(hourClickTemp.values())))
            hourClickTemp[dt] = click
        else:
            hourClickList.append(0)
            hourClickTemp = {dt:click}
    tempDf['item_lasthour_click'] = hourClickList
    df = df.merge(tempDf[['item_id','context_timestamp','item_last_click_timedelta','item_lasthour_click']], how='left', on=['item_id','context_timestamp'])
    return df

# 添加用户维度特征
def addUserFea(df, hisDf=None):
    if isinstance(hisDf, pd.DataFrame):
        originDf = pd.concat([hisDf,df],ignore_index=True)
    else:
        originDf = df.copy()
    tempDf = pd.pivot_table(originDf, index=['user_id','date'], values=['is_trade'], aggfunc=[len,np.sum])
    tempDf.columns = ['click', 'trade']
    tempDf.reset_index(inplace=True)
    tempDf['last_user'] = tempDf['user_id'].shift(1)
    tempDf['last_user'] = tempDf['user_id']==tempDf['last_user']
    clickList,tradeList = [[] for i in range(2)]
    clickTemp = tradeTemp = 0
    for same,click,trade in tempDf[['last_user','click', 'trade']].values:
        clickList.append(clickTemp if same else 0)
        clickTemp = clickTemp+click if same else click
        tradeList.append(tradeTemp if same else 0)
        tradeTemp = tradeTemp+trade if same else trade
    tempDf['user_his_click'] = clickList
    tempDf['user_his_trade'] = tradeList
    tempDf['user_his_trade_ratio'] = tempDf.user_his_trade / tempDf.user_his_click
    df = df.merge(tempDf[['user_id','date','user_his_click', 'user_his_trade','user_his_trade_ratio']], how='left', on=['user_id','date'])
    df['user_his_trade_ratio'].fillna(0, inplace=True)

    tempDf = pd.pivot_table(originDf, index=['user_id','context_timestamp'], values=['is_trade'], aggfunc=len)
    tempDf.columns = ['click']
    tempDf.reset_index(inplace=True)
    tempDf['last_user_id'] = tempDf['user_id'].shift(1)
    tempDf['last_user_id'] = tempDf['last_user_id']==tempDf['user_id']
    tempDf['last_click_time'] = tempDf['context_timestamp'].shift(1)
    tempDf.loc[~tempDf.last_user_id, 'last_click_time'] = np.nan
    tempDf['user_last_click_timedelta'] = tempDf['context_timestamp'] - tempDf['last_click_time']
    tempDf['user_last_click_timedelta'] = tempDf['user_last_click_timedelta'].dt.seconds
    tempDf['user_last_click_timedelta'].fillna(999999, inplace=True)
    hourClickList = []
    hourClickTemp = {}
    for same, dt, click in tempDf[['last_user_id','context_timestamp','click']].values:
        if same:
            [hourClickTemp.pop(k) for k in list(hourClickTemp) if k<dt-timedelta(hours=1)]
            hourClickList.append(np.sum(list(hourClickTemp.values())))
            hourClickTemp[dt] = click
        else:
            hourClickList.append(0)
            hourClickTemp = {dt:click}
    tempDf['user_lasthour_click'] = hourClickList
    df = df.merge(tempDf[['user_id','context_timestamp','user_last_click_timedelta','user_lasthour_click']], how='left', on=['user_id','context_timestamp'])
    return df

# 添加用户商品关联维度的特征
def addUserItemFea(df, hisDf=None):
    if isinstance(hisDf, pd.DataFrame):
        originDf = pd.concat([hisDf,df],ignore_index=True)
    else:
        originDf = df.copy()
    tempDf = pd.pivot_table(originDf, index=['user_item','context_timestamp'], values=['is_trade'], aggfunc=len)
    tempDf.columns = ['click']
    tempDf.reset_index(inplace=True)
    tempDf['last_user_item'] = tempDf['user_item'].shift(1)
    tempDf['last_user_item'] = tempDf['last_user_item']==tempDf['user_item']
    tempDf['last_click_time'] = tempDf['context_timestamp'].shift(1)
    tempDf.loc[~tempDf.last_user_item, 'last_click_time'] = np.nan
    tempDf['ui_last_click_timedelta'] = tempDf['context_timestamp'] - tempDf['last_click_time']
    tempDf['ui_last_click_timedelta'] = tempDf['ui_last_click_timedelta'].dt.seconds
    tempDf['ui_last_click_timedelta'].fillna(999999, inplace=True)
    hourClickList = []
    hourClickTemp = {}
    for same, dt, click in tempDf[['last_user_item','context_timestamp','click']].values:
        if same:
            [hourClickTemp.pop(k) for k in list(hourClickTemp) if k<dt-timedelta(hours=1)]
            hourClickList.append(np.sum(list(hourClickTemp.values())))
            hourClickTemp[dt] = click
        else:
            hourClickList.append(0)
            hourClickTemp = {dt:click}
    tempDf['ui_lasthour_click'] = hourClickList
    df = df.merge(tempDf[['user_item','context_timestamp','ui_last_click_timedelta','ui_lasthour_click']], how='left', on=['user_item','context_timestamp'])

    tempDf = pd.pivot_table(originDf, index=['user_item','date'], values=['is_trade'], aggfunc=[len,np.sum])
    tempDf.columns = ['ui_lastdate_click', 'ui_lastdate_trade']
    tempDf.reset_index(inplace=True)
    tempDf['date'] = tempDf['date'] + timedelta(days=1)
    df = df.merge(tempDf[['user_item','date','ui_lastdate_click', 'ui_lastdate_trade']], how='left', on=['user_item','date'])
    df.fillna({k:0 for k in ['ui_lastdate_click', 'ui_lastdate_trade']}, inplace=True)
    return df

# 统计用户各价格段的浏览特征
def addUserPriceFea(df, hisDf=None):
    if isinstance(hisDf, pd.DataFrame):
        df['is_trade'] = 0
        tempDf = pd.concat([hisDf,df], ignore_index=True)
    else:
        tempDf = df.copy()
    temp = tempDf.groupby(['user_id', 'item_price_level']).size(
        ).reset_index().rename(columns={0: 'user_price_times'})
    df = df.merge(temp, how='left', on=['user_id', 'item_price_level'])
    temp = tempDf.groupby(['user_id']).size(
        ).reset_index().rename(columns={0: 'user_times'})
    df = df.merge(temp, how='left', on=['user_id'])
    df['user_price_ratio'] = df.user_price_times / df.user_times
    return df

# 添加广告商品与查询词的相关性特征
def addContextRelatedFea(df):
    cate1List = []
    cate2List = []
    propList = []
    for cate1,cate2,prop,pred in df[['item_category1','item_category2','item_property_list','predict_category_property']].values:
        if isinstance(pred, float):
            cate1List.append(0)
            cate2List.append(0)
            propList.append(0)
        else:
            hasCate1 = cate1 in pred.keys()
            hasCate2 = cate2 in pred.keys()
            cate1List.append(1 if hasCate1 else 0)
            cate2List.append(1 if hasCate2 else 0)
            if hasCate2:
                propList.append(len(set(prop).intersection(set(pred[cate2]))))
            elif hasCate1:
                propList.append(len(set(prop).intersection(set(pred[cate1]))))
            else:
                propList.append(0)
    df['same_category1'] = cate1List
    df['same_category2'] = cate2List
    df['same_prop_num'] = propList
    return df

class XgbModel:
    def __init__(self, feaNames=None, params={}):
        self.feaNames = feaNames
        self.params = {
            'objective': 'binary:logistic',
            'eval_metric':'logloss',
            'silent': True,
            'eta': 0.1,
            'max_depth': 4,
            'gamma': 0.01,
            'subsample': 0.7,
            'colsample_bytree': 0.75,
            'min_child_weight': 8,
            'max_delta_step': 2,
            # 'alpha':1600
        }
        for k,v in params.items():
            self.params[k] = v
        self.clf = None

    def train(self, X, y, train_size=1, test_size=0.1, verbose=True, num_boost_round=1000, early_stopping_rounds=3):
        if train_size==1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            X_train, y_train = X, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feaNames)
        dval = xgb.DMatrix(X_test, label=y_test, feature_names=self.feaNames)
        watchlist = [(dtrain,'train'),(dval,'val')]
        clf = xgb.train(
            self.params, dtrain, 
            num_boost_round = num_boost_round, 
            evals = watchlist, 
            early_stopping_rounds = early_stopping_rounds,
            verbose_eval=verbose
        )
        self.clf = clf

    def trainCV(self, X, y, nFold=3, verbose=True, num_boost_round=1500, early_stopping_rounds=10):
        dtrain = xgb.DMatrix(X, label=y, feature_names=self.feaNames)
        cvResult = xgb.cv(
            self.params, dtrain, 
            num_boost_round = num_boost_round, 
            nfold = nFold,
            early_stopping_rounds = early_stopping_rounds,
            verbose_eval=verbose
        )
        clf = xgb.train(
            self.params, dtrain, 
            num_boost_round = cvResult.shape[0], 
        )
        self.clf = clf

    def gridSearch(self, X, y, nFold=3, verbose=1, num_boost_round=120):
        paramsGrids = {
            # 'n_estimators': [50+5*i for i in range(0,30)],
            # 'gamma': [0+10*i for i in range(0,10)],
            # 'max_depth': list(range(3,10)),
            'min_child_weight': list(range(1,10)),
            'subsample': [1-0.05*i for i in range(0,8)],
            'colsample_bytree': [1-0.05*i for i in range(0,10)],
            # 'reg_alpha':[1000+100*i for i in range(0,20)]
            'max_delta_step': [0+1*i for i in range(0,8)],
        }
        for k,v in paramsGrids.items():
            gsearch = GridSearchCV(
                estimator = xgb.XGBClassifier(
                    max_depth = self.params['max_depth'], 
                    # gamma = self.params['gamma'],
                    learning_rate = self.params['eta'],
                    max_delta_step = self.params['max_delta_step'],
                    min_child_weight = self.params['min_child_weight'],
                    subsample = self.params['subsample'],
                    colsample_bytree = self.params['colsample_bytree'],
                    silent = self.params['silent'],
                    # reg_alpha = self.params['alpha'],
                    n_estimators = num_boost_round
                ),
                # param_grid = paramsGrids,
                param_grid = {k:v},
                scoring = 'neg_log_loss',
                cv = nFold,
                verbose = verbose,
                n_jobs = 4
            )
            gsearch.fit(X, y)
            print(pd.DataFrame(gsearch.cv_results_))
            print(gsearch.best_params_)
        exit()

    def predict(self, X):
        return self.clf.predict(xgb.DMatrix(X, feature_names=self.feaNames))

    def getFeaScore(self, show=False):
        fscore = self.clf.get_score()
        feaNames = fscore.keys()
        scoreDf = pd.DataFrame(index=feaNames, columns=['importance'])
        for k,v in fscore.items():
            scoreDf.loc[k, 'importance'] = v
        if show:
            print(scoreDf.sort_index(by=['importance'], ascending=False))
        return scoreDf

# 特征方法汇总
def feaFactory(df, hisDf=None):
    startTime = datetime.now()
    df = formatDf(df)
    df = splitMultiFea(df)
    df = combineKey(df)
    df = addTimeFea(df)
    df = addUserFea(df, hisDf)
    df = addItemFea(df, hisDf)
    df = addUserItemFea(df, hisDf)
    # df = addContextRelatedFea(df)
    # df = addUserPriceFea(df, hisDf)
    print('finished feaFactory: ', datetime.now() - startTime)
    return df

# 划分训练集和测试集
def trainTestSplit(df, splitDate=pd.to_datetime('2018-09-23'), trainPeriod=3, testPeriod=1):
    trainDf = df[(df.context_timestamp<splitDate)&(df.context_timestamp>=splitDate-timedelta(days=trainPeriod))]
    testDf = df[(df.context_timestamp>=splitDate)&(df.context_timestamp<splitDate+timedelta(days=testPeriod))]
    return (trainDf, testDf)

# 导出预测结果
def exportResult(df, fileName, header=True, index=False, sep=' '):
    df.to_csv('./%s' % fileName, sep=sep, header=header, index=index)

# 统计预测误差
def countDeltaY(predictSeries, labelSeries, show=True, title='', subplot=None):
    deltaSeries = predictSeries - labelSeries
    if subplot!=None:
        plt.subplot(subplot[0], subplot[1], subplot[2])
    deltaSeries.plot(style='b-')
    plt.title(title)
    if show:
        plt.show()
    return deltaSeries

# 获取stacking下一层数据集
def getOof(clf, trainX, trainY, testX, nFold=10):
    oofTrain = np.zeros(trainX.shape[0])
    oofTest = np.zeros(testX.shape[0])
    oofTestSkf = np.zeros((testX.shape[0], nFold))
    kf = KFold(n_splits=nFold, shuffle=True)
    for i, (trainIdx, testIdx) in enumerate(kf.split(trainX)):
        kfTrainX = trainX[trainIdx]
        kfTrainY = trainY[trainIdx]
        kfTestX = trainX[testIdx]
        clf.trainCV(kfTrainX, kfTrainY, verbose=False)
        oofTrain[testIdx] = clf.predict(kfTestX)
    clf.trainCV(trainX,trainY, verbose=False)
    oofTest = clf.predict(testX)
    return oofTrain, oofTest


if __name__ == '__main__':
    # 导入数据
    df = importDf('../data/round1_ijcai_18_train_20180301.txt')

    # 特征处理
    df = feaFactory(df)

    # 特征筛选
    # tempCol = ['item_his_click','item_his_trade','item_his_trade_ratio','item_last_click_timedelta','item_lasthour_click']
    # resultDf = getFeaScore(df[tempCol].values, df['is_trade'].values, tempCol)
    # print(resultDf[resultDf.scores>10])
    fea = [
        'item_category1','item_category2','item_city_id', 'item_sales_level','item_collected_level','item_price_level',#'item_id',
        'user_gender_id','user_age_level','user_star_level',#'user_id',
        'hour2','hour','context_page_id',
        'shop_review_positive_rate','shop_score_service','shop_score_delivery','shop_score_description',#'shop_id',
        'user_his_click', 'user_his_trade','user_his_trade_ratio','user_last_click_timedelta','user_lasthour_click',
        'item_his_click','item_his_trade','item_his_trade_ratio',#'item_lasthour_click','item_last_click_timedelta',
        'ui_last_click_timedelta','ui_lasthour_click','ui_lastdate_click','ui_lastdate_trade',
        # 'user_cate_query_day','user_cate_query_day_hour'
    ]
    # exit()

    # 测试模型效果
    costDf = pd.DataFrame(index=fea+['cost'])
    xgbModel = XgbModel(feaNames=fea)
    for dt in pd.date_range(start='2018-09-21', end='2018-09-23', freq='D'):
        trainDf, testDf = trainTestSplit(df, dt, trainPeriod=7)
        # xgbModel.gridSearch(trainDf[fea].values, trainDf['is_trade'].values)
        xgbModel.trainCV(trainDf[fea].values, trainDf['is_trade'].values)
        testDf.loc[:,'predict'] = xgbModel.predict(testDf[fea].values)
        scoreDf = xgbModel.getFeaScore()
        scoreDf.columns = [dt.strftime('%Y-%m-%d')]
        costDf = costDf.merge(scoreDf, how='left', left_index=True, right_index=True)
        cost = metrics.log_loss(testDf['is_trade'].values, testDf['predict'].values)
        costDf.loc['cost',dt.strftime('%Y-%m-%d')] = cost
    print(costDf)
    exit()

    # 正式模型
    modelName = "xgboost1A"
    xgbModel.trainCV(df[fea].values, df['is_trade'].values)
    xgbModel.getFeaScore(show=True)
    xgbModel.clf.save_model('%s.model'%modelName)

    # 预测集准备
    startTime = datetime.now()
    predictDf = importDf('../data/round1_ijcai_18_test_a_20180301.txt')
    predictDf = feaFactory(predictDf, hisDf=df)
    # 填补缺失字段
    print('缺失字段：', [x for x in fea if x not in predictDf.columns])
    for x in [x for x in fea if x not in predictDf.columns]:
        predictDf[x] = 0
    print("预测集：\n",predictDf.head())
    print(predictDf[fea].info())

    # 开始预测
    predictDf.loc[:,'predicted_score'] = xgbModel.predict(predictDf[fea].values)
    print("预测结果：\n",predictDf[['instance_id','predicted_score']].head())
    # exportResult(predictDf, "%s_predict.csv" % modelName)
    exportResult(predictDf[['instance_id','predicted_score']], "%s.txt" % modelName)
