# -*- coding: utf-8 -*-
from __future__ import unicode_literals

SEED = 123456
import gc
import sys
import os
import csv

import numpy as np
np.random.seed(SEED)
import pandas as pd
import itertools
import operator
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import random
random.seed(1)
from datetime import datetime

import codecs
import json

import xgboost as xgb
from sklearn.cross_validation import train_test_split  
from sklearn.cross_validation import KFold

from sklearn.metrics import mean_absolute_error,log_loss

directory = './'
import scipy as sp

def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()
    
def save(ids,score,prediction):
    print("Writing results")
    result = pd.DataFrame({'id':ids,'loss':prediction})
    
    print("score: ",score)

    print (result.head())
    now = datetime.now()

    name=str(now.strftime("%d-%m-%y_%H-%M"))+"_"+str(score)
    print("name: ",name)
    
    os.system("cp "+sys.argv[0]+" ./ext/scr"+name+".py")
    result.to_csv('./ext/sub'+name+'.csv', index=False)

def score(params):
    print ("Training with params : ")
    print (params)
    num_round = int(params['n_estimators'])
    params['max_depth']=int(params['max_depth'])
    del params['n_estimators']
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)
    # watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    model = xgb.train(params, dtrain, num_round)
    
    predictions = model.predict(dvalid)
    
    score_tmp=pd.DataFrame(predictions,columns=['score'])

    score_tmp[score_tmp['score']>=0.9999] = 0.9999999
    score_tmp[score_tmp['score']<=0] = 0.0000001
    predictions=score_tmp[['score']].values
    score = log_loss(y_test, predictions)
    print ("\tScore {0}\n\n".format(score))
    params['n_estimators'] = num_round
    params['score'] = score
    
    optim.append(params)
    return {'loss': score, 'status': STATUS_OK}

def sortByEl(d):
     return d['score']
 
def optimize(trials):
    space = {
             'n_estimators' : hp.quniform('n_estimators', 300, 500, 1),
             'eta' : hp.quniform('eta', 0.025, 0.1, 0.025),
             'max_depth' : hp.quniform('max_depth', 3, 8, 1),
             'min_child_weight' : hp.quniform('min_child_weight', 1, 6, 1),
             'subsample' : hp.quniform('subsample', 0.5, 1, 0.05),
             'gamma' : hp.quniform('gamma', 0.3, 1, 0.05),
             'colsample_bytree' : hp.quniform('colsample_bytree', 0.3, 1, 0.05),
             'eval_metric': 'logloss',
             'objective': 'reg:logistic',
             'nthread' : -1,
             'silent' : 1
             }

    best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=400)

    print (best)
    
if __name__ == "__main__":
    optim = []
    
    log=['attemptsOnTheHighestLevel','totalNumOfAttempts']
    log1 = ['maxPlayerLevel' ,'totalScore','totalBonusScore','totalStarsCount']
    COMB_FEATURE = [u'maxPlayerLevel', u'numberOfAttemptedLevels',
       u'totalNumOfAttempts',
       u'averageNumOfTurnsPerCompletedLevel', u'doReturnOnLowerLevels',
       u'numberOfBoostersUsed', u'fractionOfUsefullBoosters', u'numberOfDaysActuallyPlayed']
       
    train_X = pd.read_csv(directory + 'x_train.csv',header=0,sep=';')
    train_y = pd.read_csv(directory + 'y_train.csv',header=-1,sep=';')
    train_test = train_X
#    test_X =  pd.read_csv(directory + 'x_test.csv',header=0,sep=';')
#    train_test = pd.concat((train_X, test_X)).reset_index(drop=True)
    
#    combine = []
#    for name in COMB_FEATURE:
#        line = train_test[name]
#        combine.append(name)
#        #print name
#        for k in range(int (line.max())):
#            name_k=name+'_'+str(k)
#            train_test[name_k]= (line == k )
#            combine.append(name_k)
#            
#    for comb in itertools.combinations(COMB_FEATURE, 2):
#        feat1 = 'ab_'+comb[0] + "_" + comb[1]
#        feat2 = 'ba_'+comb[1] + "_" + comb[0]
#        
#        line1 = train_test[comb[0]]*train_test[comb[1]].max() + train_test[comb[1]]
#        line2 = train_test[comb[1]]*train_test[comb[0]].max() + train_test[comb[0]]
#        
#        train_test[feat1] = line1
#        train_test[feat2] = line2
#        #print('Combining Columns:', feat)
#
#    #print train_test.describe(),train_test.info()
#    for k in log:
#        train_test[k+'_log']=np.log(train_test[k])
#    for k in log1:
#        train_test[k+'_log1']=np.log(train_test[k]+1)
#    train_X =  train_test.iloc[0:train_X.shape[0],:]   
#    test_X  =  train_test.iloc[test_X.shape[0]:,:]   
#    
#       
#    xgb_params = {
#            'seed': 0,
#            'colsample_bytree': 0.9,
#            'silent': 1,
#            'subsample': 0.9,
#            'learning_rate': 0.075,
#            'objective': 'reg:linear',
#            'max_depth': 8,
#            'num_parallel_tree': 1,
#            'min_child_weight': 1,
#            'eval_metric': 'logloss',
#            'nrounds': 50,
#            'verbose': True,
#        }
#    print train_test.columns.shape
#    ceate_feature_map(list(train_test.columns))
#    dtrain = xgb.DMatrix(train_X, label=train_y)
#    temp_model = xgb.train(xgb_params, dtrain, 100, verbose_eval=10)
#    importance = temp_model.get_fscore(fmap='xgb.fmap')
##    print 111,len(importance)
##    exit(1)
##    
#    importance = sorted(importance.items(), key=operator.itemgetter(1))
##    print 222,importance
##    
#    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
#    df['fscore'] = df['fscore'] / df['fscore'].sum()
#    #print df.values
#    maxFuture=-100
#    features=df[['feature']].values[maxFuture:].ravel()
#    #print featuresDrop.shape
#    #train_test.drop(featuresDrop,axis=1,inplace = True)   
#    train_test = train_test[features]
    
#    train_X =  train_test.iloc[0:train_X.shape[0],:]   
#    test_X  =  train_test.iloc[test_X.shape[0]:,:]   
       
#    print train_X.describe()
    #exit()        
#    test_X = test_X.drop(test_X.columns[0],axis=1)
#    test_X = test_X.set_index(test_X.columns[0])    
    #print test_X.head()
        
#    print "train shape",(train_X.shape),(train_y.shape)
#    print "test shape",(test_X.shape)
    print("Building model..")
    
    X_train, X_test, y_train, y_test = train_test_split(
    train_X, train_y, test_size=0.1, random_state=SEED)
    
    trials = Trials()

    optimize(trials)
    
    pd.DataFrame(optim).to_csv('clear.csv')
    print (sorted(optim, key=sortByEl, reverse=True)[:-10])
