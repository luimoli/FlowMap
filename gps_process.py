import os
import pandas as pd
import time
import numpy as np
from pandas import Series, DataFrame
from tqdm import tqdm

def openfile_gps(path):
    with open(path,'r') as f:
        df = pd.read_csv(f,names=['driver','order','time','lon','lat'])
        # df = df.drop(['driver'],axis=1)
        df['d_time_n'] = pd.to_datetime(df['time'],unit='s',origin=pd.Timestamp('1970-01-01 08:00:00'))
    return df

def openfile_order(hpath):
    with open(hpath,'r') as f:
        data = pd.read_csv(f,names=['order','time','time_down','lon','lat','lon1','lat1'])
    a = data.drop(['time_down','lon1','lat1'],axis=1)
    # a['d_time_up'] =  pd.to_datetime(a['time_up']) #把time改为时间格式，新增一列
    # b = a.sort_values('d_time_n')
    return a

def group(df):
    grouped = df.groupby('order')
    data =pd.DataFrame(columns=('order','time','lon','lat'))
    for name,group in tqdm(grouped):
        mintime = min(group['time'])
        ndf = group.loc[group['time'] == mintime]
        data = pd.concat([data,ndf],ignore_index=True)
        # import ipdb; ipdb.set_trace()
    return data



def daylist(ns,ne):#ne的值需要多加一天
    nlist = []
    for i in range(ns,ne):
        if i < 10:nlist.append( '0' + str(i))
        else:nlist.append(str(i))
    return nlist

def time_index(mindtime,maxdtime,freq_str):#生成时间间隔的index
    # mindtime = '2016-11-01 00:00:00'
    # maxdtime = '2016-12-01 00:00:00'
    tm_index = pd.date_range(start=mindtime, end=maxdtime,freq=freq_str) # 生成时间间隔的index
    return tm_index
# mindtime = '2016-11-01 00:00:00'
# maxdtime = '2016-11-02 00:00:00'
# tt = time_index(mindtime,maxdtime,'15min')
# print(tt.shape)


def select_by_geo(data,maxlon,minlon,maxlat,minlat):
    # data_s = data[(data['d_time_n'] >= '2017-05-01 00:00:00') & (data['d_time_n'] <= '2017-10-31 23:59:59')]
    data_lonlat_s = data[(data['lon'] >= minlon) & (data['lon'] <= maxlon) & (data['lat']>=minlat) & (data['lat']<=maxlat)]
    # print(data.shape)
    return data_lonlat_s

def unix_to_date(df,old_name,new_name):
    # df['d_time_n'] = pd.to_datetime(df['time'],unit='s',origin=pd.Timestamp('1970-01-01 08:00:00'))
    df[new_name] = pd.to_datetime(df[old_name],unit='s',origin=pd.Timestamp('1970-01-01 08:00:00'))
    return df
