import os
import pandas as pd
import time
# import datatime
import numpy as np
from pandas import Series, DataFrame
# from geographiclib.geodesic import Geodesic
from math import radians, cos, sin, asin, sqrt
from datetime import datetime
from tqdm import tqdm

def grid_fill_s(lon,lat,n,maxlon=110.6100,minlon=110.2200,maxlat=20.0700,minlat=19.6900):
    xlen = maxlon - minlon  #cols
    ylen = maxlat - minlat  #rows
    xfac,yfac = xlen/n, ylen/n
    grid = np.zeros((1,n,n))
    assert len(lon) == len(lat)
    for i in range(len(lon)):
        posx = int((lon[i]-minlon) // xfac)
        posy = int((lat[i]-minlat) // yfac)
        # import ipdb; ipdb.set_trace()
        if posx < n and posy < n:
            grid[0][posy][posx] += 1.0
        elif posx == n and posy < n:
            grid[0][posy][n-1] += 1.0
        elif posx < n and posy == n:
            grid[0][n-1][posx] += 1.0
        else:
            print(lon[i],lat[i])
            print(posx,posy)
            print(xfac,yfac)
            print('---------------')
    return grid



def grid_fill(lon,lat,nx,ny,maxlon,minlon,maxlat,minlat):
    xlen = maxlon - minlon  #cols
    ylen = maxlat - minlat  #rows
    xfac,yfac = xlen/nx, ylen/ny
    grid = np.zeros((1,ny,nx))
    assert len(lon) == len(lat)
    for i in range(len(lon)):
        posx = int((lon[i]-minlon) // xfac)
        posy = int((lat[i]-minlat) // yfac)
        # import ipdb; ipdb.set_trace()
        if posx < nx and posy < ny:
            grid[0][posy][posx] += 1.0
        elif posx == nx and posy < ny:
            grid[0][posy][nx-1] += 1.0
        elif posx < nx and posy == ny:
            grid[0][ny-1][posx] += 1.0
        else:
            print(lon[i],lat[i])
            print(posx,posy)
            print(xfac,yfac)
            print('---------------')
    return grid


def map_fill(a,tm_index,save_npy_path,nx,ny,maxlon,minlon,maxlat,minlat):
    '''
    a:data ; tm_index: time interval
    '''
    tmp = a[(a['d_time_n'] >= tm_index[0]) & (a['d_time_n'] < tm_index[1])]
    lonlist,latlist = tmp['s_lng'].tolist(),tmp['s_lat'].tolist()
    nres= grid_fill(lonlist,latlist,nx,ny,maxlon,minlon,maxlat,minlat)

    for i in tqdm(range(1,len(tm_index)-1)):
        tmp = a[(a['d_time_n'] >= tm_index[i]) & (a['d_time_n'] < tm_index[i+1])] #选择出的在这个时间段的data
        lonlist,latlist = tmp['s_lng'].tolist(),tmp['s_lat'].tolist()
        res= grid_fill(lonlist,latlist,nx,ny,maxlon,minlon,maxlat,minlat)
        nres = np.row_stack((nres,res))

    np.save(save_npy_path, nres)
    return nres


def time_index(freq_str):#生成时间间隔的index
    # mindtime = min(a['d_time'])
    # maxdtime = max(a['d_time'])
    # print(mindtime,maxdtime)
    mindtime = '2017-05-01 00:00:00'
    maxdtime = '2017-11-01 00:00:00'
    tm_index = pd.date_range(start=mindtime, end=maxdtime,freq=freq_str) # 生成时间间隔的index
    return tm_index

# tmp = time_index()
# print(len(tmp))
# file(hpath,n)


def openfile(hpath,n): #pandas打开第n个文件
    with open(hpath,'r') as f:
        data = pd.read_csv(f,sep='\t')
        # data = pd.read_csv(f,names=['id','time_up','time_down','lon','lat','lon1','lat1'])
    cols = ['dwv_order_make_haikou_'+n+'.order_id','dwv_order_make_haikou_'+n+'.departure_time',
            'dwv_order_make_haikou_'+n+'.starting_lng','dwv_order_make_haikou_'+n+'.starting_lat']
    a = data.loc[:,cols]
    a.columns = ['order_id','d_time','s_lng','s_lat']

    a['d_time_n'] =  pd.to_datetime(a['d_time']) #把time改为时间格式，新增一列
    # b = a.sort_values('d_time_n')
    print(n, a.shape)
    return a

def all_file(): #pandas合并所有文件数据
    nlist = list(map(str,range(1,9)))
    # print(nlist)
    hpath = '/data3/liumengmeng/data_haikou/dwv_order_make_haikou_' + '1' + '.txt'
    data = openfile(hpath,'1')
    for i in range(1,len(nlist)):
        hpath = '/data3/liumengmeng/data_haikou/dwv_order_make_haikou_' + nlist[i] + '.txt'
        new_data = openfile(hpath, nlist[i])
        data = pd.concat([data,new_data])
        # print(data.shape)
    data_s = data[(data['d_time_n'] >= '2017-05-01 00:00:00') & (data['d_time_n'] <= '2017-10-31 23:59:59')]
    maxlon=110.5200
    minlon=110.1600
    maxlat=20.0850
    minlat=19.9000
    data_lonlat_s = data_s[(data_s['s_lng'] >= minlon) & (data_s['s_lng'] <= maxlon) & (data_s['s_lat']>=minlat) & (data_s['s_lat']<=maxlat)]
    print(data.shape)
    print(data_s.shape)
    print(data_lonlat_s.shape)
    return data_lonlat_s
#-------------------------------------------------------------

# data = all_file()

# #---------可视化经纬度-------
# ax = data.plot.scatter(x='s_lng', y='s_lat',s=1)
# fig=ax.get_figure()
# fig.savefig('./test/name5.png')

# hpath = '/data3/liumengmeng/data_haikou/dwv_order_make_haikou_' + '1' + '.txt'
# df = openfile(hpath,'1')
# data = df[(df['d_time_n'] >= '2017-05-01 00:00:00') & (df['d_time_n'] <= '2017-10-31 23:59:59')]

# # ----用pandas数据填充flow map----
# freq_str = '30min'
# nx,ny = 48,24
# tm_index = time_index(freq_str)
# save_npy_path = './itv_' + freq_str + '_' + str(nx) + '-' + str(ny) + '.npy' #设置保存flow map（array）的路径
# res = map_fill(data,tm_index,save_npy_path,nx,ny)
# print(res.shape)


#=========================================================================================

# hpath = '/data3/liumengmeng/data_haikou/dwv_order_make_haikou_' + '1' + '.txt'
# data = openfile(hpath,'1')

# # data.to_csv('./data.csv',index=None)
# data = pd.read_csv('./data.csv')
# print(data.head())
# print(data.shape)


# data[(data['d_time_n'] >= '2017-05-01 00:00:00') & (data['d_time_n'] <= '2017-05-01 23:59:59')].to_csv('./0501.csv')
# print(tmp[(tmp['d_time'] < '2017-05-01 00:00:00')])
# print(tmp[(tmp['d_time'] > '2017-11-01 00:00:00')])
# maxlon,minlon = max(tmp['s_lng']),min(tmp['s_lng'])
# maxlat,minlat = max(tmp['s_lat']),min(tmp['s_lat'])
# print(maxlon,minlon,maxlat,minlat)
# print(tmp[(tmp['d_time'] > '2017-05-01 00:00:00') & (tmp['d_time'] < '2017-05-01 23:59:59')])



# hpath1 = '/data3/liumengmeng/data_haikou/dwv_order_make_haikou_' + '1' + '.txt'
# final1 = file(hpath1,'1')
# hpath2 = '/data3/liumengmeng/data_haikou/dwv_order_make_haikou_' + '2' + '.txt'
# final2 = file(hpath2,'2')
# tmp = pd.concat([final1,final2])
# print(tmp[(tmp['d_time'] > '2017-05-19 18:07:09') & (tmp['d_time'] < '2017-05-19 19:07:09')])
# print(tmp.loc[1300000])







#------------------------------------------------------------------------------------------
def calc_dis(lat1_in_degrees,long1_in_degrees,lat2_in_degrees,long2_in_degrees):
    coords_1 = (float(lat1_in_degrees), float(long1_in_degrees))
    coords_2 = (float(lat2_in_degrees), float(long2_in_degrees))
    # from geographiclib.geodesic import Geodesic
    import geopy
    from geopy.distance import geodesic
    return geopy.distance.geodesic(coords_1, coords_2).m
    #以M为单位显示
    # print(geopy.distance.geodesic(coords_1, coords_2).m)
    #以KM为单位显示
    # print(geopy.distance.geodesic(coords_1, coords_2).km)

def geodistance(lat1,lng1,lat2,lng2):
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2 
    dis=2*asin(sqrt(a))*6371*1000
    return dis

# maxlon=110.6100
# minlon=110.2200
# maxlat=20.0700
# minlat=19.6900

# print(geodistance(maxlat,minlon,maxlat,maxlon))
# print(calc_dis(maxlat,minlon,minlat,minlon))