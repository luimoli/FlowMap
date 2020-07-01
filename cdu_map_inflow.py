#-------------现在使用的flow图，inflow和outflow一起concat，变为2channel的图--------------------

import os
import pandas as pd
import time
import numpy as np
from pandas import Series, DataFrame
from tqdm import tqdm
from map_hk import grid_fill,grid_fill_s

# path = '/data3/liumengmeng/data_cdu/gps_20161101'

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


def grid_pos_add(posx,posy,grid,nx,ny):
    if posx < nx and posy < ny:
        grid[0][posy][posx] += 1.0
    elif posx == nx and posy < ny:
        grid[0][posy][nx-1] += 1.0
    elif posx < nx and posy == ny:
        grid[0][ny-1][posx] += 1.0
    else:
        print(f'wrong:{posx},{posy}')
    return grid

def grid_pos_minus(posx,posy,grid,nx,ny):
    if posx < nx and posy < ny:
        grid[0][posy][posx] -= 1.0
    elif posx == nx and posy < ny:
        grid[0][posy][nx-1] -= 1.0
    elif posx < nx and posy == ny:
        grid[0][ny-1][posx] -= 1.0
    else:
        print(f'wrong:{posx},{posy}')
    return grid


def grid_fill_s(gridin,gridout,lon,lat,nx,ny,maxlon,minlon,maxlat,minlat):
    xlen = maxlon - minlon  #cols
    ylen = maxlat - minlat  #rows
    xfac,yfac = xlen/nx, ylen/ny
    # grid = np.zeros((1,ny,nx))
    assert len(lon) == len(lat)
    posx = [int((i-minlon) // xfac) for i in lon]
    posy = [int((i-minlat) // yfac) for i in lat]
    dic = {}
    for i in range(len(lon)-1):
        if not (posx[i+1],posy[i+1]) == (posx[i],posy[i]):
            gridin = grid_pos_add(posx[i+1],posy[i+1],gridin,nx,ny)
            gridout = grid_pos_minus(posx[i],posy[i],gridout,nx,ny)

        # if (posx[i],posy[i]) in dic:pass
        # else:
        #     dic[(posx[i],posy[i])] = 1
        #     grid = grid_pos(posx[i],posy[i],grid,nx,ny)

    return gridin,gridout



#-------------------------------------------------------------------------------
def map_fill(a,tm_index,save_npy_path,nx,ny,maxlon,minlon,maxlat,minlat):
    final = np.zeros((len(tm_index)-1,ny,nx))

    for i in range(len(tm_index)-1):
        tmp = a[(a['d_time_n'] >= tm_index[i]) & (a['d_time_n'] < tm_index[i+1])] #选择出的在这个时间段的data
        lonlist,latlist = tmp['lon'].tolist(),tmp['lat'].tolist()
        res= grid_fill(lonlist,latlist,nx,ny,maxlon,minlon,maxlat,minlat)
        # nres = np.row_stack((nres,res))
        final[i] = res[0]
    # np.save(save_npy_path, nres)
    return final

def map_fill_s(a,tm_index,save_npy_path,nx,ny,maxlon,minlon,maxlat,minlat):
    final_in = np.zeros((len(tm_index)-1,ny,nx)) #(interval,48,48)
    final_out = np.zeros((len(tm_index)-1,ny,nx))
    for i in (range(len(tm_index)-1)):
        df = a[(a['d_time_n'] >= tm_index[i]) & (a['d_time_n'] < tm_index[i+1])] #选择出的在这个时间段的data
        # gp = df.groupby('order')
        gp = df.groupby('driver')
        gridin = np.zeros((1,ny,nx))
        gridout = np.zeros((1,ny,nx))
        for name,tmp in gp:
            lonlist,latlist = tmp['lon'].tolist(),tmp['lat'].tolist()
            gridin,gridout = grid_fill_s(gridin,gridout,lonlist,latlist,nx,ny,maxlon,minlon,maxlat,minlat)
        final_in[i] = gridin[0]
        final_out[i] = gridout[0]
    # np.save(save_npy_path, nres)
    return final_in,final_out

def map_avg(arr):
    a,b,c = arr.shape
    grid = np.zeros((b,c))
    for i in range(b):
        for j in range(c):
            grid[i][j]=np.mean(arr[:,i,j])
    # res = np.expand_dims(grid,0)
    return grid

def min_map_none0(y):
    tmp1,tmp2 = y[0][y[0]>0].shape[0],y[0][y[0]>=1].shape[0]
    for i in range(1,len(y)):
        sha1 = y[i][y[i]>0].shape[0]
        sha2 = y[i][y[i]>=1].shape[0]
        if sha1 < tmp1:tmp1 = sha1
        if sha2 < tmp2:tmp2 = sha2
    return tmp1,tmp2

def max_map_none0(y):
    tmp = 0
    for i in range(len(y)):
    	sha = y[i][y[i]>0].shape[0]
    	if sha > tmp:
            tmp = sha
    return tmp


#-----------------------------------------------------------------------------

def all_file_gps(t_index, day_start,day_end_1,dataset,month,save_npy_path,nx,ny,maxlon,minlon,maxlat,minlat): #pandas合并所有文件数据
    nlist = daylist(day_start,day_end_1) ##TODO!!!!!!
    # t_index = time_index('2016-11-01 00:00:00','2016-12-01 00:00:00','D')
    # data =pd.DataFrame(columns=('driver','order','time','lon','lat'))
    daymapnum = len(time_index(t_index[0],t_index[1],'15min')) - 1
    n_d_grid_in = np.zeros(((len(nlist)*daymapnum,ny,nx))) #存储一个月内所有天的map
    n_d_grid_out = np.zeros(((len(nlist)*daymapnum,ny,nx))) #存储一个月内所有天的map
    for i in tqdm(range(len(nlist))):
        hpath = '/data3/liumengmeng/'+dataset+'/gps_2016'+str(month) + nlist[i]
        ndata = openfile_gps(hpath)
        tqdm.write(f'{ndata.shape}')
        ndata = select_by_geo(ndata,maxlon,minlon,maxlat,minlat)
        tqdm.write(f'{ndata.shape}')
        d_index = time_index(t_index[i],t_index[i+1],'15min') #把一天以15min分时间戳

        res_in,res_out = map_fill_s(ndata,d_index,save_npy_path,nx,ny,maxlon,minlon,maxlat,minlat)

        d_map_num = len(d_index)-1
        # d_grid = np.zeros((d_map_num,ny,nx)) ########nx ny position!TODO 存储一天的96张map
        # for j in range(d_map_num):
        #     m_index = time_index(d_index[j],d_index[j+1],'15s') #把15min以4s分时间戳 TODO
        #     # res = map_fill(ndata,m_index,save_npy_path,nx,ny,maxlon,minlon,maxlat,minlat) #15min内的所有map
        #     res = map_fill_s(ndata,m_index,save_npy_path,nx,ny,maxlon,minlon,maxlat,minlat)
        #     d_grid[j] = map_avg(res) #取平均值作为这15min的一个map
        
        # tqdm.write(f'{str(month) + nlist[i]}:{res.shape}')
        # tqdm.write(f'min:{min_map_none0(res)},max:{max_map_none0(res)}')

        # np.save('./xian_in/'+str(month) + '_' + nlist[i]+'_64.npy',res_in) #TODO
        # np.save('./xian_out/'+str(month) + '_' + nlist[i]+'_64.npy',res_out) #TODO
        # import ipdb; ipdb.set_trace()
        
        n_d_grid_in[i*d_map_num:(i+1)*d_map_num] = res_in
        n_d_grid_out[i*d_map_num:(i+1)*d_map_num] = res_out

    np.save(save_npy_path[0],n_d_grid_in)
    np.save(save_npy_path[1],n_d_grid_out)
    return n_d_grid_in,n_d_grid_out

    # # orderframe = group(ndata)#######################!!!!!!!!!!!!!!!!
    # data = pd.concat([data,ndata],ignore_index=True)
    # data.to_csv('./cdu_10_order_full.csv',index=False)
    # return data


def all_file_order(day_start,day_end_1,dataset,month): #pandas合并所有文件数据
    nlist = daylist(day_start,day_end_1)
    data =pd.DataFrame(columns=('order','time','lon','lat'))
    for i in tqdm(range(len(nlist))):
        hpath = '/data3/liumengmeng/'+dataset+'/order_2016'+str(month) + nlist[i]
        new_data = openfile_order(hpath)
        data = pd.concat([data,new_data],ignore_index=True)
        # print(data.shape) #(209423, 4)
    print(data.shape)
    return data

def all_data(df10,df11):
    data = pd.concat([df10,df11],ignore_index=True)
    maxlon=104.2
    minlon=103.9
    maxlat=30.8
    minlat=30.5
    df = data[(data['lon'] >= minlon) & (data['lon'] <= maxlon) & (data['lat']>=minlat) & (data['lat']<=maxlat)]
    df['d_time_n'] = pd.to_datetime(df['time'],unit='s',origin=pd.Timestamp('1970-01-01 08:00:00'))
    print(df.shape)
    return df



# df10 = pd.read_csv('./cdu_10_order_full.csv')
# df11 = all_file_order()
# data = all_data()


# ----用pandas数据填充flow map-----------------------------------------
# maxlon=104.13
# minlon=103.99
# maxlat=30.72
# minlat=30.61

# lonlat = [104.129076,104.043333,30.726490,30.655191] #chengdu
lonlat = [109.008833,108.92309,34.278608,34.207309] #xian
freq_str = '15min'
nx,ny = 64,64

t_index = time_index('2016-10-01 00:00:00','2016-11-01 00:00:00','D')
# save_npy_path = './cdu/cdu_10_' + freq_str + '_' + str(nx) + '-' + str(ny) + '.npy' #设置保存flow map（array）的路径
# nn = all_file_gps(t_index,1,32,'data_cdu_10/chengdu',10,save_npy_path,nx,ny,*lonlat)

# t_index = time_index('2016-11-01 00:00:00','2016-12-01 00:00:00','D')

save_npy_path0 = './xian/xian_11_in_' + freq_str + '_' + str(nx) + '-' + str(ny) + '.npy'
save_npy_path1 = './xian/xian_11_out_' + freq_str + '_' + str(nx) + '-' + str(ny) + '.npy'
save_npy_path = [save_npy_path0,save_npy_path1]
nn = all_file_gps(t_index,1,32,'data_xian_10/xian',10,save_npy_path,nx,ny,*lonlat)


# nn = select_by_geo_time(nn,*lonlat)

# # ----------可视化------------------
# ax = nn.plot.scatter(x='lon', y='lat',s=0.5)
# fig=ax.get_figure()
# fig.savefig('./test/test_11_1.png')

# freq_str = '30min'
# nx,ny = 48,48
# tm_index = time_index(freq_str)
# save_npy_path = './itv/cdu_' + freq_str + '_' + str(nx) + '-' + str(ny) + '.npy' #设置保存flow map（array）的路径
# res = map_fill(nn,tm_index,save_npy_path,nx,ny,maxlon=104.2,minlon=103.9,maxlat=30.8,minlat=30.5)
# print(res.shape)