#-------------现在使用的flow图，inflow和outflow一起concat，变为2channel的图--------------------

import os
import pandas as pd
import time
import numpy as np
from pandas import Series, DataFrame
from tqdm import tqdm

from gps_process import *


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

def map_fill_s(a,tm_index,nx,ny,maxlon,minlon,maxlat,minlat):
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

def all_file_gps_nosave_v1(t_index, day_start,day_end_1,dataset,month,nx,ny,maxlon,minlon,maxlat,minlat): #pandas合并所有文件数据
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

        res_in,res_out = map_fill_s(ndata,d_index,nx,ny,maxlon,minlon,maxlat,minlat)

        d_map_num = len(d_index)-1
        n_d_grid_in[i*d_map_num:(i+1)*d_map_num] = res_in
        n_d_grid_out[i*d_map_num:(i+1)*d_map_num] = res_out

    # np.save(save_npy_path[0],n_d_grid_in)
    # np.save(save_npy_path[1],n_d_grid_out)
    return n_d_grid_in,n_d_grid_out



def all_file_gps_nosave(t_index, day_start,day_end_1,dataset,month,nxc,nyc,nxf,nyf,maxlon,minlon,maxlat,minlat): #pandas合并所有文件数据
    nlist = daylist(day_start,day_end_1) ##TODO!!!!!!
    # t_index = time_index('2016-11-01 00:00:00','2016-12-01 00:00:00','D')
    # data =pd.DataFrame(columns=('driver','order','time','lon','lat'))
    daymapnum = len(time_index(t_index[0],t_index[1],'15min')) - 1
    grid_in_c = np.zeros(((len(nlist)*daymapnum,nyc,nxc))) #存储一个月内所有天的corase map
    grid_out_c = np.zeros(((len(nlist)*daymapnum,nyc,nxc))) #存储一个月内所有天的coase map
    
    grid_in_f = np.zeros(((len(nlist)*daymapnum,nyf,nxf))) #存储一个月内所有天的fine map
    grid_out_f = np.zeros(((len(nlist)*daymapnum,nyf,nxf))) #存储一个月内所有天的fine map
    for i in tqdm(range(len(nlist))):
        hpath = '/data3/liumengmeng/'+dataset+'/gps_2016'+str(month) + nlist[i]
        ndata = openfile_gps(hpath)
        tqdm.write(f'{ndata.shape}')
        ndata = select_by_geo(ndata,maxlon,minlon,maxlat,minlat)
        tqdm.write(f'{ndata.shape}')
        d_index = time_index(t_index[i],t_index[i+1],'15min') #把一天以15min分时间戳

        d_map_num = len(d_index)-1
        res_in_c,res_out_c = map_fill_s(ndata,d_index,nxc,nyc,maxlon,minlon,maxlat,minlat)
        grid_in_c[i*d_map_num:(i+1)*d_map_num] = res_in_c
        grid_out_c[i*d_map_num:(i+1)*d_map_num] = res_out_c

        res_in_f,res_out_f = map_fill_s(ndata,d_index,nxf,nyf,maxlon,minlon,maxlat,minlat)
        grid_in_f[i*d_map_num:(i+1)*d_map_num] = res_in_f
        grid_out_f[i*d_map_num:(i+1)*d_map_num] = res_out_f

    # np.save(save_npy_path[0],n_d_grid_in)
    # np.save(save_npy_path[1],n_d_grid_out)
    return grid_in_c, grid_out_c, grid_in_f, grid_out_f


# df10 = pd.read_csv('./cdu_10_order_full.csv')
# df11 = all_file_order()
# data = all_data()




# ----用pandas数据填充flow map-----------------------------------------

# # lonlat = [104.129076,104.043333,30.726490,30.655191] #chengdu
# # lonlat = [109.008833,108.92309,34.278608,34.207309] #xian
# freq_str = '15min'
# nx,ny = 64, 64

# # t_index = time_index('2016-10-01 00:00:00','2016-11-01 00:00:00','D')
# t_index = time_index('2016-11-01 00:00:00','2016-12-01 00:00:00','D')


# save_npy_path0 = '../cdunew/cdu_11_in_' + freq_str + '_' + str(nx) + '-' + str(ny) + '.npy'
# save_npy_path1 = '../cdunew/cdu_11_out_' + freq_str + '_' + str(nx) + '-' + str(ny) + '.npy'
# save_npy_path = [save_npy_path0,save_npy_path1]
# nn = all_file_gps(t_index,1,31,'data_cdu',11,save_npy_path,nx,ny,*lonlat)




# # ----------可视化------------------
lonlat = [109.008833,108.92309,34.278608,34.207309] #xian
hpath = '/data3/liumengmeng/'+'data_xian/xian'+'/gps_2016'+str(1130)
# lonlat = [104.129076,104.043333,30.726490,30.655191] #chengdu
# hpath = '/data3/liumengmeng/'+'data_cdu_10/chengdu'+'/gps_2016'+str(1001)
ndata = openfile_gps(hpath)
nn = select_by_geo(ndata,*lonlat)

ax = ndata.plot.scatter(x='lon', y='lat',s=0.5)
fig=ax.get_figure()
fig.savefig('./test_xian_1130.png')

bx = nn.plot.scatter(x='lon', y='lat',s=0.5)
fig=bx.get_figure()
fig.savefig('./test_xian_1130_select.png')



# freq_str = '30min'
# nx,ny = 48,48
# tm_index = time_index(freq_str)
# save_npy_path = './itv/cdu_' + freq_str + '_' + str(nx) + '-' + str(ny) + '.npy' #设置保存flow map（array）的路径
# res = map_fill(nn,tm_index,save_npy_path,nx,ny,maxlon=104.2,minlon=103.9,maxlat=30.8,minlat=30.5)
# print(res.shape)