import os
import pandas as pd
import time
import numpy as np
from pandas import Series, DataFrame
from tqdm import tqdm

from cdu_map_inflow import all_file_gps,all_file_gps_nosave,time_index
from split_datasets import split_dataset

freq_str = '15min'
# nx,ny = 64, 64
t_index_10 = time_index('2016-10-01 00:00:00','2016-11-01 00:00:00','D')
t_index_11 = time_index('2016-11-01 00:00:00','2016-12-01 00:00:00','D')


def concat_2channel(inflowmap,outflowmap,nx,ny):
    '''
    concat channel:
    例如:把[5xxx,16,16]的inflowmap和[5xxx,16,16]的outflowmap concat起来生成[5xxx,2,16,16]
    '''
    re1 = np.expand_dims(inflowmap,1)
    re2 = np.expand_dims(outflowmap,1)
    assert len(re1) == len(re2)
    print(re1.shape,re2.shape) 
    res = np.zeros((len(re1),2,ny,nx)) #TODO
    for i in range(len(re1)):
        res[i] = np.row_stack((re1[i],re2[i]))
    print(res.shape)
    return res

def gene_by_nxny(data_10_folder,data_11_folder,xycf,lonlat):
    '''
    分别生成10月和11月的inflowmap和outflowmap，然后把10月和11月的数据连接起来，然后concat channel得到最终的map
    '''
    in10_c,out10_c,in10_f,out10_f = all_file_gps_nosave(t_index_10,1,32,data_10_folder,10,*xycf,*lonlat)
    in11_c,out11_c,in11_f,out11_f = all_file_gps_nosave(t_index_11,1,31,data_11_folder,11,*xycf,*lonlat)
    in_c, out_c = np.row_stack((in10_c,in11_c)),np.row_stack((out10_c,out11_c))
    in_f, out_f = np.row_stack((in10_f,in11_f)),np.row_stack((out10_f,out11_f))
    map_c = concat_2channel(in_c,out_c, xycf[0],xycf[1])  #corse map [5xxx,2,16,16]  
    map_f = concat_2channel(in_f,out_f, xycf[2],xycf[3])  #high map [5xxx,2,64,64]
    return map_c,map_f


def generate(city, npy_path, data_path):
    '''
    city: 'xian' or 'cdu' ---确定gps数据位置以及select的经纬度范围
    npy_path: ---保存corse map和high map的npy文件的位置
    data_path: ---把map划分为数据集后存储的位置
    '''
    if city == 'cdu':
        data_10_folder = 'data_cdu_10/chengdu'
        data_11_folder = 'data_cdu'
        lonlat = [104.129076,104.043333,30.726490,30.655191] #chengdu
    elif city == 'xian':
        data_10_folder = 'data_xian_10/xian'
        data_11_folder = 'data_xian/xian'
        lonlat = [109.008833,108.92309,34.278608,34.207309] #xian
    else:
        print('wrong city choice!')

    xycf = [16, 16, 64, 64]
    map_16, map_64 = gene_by_nxny(data_10_folder,data_11_folder,xycf,lonlat)

    np.save(f'{npy_path}{city}_16-16.npy',map_16)
    np.save(f'{npy_path}{city}_64-64.npy',map_64)

    namelist = ['train','valid','test']
    fullpath = [os.path.join(data_path,i) for i in namelist]
    for i in fullpath:os.makedirs(i)
    split_dataset(map_16,map_64,*fullpath)





data_path ='/data3/liumengmeng/ORI/FM/data/cdu_test_1'
# data_path = '/data3/liumengmeng/UrbanFM/data/xian_15min_2-16-16_2-64-64'
npy_path = '/data3/liumengmeng/FlowMap/data_test/'
city = 'cdu'

generate(city, npy_path, data_path)
