import os
import pandas as pd
import time
import numpy as np

#----------把细粒度图按照factor大小聚合成粗粒度图-----------------------
def coarse(factor,array,save_coarse_path):
    a,b,c = array.shape
    bn,cn = b//factor,c//factor
    narray = np.zeros((a,bn,cn))
    for k in range(a):
        for i in range(0,b,factor):
            for j in range(0,c,factor):
                tmp = array[k][i:i+factor,j:j+factor]
                narray[k][i//factor][j//factor] += tmp.sum()
    np.save(save_coarse_path,narray)
    return narray


freq_str = '15min'


#-------------把10月和11月的数据连接起来-------------------------------
# nx,ny = 16,16

# mon1_path = '../xian/xian_10_in_' + freq_str + '_' + str(nx) + '-' + str(ny) + '.npy'
# mon2_path = '../xian/xian_11_in_' + freq_str + '_' + str(nx) + '-' + str(ny) + '.npy'
# mon1,mon2 = np.load(mon1_path),np.load(mon2_path)
# print(mon1.shape,mon2.shape)
# res = np.row_stack((mon1,mon2))
# np.save('../xian/xian_in_' + freq_str + '_' + str(nx) + '-' + str(ny) + '.npy',res)
# print(res.shape)

# mon1_path = '../cdunew/cdu_10_out_' + freq_str + '_' + str(nx) + '-' + str(ny) + '.npy'
# mon2_path = '../cdunew/cdu_11_out_' + freq_str + '_' + str(nx) + '-' + str(ny) + '.npy'
# mon1,mon2 = np.load(mon1_path),np.load(mon2_path)
# res = np.row_stack((mon1,mon2))
# np.save('../cdunew/cdu_out_' + freq_str + '_' + str(nx) + '-' + str(ny) + '.npy',res)
# print(mon1.shape,mon2.shape)
# print(res.shape)



#-------------把10月11月连起来的in out数据concat起来生成2 channel 的map-------------------------------
from numpy import set_printoptions
np.set_printoptions(threshold=np.inf)

nx,ny = 64,64

# city = 'xian'
cityfolder = 'cdunew'
city = 'cdu'
res1 = np.load(f'../{cityfolder}/{city}_in_{freq_str}_{str(nx)}-{str(ny)}.npy')
res2 = np.load(f'../{cityfolder}/{city}_out_{freq_str}_{str(nx)}-{str(ny)}.npy')
assert len(res1) == len(res2)
print(res1.shape,res2.shape)  

re1 = np.expand_dims(res1,1)
re2 = np.expand_dims(res2,1)
print(re1.shape,re2.shape) 
res = np.zeros((len(res1),2,ny,nx)) #TODO
for i in range(len(res1)):
	res[i] = np.row_stack((re1[i],re2[i]))
print(res.shape)
# print(res[-1])

np.save('../cdunew/cdu_' + freq_str + '_' + str(nx) + '-' + str(ny) + '.npy',res)





# # 横向concat 错误
# res = np.zeros((len(res1),ny,nx*2)) #TODO
# for i in range(len(res1)):
# 	res[i] = np.column_stack((res1[i],res2[i]))
# print(res.shape)

# lis = []
# for i in range(len(res)):
#     lis.append(res[i][res[i]!=0].shape)

# import ipdb; ipdb.set_trace()
# t1 = res1[44][10][0] + res1[44][10][1] + res1[45][11][0] + res1[45][11][1]
# t2 = res2[44][10][0] + res2[44][10][1] + res2[45][11][0] + res2[45][11][1]
# print(t1+t2)
# t = res2[44][5][0]+res1[44][5][0]
# print(t)
# print(max(lis))
# print(min(lis))
# print(res[3000][res[3000]!=0].shape)

# np.save('./xian/xian_' + freq_str + '_' + str(nx) + '-' + str(ny) + '.npy',res)


#-------------------------------------------------------------------------------------
# city = 'xian'
# # high_path = './itv_' + freq_str + '_' + str(nx) + '-' + str(ny) + '.npy'
# high_path = f'./{city}/{city}_out_{freq_str}_{str(nx)}-{str(ny)}.npy'
# high_path = f'./xian_in/10_01_64.npy'
# array = np.load(high_path)
# print(array.shape)


# factor = 4   #需要生成/factor的大小的粗粒度图
# # save_coarse_path = './itv_' + freq_str + '_' + str(nx) + '-' + str(ny) + '_c' + str(factor) +'.npy' #设置保存路径
# save_coarse_path = f'./{city}/{city}_out_{freq_str}_{str(nx)}-{str(ny)}_c4.npy'
# save_coarse_path = f'./xian_in/10_01_64_c4.npy'
# carray = coarse(factor,array,save_coarse_path)
# print(carray.shape)








# array = np.array([[[5, 10, 15,1,1,1],
# 	        [20, 25, 30,2,2,2],
# 	        [35, 40, 45,3,3,3],
#             [35, 40, 45,4,4,4]]])
# #             [5, 10, 15,1,1,1],
# #             [5, 10, 15,1,1,1]]])

# print(coarse(2,array,'./test/test.npy'))