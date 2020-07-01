# import pandas as pd
# with open(path,'r') as f:
#     # a = [line.rstrip() for line in f]
#     data = pd.read_csv(f,names=['id','time_up','time_down','lon','lat','lon1','lat1'])
# a = data.drop(['time_down','lon1','lat1'],axis=1)
# print(a.shape)
# print(a.head(11))
# print(a.loc[(a['id'] == '5feeae0307e15203484b9ffceef89855')])
# a = data.drop(['dwv_order_make_haikou_1.product_id','dwv_order_make_haikou_1.district','dwv_order_make_haikou_1.county','dwv_order_make_haikou_1.combo_type','dwv_order_make_haikou_1.combo_type','dwv_order_make_haikou_1.driver_product_id','dwv_order_make_haikou_1.start_dest_distance','dwv_order_make_haikou_1.pre_total_fee','dwv_order_make_haikou_1.product_1level'],axis=1)
# a = data.drop(['dwv_order_make_haikou_1.product_id','dwv_order_make_haikou_1.city_id','dwv_order_make_haikou_1.district','dwv_order_make_haikou_1.county','dwv_order_make_haikou_1.type','dwv_order_make_haikou_1.combo_type','dwv_order_make_haikou_1.traffic_type','dwv_order_make_haikou_1.passenger_count','dwv_order_make_haikou_1.driver_product_id','dwv_order_make_haikou_1.start_dest_distance','dwv_order_make_haikou_1.arrive_time','dwv_order_make_haikou_1.pre_total_fee','dwv_order_make_haikou_1.normal_time','dwv_order_make_haikou_1.product_1level','dwv_order_make_haikou_1.dest_lng','dwv_order_make_haikou_1.dest_lat','dwv_order_make_haikou_1.year','dwv_order_make_haikou_1.month','dwv_order_make_haikou_1.day'],axis=1)


import numpy as np
a = np.zeros((4,4,4))
b1 = np.ones((2,4,4))
b = np.array([[[5, 10, 15,1],
	        [20, 25, 30,2],
	        [35, 40, 45,3],
            [35, 40, 45,4]],

            [[5, 1, 1,1],
	        [2, 2, 2,2],
	        [9, 4, 4,3],
            [3, 4, 4,4]]])
c = np.array([[5, 1, 1,1],
	        [2, 2, 2,2],
	        [9, 4, 4,3],
            [3, 4, 4,4]])

ar1 = np.array([[1,2,3], [4,5,6],[7,8,9]])
ar2 = np.array([[7,8,9], [11,12,13],[7,8,9]])

res = np.expand_dims(ar1,0)
print(ar1.shape,ar2.shape)
print(res)
print(res.shape)

# a[2:4] = b
# print(c.mean())
# print(b[:,2,0])
# print(np.append(b,a,0).shape)
# print(np.column_stack((b[0],b1[0])))


# import numpy as np
# N = 4
# a = np.random.rand(N,N)
# b = np.zeros((N,N+1))
# b[:,:-1] = a
# print(b)

# print(np.row_stack((a,b)).shape)
# tmp = np.row_stack((b,b1,a))



# df = pd.DataFrame(['1480521598','1480521598','1480521598'], columns=['time'])
# df['time'] = pd.to_numeric(df['time'])
# df['d_time_n'] = pd.to_datetime(df['time'],unit='s',origin=pd.Timestamp('1970-01-01 08:00:00'))
# print(df.head())

# ax = df.plot.scatter(x='a', y='b')
# # ax=df.plot()
# fig=ax.get_figure()
# fig.savefig('./name.png')




# import numpy as np
# from numpy import set_printoptions
# np.set_printoptions(threshold=np.inf)
# x = np.load('/data3/liumengmeng/ORI/FM/data/P1/train/X.npy')
# y = np.load('/data3/liumengmeng/ORI/FM/data/P1/train/Y.npy')
# print(x.shape,y.shape)
# print(x[1000][x[1000]>0].shape)

# print(y[1000][y[1000]>1].shape)
# print(y[1000][y[1000]>0].shape)
# # print(y[1000])

# tmp = y[0][y[0]>0].shape[0]
# for i in range(1,len(y)):
#     sha = y[i][y[i]>0].shape[0]
#     if sha < tmp:
#         tmp = sha
# print(tmp)

# lis = []
# for i in range(len(y)):
# 	sha = y[i][y[i]>0].shape[0]
# 	lis.append(sha)
# lis.sort()
# print(lis)
# print(sum(lis)/len(lis) /128/128)


#------------------about time -----------------------------------------
# import time
# import datetime
# a = time.gmtime(1480521598)
# b = datetime.datetime.fromtimestamp(1480521598)
# time_local = time.localtime(1480521598)
# c = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
# print(a)
# print(b)
# print(c)

from numpy import set_printoptions
np.set_printoptions(threshold=np.inf)

#--------------------test npy -------------------------------------
city = 'cdu'
freq_str = '15min'
# nx,ny = 64,64
high_path = f'./{city}/{city}_in_{freq_str}_{str(64)}-{str(64)}.npy'
high_path1 = f'./{city}/{city}_in_{freq_str}_{str(16)}-{str(16)}.npy'
# high_path2 = f'./{city}/{city}_in_{freq_str}_{str(nx)}-{str(ny)}_c4.npy'
# high_path3 = f'./{city}/{city}_out_{freq_str}_{str(nx)}-{str(ny)}_c4.npy'

# high_path = f'./xian_in/10_01.npy'
# high_path1 = f'./xian_out/10_01.npy'
# high_path2 = f'./xian_in/10_01_64.npy'
# high_path3 = f'./xian_out/10_01_64.npy'

# y = np.load(high_path)
# y1 = np.load(high_path1)

# print(y[1000].mean())
# print(y1[1000].mean())
# s16 = np.add(y,y1)

# y2 = np.load(high_path2)
# y3 = np.load(high_path3)
# sc4 = np.add(y2,y3)


# res = np.subtract(s16,sc4)
# res = np.subtract(y[50],y2[50])
# print(res[100])

# tmp = y[0][y[0]>1].shape[0]
# for i in range(len(y)):
#     sha = y[i][y[i]>1].shape[0]
#     if sha < tmp:
#         tmp = sha
# print(tmp)

# lis = []
# for i in range(len(y)):
#     # sha = y[i][y[i]>0].shape[0]
#     lis.append(y[i][y[i]>1].shape[0])
#     lis.append()
# lis.sort()
# print(lis)
# print(max(lis))
# print(sum(lis)/len(lis))

# a,b,c= y.shape
# lis,alis = [],[]
# for i in range(a):
#     lis.append(y[i].sum() / y1[i].sum())


# import torch
# import torch.nn as nn

# # 定义一个单步的rnn
# rnn_single = nn.RNNCell(input_size=100, hidden_size=200)
# # 访问其中的参数
# print(rnn_single.weight_hh.size())
# # 构造一个序列，长为6，batch是5，特征是100
# x = (torch.randn(6, 5, 100))
# # 定义初始的记忆状态
# h_t = (torch.zeros(5, 200))
# # 传入 rnn
# out = []
# for i in range(6):   # 通过循环6次作用在整个序列上
#     h_t = rnn_single(x[i], h_t)
#     out.append(h_t)
# import ipdb; ipdb.set_trace()
