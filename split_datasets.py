import os
import numpy as np

# 把两个月的inflow-outflow map 按照tain valid test划分成可用的数据集
#-----------------------------------------------------------------------
def split_dataset(x,y,train_path,val_path,test_path):
    assert len(x) == len(y)
    lentrain = (len(x)//6)*4
    lenval = (len(x) - lentrain)//2
    lentest = len(x) - lentrain - lenval
    xtrain,ytrain = x[:lentrain],y[:lentrain]
    xval,yval = x[lentrain:lentrain+lenval],y[lentrain:lentrain+lenval]
    xtest,ytest = x[lentrain+lenval:],y[lentrain+lenval:]
    np.save(os.path.join(train_path,'X.npy'),xtrain)
    np.save(os.path.join(train_path,'Y.npy'),ytrain)
    np.save(os.path.join(val_path,'X.npy'),xval)
    np.save(os.path.join(val_path,'Y.npy'),yval)
    np.save(os.path.join(test_path,'X.npy'),xtest)
    np.save(os.path.join(test_path,'Y.npy'),ytest)

from numpy import set_printoptions
np.set_printoptions(threshold=np.inf)


freq_str = '15min'
# nx,ny = 64,64
datafolder = 'adata'
# nsize = 128
# factor = 4
#--------------------------------------------------------------------------
city = 'xian'

# high_path = './adata/cdu_' + freq_str + '_' + str(64) + '-' + str(64) + '.npy'
high_path = f'./{datafolder}/{city}_{freq_str}_{str(64)}-{str(64)}.npy'
y = np.load(high_path)
print('y:',y.shape)
# print(y[4200])
#--------------------------------------------------------------------

coarse_path = f'./{datafolder}/{city}_{freq_str}_{str(16)}-{str(16)}.npy'
x = np.load(coarse_path)
print('x:',x.shape)
# print(x[0])

#----------------------------------------------------------------------

namelist = ['train','valid','test']
# datapath ='/data3/liumengmeng/ORI/FM/data/xian_15min_2-16-16_2-64-64/'
# datapath = '/data3/liumengmeng/UrbanFM/data/xian_15min_2-16-16_2-64-64'
fullpath = [os.path.join(datapath,i) for i in namelist]

# for i in fullpath:os.makedirs(i)

split_dataset(x,y,*fullpath)