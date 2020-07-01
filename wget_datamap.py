#------------生成wget命令，依次下载GPS数据---------------------

import time
import os
import sys
# print(time.gmtime(1501584540)[2])

txt_path = '/data3/liumengmeng/data_xian_10/file_xian_10.txt'
pathname = 'xian_10_'
# save_path = 'E:/DataMap/file1.txt'
with open(txt_path) as f:
    a = f.read().split()
    # a = [line.rstrip() for line in f]
fn = list(map(str,range(1,32))) #按1-31给每个下载文件命名

#生成下载文件的指令 #链接需要加上""才行
# orders = []
# for i in range(len(a)):
#     print('wget'+' '+'"'+ a[i] +'"' + ' ' + '-O'+' ' + pathname + fn[i]+'.tar.gz')
    # orders.append('wget'+' '+'"'+ a[i] +'"' + ' ' + '-O'+' '+ fn[i]+'.tar.gz')

# #生成解压文件的指令
for i in fn:
    print('tar -zxvf '+ pathname + i + '.tar.gz')
    # print('tar -zxvf '+ i + '.tar.gz')
#     # print(i)

# with open(save_path) as f:
#     for i in orders:
#         f.write(i+'\n')

