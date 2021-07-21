# 改变log中的图片名字，更好可视化
import os
import glob
log = 'logs/items_3d_update_4points.txt'
log_o = 'logs/items_3d_expand4.txt'
f = open(log,"r") 
# f_o = open(log_o, "w")
lines = f.readlines()      #读取全部内容 ，并以列表方式返回
for line in lines:
    if(line[0] == '*'):
        for name in glob.glob(line[3:-5] + '_*.jpg'):
            # print(name)
            if('expand_4' in name):
                print("**", name)
    else:
        # f_o.write()
        print(line[:-1])
