# 改变log中的图片名字，更好可视化
import os
import glob
import re
log = 'logs/items_3d_update_4points.txt'
log_o = 'logs/items_3d_expand4.txt'
log = 'saved/hfnet_nn_B1.txt'
f = open(log,"r") 
# f_o = open(log_o, "w")
lines = f.readlines()
query_map = {}
for line in lines:
    q, d = line.strip().split(' ')
    q = re.sub('[0-9]*_txt.*', '', q)
    d = re.sub('[0-9]*_txt.*', '', d)
    # query_map[q]
    if q in query_map:
        query_map[q].append(d)
    else:
        query_map[q] = [d]
for q in query_map:
    ds = set(query_map[q])  
    for d in ds:  
        print(q,d)

