import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import pickle
import os

for pyramid_id in [0, 1, 2, 4]:
    globaldesc = np.array([[]])
    globalindex = np.array([[]])
    for floor_id in [1, 2]:
        globaldesc_name = "saved/db_f" + \
            str(floor_id) + "_pyramid" + \
            str(pyramid_id) + "_hfnet_globaldesc.npy"
        # saved/db_f1_pyramid0_hfnet_globaldesc.npy
        globalindex_name = "saved/db_f" + \
            str(floor_id) + "_pyramid" + \
            str(pyramid_id) + "_hfnet_globalindex.npy"
        # saved/db_f1_pyramid0_hfnet_globalindex.npy
        print(globaldesc_name)
        if(globaldesc.size):
            globaldesc = np.append(
                globaldesc, np.load(globaldesc_name), axis=0)
            globalindex = np.append(
                globalindex, np.load(globalindex_name), axis=0)
        else:
            globaldesc = np.load(globaldesc_name)
            globalindex = np.load(globalindex_name)
        print(globaldesc.shape)
        print(globalindex.shape)
    np.save("saved/db_pyramid" + str(pyramid_id) +
            "_hfnet_globalindex.npy", globalindex)
    np.save("saved/db_pyramid" + str(pyramid_id) +
            "_hfnet_globaldesc.npy", globaldesc)
