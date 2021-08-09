import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import pickle 
import os


image_names = np.load('/home/ezxr/Downloads/PaddleClas/deploy/save/db_gallery_images_ibl.npy')
globaldesc = np.load('/home/ezxr/Downloads/PaddleClas/deploy/save/db_gallery_features_ibl.npy')

# image_names_pyram = np.load('saved/database_hfnet_globalindex.npy')
# globaldesc = np.load('saved/database_hfnet_globaldesc.npy')

image_names_query = np.load('/home/ezxr/Downloads/PaddleClas/deploy/save/gallery_images_ibl.npy')
globaldesc_query = np.load('/home/ezxr/Downloads/PaddleClas/deploy/save/gallery_features_ibl.npy')

## fit
nbrs = NearestNeighbors(n_neighbors=50, algorithm='auto').fit(globaldesc)
knnPickle = open('saved/knn_model50', 'wb') 
pickle.dump(nbrs, knnPickle)
nn_model = nbrs

## load
# nn_model = pickle.load(open('saved/knn_model_paddle_clas', 'rb'))

distances, indices = nn_model.kneighbors(globaldesc_query) 
for i in range(0,image_names_query.size):
    names = [image_names[indices_now] for indices_now in indices[i]][0:50]
    print(i, image_names_query[i])
    for ii in range(0, 19):
        print("**", names[ii], distances[i][ii+1] )