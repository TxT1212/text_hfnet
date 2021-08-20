import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import pickle 
import re

image_names = np.load('saved_ibl/db_pyramid4_hfnet_globalindex.npy')
image_names_query = np.load('saved_ibl/query_pyramid4_hfnet_globalindex.npy')

globaldesc = np.load('saved_ibl/db_pyramid4_hfnet_globaldesc.npy')
globaldesc_query = np.load('saved_ibl/query_pyramid4_hfnet_globaldesc.npy')

query_img_ocr_path = '/home/ezxr/Documents/ibl_dataset_cvpr17_3852/query_images_ocr/'
query_img_path = '/home/ezxr/Documents/ibl_dataset_cvpr17_3852/query_images_undistort/'
db_img_ocr_path = '/home/ezxr/Documents/ibl_dataset_cvpr17_3852/training_image_ocr/'
db_img_path = '/home/ezxr/Documents/ibl_dataset_cvpr17_3852/training_images_undistort/'
## fit
# nbrs = NearestNeighbors(n_neighbors=20, algorithm='auto').fit(globaldesc)
# knnPickle = open('saved_ibl/knn_model_pyramid4', 'wb') 
# pickle.dump(nbrs, knnPickle)


### load
nn_model = pickle.load(open('saved_ibl/knn_model_pyramid4', 'rb'))
distances, indices = nn_model.kneighbors(globaldesc_query) 
# distances, indices = nbrs.kneighbors(globaldesc)



for i in range(0,image_names_query.size):
    pass
    names = [image_names[indices_now] for indices_now in indices[i]][0:20]
    # print(i, image_names_query[i])
    for ii in range(0, 5):
        # print("**", names[ii], distances[i][ii])
        q = re.sub('[0-9]*_isFlip.*', '', image_names_query[i]).replace(query_img_ocr_path, "")
        d = re.sub('[0-9]*_isFlip.*', '', names[ii]).replace(db_img_ocr_path, "")
        print(q, d)

