import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import pickle 
import os


image_names = np.load('saved/database_hfnet_globalindex.npy')
image_names_query = np.load('saved/query_hfnet_globalindex.npy')

globaldesc = np.load('saved/database_hfnet_globaldesc.npy')
globaldesc_query = np.load('saved/query_hfnet_globaldesc.npy')

## fit
nbrs = NearestNeighbors(n_neighbors=50, algorithm='auto').fit(globaldesc)
knnPickle = open('saved/knn_model50', 'wb') 
pickle.dump(nbrs, knnPickle)
nn_model = nbrs

### load
# nn_model = pickle.load(open('saved/knn_model1', 'rb'))

distances, indices = nn_model.kneighbors(globaldesc_query) 
for i in range(0,image_names_query.size):
    names = [image_names[indices_now] for indices_now in indices[i]][0:50]
    print(i, image_names_query[i])
    for ii in range(0, 19):
        print(indices[i][ii], "*", names[ii], distances[i][ii+1] )

# distances, indices = nn_model.kneighbors(globaldesc) 
# for i in range(0,image_names.size):
#     names = [image_names[indices_now] for indices_now in indices[i]][0:50]
#     print(i, image_names[i])
#     for ii in range(0, 10):

#         print("*****", names[ii], distances[i][ii+1] )

# image_names_new = []
# for img in image_names:
#     img_new = img.replace(":","_")
#     print(img_new)
#     image_names_new.append(img_new)
#     os.rename(img, img_new)
# np.save('saved/database_hfnet_globalindex.npy', np.array(image_names_new))
