import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import pickle 


image_names = np.load('saved/database_hfnet_globalindex.npy')
# image_names_query = np.load('saved/database_hfnet_globalindex_2.npy')

globaldesc = np.load('saved/database_hfnet_globaldesc.npy')
# globaldesc_query = np.load('saved/database_hfnet_globaldesc_2.npy')

print(globaldesc.shape)
## fit
nbrs = NearestNeighbors(n_neighbors=20, algorithm='auto').fit(globaldesc)
knnPickle = open('saved/knn_model1', 'wb') 
pickle.dump(nbrs, knnPickle)


### load
# nn_model = pickle.load(open('saved/knn_model1', 'rb'))
# distances, indices = nn_model.kneighbors(globaldesc_query) 
# distances, indices = nbrs.kneighbors(globaldesc)
# np.save("saved/data_base_hfnet_nn20_index.npy", indices)
# np.save("saved/data_base_hfnet_nn20_distance.npy", distances)

# distances = np.load('saved/data_base_hfnet_nn20_distance.npy')
# indices = np.load('saved/data_base_hfnet_nn20_index.npy')
# print("indices.shape:", indices.shape)



# for i in range(0,image_names_query.size):
#     pass
#     names = [image_names[indices_now] for indices_now in indices[i]][1:20]
#     print(i, image_names_query[i])
#     for ii in range(0, 19):
#         print("*****", names[ii], distances[i][ii+1])
