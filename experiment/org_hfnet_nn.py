import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import pickle 


image_names = np.load('saved_ibl/db_whole_image_hfnet_globalindex.npy')
image_names_query = np.load('saved_ibl/query_whole_image_hfnet_globalindex.npy')

globaldesc = np.load('saved_ibl/db_whole_image_hfnet_globaldesc.npy')
globaldesc_query = np.load('saved_ibl/query_whole_image_hfnet_globaldesc.npy')

print(globaldesc.shape)
print(image_names_query.shape)
## fit
# nbrs = NearestNeighbors(n_neighbors=20, algorithm='auto').fit(globaldesc)
# knnPickle = open('saved_ibl/knn_model_org', 'wb') 
# pickle.dump(nbrs, knnPickle)


# ### load
nn_model = pickle.load(open('saved_ibl/knn_model_org', 'rb'))
distances, indices = nn_model.kneighbors(globaldesc_query) 
# distances, indices = nbrs.kneighbors(globaldesc)



for i in range(0,image_names_query.size):
    pass
    names = [image_names[indices_now] for indices_now in indices[i]][0:20]
    # print(i, image_names_query[i])
    for ii in range(0, 19):
        # print("**", names[ii], distances[i][ii])
        print(image_names_query[i], names[ii]) # >> logs_ibl/log_org.txt
