import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import pickle
import os

# for pyramid_id in [0, 1, 2, 4]:
#     image_names = np.load("saved/db_pyramid" + str(pyramid_id) +
#                           "_hfnet_globalindex.npy")
#     globaldesc = np.load("saved/db_pyramid" + str(pyramid_id) +
#                          "_hfnet_globaldesc.npy")
#     nbrs = NearestNeighbors(n_neighbors=30, algorithm='auto').fit(globaldesc)
#     knnPickle = open('saved/knn_model_pyramid' + str(pyramid_id), 'wb')
#     pickle.dump(nbrs, knnPickle)


# image_names = np.load('saved/db_pyramid_hfnet_globalindex.npy')
# # globaldesc = np.load('saved/db_pyramid_hfnet_globaldesc.npy')


# image_names_query = np.load('saved/query_pyramid_hfnet_globalindex.npy')
# globaldesc_query = np.load('saved/query_pyramid_hfnet_globaldesc.npy')
# print(image_names.shape)
# # print(globaldesc.shape)
# print(image_names_query.shape)
# print(globaldesc_query.shape)
# # fit
# # nbrs = NearestNeighbors(n_neighbors=35, algorithm='kd_tree').fit(globaldesc)
# # knnPickle = open('saved/knn_model_pyramid', 'wb')
# # pickle.dump(nbrs, knnPickle)
# # nn_model = nbrs

# # load
# nn_model = pickle.load(open('saved/knn_model_pyramid', 'rb'))

# distances, indices = nn_model.kneighbors(globaldesc_query)
# for i in range(0,image_names_query.size):
#     names = [image_names[indices_now] for indices_now in indices[i]][0:50]
#     print(i, image_names_query[i])
#     for ii in range(1, 19):
#         print("**", names[ii], distances[i][ii])

# distances, indices = nn_model.kneighbors(globaldesc)
# for i in range(0,image_names.size):
#     names = [image_names[indices_now] for indices_now in indices[i]][0:50]
#     print(i, image_names[i])
#     for ii in range(0, 10):

#         print("*****", names[ii], distances[i][ii+1] )
# print(len(image_names_query))
# for pyramid_id in [0, 1, 2, 4]:

#     image_names_new = []
#     globaldesc_new = []
#     for img_id in range(0, len(image_names_query)):
#         img = image_names_query[img_id]
#         if('expand_' + str(pyramid_id) in img):
#         # /home/ezxr/Documents/wxc/query_ocr/2021-01-19/2021-01-19-19-58-18_Fail.jpg0_isFlip_0_txt_KC_wh-ratio_1.26_confidence_0.872_expand_1.jpg
#             img_new = img.replace(":", "_")
#             print(img_new)
#             globaldesc_new.append(globaldesc_query[img_id])
#             image_names_new.append(img_new)
#             os.rename(img, img_new)
#     np.save('saved/query_pyramid'+str(pyramid_id) +
#             '_hfnet_globalindex.npy', np.array(image_names_new))
#     np.save('saved/query_pyramid'+str(pyramid_id) +
#             '_hfnet_globaldesc.npy', np.array(globaldesc_new).squeeze())
for pyramid_id in [0, 1, 2, 4]:

    image_names_query = np.load('saved/query_pyramid'+str(pyramid_id) +
            '_hfnet_globalindex.npy')
    globaldesc_query =  np.load('saved/query_pyramid'+str(pyramid_id) +
            '_hfnet_globaldesc.npy')
    nn_model = pickle.load(open('saved/knn_model_pyramid' + str(pyramid_id), 'rb'))
    distances, indices = nn_model.kneighbors(globaldesc_query)
    image_names = np.load('saved/db_pyramid'+str(pyramid_id) +
            '_hfnet_globalindex.npy')
    for i in range(0,image_names_query.size):
        names = [image_names[indices_now] for indices_now in indices[i]][0:50]
        print(i, image_names_query[i])
        for ii in range(1, 19):
            print("**", names[ii], distances[i][ii])