###### orc /
# conda activate ocr
image_pathq=/home/ezxr/Documents/ibl_dataset_cvpr17_3852/query_images_undistort/
ocr_pathq=/home/ezxr/Documents/ibl_dataset_cvpr17_3852/query_images_ocr/
# python pipeline/padder_ocr.py --input_path $image_pathq --output_path $ocr_pathq
# python pipeline/ocr_area_extract.py --input_path $image_pathq --output_path $ocr_pathq

image_pathdb=/home/ezxr/Documents/ibl_dataset_cvpr17_3852/training_images_undistort/
ocr_pathdb=/home/ezxr/Documents/ibl_dataset_cvpr17_3852/training_image_ocr/
# python pipeline/padder_ocr.py --input_path $image_pathdb --output_path $ocr_pathdb
# python pipeline/ocr_area_extract.py --input_path $image_pathdb --output_path $ocr_pathdb


# hfnet
desc_save_path=/home/ezxr/Downloads/ocr/text_hfnet/saved_ibl/

# python pipeline/hfnet.py --input_path $ocr_pathdb --output_path $desc_save_path --save_prefix db_p8_hfnet --image_suffix expand_8.jpg
# python pipeline/hfnet.py --input_path $ocr_pathq --output_path $desc_save_path --save_prefix query_p8_hfnet --image_suffix expand_8.jpg
hfnet_pairs_list=saved_ibl/hfnet_pairs_list.txt
# python pipeline/hfnet_nn.py --input_path $desc_save_path --save_query_prefix query_p8_hfnet --save_db_prefix db_p8_hfnet >> $hfnet_pairs_list

# # paddleClas
paddle_clas_path=/home/ezxr/Downloads/PaddleClas/deploy
cd $paddle_clas_path
# python python/get_descriptor.py  -c configs/inference_logo.yaml -o IndexProcess.gallery_features_outpath="$desc_save_path/db_logo_globaldesc.npy" -o IndexProcess.gallery_images_outpath="$desc_save_path/db_logo_globalindex.npy" -o IndexProcess.image_root="$ocr_pathdb" 
# python python/get_descriptor.py  -c configs/inference_logo.yaml -o IndexProcess.gallery_features_outpath="$desc_save_path/query_logo_globaldesc.npy" -o IndexProcess.gallery_images_outpath="$desc_save_path/query_logo_globalindex.npy" -o IndexProcess.image_root="$ocr_pathq" 
cd - # 回到当前文件夹
paddle_clas_pairs_list=saved_ibl/paddle_clas_pairs_list.txt
python pipeline/hfnet_nn.py --input_path $desc_save_path --save_query_prefix query_logo --save_db_prefix db_logo >> $paddle_clas_pairs_list --threshold 0.85
# subFolder=4F 
# ws=/media/txt/data2/naver/HyundaiDepartmentStore/
# # kapture_export_colmap.py -i ${ws}${subFolder}/release/mapping -db ${ws}${subFolder}/release/database.db -txt ${ws}${subFolder}/release/
# cd ${ws}${subFolder}/release
# # colmap feature_extractor --database_path database.db --image_path mapping/sensors/records_data/
# # colmap spatial_matcher --database_path database.db   --SpatialMatching.is_gps 0 --SpatialMatching.ignore_z 0

# # mkdir sfm_sift
# # colmap point_triangulator --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_extra_param 0 --database_path database.db --image_path ./mapping/sensors/records_data/ --input_path . --output_path sfm_sift/

# subFolder=B1
# # kapture_export_colmap.py -i ${ws}${subFolder}/release/mapping -db ${ws}${subFolder}/release/database.db -txt ${ws}${subFolder}/release/
# cd ${ws}${subFolder}/release
# # colmap feature_extractor --database_path database.db --image_path mapping/sensors/records_data/
# # colmap spatial_matcher --database_path database.db   --SpatialMatching.is_gps 0 --SpatialMatching.ignore_z 0
# # mkdir sfm_sift
# # colmap point_triangulator --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_extra_param 0 --database_path database.db --image_path ./mapping/sensors/records_data/ --input_path . --output_path sfm_sift/
# cd /home/txt/Downloads/ezxr_loc/Hierarchical-Localization/
# python -m hloc.pipelines.naver.pipeline
# python -m hloc.pipelines.naverF4.pipeline
# python -m hloc.pipelines.naverB1.pipeline


