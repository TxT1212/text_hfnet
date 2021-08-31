###### orc 
image_pathq=/home/ezxr/Documents/ibl_dataset_cvpr17_3852/query_images_undistort
ocr_pathq=/home/ezxr/Documents/ibl_dataset_cvpr17_3852/query_images_ocr
# python pipeline/padder_ocr.py --input_path $image_pathq --output_path $ocr_pathq
# python pipeline/ocr_area_extract.py --input_path $image_pathq --output_path $ocr_pathq

image_pathdb=/home/ezxr/Documents/ibl_dataset_cvpr17_3852/training_images_undistort
ocr_pathdb=/home/ezxr/Documents/ibl_dataset_cvpr17_3852/training_image_ocr
# python pipeline/padder_ocr.py --input_path $image_pathdb --output_path $ocr_pathdb
# python pipeline/ocr_area_extract.py --input_path $image_pathdb --output_path $ocr_pathdb

desc_save_path=/home/ezxr/Downloads/ocr/text_hfnet/saved_ibl/

python pipeline/hfnet.py --input_path $ocr_pathdb --output_path $desc_save_path --save_prefix db_p8 --image_suffix expand_8.jpg
python pipeline/hfnet.py --input_path $ocr_pathq --output_path $desc_save_path --save_prefix query_p8 --image_suffix expand_8.jpg

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


# which python
# python pipeline/padder_ocr.py --output_path /media/txt/data2/data2/naver/ocr/ --input_path /media/txt/data2/data2/naver/HyundaiDepartmentStore
# python pipeline/ocr_area_extract.py --output_path /media/txt/data2/data2/naver/ocr/ --input_path /media/txt/data2/data2/naver/HyundaiDepartmentStore/