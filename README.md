# Text-Image Driven 4D Scene Model Generation
## TID4DSMG

### Run GSPLAT

''' python train_gaussian.py default --data_dir /home/dusan/Desktop/ML_PROJECTS/tid4dsmg/data/imgs_sparse/ --data_factor 1 --result_dir ./results/test_01_pycolmap_train '''

### Run COLMAP

''' pytohn colmap_recontruction.py --image_path /path/ --output_path /outputpath/ '''

### Run Metric3D
- Go to metric 3d dir
''' bash test_vit.sh '''