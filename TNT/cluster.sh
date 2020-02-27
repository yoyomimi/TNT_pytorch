CUDA_VISIBLE_DEVICE=0 python3 TNT/generate_clusters.py --cfg configs/cluster_generation.yaml \
 --frame_dir /mnt/cephfs_new_wj/cv/chenmingfei.lasia/TNT_pytorch/data/training/objtrack/images/0012 \
 MODEL.RESUME_PATH /mnt/cephfs_new_wj/cv/chenmingfei.lasia/TNT_pytorch/work_dirs/fcos_detector/2020-02-21-22-54/FCOS_epoch222_iter002000_checkpoint.pth \
 MODEL.BACKBONE.PRETRAINED False 
