CUDA_VISIBLE_DEVICE=0 python3 TNT/generate_clusters.py --cfg configs/cluster_generation.yaml \
 --frame_dir /mnt/cephfs_new_wj/cv/chenmingfei.lasia/TNT_pytorch/data/training/objtrack/images/0012 \
 MODEL.BACKBONE.PRETRAINED False \
 MODEL.RESUME_PATH /mnt/cephfs_new_wj/cv/chenmingfei.lasia/TNT_pytorch/work_dirs/trackletpair_connectivity/2020-03-01-12-13/TrackletConnectivity_epoch000_iter000010_checkpoint.pth \
 
