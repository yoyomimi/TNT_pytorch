# training
CUDA_VISIBLE_DEVICE=0 python3 detection/train_det.py --cfg configs/fcos_detector.yaml WORKERS 10 MODEL.BACKBONE.WEIGHTS /mnt/cephfs_new_wj/cv/chenmingfei.lasia/TNT_pytorch/data/R-50.pkl

# eval
CUDA_VISIBLE_DEVICE=0 python3 detection/eval_det.py --cfg configs/fcos_detector.yaml \
 MODEL.RESUME_PATH /mnt/cephfs_new_wj/cv/chenmingfei.lasia/TNT_pytorch/data/fcos_res50_checkpoint.pth \
 TEST.OUT_DIR eval_outdir \
 DATASET.IMG_NUM_PER_GPU 1 \
 MODEL.BACKBONE.PRETRAINED False

# test
CUDA_VISIBLE_DEVICE=1 python3 detection/test_det.py --cfg configs/fcos_detector.yaml \
 --jpg_path /mnt/cephfs_new_wj/cv/chenmingfei.lasia/TNT_pytorch/data/training/objtrack/images/0012 \
 MODEL.RESUME_PATH /mnt/cephfs_new_wj/cv/chenmingfei.lasia/TNT_pytorch/data/fcos_res50_checkpoint.pth \
 MODEL.BACKBONE.PRETRAINED False 

# TODO change the model resume_path


