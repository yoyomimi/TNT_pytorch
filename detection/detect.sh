export PATH=/usr/local/cuda-9.0/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH &&\
# training

#CUDA_VISIBLE_DEVICE=0,1 python3 detection/train_det.py --cfg configs/fcos_detector.yaml WORKERS 10 MODEL.BACKBONE.WEIGHTS data/R-50.pkl

# # eval
# CUDA_VISIBLE_DEVICE=0 python3 detection/eval_det.py --cfg configs/fcos_detector.yaml \
#  MODEL.RESUME_PATH /mnt/cephfs_new_wj/cv/chenmingfei.lasia/TNT_pytorch/data/fcos_res50_checkpoint.pth \
#  TEST.OUT_DIR eval_outdir \
#  DATASET.IMG_NUM_PER_GPU 1 \
#  MODEL.BACKBONE.PRETRAINED False

# # test
CUDA_VISIBLE_DEVICE=1 python3 detection/test_det.py --cfg configs/fcos_detector.yaml \
 --jpg_path data/testing/image_02/0011 \
 MODEL.RESUME_PATH /mnt/ht/students/cmf/TNT_pytorch/work_dirs/fcos_detector/2020-03-10-05-51/FCOS_epoch007_iter014000_checkpoint.pth\
 MODEL.BACKBONE.PRETRAINED False 

# # TODO change the model resume_path


