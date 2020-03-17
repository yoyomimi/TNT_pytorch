# CUDA_VISIBLE_DEVICE=1 python3 TNT/generate_clusters.py --cfg configs/cluster_generation.yaml \
# --frame_dir /mnt/ht/students/cmf/TNT_pytorch/data/training/left_02/images/0011
python3 utils/outwrite.py
ffmpeg -f image2 -i test_out/track_%06d.jpg -vf  "scale=1024:-2"  -r 10 video.mp4 -y



 
