# TNT_pytorch
This is a PyTorch version for MOT TrackletNet

- implementation timeline http://note.youdao.com/noteshare?id=fde3925cdd1b82d39a4124c961737c8e

## detection

FCOS to detect objects with class on images.

Start to train or eval or test:

see detection/detect.sh


## crop images using labels

Use tracking labels to generate crop images. Destination dir structure is as data/crop_data dir in tree.md.

Start to crop:

python3 datasets/crop_gt_box.py --data_root data/ --dst_root data/crop_data   


## appearance

Facenet + triplet loss to train feature embeddings generation based on cropped images.

Start to train:

python3 appearance/train_appearance.py --cfg configs/facenet_triplet_appearance.yaml


## data dir structure

see tree.md


## environments

see requirements.txt

pip install -r requirements.txt
       
                    
