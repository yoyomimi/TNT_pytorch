# TNT_pytorch
This is a PyTorch version for MOT TrackletNet

- implementation timeline http://note.youdao.com/noteshare?id=fde3925cdd1b82d39a4124c961737c8e

## detection

FCOS to detect objects with class on images.

Start to train or eval or test:

see detection/detect.sh


## appearance

Facenet + triplet loss to train feature embeddings generation based on cropped images.

Start to train:

python3 appearance/train_appearance.py --cfg configs/facenet_triplet_appearance.yaml


## data dir structure

see tree.md


## environments

see requirements.txt

pip install -r requirements.txt
       
                    
