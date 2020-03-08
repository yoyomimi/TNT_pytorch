# TNT_pytorch
This is a PyTorch version for MOT TrackletNet

## Step 1: detection

We use FCOS as an example detector to detect objects with class on images.

**Start to train or eval or test:**

see detection/detect.sh

Remember to change DATASET.ROOT in configs/fcos_detector.yaml to the root of your own data.


## Step 2: crop images using labels

We use tracking labels to generate crop images. Destination dir structure is  data/crop_data dir in tree.md. The cropped images will be used to train the appearance model and tracklet connectivity model.

**Start to crop:**

python3 datasets/crop_gt_box.py --data_root data/ --dst_root data/crop_data   


## Step 3: appearance

We use Facenet + triplet loss as an example to train feature embeddings generation based on cropped images.

**Start to train:**

python3 appearance/train_appearance.py --cfg configs/facenet_triplet_appearance.yaml

Remember to change DATASET.ROOT in configs/facenet_triplet_appearance.yaml to the root of your own cropped  data, change MODEL.FEATURE_PATH and MODEL.LOGITS_PATH to the pretrained weights path of InceptionResnetV1 (20180402-114759-vggface2-features.pt and 20180402-114759-vggface2-logits.pt)

## Step 4: tracklets

Use feature embeddings inferenced by the **appearance** model in step 3 to train the tracklet connectivity model (referring to the paper of TNT) . 

**Start to train:**

python3 tracklets/train_trackletpair_fushion.py --cfg configs/tracklet_pair_connectivity.yaml

Remember to change DATASET.ROOT in configs/tracklet_pair_connectivity.yaml to the root of your own cropped  data, change MODEL.APPEARANCE.WEIGHTS to the path of your own well trained appearance model path file (e.g. Facenet). TRACKLET.WINDOW_lEN indicates the length of the silding window for trackletpair sampling.

## Step 5: clusters




## data dir structure

See tree.md to get the example data dir structure when running the project.


## environments

See requirements.txt.

**Start to set:**

pip install -r requirements.txt

​                    
