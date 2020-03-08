# TNT_pytorch
This is a PyTorch version for MOT TrackletNet

## Step 1: detection

We use FCOS as an example detector to detect objects with class on images.

**Start to train or eval or test:**

see **detection/detect.sh**

Remember to change DATASET.ROOT in configs/fcos_detector.yaml to the root of your own data.


## Step 2: crop images using labels

We use tracking labels to generate crop images. Destination dir structure is  data/crop_data dir in tree.md. The cropped images will be used to train the appearance model and tracklet connectivity model.

**Start to crop:**

`python3 datasets/crop_gt_box.py --data_root data/ --dst_root data/crop_data`   


## Step 3: appearance

We use Facenet + triplet loss as an example to train feature embeddings generation based on cropped images.

**Start to train:**

`python3 appearance/train_appearance.py --cfg configs/facenet_triplet_appearance.yaml`

Remember to change DATASET.ROOT in configs/facenet_triplet_appearance.yaml to the root of your own cropped data, change MODEL.FEATURE_PATH and MODEL.LOGITS_PATH to the pretrained weights path of InceptionResnetV1 (20180402-114759-vggface2-features.pt and 20180402-114759-vggface2-logits.pt)

## Step 4: tracklets

Use feature embeddings inferenced by the **appearance** model in step 3 to train the tracklet connectivity model (referring to the paper of TNT) . 

**Start to train:**

`python3 tracklets/train_trackletpair_fushion.py --cfg configs/tracklet_pair_connectivity.yaml`

Remember to change DATASET.ROOT in configs/tracklet_pair_connectivity.yaml to the root of your own cropped  data, change MODEL.APPEARANCE.WEIGHTS to the path of your own well trained appearance model .pt file (e.g. Facenet). TRACKLET.WINDOW_lEN indicates the length of the silding window for trackletpair sampling.

## Step 5: clusters

Generate tracklet clusters with interpolation among discrete tracklets in one cluster, based on frame images. The pipeline constructs as below:

- We firstly use well trained detector in step 1 to get detected results on each frame.  

- Secondly, we apply well trained appearance model in step 3 to get apearance embedding for each detected objects.

- Thirdly, we use geometry constriants and advanced location prediction to merge the neighbor detected resultes as coarse tracklets.

- Fourthly, we analyze the temporal connections between coarse tracklet pairs and the object location information to cluster the tracklets into coarse clusters. We use well trained tracklet connectivity model in step 4 to update the coarse clusters, compute the cost between tracklet pairs  in one cluster.

- Fifthly, we consider the cost between tracklet pairs as edge weights in a tracklet graph. We use graph optimal method to generate the optimal clusters from the tracklets.

- Lastly, we interpolate  among discrete tracklets in one cluster, saving results into data/visualize.json:

  `dict{`

  ​    `frame_id: {`

  ​        `track_id: {`

  ​            `label: 0`

  ​            `loc: [xmin, ymin, xmax, ymax]`

  ​        `}`

  ​    `}`

  `}`

**Start the pipeline:**

`python3 TNT/generate_clusters.py --cfg configs/cluster_generation.yaml  --frame_dir [the root path for your own frame images of one video]` 

Remember to change DATASET.ROOT in configs/cluster_generation.yaml to the root of your own data,  change MODEL.DETECTION_WEIGHTS to the path of your own well trained detection model .pt file in step 1 (e.g. FCOS), change MODEL.APPEARANCE.WEIGHTS to the path of your own well trained appearance model .pt file in step 3 (e.g. Facenet), change MODEL.RESUME_PATH to the path of your own well trained tracklet connectivity model .pt file in step 4.


## data dir structure

See tree.md to get the example data dir structure when running the project.


## environments

See requirements.txt.

**Start to set:**

pip install -r requirements.txt

​                    
