.
├── appearance
│   ├── backbones
│   │   ├── inception_resnet_v1.py
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── inception_resnet_v1.cpython-37.pyc
│   │   │   ├── __init__.cpython-37.pyc
│   │   │   └── resnet.cpython-37.pyc
│   │   ├── resnet.py
│   │   └── vgg.py
│   ├── _init_paths.py
│   ├── __init__.py
│   ├── loss
│   │   ├── facenet_loss.py
│   │   ├── __init__.py
│   │   └── __pycache__
│   │       ├── facenet_loss.cpython-37.pyc
│   │       └── __init__.cpython-37.pyc
│   ├── __pycache__
│   │   ├── __init__.cpython-37.pyc
│   │   └── _init_paths.cpython-37.pyc
│   ├── train_appearance.py
│   ├── trainer
│   │   ├── appearanceTrainer.py
│   │   ├── __init__.py
│   │   └── __pycache__
│   │       ├── appearanceTrainer.cpython-37.pyc
│   │       ├── FCOStrainer.cpython-37.pyc
│   │       ├── __init__.cpython-37.pyc
│   │       └── trainer.cpython-37.pyc
│   └── utils
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── __init__.cpython-37.pyc
│       │   └── resnet_load_pkl.cpython-37.pyc
│       └── resnet_load_pkl.py
├── configs
│   ├── default.py
│   ├── facenet_triplet_appearance.yaml
│   ├── fcos_detector.yaml
│   ├── __init__.py
│   └── __pycache__
│       ├── default.cpython-37.pyc
│       └── __init__.cpython-37.pyc
├── data
│   ├── 20180402-114759-vggface2-features.pt
│   ├── 20180402-114759-vggface2-logits.pt
│   ├── crop_data
│   │   ├── eval
│   │   │   ├── gt_eval_track_dict.json
│   │   │   ├── objtrack_0012
│   │   │   │   ├── 0
│   │   │   │   │   ├── Cyclist_0_crop.jpg
│   │   │   │   │   ├── Cyclist_10_crop.jpg
│   │   │   │   │   ├── Cyclist_11_crop.jpg
│   │   │   │   │   ├── Cyclist_12_crop.jpg
│   │   │   │   │   ├── Cyclist_13_crop.jpg
│   │   │   │   │   ├── Cyclist_14_crop.jpg
│   │   │   │   │   ├── Cyclist_15_crop.jpg
│   │   │   │   │   ├── Cyclist_16_crop.jpg
│   │   │   │   │   ├── Cyclist_17_crop.jpg
│   │   │   │   │   ├── Cyclist_18_crop.jpg
│   │   │   │   │   ├── Cyclist_19_crop.jpg
│   │   │   │   │   ├── Cyclist_1_crop.jpg
│   │   │   │   │   ├── Cyclist_20_crop.jpg
│   │   │   │   │   ├── Cyclist_21_crop.jpg
│   │   │   │   │   ├── Cyclist_22_crop.jpg
│   │   │   │   │   ├── Cyclist_23_crop.jpg
│   │   │   │   │   ├── Cyclist_24_crop.jpg
│   │   │   │   │   ├── Cyclist_25_crop.jpg
│   │   │   │   │   ├── Cyclist_26_crop.jpg
│   │   │   │   │   ├── Cyclist_27_crop.jpg
│   │   │   │   │   ├── Cyclist_28_crop.jpg
│   │   │   │   │   ├── Cyclist_29_crop.jpg
│   │   │   │   │   ├── Cyclist_2_crop.jpg
│   │   │   │   │   ├── Cyclist_30_crop.jpg
│   │   │   │   │   ├── Cyclist_31_crop.jpg
│   │   │   │   │   ├── Cyclist_32_crop.jpg
│   │   │   │   │   ├── Cyclist_33_crop.jpg
│   │   │   │   │   ├── Cyclist_34_crop.jpg
│   │   │   │   │   ├── Cyclist_35_crop.jpg
│   │   │   │   │   ├── Cyclist_36_crop.jpg
│   │   │   │   │   ├── Cyclist_37_crop.jpg
│   │   │   │   │   ├── Cyclist_38_crop.jpg
│   │   │   │   │   ├── Cyclist_39_crop.jpg
│   │   │   │   │   ├── Cyclist_3_crop.jpg
│   │   │   │   │   ├── Cyclist_40_crop.jpg
│   │   │   │   │   ├── Cyclist_4_crop.jpg
│   │   │   │   │   ├── Cyclist_5_crop.jpg
│   │   │   │   │   ├── Cyclist_6_crop.jpg
│   │   │   │   │   ├── Cyclist_7_crop.jpg
│   │   │   │   │   ├── Cyclist_8_crop.jpg
│   │   │   │   │   └── Cyclist_9_crop.jpg
│   │   │   │   ├── 1
│   │   │   │   │   ├── Car_0_crop.jpg
│   │   │   │   │   ├── Car_10_crop.jpg
│   │   │   │   │   ├── Car_11_crop.jpg
│   │   │   │   │   ├── Car_12_crop.jpg
│   │   │   │   │   ├── Car_13_crop.jpg
│   │   │   │   │   ├── Car_14_crop.jpg
│   │   │   │   │   ├── Car_15_crop.jpg
│   │   │   │   │   ├── Car_16_crop.jpg
│   │   │   │   │   ├── Car_17_crop.jpg
│   │   │   │   │   ├── Car_18_crop.jpg
│   │   │   │   │   ├── Car_19_crop.jpg
│   │   │   │   │   ├── Car_1_crop.jpg
│   │   │   │   │   ├── Car_20_crop.jpg
│   │   │   │   │   ├── Car_21_crop.jpg
│   │   │   │   │   ├── Car_22_crop.jpg
│   │   │   │   │   ├── Car_23_crop.jpg
│   │   │   │   │   ├── Car_24_crop.jpg
│   │   │   │   │   ├── Car_25_crop.jpg
│   │   │   │   │   ├── Car_26_crop.jpg
│   │   │   │   │   ├── Car_27_crop.jpg
│   │   │   │   │   ├── Car_28_crop.jpg
│   │   │   │   │   ├── Car_29_crop.jpg
│   │   │   │   │   ├── Car_2_crop.jpg
│   │   │   │   │   ├── Car_30_crop.jpg
│   │   │   │   │   ├── Car_31_crop.jpg
│   │   │   │   │   ├── Car_32_crop.jpg
│   │   │   │   │   ├── Car_33_crop.jpg
│   │   │   │   │   ├── Car_34_crop.jpg
│   │   │   │   │   ├── Car_35_crop.jpg
│   │   │   │   │   ├── Car_36_crop.jpg
│   │   │   │   │   ├── Car_37_crop.jpg
│   │   │   │   │   ├── Car_38_crop.jpg
│   │   │   │   │   ├── Car_39_crop.jpg
│   │   │   │   │   ├── Car_3_crop.jpg
│   │   │   │   │   ├── Car_40_crop.jpg
│   │   │   │   │   ├── Car_41_crop.jpg
│   │   │   │   │   ├── Car_42_crop.jpg
│   │   │   │   │   ├── Car_43_crop.jpg
│   │   │   │   │   ├── Car_44_crop.jpg
│   │   │   │   │   ├── Car_45_crop.jpg
│   │   │   │   │   ├── Car_46_crop.jpg
│   │   │   │   │   ├── Car_47_crop.jpg
│   │   │   │   │   ├── Car_48_crop.jpg
│   │   │   │   │   ├── Car_49_crop.jpg
│   │   │   │   │   ├── Car_4_crop.jpg
│   │   │   │   │   ├── Car_50_crop.jpg
│   │   │   │   │   ├── Car_51_crop.jpg
│   │   │   │   │   ├── Car_52_crop.jpg
│   │   │   │   │   ├── Car_53_crop.jpg
│   │   │   │   │   ├── Car_54_crop.jpg
│   │   │   │   │   ├── Car_55_crop.jpg
│   │   │   │   │   ├── Car_56_crop.jpg
│   │   │   │   │   ├── Car_57_crop.jpg
│   │   │   │   │   ├── Car_58_crop.jpg
│   │   │   │   │   ├── Car_59_crop.jpg
│   │   │   │   │   ├── Car_5_crop.jpg
│   │   │   │   │   ├── Car_60_crop.jpg
│   │   │   │   │   ├── Car_61_crop.jpg
│   │   │   │   │   ├── Car_62_crop.jpg
│   │   │   │   │   ├── Car_63_crop.jpg
│   │   │   │   │   ├── Car_64_crop.jpg
│   │   │   │   │   ├── Car_65_crop.jpg
│   │   │   │   │   ├── Car_6_crop.jpg
│   │   │   │   │   ├── Car_7_crop.jpg
│   │   │   │   │   ├── Car_8_crop.jpg
│   │   │   │   │   └── Car_9_crop.jpg
│   │   │   │   ├── 2
│   │   │   │   │   ├── Pedestrian_13_crop.jpg
│   │   │   │   │   ├── Pedestrian_14_crop.jpg
│   │   │   │   │   ├── Pedestrian_15_crop.jpg
│   │   │   │   │   ├── Pedestrian_16_crop.jpg
│   │   │   │   │   ├── Pedestrian_17_crop.jpg
│   │   │   │   │   ├── Pedestrian_18_crop.jpg
│   │   │   │   │   ├── Pedestrian_19_crop.jpg
│   │   │   │   │   ├── Pedestrian_20_crop.jpg
│   │   │   │   │   ├── Pedestrian_21_crop.jpg
│   │   │   │   │   ├── Pedestrian_22_crop.jpg
│   │   │   │   │   ├── Pedestrian_23_crop.jpg
│   │   │   │   │   ├── Pedestrian_24_crop.jpg
│   │   │   │   │   ├── Pedestrian_25_crop.jpg
│   │   │   │   │   ├── Pedestrian_26_crop.jpg
│   │   │   │   │   ├── Pedestrian_27_crop.jpg
│   │   │   │   │   ├── Pedestrian_28_crop.jpg
│   │   │   │   │   ├── Pedestrian_29_crop.jpg
│   │   │   │   │   ├── Pedestrian_30_crop.jpg
│   │   │   │   │   ├── Pedestrian_31_crop.jpg
│   │   │   │   │   ├── Pedestrian_32_crop.jpg
│   │   │   │   │   ├── Pedestrian_33_crop.jpg
│   │   │   │   │   ├── Pedestrian_34_crop.jpg
│   │   │   │   │   ├── Pedestrian_35_crop.jpg
│   │   │   │   │   ├── Pedestrian_36_crop.jpg
│   │   │   │   │   ├── Pedestrian_37_crop.jpg
│   │   │   │   │   ├── Pedestrian_38_crop.jpg
│   │   │   │   │   ├── Pedestrian_39_crop.jpg
│   │   │   │   │   ├── Pedestrian_40_crop.jpg
│   │   │   │   │   ├── Pedestrian_41_crop.jpg
│   │   │   │   │   ├── Pedestrian_42_crop.jpg
│   │   │   │   │   ├── Pedestrian_43_crop.jpg
│   │   │   │   │   ├── Pedestrian_44_crop.jpg
│   │   │   │   │   ├── Pedestrian_45_crop.jpg
│   │   │   │   │   ├── Pedestrian_46_crop.jpg
│   │   │   │   │   ├── Pedestrian_47_crop.jpg
│   │   │   │   │   ├── Pedestrian_48_crop.jpg
│   │   │   │   │   ├── Pedestrian_49_crop.jpg
│   │   │   │   │   ├── Pedestrian_50_crop.jpg
│   │   │   │   │   ├── Pedestrian_51_crop.jpg
│   │   │   │   │   ├── Pedestrian_52_crop.jpg
│   │   │   │   │   ├── Pedestrian_53_crop.jpg
│   │   │   │   │   ├── Pedestrian_54_crop.jpg
│   │   │   │   │   ├── Pedestrian_55_crop.jpg
│   │   │   │   │   ├── Pedestrian_56_crop.jpg
│   │   │   │   │   ├── Pedestrian_57_crop.jpg
│   │   │   │   │   ├── Pedestrian_58_crop.jpg
│   │   │   │   │   ├── Pedestrian_59_crop.jpg
│   │   │   │   │   ├── Pedestrian_60_crop.jpg
│   │   │   │   │   ├── Pedestrian_61_crop.jpg
│   │   │   │   │   ├── Pedestrian_62_crop.jpg
│   │   │   │   │   ├── Pedestrian_63_crop.jpg
│   │   │   │   │   ├── Pedestrian_64_crop.jpg
│   │   │   │   │   ├── Pedestrian_65_crop.jpg
│   │   │   │   │   ├── Pedestrian_66_crop.jpg
│   │   │   │   │   ├── Pedestrian_67_crop.jpg
│   │   │   │   │   ├── Pedestrian_68_crop.jpg
│   │   │   │   │   ├── Pedestrian_69_crop.jpg
│   │   │   │   │   ├── Pedestrian_70_crop.jpg
│   │   │   │   │   ├── Pedestrian_71_crop.jpg
│   │   │   │   │   ├── Pedestrian_72_crop.jpg
│   │   │   │   │   ├── Pedestrian_73_crop.jpg
│   │   │   │   │   ├── Pedestrian_74_crop.jpg
│   │   │   │   │   ├── Pedestrian_75_crop.jpg
│   │   │   │   │   └── Pedestrian_76_crop.jpg
│   │   │   │   └── 3
│   │   │   │       ├── Car_0_crop.jpg
│   │   │   │       ├── Car_10_crop.jpg
│   │   │   │       ├── Car_11_crop.jpg
│   │   │   │       ├── Car_12_crop.jpg
│   │   │   │       ├── Car_13_crop.jpg
│   │   │   │       ├── Car_14_crop.jpg
│   │   │   │       ├── Car_15_crop.jpg
│   │   │   │       ├── Car_16_crop.jpg
│   │   │   │       ├── Car_17_crop.jpg
│   │   │   │       ├── Car_18_crop.jpg
│   │   │   │       ├── Car_19_crop.jpg
│   │   │   │       ├── Car_1_crop.jpg
│   │   │   │       ├── Car_20_crop.jpg
│   │   │   │       ├── Car_21_crop.jpg
│   │   │   │       ├── Car_22_crop.jpg
│   │   │   │       ├── Car_23_crop.jpg
│   │   │   │       ├── Car_24_crop.jpg
│   │   │   │       ├── Car_25_crop.jpg
│   │   │   │       ├── Car_26_crop.jpg
│   │   │   │       ├── Car_27_crop.jpg
│   │   │   │       ├── Car_28_crop.jpg
│   │   │   │       ├── Car_29_crop.jpg
│   │   │   │       ├── Car_2_crop.jpg
│   │   │   │       ├── Car_30_crop.jpg
│   │   │   │       ├── Car_31_crop.jpg
│   │   │   │       ├── Car_32_crop.jpg
│   │   │   │       ├── Car_33_crop.jpg
│   │   │   │       ├── Car_34_crop.jpg
│   │   │   │       ├── Car_35_crop.jpg
│   │   │   │       ├── Car_36_crop.jpg
│   │   │   │       ├── Car_37_crop.jpg
│   │   │   │       ├── Car_38_crop.jpg
│   │   │   │       ├── Car_39_crop.jpg
│   │   │   │       ├── Car_3_crop.jpg
│   │   │   │       ├── Car_40_crop.jpg
│   │   │   │       ├── Car_41_crop.jpg
│   │   │   │       ├── Car_42_crop.jpg
│   │   │   │       ├── Car_43_crop.jpg
│   │   │   │       ├── Car_44_crop.jpg
│   │   │   │       ├── Car_45_crop.jpg
│   │   │   │       ├── Car_46_crop.jpg
│   │   │   │       ├── Car_47_crop.jpg
│   │   │   │       ├── Car_48_crop.jpg
│   │   │   │       ├── Car_49_crop.jpg
│   │   │   │       ├── Car_4_crop.jpg
│   │   │   │       ├── Car_50_crop.jpg
│   │   │   │       ├── Car_51_crop.jpg
│   │   │   │       ├── Car_52_crop.jpg
│   │   │   │       ├── Car_53_crop.jpg
│   │   │   │       ├── Car_54_crop.jpg
│   │   │   │       ├── Car_55_crop.jpg
│   │   │   │       ├── Car_56_crop.jpg
│   │   │   │       ├── Car_57_crop.jpg
│   │   │   │       ├── Car_58_crop.jpg
│   │   │   │       ├── Car_59_crop.jpg
│   │   │   │       ├── Car_5_crop.jpg
│   │   │   │       ├── Car_60_crop.jpg
│   │   │   │       ├── Car_61_crop.jpg
│   │   │   │       ├── Car_62_crop.jpg
│   │   │   │       ├── Car_63_crop.jpg
│   │   │   │       ├── Car_64_crop.jpg
│   │   │   │       ├── Car_65_crop.jpg
│   │   │   │       ├── Car_66_crop.jpg
│   │   │   │       ├── Car_67_crop.jpg
│   │   │   │       ├── Car_68_crop.jpg
│   │   │   │       ├── Car_69_crop.jpg
│   │   │   │       ├── Car_6_crop.jpg
│   │   │   │       ├── Car_70_crop.jpg
│   │   │   │       ├── Car_71_crop.jpg
│   │   │   │       ├── Car_72_crop.jpg
│   │   │   │       ├── Car_73_crop.jpg
│   │   │   │       ├── Car_74_crop.jpg
│   │   │   │       ├── Car_75_crop.jpg
│   │   │   │       ├── Car_76_crop.jpg
│   │   │   │       ├── Car_77_crop.jpg
│   │   │   │       ├── Car_7_crop.jpg
│   │   │   │       ├── Car_8_crop.jpg
│   │   │   │       └── Car_9_crop.jpg
│   │   │   ├── objtrack_0012_1
│   │   │   │   ├── 0
│   │   │   │   │   ├── Cyclist_0_crop.jpg
│   │   │   │   │   ├── Cyclist_10_crop.jpg
│   │   │   │   │   ├── Cyclist_11_crop.jpg
│   │   │   │   │   ├── Cyclist_12_crop.jpg
│   │   │   │   │   ├── Cyclist_13_crop.jpg
│   │   │   │   │   ├── Cyclist_14_crop.jpg
│   │   │   │   │   ├── Cyclist_15_crop.jpg
│   │   │   │   │   ├── Cyclist_16_crop.jpg
│   │   │   │   │   ├── Cyclist_17_crop.jpg
│   │   │   │   │   ├── Cyclist_18_crop.jpg
│   │   │   │   │   ├── Cyclist_19_crop.jpg
│   │   │   │   │   ├── Cyclist_1_crop.jpg
│   │   │   │   │   ├── Cyclist_20_crop.jpg
│   │   │   │   │   ├── Cyclist_21_crop.jpg
│   │   │   │   │   ├── Cyclist_22_crop.jpg
│   │   │   │   │   ├── Cyclist_23_crop.jpg
│   │   │   │   │   ├── Cyclist_24_crop.jpg
│   │   │   │   │   ├── Cyclist_25_crop.jpg
│   │   │   │   │   ├── Cyclist_26_crop.jpg
│   │   │   │   │   ├── Cyclist_27_crop.jpg
│   │   │   │   │   ├── Cyclist_28_crop.jpg
│   │   │   │   │   ├── Cyclist_29_crop.jpg
│   │   │   │   │   ├── Cyclist_2_crop.jpg
│   │   │   │   │   ├── Cyclist_30_crop.jpg
│   │   │   │   │   ├── Cyclist_31_crop.jpg
│   │   │   │   │   ├── Cyclist_32_crop.jpg
│   │   │   │   │   ├── Cyclist_33_crop.jpg
│   │   │   │   │   ├── Cyclist_34_crop.jpg
│   │   │   │   │   ├── Cyclist_35_crop.jpg
│   │   │   │   │   ├── Cyclist_36_crop.jpg
│   │   │   │   │   ├── Cyclist_37_crop.jpg
│   │   │   │   │   ├── Cyclist_38_crop.jpg
│   │   │   │   │   ├── Cyclist_39_crop.jpg
│   │   │   │   │   ├── Cyclist_3_crop.jpg
│   │   │   │   │   ├── Cyclist_40_crop.jpg
│   │   │   │   │   ├── Cyclist_4_crop.jpg
│   │   │   │   │   ├── Cyclist_5_crop.jpg
│   │   │   │   │   ├── Cyclist_6_crop.jpg
│   │   │   │   │   ├── Cyclist_7_crop.jpg
│   │   │   │   │   ├── Cyclist_8_crop.jpg
│   │   │   │   │   └── Cyclist_9_crop.jpg
│   │   │   │   ├── 1
│   │   │   │   │   ├── Car_0_crop.jpg
│   │   │   │   │   ├── Car_10_crop.jpg
│   │   │   │   │   ├── Car_11_crop.jpg
│   │   │   │   │   ├── Car_12_crop.jpg
│   │   │   │   │   ├── Car_13_crop.jpg
│   │   │   │   │   ├── Car_14_crop.jpg
│   │   │   │   │   ├── Car_15_crop.jpg
│   │   │   │   │   ├── Car_16_crop.jpg
│   │   │   │   │   ├── Car_17_crop.jpg
│   │   │   │   │   ├── Car_18_crop.jpg
│   │   │   │   │   ├── Car_19_crop.jpg
│   │   │   │   │   ├── Car_1_crop.jpg
│   │   │   │   │   ├── Car_20_crop.jpg
│   │   │   │   │   ├── Car_21_crop.jpg
│   │   │   │   │   ├── Car_22_crop.jpg
│   │   │   │   │   ├── Car_23_crop.jpg
│   │   │   │   │   ├── Car_24_crop.jpg
│   │   │   │   │   ├── Car_25_crop.jpg
│   │   │   │   │   ├── Car_26_crop.jpg
│   │   │   │   │   ├── Car_27_crop.jpg
│   │   │   │   │   ├── Car_28_crop.jpg
│   │   │   │   │   ├── Car_29_crop.jpg
│   │   │   │   │   ├── Car_2_crop.jpg
│   │   │   │   │   ├── Car_30_crop.jpg
│   │   │   │   │   ├── Car_31_crop.jpg
│   │   │   │   │   ├── Car_32_crop.jpg
│   │   │   │   │   ├── Car_33_crop.jpg
│   │   │   │   │   ├── Car_34_crop.jpg
│   │   │   │   │   ├── Car_35_crop.jpg
│   │   │   │   │   ├── Car_36_crop.jpg
│   │   │   │   │   ├── Car_37_crop.jpg
│   │   │   │   │   ├── Car_38_crop.jpg
│   │   │   │   │   ├── Car_39_crop.jpg
│   │   │   │   │   ├── Car_3_crop.jpg
│   │   │   │   │   ├── Car_40_crop.jpg
│   │   │   │   │   ├── Car_41_crop.jpg
│   │   │   │   │   ├── Car_42_crop.jpg
│   │   │   │   │   ├── Car_43_crop.jpg
│   │   │   │   │   ├── Car_44_crop.jpg
│   │   │   │   │   ├── Car_45_crop.jpg
│   │   │   │   │   ├── Car_46_crop.jpg
│   │   │   │   │   ├── Car_47_crop.jpg
│   │   │   │   │   ├── Car_48_crop.jpg
│   │   │   │   │   ├── Car_49_crop.jpg
│   │   │   │   │   ├── Car_4_crop.jpg
│   │   │   │   │   ├── Car_50_crop.jpg
│   │   │   │   │   ├── Car_51_crop.jpg
│   │   │   │   │   ├── Car_52_crop.jpg
│   │   │   │   │   ├── Car_53_crop.jpg
│   │   │   │   │   ├── Car_54_crop.jpg
│   │   │   │   │   ├── Car_55_crop.jpg
│   │   │   │   │   ├── Car_56_crop.jpg
│   │   │   │   │   ├── Car_57_crop.jpg
│   │   │   │   │   ├── Car_58_crop.jpg
│   │   │   │   │   ├── Car_59_crop.jpg
│   │   │   │   │   ├── Car_5_crop.jpg
│   │   │   │   │   ├── Car_60_crop.jpg
│   │   │   │   │   ├── Car_61_crop.jpg
│   │   │   │   │   ├── Car_62_crop.jpg
│   │   │   │   │   ├── Car_63_crop.jpg
│   │   │   │   │   ├── Car_64_crop.jpg
│   │   │   │   │   ├── Car_65_crop.jpg
│   │   │   │   │   ├── Car_6_crop.jpg
│   │   │   │   │   ├── Car_7_crop.jpg
│   │   │   │   │   ├── Car_8_crop.jpg
│   │   │   │   │   └── Car_9_crop.jpg
│   │   │   │   ├── 2
│   │   │   │   │   ├── Pedestrian_13_crop.jpg
│   │   │   │   │   ├── Pedestrian_14_crop.jpg
│   │   │   │   │   ├── Pedestrian_15_crop.jpg
│   │   │   │   │   ├── Pedestrian_16_crop.jpg
│   │   │   │   │   ├── Pedestrian_17_crop.jpg
│   │   │   │   │   ├── Pedestrian_18_crop.jpg
│   │   │   │   │   ├── Pedestrian_19_crop.jpg
│   │   │   │   │   ├── Pedestrian_20_crop.jpg
│   │   │   │   │   ├── Pedestrian_21_crop.jpg
│   │   │   │   │   ├── Pedestrian_22_crop.jpg
│   │   │   │   │   ├── Pedestrian_23_crop.jpg
│   │   │   │   │   ├── Pedestrian_24_crop.jpg
│   │   │   │   │   ├── Pedestrian_25_crop.jpg
│   │   │   │   │   ├── Pedestrian_26_crop.jpg
│   │   │   │   │   ├── Pedestrian_27_crop.jpg
│   │   │   │   │   ├── Pedestrian_28_crop.jpg
│   │   │   │   │   ├── Pedestrian_29_crop.jpg
│   │   │   │   │   ├── Pedestrian_30_crop.jpg
│   │   │   │   │   ├── Pedestrian_31_crop.jpg
│   │   │   │   │   ├── Pedestrian_32_crop.jpg
│   │   │   │   │   ├── Pedestrian_33_crop.jpg
│   │   │   │   │   ├── Pedestrian_34_crop.jpg
│   │   │   │   │   ├── Pedestrian_35_crop.jpg
│   │   │   │   │   ├── Pedestrian_36_crop.jpg
│   │   │   │   │   ├── Pedestrian_37_crop.jpg
│   │   │   │   │   ├── Pedestrian_38_crop.jpg
│   │   │   │   │   ├── Pedestrian_39_crop.jpg
│   │   │   │   │   ├── Pedestrian_40_crop.jpg
│   │   │   │   │   ├── Pedestrian_41_crop.jpg
│   │   │   │   │   ├── Pedestrian_42_crop.jpg
│   │   │   │   │   ├── Pedestrian_43_crop.jpg
│   │   │   │   │   ├── Pedestrian_44_crop.jpg
│   │   │   │   │   ├── Pedestrian_45_crop.jpg
│   │   │   │   │   ├── Pedestrian_46_crop.jpg
│   │   │   │   │   ├── Pedestrian_47_crop.jpg
│   │   │   │   │   ├── Pedestrian_48_crop.jpg
│   │   │   │   │   ├── Pedestrian_49_crop.jpg
│   │   │   │   │   ├── Pedestrian_50_crop.jpg
│   │   │   │   │   ├── Pedestrian_51_crop.jpg
│   │   │   │   │   ├── Pedestrian_52_crop.jpg
│   │   │   │   │   ├── Pedestrian_53_crop.jpg
│   │   │   │   │   ├── Pedestrian_54_crop.jpg
│   │   │   │   │   ├── Pedestrian_55_crop.jpg
│   │   │   │   │   ├── Pedestrian_56_crop.jpg
│   │   │   │   │   ├── Pedestrian_57_crop.jpg
│   │   │   │   │   ├── Pedestrian_58_crop.jpg
│   │   │   │   │   ├── Pedestrian_59_crop.jpg
│   │   │   │   │   ├── Pedestrian_60_crop.jpg
│   │   │   │   │   ├── Pedestrian_61_crop.jpg
│   │   │   │   │   ├── Pedestrian_62_crop.jpg
│   │   │   │   │   ├── Pedestrian_63_crop.jpg
│   │   │   │   │   ├── Pedestrian_64_crop.jpg
│   │   │   │   │   ├── Pedestrian_65_crop.jpg
│   │   │   │   │   ├── Pedestrian_66_crop.jpg
│   │   │   │   │   ├── Pedestrian_67_crop.jpg
│   │   │   │   │   ├── Pedestrian_68_crop.jpg
│   │   │   │   │   ├── Pedestrian_69_crop.jpg
│   │   │   │   │   ├── Pedestrian_70_crop.jpg
│   │   │   │   │   ├── Pedestrian_71_crop.jpg
│   │   │   │   │   ├── Pedestrian_72_crop.jpg
│   │   │   │   │   ├── Pedestrian_73_crop.jpg
│   │   │   │   │   ├── Pedestrian_74_crop.jpg
│   │   │   │   │   ├── Pedestrian_75_crop.jpg
│   │   │   │   │   └── Pedestrian_76_crop.jpg
│   │   │   │   └── 3
│   │   │   │       ├── Car_0_crop.jpg
│   │   │   │       ├── Car_10_crop.jpg
│   │   │   │       ├── Car_11_crop.jpg
│   │   │   │       ├── Car_12_crop.jpg
│   │   │   │       ├── Car_13_crop.jpg
│   │   │   │       ├── Car_14_crop.jpg
│   │   │   │       ├── Car_15_crop.jpg
│   │   │   │       ├── Car_16_crop.jpg
│   │   │   │       ├── Car_17_crop.jpg
│   │   │   │       ├── Car_18_crop.jpg
│   │   │   │       ├── Car_19_crop.jpg
│   │   │   │       ├── Car_1_crop.jpg
│   │   │   │       ├── Car_20_crop.jpg
│   │   │   │       ├── Car_21_crop.jpg
│   │   │   │       ├── Car_22_crop.jpg
│   │   │   │       ├── Car_23_crop.jpg
│   │   │   │       ├── Car_24_crop.jpg
│   │   │   │       ├── Car_25_crop.jpg
│   │   │   │       ├── Car_26_crop.jpg
│   │   │   │       ├── Car_27_crop.jpg
│   │   │   │       ├── Car_28_crop.jpg
│   │   │   │       ├── Car_29_crop.jpg
│   │   │   │       ├── Car_2_crop.jpg
│   │   │   │       ├── Car_30_crop.jpg
│   │   │   │       ├── Car_31_crop.jpg
│   │   │   │       ├── Car_32_crop.jpg
│   │   │   │       ├── Car_33_crop.jpg
│   │   │   │       ├── Car_34_crop.jpg
│   │   │   │       ├── Car_35_crop.jpg
│   │   │   │       ├── Car_36_crop.jpg
│   │   │   │       ├── Car_37_crop.jpg
│   │   │   │       ├── Car_38_crop.jpg
│   │   │   │       ├── Car_39_crop.jpg
│   │   │   │       ├── Car_3_crop.jpg
│   │   │   │       ├── Car_40_crop.jpg
│   │   │   │       ├── Car_41_crop.jpg
│   │   │   │       ├── Car_42_crop.jpg
│   │   │   │       ├── Car_43_crop.jpg
│   │   │   │       ├── Car_44_crop.jpg
│   │   │   │       ├── Car_45_crop.jpg
│   │   │   │       ├── Car_46_crop.jpg
│   │   │   │       ├── Car_47_crop.jpg
│   │   │   │       ├── Car_48_crop.jpg
│   │   │   │       ├── Car_49_crop.jpg
│   │   │   │       ├── Car_4_crop.jpg
│   │   │   │       ├── Car_50_crop.jpg
│   │   │   │       ├── Car_51_crop.jpg
│   │   │   │       ├── Car_52_crop.jpg
│   │   │   │       ├── Car_53_crop.jpg
│   │   │   │       ├── Car_54_crop.jpg
│   │   │   │       ├── Car_55_crop.jpg
│   │   │   │       ├── Car_56_crop.jpg
│   │   │   │       ├── Car_57_crop.jpg
│   │   │   │       ├── Car_58_crop.jpg
│   │   │   │       ├── Car_59_crop.jpg
│   │   │   │       ├── Car_5_crop.jpg
│   │   │   │       ├── Car_60_crop.jpg
│   │   │   │       ├── Car_61_crop.jpg
│   │   │   │       ├── Car_62_crop.jpg
│   │   │   │       ├── Car_63_crop.jpg
│   │   │   │       ├── Car_64_crop.jpg
│   │   │   │       ├── Car_65_crop.jpg
│   │   │   │       ├── Car_66_crop.jpg
│   │   │   │       ├── Car_67_crop.jpg
│   │   │   │       ├── Car_68_crop.jpg
│   │   │   │       ├── Car_69_crop.jpg
│   │   │   │       ├── Car_6_crop.jpg
│   │   │   │       ├── Car_70_crop.jpg
│   │   │   │       ├── Car_71_crop.jpg
│   │   │   │       ├── Car_72_crop.jpg
│   │   │   │       ├── Car_73_crop.jpg
│   │   │   │       ├── Car_74_crop.jpg
│   │   │   │       ├── Car_75_crop.jpg
│   │   │   │       ├── Car_76_crop.jpg
│   │   │   │       ├── Car_77_crop.jpg
│   │   │   │       ├── Car_7_crop.jpg
│   │   │   │       ├── Car_8_crop.jpg
│   │   │   │       └── Car_9_crop.jpg
│   │   │   └── triplet_pair.txt
│   │   └── training
│   │       ├── gt_train_track_dict.json
│   │       ├── objtrack_0012
│   │       │   ├── 0
│   │       │   │   ├── Cyclist_0_crop.jpg
│   │       │   │   ├── Cyclist_10_crop.jpg
│   │       │   │   ├── Cyclist_11_crop.jpg
│   │       │   │   ├── Cyclist_12_crop.jpg
│   │       │   │   ├── Cyclist_13_crop.jpg
│   │       │   │   ├── Cyclist_14_crop.jpg
│   │       │   │   ├── Cyclist_15_crop.jpg
│   │       │   │   ├── Cyclist_16_crop.jpg
│   │       │   │   ├── Cyclist_17_crop.jpg
│   │       │   │   ├── Cyclist_18_crop.jpg
│   │       │   │   ├── Cyclist_19_crop.jpg
│   │       │   │   ├── Cyclist_1_crop.jpg
│   │       │   │   ├── Cyclist_20_crop.jpg
│   │       │   │   ├── Cyclist_21_crop.jpg
│   │       │   │   ├── Cyclist_22_crop.jpg
│   │       │   │   ├── Cyclist_23_crop.jpg
│   │       │   │   ├── Cyclist_24_crop.jpg
│   │       │   │   ├── Cyclist_25_crop.jpg
│   │       │   │   ├── Cyclist_26_crop.jpg
│   │       │   │   ├── Cyclist_27_crop.jpg
│   │       │   │   ├── Cyclist_28_crop.jpg
│   │       │   │   ├── Cyclist_29_crop.jpg
│   │       │   │   ├── Cyclist_2_crop.jpg
│   │       │   │   ├── Cyclist_30_crop.jpg
│   │       │   │   ├── Cyclist_31_crop.jpg
│   │       │   │   ├── Cyclist_32_crop.jpg
│   │       │   │   ├── Cyclist_33_crop.jpg
│   │       │   │   ├── Cyclist_34_crop.jpg
│   │       │   │   ├── Cyclist_35_crop.jpg
│   │       │   │   ├── Cyclist_36_crop.jpg
│   │       │   │   ├── Cyclist_37_crop.jpg
│   │       │   │   ├── Cyclist_38_crop.jpg
│   │       │   │   ├── Cyclist_39_crop.jpg
│   │       │   │   ├── Cyclist_3_crop.jpg
│   │       │   │   ├── Cyclist_40_crop.jpg
│   │       │   │   ├── Cyclist_4_crop.jpg
│   │       │   │   ├── Cyclist_5_crop.jpg
│   │       │   │   ├── Cyclist_6_crop.jpg
│   │       │   │   ├── Cyclist_7_crop.jpg
│   │       │   │   ├── Cyclist_8_crop.jpg
│   │       │   │   └── Cyclist_9_crop.jpg
│   │       │   ├── 1
│   │       │   │   ├── Car_0_crop.jpg
│   │       │   │   ├── Car_10_crop.jpg
│   │       │   │   ├── Car_11_crop.jpg
│   │       │   │   ├── Car_12_crop.jpg
│   │       │   │   ├── Car_13_crop.jpg
│   │       │   │   ├── Car_14_crop.jpg
│   │       │   │   ├── Car_15_crop.jpg
│   │       │   │   ├── Car_16_crop.jpg
│   │       │   │   ├── Car_17_crop.jpg
│   │       │   │   ├── Car_18_crop.jpg
│   │       │   │   ├── Car_19_crop.jpg
│   │       │   │   ├── Car_1_crop.jpg
│   │       │   │   ├── Car_20_crop.jpg
│   │       │   │   ├── Car_21_crop.jpg
│   │       │   │   ├── Car_22_crop.jpg
│   │       │   │   ├── Car_23_crop.jpg
│   │       │   │   ├── Car_24_crop.jpg
│   │       │   │   ├── Car_25_crop.jpg
│   │       │   │   ├── Car_26_crop.jpg
│   │       │   │   ├── Car_27_crop.jpg
│   │       │   │   ├── Car_28_crop.jpg
│   │       │   │   ├── Car_29_crop.jpg
│   │       │   │   ├── Car_2_crop.jpg
│   │       │   │   ├── Car_30_crop.jpg
│   │       │   │   ├── Car_31_crop.jpg
│   │       │   │   ├── Car_32_crop.jpg
│   │       │   │   ├── Car_33_crop.jpg
│   │       │   │   ├── Car_34_crop.jpg
│   │       │   │   ├── Car_35_crop.jpg
│   │       │   │   ├── Car_36_crop.jpg
│   │       │   │   ├── Car_37_crop.jpg
│   │       │   │   ├── Car_38_crop.jpg
│   │       │   │   ├── Car_39_crop.jpg
│   │       │   │   ├── Car_3_crop.jpg
│   │       │   │   ├── Car_40_crop.jpg
│   │       │   │   ├── Car_41_crop.jpg
│   │       │   │   ├── Car_42_crop.jpg
│   │       │   │   ├── Car_43_crop.jpg
│   │       │   │   ├── Car_44_crop.jpg
│   │       │   │   ├── Car_45_crop.jpg
│   │       │   │   ├── Car_46_crop.jpg
│   │       │   │   ├── Car_47_crop.jpg
│   │       │   │   ├── Car_48_crop.jpg
│   │       │   │   ├── Car_49_crop.jpg
│   │       │   │   ├── Car_4_crop.jpg
│   │       │   │   ├── Car_50_crop.jpg
│   │       │   │   ├── Car_51_crop.jpg
│   │       │   │   ├── Car_52_crop.jpg
│   │       │   │   ├── Car_53_crop.jpg
│   │       │   │   ├── Car_54_crop.jpg
│   │       │   │   ├── Car_55_crop.jpg
│   │       │   │   ├── Car_56_crop.jpg
│   │       │   │   ├── Car_57_crop.jpg
│   │       │   │   ├── Car_58_crop.jpg
│   │       │   │   ├── Car_59_crop.jpg
│   │       │   │   ├── Car_5_crop.jpg
│   │       │   │   ├── Car_60_crop.jpg
│   │       │   │   ├── Car_61_crop.jpg
│   │       │   │   ├── Car_62_crop.jpg
│   │       │   │   ├── Car_63_crop.jpg
│   │       │   │   ├── Car_64_crop.jpg
│   │       │   │   ├── Car_65_crop.jpg
│   │       │   │   ├── Car_6_crop.jpg
│   │       │   │   ├── Car_7_crop.jpg
│   │       │   │   ├── Car_8_crop.jpg
│   │       │   │   └── Car_9_crop.jpg
│   │       │   ├── 2
│   │       │   │   ├── Pedestrian_13_crop.jpg
│   │       │   │   ├── Pedestrian_14_crop.jpg
│   │       │   │   ├── Pedestrian_15_crop.jpg
│   │       │   │   ├── Pedestrian_16_crop.jpg
│   │       │   │   ├── Pedestrian_17_crop.jpg
│   │       │   │   ├── Pedestrian_18_crop.jpg
│   │       │   │   ├── Pedestrian_19_crop.jpg
│   │       │   │   ├── Pedestrian_20_crop.jpg
│   │       │   │   ├── Pedestrian_21_crop.jpg
│   │       │   │   ├── Pedestrian_22_crop.jpg
│   │       │   │   ├── Pedestrian_23_crop.jpg
│   │       │   │   ├── Pedestrian_24_crop.jpg
│   │       │   │   ├── Pedestrian_25_crop.jpg
│   │       │   │   ├── Pedestrian_26_crop.jpg
│   │       │   │   ├── Pedestrian_27_crop.jpg
│   │       │   │   ├── Pedestrian_28_crop.jpg
│   │       │   │   ├── Pedestrian_29_crop.jpg
│   │       │   │   ├── Pedestrian_30_crop.jpg
│   │       │   │   ├── Pedestrian_31_crop.jpg
│   │       │   │   ├── Pedestrian_32_crop.jpg
│   │       │   │   ├── Pedestrian_33_crop.jpg
│   │       │   │   ├── Pedestrian_34_crop.jpg
│   │       │   │   ├── Pedestrian_35_crop.jpg
│   │       │   │   ├── Pedestrian_36_crop.jpg
│   │       │   │   ├── Pedestrian_37_crop.jpg
│   │       │   │   ├── Pedestrian_38_crop.jpg
│   │       │   │   ├── Pedestrian_39_crop.jpg
│   │       │   │   ├── Pedestrian_40_crop.jpg
│   │       │   │   ├── Pedestrian_41_crop.jpg
│   │       │   │   ├── Pedestrian_42_crop.jpg
│   │       │   │   ├── Pedestrian_43_crop.jpg
│   │       │   │   ├── Pedestrian_44_crop.jpg
│   │       │   │   ├── Pedestrian_45_crop.jpg
│   │       │   │   ├── Pedestrian_46_crop.jpg
│   │       │   │   ├── Pedestrian_47_crop.jpg
│   │       │   │   ├── Pedestrian_48_crop.jpg
│   │       │   │   ├── Pedestrian_49_crop.jpg
│   │       │   │   ├── Pedestrian_50_crop.jpg
│   │       │   │   ├── Pedestrian_51_crop.jpg
│   │       │   │   ├── Pedestrian_52_crop.jpg
│   │       │   │   ├── Pedestrian_53_crop.jpg
│   │       │   │   ├── Pedestrian_54_crop.jpg
│   │       │   │   ├── Pedestrian_55_crop.jpg
│   │       │   │   ├── Pedestrian_56_crop.jpg
│   │       │   │   ├── Pedestrian_57_crop.jpg
│   │       │   │   ├── Pedestrian_58_crop.jpg
│   │       │   │   ├── Pedestrian_59_crop.jpg
│   │       │   │   ├── Pedestrian_60_crop.jpg
│   │       │   │   ├── Pedestrian_61_crop.jpg
│   │       │   │   ├── Pedestrian_62_crop.jpg
│   │       │   │   ├── Pedestrian_63_crop.jpg
│   │       │   │   ├── Pedestrian_64_crop.jpg
│   │       │   │   ├── Pedestrian_65_crop.jpg
│   │       │   │   ├── Pedestrian_66_crop.jpg
│   │       │   │   ├── Pedestrian_67_crop.jpg
│   │       │   │   ├── Pedestrian_68_crop.jpg
│   │       │   │   ├── Pedestrian_69_crop.jpg
│   │       │   │   ├── Pedestrian_70_crop.jpg
│   │       │   │   ├── Pedestrian_71_crop.jpg
│   │       │   │   ├── Pedestrian_72_crop.jpg
│   │       │   │   ├── Pedestrian_73_crop.jpg
│   │       │   │   ├── Pedestrian_74_crop.jpg
│   │       │   │   ├── Pedestrian_75_crop.jpg
│   │       │   │   └── Pedestrian_76_crop.jpg
│   │       │   └── 3
│   │       │       ├── Car_0_crop.jpg
│   │       │       ├── Car_10_crop.jpg
│   │       │       ├── Car_11_crop.jpg
│   │       │       ├── Car_12_crop.jpg
│   │       │       ├── Car_13_crop.jpg
│   │       │       ├── Car_14_crop.jpg
│   │       │       ├── Car_15_crop.jpg
│   │       │       ├── Car_16_crop.jpg
│   │       │       ├── Car_17_crop.jpg
│   │       │       ├── Car_18_crop.jpg
│   │       │       ├── Car_19_crop.jpg
│   │       │       ├── Car_1_crop.jpg
│   │       │       ├── Car_20_crop.jpg
│   │       │       ├── Car_21_crop.jpg
│   │       │       ├── Car_22_crop.jpg
│   │       │       ├── Car_23_crop.jpg
│   │       │       ├── Car_24_crop.jpg
│   │       │       ├── Car_25_crop.jpg
│   │       │       ├── Car_26_crop.jpg
│   │       │       ├── Car_27_crop.jpg
│   │       │       ├── Car_28_crop.jpg
│   │       │       ├── Car_29_crop.jpg
│   │       │       ├── Car_2_crop.jpg
│   │       │       ├── Car_30_crop.jpg
│   │       │       ├── Car_31_crop.jpg
│   │       │       ├── Car_32_crop.jpg
│   │       │       ├── Car_33_crop.jpg
│   │       │       ├── Car_34_crop.jpg
│   │       │       ├── Car_35_crop.jpg
│   │       │       ├── Car_36_crop.jpg
│   │       │       ├── Car_37_crop.jpg
│   │       │       ├── Car_38_crop.jpg
│   │       │       ├── Car_39_crop.jpg
│   │       │       ├── Car_3_crop.jpg
│   │       │       ├── Car_40_crop.jpg
│   │       │       ├── Car_41_crop.jpg
│   │       │       ├── Car_42_crop.jpg
│   │       │       ├── Car_43_crop.jpg
│   │       │       ├── Car_44_crop.jpg
│   │       │       ├── Car_45_crop.jpg
│   │       │       ├── Car_46_crop.jpg
│   │       │       ├── Car_47_crop.jpg
│   │       │       ├── Car_48_crop.jpg
│   │       │       ├── Car_49_crop.jpg
│   │       │       ├── Car_4_crop.jpg
│   │       │       ├── Car_50_crop.jpg
│   │       │       ├── Car_51_crop.jpg
│   │       │       ├── Car_52_crop.jpg
│   │       │       ├── Car_53_crop.jpg
│   │       │       ├── Car_54_crop.jpg
│   │       │       ├── Car_55_crop.jpg
│   │       │       ├── Car_56_crop.jpg
│   │       │       ├── Car_57_crop.jpg
│   │       │       ├── Car_58_crop.jpg
│   │       │       ├── Car_59_crop.jpg
│   │       │       ├── Car_5_crop.jpg
│   │       │       ├── Car_60_crop.jpg
│   │       │       ├── Car_61_crop.jpg
│   │       │       ├── Car_62_crop.jpg
│   │       │       ├── Car_63_crop.jpg
│   │       │       ├── Car_64_crop.jpg
│   │       │       ├── Car_65_crop.jpg
│   │       │       ├── Car_66_crop.jpg
│   │       │       ├── Car_67_crop.jpg
│   │       │       ├── Car_68_crop.jpg
│   │       │       ├── Car_69_crop.jpg
│   │       │       ├── Car_6_crop.jpg
│   │       │       ├── Car_70_crop.jpg
│   │       │       ├── Car_71_crop.jpg
│   │       │       ├── Car_72_crop.jpg
│   │       │       ├── Car_73_crop.jpg
│   │       │       ├── Car_74_crop.jpg
│   │       │       ├── Car_75_crop.jpg
│   │       │       ├── Car_76_crop.jpg
│   │       │       ├── Car_77_crop.jpg
│   │       │       ├── Car_7_crop.jpg
│   │       │       ├── Car_8_crop.jpg
│   │       │       └── Car_9_crop.jpg
│   │       ├── objtrack_0012_1
│   │       │   ├── 0
│   │       │   │   ├── Cyclist_0_crop.jpg
│   │       │   │   ├── Cyclist_10_crop.jpg
│   │       │   │   ├── Cyclist_11_crop.jpg
│   │       │   │   ├── Cyclist_12_crop.jpg
│   │       │   │   ├── Cyclist_13_crop.jpg
│   │       │   │   ├── Cyclist_14_crop.jpg
│   │       │   │   ├── Cyclist_15_crop.jpg
│   │       │   │   ├── Cyclist_16_crop.jpg
│   │       │   │   ├── Cyclist_17_crop.jpg
│   │       │   │   ├── Cyclist_18_crop.jpg
│   │       │   │   ├── Cyclist_19_crop.jpg
│   │       │   │   ├── Cyclist_1_crop.jpg
│   │       │   │   ├── Cyclist_20_crop.jpg
│   │       │   │   ├── Cyclist_21_crop.jpg
│   │       │   │   ├── Cyclist_22_crop.jpg
│   │       │   │   ├── Cyclist_23_crop.jpg
│   │       │   │   ├── Cyclist_24_crop.jpg
│   │       │   │   ├── Cyclist_25_crop.jpg
│   │       │   │   ├── Cyclist_26_crop.jpg
│   │       │   │   ├── Cyclist_27_crop.jpg
│   │       │   │   ├── Cyclist_28_crop.jpg
│   │       │   │   ├── Cyclist_29_crop.jpg
│   │       │   │   ├── Cyclist_2_crop.jpg
│   │       │   │   ├── Cyclist_30_crop.jpg
│   │       │   │   ├── Cyclist_31_crop.jpg
│   │       │   │   ├── Cyclist_32_crop.jpg
│   │       │   │   ├── Cyclist_33_crop.jpg
│   │       │   │   ├── Cyclist_34_crop.jpg
│   │       │   │   ├── Cyclist_35_crop.jpg
│   │       │   │   ├── Cyclist_36_crop.jpg
│   │       │   │   ├── Cyclist_37_crop.jpg
│   │       │   │   ├── Cyclist_38_crop.jpg
│   │       │   │   ├── Cyclist_39_crop.jpg
│   │       │   │   ├── Cyclist_3_crop.jpg
│   │       │   │   ├── Cyclist_40_crop.jpg
│   │       │   │   ├── Cyclist_4_crop.jpg
│   │       │   │   ├── Cyclist_5_crop.jpg
│   │       │   │   ├── Cyclist_6_crop.jpg
│   │       │   │   ├── Cyclist_7_crop.jpg
│   │       │   │   ├── Cyclist_8_crop.jpg
│   │       │   │   └── Cyclist_9_crop.jpg
│   │       │   ├── 1
│   │       │   │   ├── Car_0_crop.jpg
│   │       │   │   ├── Car_10_crop.jpg
│   │       │   │   ├── Car_11_crop.jpg
│   │       │   │   ├── Car_12_crop.jpg
│   │       │   │   ├── Car_13_crop.jpg
│   │       │   │   ├── Car_14_crop.jpg
│   │       │   │   ├── Car_15_crop.jpg
│   │       │   │   ├── Car_16_crop.jpg
│   │       │   │   ├── Car_17_crop.jpg
│   │       │   │   ├── Car_18_crop.jpg
│   │       │   │   ├── Car_19_crop.jpg
│   │       │   │   ├── Car_1_crop.jpg
│   │       │   │   ├── Car_20_crop.jpg
│   │       │   │   ├── Car_21_crop.jpg
│   │       │   │   ├── Car_22_crop.jpg
│   │       │   │   ├── Car_23_crop.jpg
│   │       │   │   ├── Car_24_crop.jpg
│   │       │   │   ├── Car_25_crop.jpg
│   │       │   │   ├── Car_26_crop.jpg
│   │       │   │   ├── Car_27_crop.jpg
│   │       │   │   ├── Car_28_crop.jpg
│   │       │   │   ├── Car_29_crop.jpg
│   │       │   │   ├── Car_2_crop.jpg
│   │       │   │   ├── Car_30_crop.jpg
│   │       │   │   ├── Car_31_crop.jpg
│   │       │   │   ├── Car_32_crop.jpg
│   │       │   │   ├── Car_33_crop.jpg
│   │       │   │   ├── Car_34_crop.jpg
│   │       │   │   ├── Car_35_crop.jpg
│   │       │   │   ├── Car_36_crop.jpg
│   │       │   │   ├── Car_37_crop.jpg
│   │       │   │   ├── Car_38_crop.jpg
│   │       │   │   ├── Car_39_crop.jpg
│   │       │   │   ├── Car_3_crop.jpg
│   │       │   │   ├── Car_40_crop.jpg
│   │       │   │   ├── Car_41_crop.jpg
│   │       │   │   ├── Car_42_crop.jpg
│   │       │   │   ├── Car_43_crop.jpg
│   │       │   │   ├── Car_44_crop.jpg
│   │       │   │   ├── Car_45_crop.jpg
│   │       │   │   ├── Car_46_crop.jpg
│   │       │   │   ├── Car_47_crop.jpg
│   │       │   │   ├── Car_48_crop.jpg
│   │       │   │   ├── Car_49_crop.jpg
│   │       │   │   ├── Car_4_crop.jpg
│   │       │   │   ├── Car_50_crop.jpg
│   │       │   │   ├── Car_51_crop.jpg
│   │       │   │   ├── Car_52_crop.jpg
│   │       │   │   ├── Car_53_crop.jpg
│   │       │   │   ├── Car_54_crop.jpg
│   │       │   │   ├── Car_55_crop.jpg
│   │       │   │   ├── Car_56_crop.jpg
│   │       │   │   ├── Car_57_crop.jpg
│   │       │   │   ├── Car_58_crop.jpg
│   │       │   │   ├── Car_59_crop.jpg
│   │       │   │   ├── Car_5_crop.jpg
│   │       │   │   ├── Car_60_crop.jpg
│   │       │   │   ├── Car_61_crop.jpg
│   │       │   │   ├── Car_62_crop.jpg
│   │       │   │   ├── Car_63_crop.jpg
│   │       │   │   ├── Car_64_crop.jpg
│   │       │   │   ├── Car_65_crop.jpg
│   │       │   │   ├── Car_6_crop.jpg
│   │       │   │   ├── Car_7_crop.jpg
│   │       │   │   ├── Car_8_crop.jpg
│   │       │   │   └── Car_9_crop.jpg
│   │       │   ├── 2
│   │       │   │   ├── Pedestrian_13_crop.jpg
│   │       │   │   ├── Pedestrian_14_crop.jpg
│   │       │   │   ├── Pedestrian_15_crop.jpg
│   │       │   │   ├── Pedestrian_16_crop.jpg
│   │       │   │   ├── Pedestrian_17_crop.jpg
│   │       │   │   ├── Pedestrian_18_crop.jpg
│   │       │   │   ├── Pedestrian_19_crop.jpg
│   │       │   │   ├── Pedestrian_20_crop.jpg
│   │       │   │   ├── Pedestrian_21_crop.jpg
│   │       │   │   ├── Pedestrian_22_crop.jpg
│   │       │   │   ├── Pedestrian_23_crop.jpg
│   │       │   │   ├── Pedestrian_24_crop.jpg
│   │       │   │   ├── Pedestrian_25_crop.jpg
│   │       │   │   ├── Pedestrian_26_crop.jpg
│   │       │   │   ├── Pedestrian_27_crop.jpg
│   │       │   │   ├── Pedestrian_28_crop.jpg
│   │       │   │   ├── Pedestrian_29_crop.jpg
│   │       │   │   ├── Pedestrian_30_crop.jpg
│   │       │   │   ├── Pedestrian_31_crop.jpg
│   │       │   │   ├── Pedestrian_32_crop.jpg
│   │       │   │   ├── Pedestrian_33_crop.jpg
│   │       │   │   ├── Pedestrian_34_crop.jpg
│   │       │   │   ├── Pedestrian_35_crop.jpg
│   │       │   │   ├── Pedestrian_36_crop.jpg
│   │       │   │   ├── Pedestrian_37_crop.jpg
│   │       │   │   ├── Pedestrian_38_crop.jpg
│   │       │   │   ├── Pedestrian_39_crop.jpg
│   │       │   │   ├── Pedestrian_40_crop.jpg
│   │       │   │   ├── Pedestrian_41_crop.jpg
│   │       │   │   ├── Pedestrian_42_crop.jpg
│   │       │   │   ├── Pedestrian_43_crop.jpg
│   │       │   │   ├── Pedestrian_44_crop.jpg
│   │       │   │   ├── Pedestrian_45_crop.jpg
│   │       │   │   ├── Pedestrian_46_crop.jpg
│   │       │   │   ├── Pedestrian_47_crop.jpg
│   │       │   │   ├── Pedestrian_48_crop.jpg
│   │       │   │   ├── Pedestrian_49_crop.jpg
│   │       │   │   ├── Pedestrian_50_crop.jpg
│   │       │   │   ├── Pedestrian_51_crop.jpg
│   │       │   │   ├── Pedestrian_52_crop.jpg
│   │       │   │   ├── Pedestrian_53_crop.jpg
│   │       │   │   ├── Pedestrian_54_crop.jpg
│   │       │   │   ├── Pedestrian_55_crop.jpg
│   │       │   │   ├── Pedestrian_56_crop.jpg
│   │       │   │   ├── Pedestrian_57_crop.jpg
│   │       │   │   ├── Pedestrian_58_crop.jpg
│   │       │   │   ├── Pedestrian_59_crop.jpg
│   │       │   │   ├── Pedestrian_60_crop.jpg
│   │       │   │   ├── Pedestrian_61_crop.jpg
│   │       │   │   ├── Pedestrian_62_crop.jpg
│   │       │   │   ├── Pedestrian_63_crop.jpg
│   │       │   │   ├── Pedestrian_64_crop.jpg
│   │       │   │   ├── Pedestrian_65_crop.jpg
│   │       │   │   ├── Pedestrian_66_crop.jpg
│   │       │   │   ├── Pedestrian_67_crop.jpg
│   │       │   │   ├── Pedestrian_68_crop.jpg
│   │       │   │   ├── Pedestrian_69_crop.jpg
│   │       │   │   ├── Pedestrian_70_crop.jpg
│   │       │   │   ├── Pedestrian_71_crop.jpg
│   │       │   │   ├── Pedestrian_72_crop.jpg
│   │       │   │   ├── Pedestrian_73_crop.jpg
│   │       │   │   ├── Pedestrian_74_crop.jpg
│   │       │   │   ├── Pedestrian_75_crop.jpg
│   │       │   │   └── Pedestrian_76_crop.jpg
│   │       │   └── 3
│   │       │       ├── Car_0_crop.jpg
│   │       │       ├── Car_10_crop.jpg
│   │       │       ├── Car_11_crop.jpg
│   │       │       ├── Car_12_crop.jpg
│   │       │       ├── Car_13_crop.jpg
│   │       │       ├── Car_14_crop.jpg
│   │       │       ├── Car_15_crop.jpg
│   │       │       ├── Car_16_crop.jpg
│   │       │       ├── Car_17_crop.jpg
│   │       │       ├── Car_18_crop.jpg
│   │       │       ├── Car_19_crop.jpg
│   │       │       ├── Car_1_crop.jpg
│   │       │       ├── Car_20_crop.jpg
│   │       │       ├── Car_21_crop.jpg
│   │       │       ├── Car_22_crop.jpg
│   │       │       ├── Car_23_crop.jpg
│   │       │       ├── Car_24_crop.jpg
│   │       │       ├── Car_25_crop.jpg
│   │       │       ├── Car_26_crop.jpg
│   │       │       ├── Car_27_crop.jpg
│   │       │       ├── Car_28_crop.jpg
│   │       │       ├── Car_29_crop.jpg
│   │       │       ├── Car_2_crop.jpg
│   │       │       ├── Car_30_crop.jpg
│   │       │       ├── Car_31_crop.jpg
│   │       │       ├── Car_32_crop.jpg
│   │       │       ├── Car_33_crop.jpg
│   │       │       ├── Car_34_crop.jpg
│   │       │       ├── Car_35_crop.jpg
│   │       │       ├── Car_36_crop.jpg
│   │       │       ├── Car_37_crop.jpg
│   │       │       ├── Car_38_crop.jpg
│   │       │       ├── Car_39_crop.jpg
│   │       │       ├── Car_3_crop.jpg
│   │       │       ├── Car_40_crop.jpg
│   │       │       ├── Car_41_crop.jpg
│   │       │       ├── Car_42_crop.jpg
│   │       │       ├── Car_43_crop.jpg
│   │       │       ├── Car_44_crop.jpg
│   │       │       ├── Car_45_crop.jpg
│   │       │       ├── Car_46_crop.jpg
│   │       │       ├── Car_47_crop.jpg
│   │       │       ├── Car_48_crop.jpg
│   │       │       ├── Car_49_crop.jpg
│   │       │       ├── Car_4_crop.jpg
│   │       │       ├── Car_50_crop.jpg
│   │       │       ├── Car_51_crop.jpg
│   │       │       ├── Car_52_crop.jpg
│   │       │       ├── Car_53_crop.jpg
│   │       │       ├── Car_54_crop.jpg
│   │       │       ├── Car_55_crop.jpg
│   │       │       ├── Car_56_crop.jpg
│   │       │       ├── Car_57_crop.jpg
│   │       │       ├── Car_58_crop.jpg
│   │       │       ├── Car_59_crop.jpg
│   │       │       ├── Car_5_crop.jpg
│   │       │       ├── Car_60_crop.jpg
│   │       │       ├── Car_61_crop.jpg
│   │       │       ├── Car_62_crop.jpg
│   │       │       ├── Car_63_crop.jpg
│   │       │       ├── Car_64_crop.jpg
│   │       │       ├── Car_65_crop.jpg
│   │       │       ├── Car_66_crop.jpg
│   │       │       ├── Car_67_crop.jpg
│   │       │       ├── Car_68_crop.jpg
│   │       │       ├── Car_69_crop.jpg
│   │       │       ├── Car_6_crop.jpg
│   │       │       ├── Car_70_crop.jpg
│   │       │       ├── Car_71_crop.jpg
│   │       │       ├── Car_72_crop.jpg
│   │       │       ├── Car_73_crop.jpg
│   │       │       ├── Car_74_crop.jpg
│   │       │       ├── Car_75_crop.jpg
│   │       │       ├── Car_76_crop.jpg
│   │       │       ├── Car_77_crop.jpg
│   │       │       ├── Car_7_crop.jpg
│   │       │       ├── Car_8_crop.jpg
│   │       │       └── Car_9_crop.jpg
│   │       ├── objtrack1_0012
│   │       │   ├── 0
│   │       │   │   ├── Cyclist_0_crop.jpg
│   │       │   │   ├── Cyclist_10_crop.jpg
│   │       │   │   ├── Cyclist_11_crop.jpg
│   │       │   │   ├── Cyclist_12_crop.jpg
│   │       │   │   ├── Cyclist_13_crop.jpg
│   │       │   │   ├── Cyclist_14_crop.jpg
│   │       │   │   ├── Cyclist_15_crop.jpg
│   │       │   │   ├── Cyclist_16_crop.jpg
│   │       │   │   ├── Cyclist_17_crop.jpg
│   │       │   │   ├── Cyclist_18_crop.jpg
│   │       │   │   ├── Cyclist_19_crop.jpg
│   │       │   │   ├── Cyclist_1_crop.jpg
│   │       │   │   ├── Cyclist_20_crop.jpg
│   │       │   │   ├── Cyclist_21_crop.jpg
│   │       │   │   ├── Cyclist_22_crop.jpg
│   │       │   │   ├── Cyclist_23_crop.jpg
│   │       │   │   ├── Cyclist_24_crop.jpg
│   │       │   │   ├── Cyclist_25_crop.jpg
│   │       │   │   ├── Cyclist_26_crop.jpg
│   │       │   │   ├── Cyclist_27_crop.jpg
│   │       │   │   ├── Cyclist_28_crop.jpg
│   │       │   │   ├── Cyclist_29_crop.jpg
│   │       │   │   ├── Cyclist_2_crop.jpg
│   │       │   │   ├── Cyclist_30_crop.jpg
│   │       │   │   ├── Cyclist_31_crop.jpg
│   │       │   │   ├── Cyclist_32_crop.jpg
│   │       │   │   ├── Cyclist_33_crop.jpg
│   │       │   │   ├── Cyclist_34_crop.jpg
│   │       │   │   ├── Cyclist_35_crop.jpg
│   │       │   │   ├── Cyclist_36_crop.jpg
│   │       │   │   ├── Cyclist_37_crop.jpg
│   │       │   │   ├── Cyclist_38_crop.jpg
│   │       │   │   ├── Cyclist_39_crop.jpg
│   │       │   │   ├── Cyclist_3_crop.jpg
│   │       │   │   ├── Cyclist_40_crop.jpg
│   │       │   │   ├── Cyclist_4_crop.jpg
│   │       │   │   ├── Cyclist_5_crop.jpg
│   │       │   │   ├── Cyclist_6_crop.jpg
│   │       │   │   ├── Cyclist_7_crop.jpg
│   │       │   │   ├── Cyclist_8_crop.jpg
│   │       │   │   └── Cyclist_9_crop.jpg
│   │       │   ├── 1
│   │       │   │   ├── Car_0_crop.jpg
│   │       │   │   ├── Car_10_crop.jpg
│   │       │   │   ├── Car_11_crop.jpg
│   │       │   │   ├── Car_12_crop.jpg
│   │       │   │   ├── Car_13_crop.jpg
│   │       │   │   ├── Car_14_crop.jpg
│   │       │   │   ├── Car_15_crop.jpg
│   │       │   │   ├── Car_16_crop.jpg
│   │       │   │   ├── Car_17_crop.jpg
│   │       │   │   ├── Car_18_crop.jpg
│   │       │   │   ├── Car_19_crop.jpg
│   │       │   │   ├── Car_1_crop.jpg
│   │       │   │   ├── Car_20_crop.jpg
│   │       │   │   ├── Car_21_crop.jpg
│   │       │   │   ├── Car_22_crop.jpg
│   │       │   │   ├── Car_23_crop.jpg
│   │       │   │   ├── Car_24_crop.jpg
│   │       │   │   ├── Car_25_crop.jpg
│   │       │   │   ├── Car_26_crop.jpg
│   │       │   │   ├── Car_27_crop.jpg
│   │       │   │   ├── Car_28_crop.jpg
│   │       │   │   ├── Car_29_crop.jpg
│   │       │   │   ├── Car_2_crop.jpg
│   │       │   │   ├── Car_30_crop.jpg
│   │       │   │   ├── Car_31_crop.jpg
│   │       │   │   ├── Car_32_crop.jpg
│   │       │   │   ├── Car_33_crop.jpg
│   │       │   │   ├── Car_34_crop.jpg
│   │       │   │   ├── Car_35_crop.jpg
│   │       │   │   ├── Car_36_crop.jpg
│   │       │   │   ├── Car_37_crop.jpg
│   │       │   │   ├── Car_38_crop.jpg
│   │       │   │   ├── Car_39_crop.jpg
│   │       │   │   ├── Car_3_crop.jpg
│   │       │   │   ├── Car_40_crop.jpg
│   │       │   │   ├── Car_41_crop.jpg
│   │       │   │   ├── Car_42_crop.jpg
│   │       │   │   ├── Car_43_crop.jpg
│   │       │   │   ├── Car_44_crop.jpg
│   │       │   │   ├── Car_45_crop.jpg
│   │       │   │   ├── Car_46_crop.jpg
│   │       │   │   ├── Car_47_crop.jpg
│   │       │   │   ├── Car_48_crop.jpg
│   │       │   │   ├── Car_49_crop.jpg
│   │       │   │   ├── Car_4_crop.jpg
│   │       │   │   ├── Car_50_crop.jpg
│   │       │   │   ├── Car_51_crop.jpg
│   │       │   │   ├── Car_52_crop.jpg
│   │       │   │   ├── Car_53_crop.jpg
│   │       │   │   ├── Car_54_crop.jpg
│   │       │   │   ├── Car_55_crop.jpg
│   │       │   │   ├── Car_56_crop.jpg
│   │       │   │   ├── Car_57_crop.jpg
│   │       │   │   ├── Car_58_crop.jpg
│   │       │   │   ├── Car_59_crop.jpg
│   │       │   │   ├── Car_5_crop.jpg
│   │       │   │   ├── Car_60_crop.jpg
│   │       │   │   ├── Car_61_crop.jpg
│   │       │   │   ├── Car_62_crop.jpg
│   │       │   │   ├── Car_63_crop.jpg
│   │       │   │   ├── Car_64_crop.jpg
│   │       │   │   ├── Car_65_crop.jpg
│   │       │   │   ├── Car_6_crop.jpg
│   │       │   │   ├── Car_7_crop.jpg
│   │       │   │   ├── Car_8_crop.jpg
│   │       │   │   └── Car_9_crop.jpg
│   │       │   ├── 2
│   │       │   │   ├── Pedestrian_13_crop.jpg
│   │       │   │   ├── Pedestrian_14_crop.jpg
│   │       │   │   ├── Pedestrian_15_crop.jpg
│   │       │   │   ├── Pedestrian_16_crop.jpg
│   │       │   │   ├── Pedestrian_17_crop.jpg
│   │       │   │   ├── Pedestrian_18_crop.jpg
│   │       │   │   ├── Pedestrian_19_crop.jpg
│   │       │   │   ├── Pedestrian_20_crop.jpg
│   │       │   │   ├── Pedestrian_21_crop.jpg
│   │       │   │   ├── Pedestrian_22_crop.jpg
│   │       │   │   ├── Pedestrian_23_crop.jpg
│   │       │   │   ├── Pedestrian_24_crop.jpg
│   │       │   │   ├── Pedestrian_25_crop.jpg
│   │       │   │   ├── Pedestrian_26_crop.jpg
│   │       │   │   ├── Pedestrian_27_crop.jpg
│   │       │   │   ├── Pedestrian_28_crop.jpg
│   │       │   │   ├── Pedestrian_29_crop.jpg
│   │       │   │   ├── Pedestrian_30_crop.jpg
│   │       │   │   ├── Pedestrian_31_crop.jpg
│   │       │   │   ├── Pedestrian_32_crop.jpg
│   │       │   │   ├── Pedestrian_33_crop.jpg
│   │       │   │   ├── Pedestrian_34_crop.jpg
│   │       │   │   ├── Pedestrian_35_crop.jpg
│   │       │   │   ├── Pedestrian_36_crop.jpg
│   │       │   │   ├── Pedestrian_37_crop.jpg
│   │       │   │   ├── Pedestrian_38_crop.jpg
│   │       │   │   ├── Pedestrian_39_crop.jpg
│   │       │   │   ├── Pedestrian_40_crop.jpg
│   │       │   │   ├── Pedestrian_41_crop.jpg
│   │       │   │   ├── Pedestrian_42_crop.jpg
│   │       │   │   ├── Pedestrian_43_crop.jpg
│   │       │   │   ├── Pedestrian_44_crop.jpg
│   │       │   │   ├── Pedestrian_45_crop.jpg
│   │       │   │   ├── Pedestrian_46_crop.jpg
│   │       │   │   ├── Pedestrian_47_crop.jpg
│   │       │   │   ├── Pedestrian_48_crop.jpg
│   │       │   │   ├── Pedestrian_49_crop.jpg
│   │       │   │   ├── Pedestrian_50_crop.jpg
│   │       │   │   ├── Pedestrian_51_crop.jpg
│   │       │   │   ├── Pedestrian_52_crop.jpg
│   │       │   │   ├── Pedestrian_53_crop.jpg
│   │       │   │   ├── Pedestrian_54_crop.jpg
│   │       │   │   ├── Pedestrian_55_crop.jpg
│   │       │   │   ├── Pedestrian_56_crop.jpg
│   │       │   │   ├── Pedestrian_57_crop.jpg
│   │       │   │   ├── Pedestrian_58_crop.jpg
│   │       │   │   ├── Pedestrian_59_crop.jpg
│   │       │   │   ├── Pedestrian_60_crop.jpg
│   │       │   │   ├── Pedestrian_61_crop.jpg
│   │       │   │   ├── Pedestrian_62_crop.jpg
│   │       │   │   ├── Pedestrian_63_crop.jpg
│   │       │   │   ├── Pedestrian_64_crop.jpg
│   │       │   │   ├── Pedestrian_65_crop.jpg
│   │       │   │   ├── Pedestrian_66_crop.jpg
│   │       │   │   ├── Pedestrian_67_crop.jpg
│   │       │   │   ├── Pedestrian_68_crop.jpg
│   │       │   │   ├── Pedestrian_69_crop.jpg
│   │       │   │   ├── Pedestrian_70_crop.jpg
│   │       │   │   ├── Pedestrian_71_crop.jpg
│   │       │   │   ├── Pedestrian_72_crop.jpg
│   │       │   │   ├── Pedestrian_73_crop.jpg
│   │       │   │   ├── Pedestrian_74_crop.jpg
│   │       │   │   ├── Pedestrian_75_crop.jpg
│   │       │   │   └── Pedestrian_76_crop.jpg
│   │       │   └── 3
│   │       │       ├── Car_0_crop.jpg
│   │       │       ├── Car_10_crop.jpg
│   │       │       ├── Car_11_crop.jpg
│   │       │       ├── Car_12_crop.jpg
│   │       │       ├── Car_13_crop.jpg
│   │       │       ├── Car_14_crop.jpg
│   │       │       ├── Car_15_crop.jpg
│   │       │       ├── Car_16_crop.jpg
│   │       │       ├── Car_17_crop.jpg
│   │       │       ├── Car_18_crop.jpg
│   │       │       ├── Car_19_crop.jpg
│   │       │       ├── Car_1_crop.jpg
│   │       │       ├── Car_20_crop.jpg
│   │       │       ├── Car_21_crop.jpg
│   │       │       ├── Car_22_crop.jpg
│   │       │       ├── Car_23_crop.jpg
│   │       │       ├── Car_24_crop.jpg
│   │       │       ├── Car_25_crop.jpg
│   │       │       ├── Car_26_crop.jpg
│   │       │       ├── Car_27_crop.jpg
│   │       │       ├── Car_28_crop.jpg
│   │       │       ├── Car_29_crop.jpg
│   │       │       ├── Car_2_crop.jpg
│   │       │       ├── Car_30_crop.jpg
│   │       │       ├── Car_31_crop.jpg
│   │       │       ├── Car_32_crop.jpg
│   │       │       ├── Car_33_crop.jpg
│   │       │       ├── Car_34_crop.jpg
│   │       │       ├── Car_35_crop.jpg
│   │       │       ├── Car_36_crop.jpg
│   │       │       ├── Car_37_crop.jpg
│   │       │       ├── Car_38_crop.jpg
│   │       │       ├── Car_39_crop.jpg
│   │       │       ├── Car_3_crop.jpg
│   │       │       ├── Car_40_crop.jpg
│   │       │       ├── Car_41_crop.jpg
│   │       │       ├── Car_42_crop.jpg
│   │       │       ├── Car_43_crop.jpg
│   │       │       ├── Car_44_crop.jpg
│   │       │       ├── Car_45_crop.jpg
│   │       │       ├── Car_46_crop.jpg
│   │       │       ├── Car_47_crop.jpg
│   │       │       ├── Car_48_crop.jpg
│   │       │       ├── Car_49_crop.jpg
│   │       │       ├── Car_4_crop.jpg
│   │       │       ├── Car_50_crop.jpg
│   │       │       ├── Car_51_crop.jpg
│   │       │       ├── Car_52_crop.jpg
│   │       │       ├── Car_53_crop.jpg
│   │       │       ├── Car_54_crop.jpg
│   │       │       ├── Car_55_crop.jpg
│   │       │       ├── Car_56_crop.jpg
│   │       │       ├── Car_57_crop.jpg
│   │       │       ├── Car_58_crop.jpg
│   │       │       ├── Car_59_crop.jpg
│   │       │       ├── Car_5_crop.jpg
│   │       │       ├── Car_60_crop.jpg
│   │       │       ├── Car_61_crop.jpg
│   │       │       ├── Car_62_crop.jpg
│   │       │       ├── Car_63_crop.jpg
│   │       │       ├── Car_64_crop.jpg
│   │       │       ├── Car_65_crop.jpg
│   │       │       ├── Car_66_crop.jpg
│   │       │       ├── Car_67_crop.jpg
│   │       │       ├── Car_68_crop.jpg
│   │       │       ├── Car_69_crop.jpg
│   │       │       ├── Car_6_crop.jpg
│   │       │       ├── Car_70_crop.jpg
│   │       │       ├── Car_71_crop.jpg
│   │       │       ├── Car_72_crop.jpg
│   │       │       ├── Car_73_crop.jpg
│   │       │       ├── Car_74_crop.jpg
│   │       │       ├── Car_75_crop.jpg
│   │       │       ├── Car_76_crop.jpg
│   │       │       ├── Car_77_crop.jpg
│   │       │       ├── Car_7_crop.jpg
│   │       │       ├── Car_8_crop.jpg
│   │       │       └── Car_9_crop.jpg
│   │       ├── objtrack1_0012_1
│   │       │   ├── 0
│   │       │   │   ├── Cyclist_0_crop.jpg
│   │       │   │   ├── Cyclist_10_crop.jpg
│   │       │   │   ├── Cyclist_11_crop.jpg
│   │       │   │   ├── Cyclist_12_crop.jpg
│   │       │   │   ├── Cyclist_13_crop.jpg
│   │       │   │   ├── Cyclist_14_crop.jpg
│   │       │   │   ├── Cyclist_15_crop.jpg
│   │       │   │   ├── Cyclist_16_crop.jpg
│   │       │   │   ├── Cyclist_17_crop.jpg
│   │       │   │   ├── Cyclist_18_crop.jpg
│   │       │   │   ├── Cyclist_19_crop.jpg
│   │       │   │   ├── Cyclist_1_crop.jpg
│   │       │   │   ├── Cyclist_20_crop.jpg
│   │       │   │   ├── Cyclist_21_crop.jpg
│   │       │   │   ├── Cyclist_22_crop.jpg
│   │       │   │   ├── Cyclist_23_crop.jpg
│   │       │   │   ├── Cyclist_24_crop.jpg
│   │       │   │   ├── Cyclist_25_crop.jpg
│   │       │   │   ├── Cyclist_26_crop.jpg
│   │       │   │   ├── Cyclist_27_crop.jpg
│   │       │   │   ├── Cyclist_28_crop.jpg
│   │       │   │   ├── Cyclist_29_crop.jpg
│   │       │   │   ├── Cyclist_2_crop.jpg
│   │       │   │   ├── Cyclist_30_crop.jpg
│   │       │   │   ├── Cyclist_31_crop.jpg
│   │       │   │   ├── Cyclist_32_crop.jpg
│   │       │   │   ├── Cyclist_33_crop.jpg
│   │       │   │   ├── Cyclist_34_crop.jpg
│   │       │   │   ├── Cyclist_35_crop.jpg
│   │       │   │   ├── Cyclist_36_crop.jpg
│   │       │   │   ├── Cyclist_37_crop.jpg
│   │       │   │   ├── Cyclist_38_crop.jpg
│   │       │   │   ├── Cyclist_39_crop.jpg
│   │       │   │   ├── Cyclist_3_crop.jpg
│   │       │   │   ├── Cyclist_40_crop.jpg
│   │       │   │   ├── Cyclist_4_crop.jpg
│   │       │   │   ├── Cyclist_5_crop.jpg
│   │       │   │   ├── Cyclist_6_crop.jpg
│   │       │   │   ├── Cyclist_7_crop.jpg
│   │       │   │   ├── Cyclist_8_crop.jpg
│   │       │   │   └── Cyclist_9_crop.jpg
│   │       │   ├── 1
│   │       │   │   ├── Car_0_crop.jpg
│   │       │   │   ├── Car_10_crop.jpg
│   │       │   │   ├── Car_11_crop.jpg
│   │       │   │   ├── Car_12_crop.jpg
│   │       │   │   ├── Car_13_crop.jpg
│   │       │   │   ├── Car_14_crop.jpg
│   │       │   │   ├── Car_15_crop.jpg
│   │       │   │   ├── Car_16_crop.jpg
│   │       │   │   ├── Car_17_crop.jpg
│   │       │   │   ├── Car_18_crop.jpg
│   │       │   │   ├── Car_19_crop.jpg
│   │       │   │   ├── Car_1_crop.jpg
│   │       │   │   ├── Car_20_crop.jpg
│   │       │   │   ├── Car_21_crop.jpg
│   │       │   │   ├── Car_22_crop.jpg
│   │       │   │   ├── Car_23_crop.jpg
│   │       │   │   ├── Car_24_crop.jpg
│   │       │   │   ├── Car_25_crop.jpg
│   │       │   │   ├── Car_26_crop.jpg
│   │       │   │   ├── Car_27_crop.jpg
│   │       │   │   ├── Car_28_crop.jpg
│   │       │   │   ├── Car_29_crop.jpg
│   │       │   │   ├── Car_2_crop.jpg
│   │       │   │   ├── Car_30_crop.jpg
│   │       │   │   ├── Car_31_crop.jpg
│   │       │   │   ├── Car_32_crop.jpg
│   │       │   │   ├── Car_33_crop.jpg
│   │       │   │   ├── Car_34_crop.jpg
│   │       │   │   ├── Car_35_crop.jpg
│   │       │   │   ├── Car_36_crop.jpg
│   │       │   │   ├── Car_37_crop.jpg
│   │       │   │   ├── Car_38_crop.jpg
│   │       │   │   ├── Car_39_crop.jpg
│   │       │   │   ├── Car_3_crop.jpg
│   │       │   │   ├── Car_40_crop.jpg
│   │       │   │   ├── Car_41_crop.jpg
│   │       │   │   ├── Car_42_crop.jpg
│   │       │   │   ├── Car_43_crop.jpg
│   │       │   │   ├── Car_44_crop.jpg
│   │       │   │   ├── Car_45_crop.jpg
│   │       │   │   ├── Car_46_crop.jpg
│   │       │   │   ├── Car_47_crop.jpg
│   │       │   │   ├── Car_48_crop.jpg
│   │       │   │   ├── Car_49_crop.jpg
│   │       │   │   ├── Car_4_crop.jpg
│   │       │   │   ├── Car_50_crop.jpg
│   │       │   │   ├── Car_51_crop.jpg
│   │       │   │   ├── Car_52_crop.jpg
│   │       │   │   ├── Car_53_crop.jpg
│   │       │   │   ├── Car_54_crop.jpg
│   │       │   │   ├── Car_55_crop.jpg
│   │       │   │   ├── Car_56_crop.jpg
│   │       │   │   ├── Car_57_crop.jpg
│   │       │   │   ├── Car_58_crop.jpg
│   │       │   │   ├── Car_59_crop.jpg
│   │       │   │   ├── Car_5_crop.jpg
│   │       │   │   ├── Car_60_crop.jpg
│   │       │   │   ├── Car_61_crop.jpg
│   │       │   │   ├── Car_62_crop.jpg
│   │       │   │   ├── Car_63_crop.jpg
│   │       │   │   ├── Car_64_crop.jpg
│   │       │   │   ├── Car_65_crop.jpg
│   │       │   │   ├── Car_6_crop.jpg
│   │       │   │   ├── Car_7_crop.jpg
│   │       │   │   ├── Car_8_crop.jpg
│   │       │   │   └── Car_9_crop.jpg
│   │       │   ├── 2
│   │       │   │   ├── Pedestrian_13_crop.jpg
│   │       │   │   ├── Pedestrian_14_crop.jpg
│   │       │   │   ├── Pedestrian_15_crop.jpg
│   │       │   │   ├── Pedestrian_16_crop.jpg
│   │       │   │   ├── Pedestrian_17_crop.jpg
│   │       │   │   ├── Pedestrian_18_crop.jpg
│   │       │   │   ├── Pedestrian_19_crop.jpg
│   │       │   │   ├── Pedestrian_20_crop.jpg
│   │       │   │   ├── Pedestrian_21_crop.jpg
│   │       │   │   ├── Pedestrian_22_crop.jpg
│   │       │   │   ├── Pedestrian_23_crop.jpg
│   │       │   │   ├── Pedestrian_24_crop.jpg
│   │       │   │   ├── Pedestrian_25_crop.jpg
│   │       │   │   ├── Pedestrian_26_crop.jpg
│   │       │   │   ├── Pedestrian_27_crop.jpg
│   │       │   │   ├── Pedestrian_28_crop.jpg
│   │       │   │   ├── Pedestrian_29_crop.jpg
│   │       │   │   ├── Pedestrian_30_crop.jpg
│   │       │   │   ├── Pedestrian_31_crop.jpg
│   │       │   │   ├── Pedestrian_32_crop.jpg
│   │       │   │   ├── Pedestrian_33_crop.jpg
│   │       │   │   ├── Pedestrian_34_crop.jpg
│   │       │   │   ├── Pedestrian_35_crop.jpg
│   │       │   │   ├── Pedestrian_36_crop.jpg
│   │       │   │   ├── Pedestrian_37_crop.jpg
│   │       │   │   ├── Pedestrian_38_crop.jpg
│   │       │   │   ├── Pedestrian_39_crop.jpg
│   │       │   │   ├── Pedestrian_40_crop.jpg
│   │       │   │   ├── Pedestrian_41_crop.jpg
│   │       │   │   ├── Pedestrian_42_crop.jpg
│   │       │   │   ├── Pedestrian_43_crop.jpg
│   │       │   │   ├── Pedestrian_44_crop.jpg
│   │       │   │   ├── Pedestrian_45_crop.jpg
│   │       │   │   ├── Pedestrian_46_crop.jpg
│   │       │   │   ├── Pedestrian_47_crop.jpg
│   │       │   │   ├── Pedestrian_48_crop.jpg
│   │       │   │   ├── Pedestrian_49_crop.jpg
│   │       │   │   ├── Pedestrian_50_crop.jpg
│   │       │   │   ├── Pedestrian_51_crop.jpg
│   │       │   │   ├── Pedestrian_52_crop.jpg
│   │       │   │   ├── Pedestrian_53_crop.jpg
│   │       │   │   ├── Pedestrian_54_crop.jpg
│   │       │   │   ├── Pedestrian_55_crop.jpg
│   │       │   │   ├── Pedestrian_56_crop.jpg
│   │       │   │   ├── Pedestrian_57_crop.jpg
│   │       │   │   ├── Pedestrian_58_crop.jpg
│   │       │   │   ├── Pedestrian_59_crop.jpg
│   │       │   │   ├── Pedestrian_60_crop.jpg
│   │       │   │   ├── Pedestrian_61_crop.jpg
│   │       │   │   ├── Pedestrian_62_crop.jpg
│   │       │   │   ├── Pedestrian_63_crop.jpg
│   │       │   │   ├── Pedestrian_64_crop.jpg
│   │       │   │   ├── Pedestrian_65_crop.jpg
│   │       │   │   ├── Pedestrian_66_crop.jpg
│   │       │   │   ├── Pedestrian_67_crop.jpg
│   │       │   │   ├── Pedestrian_68_crop.jpg
│   │       │   │   ├── Pedestrian_69_crop.jpg
│   │       │   │   ├── Pedestrian_70_crop.jpg
│   │       │   │   ├── Pedestrian_71_crop.jpg
│   │       │   │   ├── Pedestrian_72_crop.jpg
│   │       │   │   ├── Pedestrian_73_crop.jpg
│   │       │   │   ├── Pedestrian_74_crop.jpg
│   │       │   │   ├── Pedestrian_75_crop.jpg
│   │       │   │   └── Pedestrian_76_crop.jpg
│   │       │   └── 3
│   │       │       ├── Car_0_crop.jpg
│   │       │       ├── Car_10_crop.jpg
│   │       │       ├── Car_11_crop.jpg
│   │       │       ├── Car_12_crop.jpg
│   │       │       ├── Car_13_crop.jpg
│   │       │       ├── Car_14_crop.jpg
│   │       │       ├── Car_15_crop.jpg
│   │       │       ├── Car_16_crop.jpg
│   │       │       ├── Car_17_crop.jpg
│   │       │       ├── Car_18_crop.jpg
│   │       │       ├── Car_19_crop.jpg
│   │       │       ├── Car_1_crop.jpg
│   │       │       ├── Car_20_crop.jpg
│   │       │       ├── Car_21_crop.jpg
│   │       │       ├── Car_22_crop.jpg
│   │       │       ├── Car_23_crop.jpg
│   │       │       ├── Car_24_crop.jpg
│   │       │       ├── Car_25_crop.jpg
│   │       │       ├── Car_26_crop.jpg
│   │       │       ├── Car_27_crop.jpg
│   │       │       ├── Car_28_crop.jpg
│   │       │       ├── Car_29_crop.jpg
│   │       │       ├── Car_2_crop.jpg
│   │       │       ├── Car_30_crop.jpg
│   │       │       ├── Car_31_crop.jpg
│   │       │       ├── Car_32_crop.jpg
│   │       │       ├── Car_33_crop.jpg
│   │       │       ├── Car_34_crop.jpg
│   │       │       ├── Car_35_crop.jpg
│   │       │       ├── Car_36_crop.jpg
│   │       │       ├── Car_37_crop.jpg
│   │       │       ├── Car_38_crop.jpg
│   │       │       ├── Car_39_crop.jpg
│   │       │       ├── Car_3_crop.jpg
│   │       │       ├── Car_40_crop.jpg
│   │       │       ├── Car_41_crop.jpg
│   │       │       ├── Car_42_crop.jpg
│   │       │       ├── Car_43_crop.jpg
│   │       │       ├── Car_44_crop.jpg
│   │       │       ├── Car_45_crop.jpg
│   │       │       ├── Car_46_crop.jpg
│   │       │       ├── Car_47_crop.jpg
│   │       │       ├── Car_48_crop.jpg
│   │       │       ├── Car_49_crop.jpg
│   │       │       ├── Car_4_crop.jpg
│   │       │       ├── Car_50_crop.jpg
│   │       │       ├── Car_51_crop.jpg
│   │       │       ├── Car_52_crop.jpg
│   │       │       ├── Car_53_crop.jpg
│   │       │       ├── Car_54_crop.jpg
│   │       │       ├── Car_55_crop.jpg
│   │       │       ├── Car_56_crop.jpg
│   │       │       ├── Car_57_crop.jpg
│   │       │       ├── Car_58_crop.jpg
│   │       │       ├── Car_59_crop.jpg
│   │       │       ├── Car_5_crop.jpg
│   │       │       ├── Car_60_crop.jpg
│   │       │       ├── Car_61_crop.jpg
│   │       │       ├── Car_62_crop.jpg
│   │       │       ├── Car_63_crop.jpg
│   │       │       ├── Car_64_crop.jpg
│   │       │       ├── Car_65_crop.jpg
│   │       │       ├── Car_66_crop.jpg
│   │       │       ├── Car_67_crop.jpg
│   │       │       ├── Car_68_crop.jpg
│   │       │       ├── Car_69_crop.jpg
│   │       │       ├── Car_6_crop.jpg
│   │       │       ├── Car_70_crop.jpg
│   │       │       ├── Car_71_crop.jpg
│   │       │       ├── Car_72_crop.jpg
│   │       │       ├── Car_73_crop.jpg
│   │       │       ├── Car_74_crop.jpg
│   │       │       ├── Car_75_crop.jpg
│   │       │       ├── Car_76_crop.jpg
│   │       │       ├── Car_77_crop.jpg
│   │       │       ├── Car_7_crop.jpg
│   │       │       ├── Car_8_crop.jpg
│   │       │       └── Car_9_crop.jpg
│   │       └── triplet_pair.txt
│   ├── eval
│   │   └── objtrack
│   │       ├── images
│   │       │   ├── 0012
│   │       │   │   ├── 000000.png
│   │       │   │   ├── 000001.png
│   │       │   │   ├── 000002.png
│   │       │   │   ├── 000003.png
│   │       │   │   ├── 000004.png
│   │       │   │   ├── 000005.png
│   │       │   │   ├── 000006.png
│   │       │   │   ├── 000007.png
│   │       │   │   ├── 000008.png
│   │       │   │   ├── 000009.png
│   │       │   │   ├── 000010.png
│   │       │   │   ├── 000011.png
│   │       │   │   ├── 000012.png
│   │       │   │   ├── 000013.png
│   │       │   │   ├── 000014.png
│   │       │   │   ├── 000015.png
│   │       │   │   ├── 000016.png
│   │       │   │   ├── 000017.png
│   │       │   │   ├── 000018.png
│   │       │   │   ├── 000019.png
│   │       │   │   ├── 000020.png
│   │       │   │   ├── 000021.png
│   │       │   │   ├── 000022.png
│   │       │   │   ├── 000023.png
│   │       │   │   ├── 000024.png
│   │       │   │   ├── 000025.png
│   │       │   │   ├── 000026.png
│   │       │   │   ├── 000027.png
│   │       │   │   ├── 000028.png
│   │       │   │   ├── 000029.png
│   │       │   │   ├── 000030.png
│   │       │   │   ├── 000031.png
│   │       │   │   ├── 000032.png
│   │       │   │   ├── 000033.png
│   │       │   │   ├── 000034.png
│   │       │   │   ├── 000035.png
│   │       │   │   ├── 000036.png
│   │       │   │   ├── 000037.png
│   │       │   │   ├── 000038.png
│   │       │   │   ├── 000039.png
│   │       │   │   ├── 000040.png
│   │       │   │   ├── 000041.png
│   │       │   │   ├── 000042.png
│   │       │   │   ├── 000043.png
│   │       │   │   ├── 000044.png
│   │       │   │   ├── 000045.png
│   │       │   │   ├── 000046.png
│   │       │   │   ├── 000047.png
│   │       │   │   ├── 000048.png
│   │       │   │   ├── 000049.png
│   │       │   │   ├── 000050.png
│   │       │   │   ├── 000051.png
│   │       │   │   ├── 000052.png
│   │       │   │   ├── 000053.png
│   │       │   │   ├── 000054.png
│   │       │   │   ├── 000055.png
│   │       │   │   ├── 000056.png
│   │       │   │   ├── 000057.png
│   │       │   │   ├── 000058.png
│   │       │   │   ├── 000059.png
│   │       │   │   ├── 000060.png
│   │       │   │   ├── 000061.png
│   │       │   │   ├── 000062.png
│   │       │   │   ├── 000063.png
│   │       │   │   ├── 000064.png
│   │       │   │   ├── 000065.png
│   │       │   │   ├── 000066.png
│   │       │   │   ├── 000067.png
│   │       │   │   ├── 000068.png
│   │       │   │   ├── 000069.png
│   │       │   │   ├── 000070.png
│   │       │   │   ├── 000071.png
│   │       │   │   ├── 000072.png
│   │       │   │   ├── 000073.png
│   │       │   │   ├── 000074.png
│   │       │   │   ├── 000075.png
│   │       │   │   ├── 000076.png
│   │       │   │   └── 000077.png
│   │       │   └── 0012_1
│   │       │       ├── 000000.png
│   │       │       ├── 000001.png
│   │       │       ├── 000002.png
│   │       │       ├── 000003.png
│   │       │       ├── 000004.png
│   │       │       ├── 000005.png
│   │       │       ├── 000006.png
│   │       │       ├── 000007.png
│   │       │       ├── 000008.png
│   │       │       ├── 000009.png
│   │       │       ├── 000010.png
│   │       │       ├── 000011.png
│   │       │       ├── 000012.png
│   │       │       ├── 000013.png
│   │       │       ├── 000014.png
│   │       │       ├── 000015.png
│   │       │       ├── 000016.png
│   │       │       ├── 000017.png
│   │       │       ├── 000018.png
│   │       │       ├── 000019.png
│   │       │       ├── 000020.png
│   │       │       ├── 000021.png
│   │       │       ├── 000022.png
│   │       │       ├── 000023.png
│   │       │       ├── 000024.png
│   │       │       ├── 000025.png
│   │       │       ├── 000026.png
│   │       │       ├── 000027.png
│   │       │       ├── 000028.png
│   │       │       ├── 000029.png
│   │       │       ├── 000030.png
│   │       │       ├── 000031.png
│   │       │       ├── 000032.png
│   │       │       ├── 000033.png
│   │       │       ├── 000034.png
│   │       │       ├── 000035.png
│   │       │       ├── 000036.png
│   │       │       ├── 000037.png
│   │       │       ├── 000038.png
│   │       │       ├── 000039.png
│   │       │       ├── 000040.png
│   │       │       ├── 000041.png
│   │       │       ├── 000042.png
│   │       │       ├── 000043.png
│   │       │       ├── 000044.png
│   │       │       ├── 000045.png
│   │       │       ├── 000046.png
│   │       │       ├── 000047.png
│   │       │       ├── 000048.png
│   │       │       ├── 000049.png
│   │       │       ├── 000050.png
│   │       │       ├── 000051.png
│   │       │       ├── 000052.png
│   │       │       ├── 000053.png
│   │       │       ├── 000054.png
│   │       │       ├── 000055.png
│   │       │       ├── 000056.png
│   │       │       ├── 000057.png
│   │       │       ├── 000058.png
│   │       │       ├── 000059.png
│   │       │       ├── 000060.png
│   │       │       ├── 000061.png
│   │       │       ├── 000062.png
│   │       │       ├── 000063.png
│   │       │       ├── 000064.png
│   │       │       ├── 000065.png
│   │       │       ├── 000066.png
│   │       │       ├── 000067.png
│   │       │       ├── 000068.png
│   │       │       ├── 000069.png
│   │       │       ├── 000070.png
│   │       │       ├── 000071.png
│   │       │       ├── 000072.png
│   │       │       ├── 000073.png
│   │       │       ├── 000074.png
│   │       │       ├── 000075.png
│   │       │       ├── 000076.png
│   │       │       └── 000077.png
│   │       └── labels
│   │           ├── 0000.txt
│   │           ├── 0001.txt
│   │           ├── 0002.txt
│   │           ├── 0003.txt
│   │           ├── 0004.txt
│   │           ├── 0005.txt
│   │           ├── 0006.txt
│   │           ├── 0007.txt
│   │           ├── 0008.txt
│   │           ├── 0009.txt
│   │           ├── 0010.txt
│   │           ├── 0011.txt
│   │           ├── 0012_1.txt
│   │           ├── 0012.txt
│   │           ├── 0013.txt
│   │           ├── 0014.txt
│   │           ├── 0015.txt
│   │           ├── 0016.txt
│   │           ├── 0017.txt
│   │           ├── 0018.txt
│   │           ├── 0019.txt
│   │           └── 0020.txt
│   ├── fcos_res50_checkpoint.pth
│   ├── R-50.pkl
│   └── training
│       ├── objtrack
│       │   ├── images
│       │   │   ├── 0012
│       │   │   │   ├── 000000.png
│       │   │   │   ├── 000001.png
│       │   │   │   ├── 000002.png
│       │   │   │   ├── 000003.png
│       │   │   │   ├── 000004.png
│       │   │   │   ├── 000005.png
│       │   │   │   ├── 000006.png
│       │   │   │   ├── 000007.png
│       │   │   │   ├── 000008.png
│       │   │   │   ├── 000009.png
│       │   │   │   ├── 000010.png
│       │   │   │   ├── 000011.png
│       │   │   │   ├── 000012.png
│       │   │   │   ├── 000013.png
│       │   │   │   ├── 000014.png
│       │   │   │   ├── 000015.png
│       │   │   │   ├── 000016.png
│       │   │   │   ├── 000017.png
│       │   │   │   ├── 000018.png
│       │   │   │   ├── 000019.png
│       │   │   │   ├── 000020.png
│       │   │   │   ├── 000021.png
│       │   │   │   ├── 000022.png
│       │   │   │   ├── 000023.png
│       │   │   │   ├── 000024.png
│       │   │   │   ├── 000025.png
│       │   │   │   ├── 000026.png
│       │   │   │   ├── 000027.png
│       │   │   │   ├── 000028.png
│       │   │   │   ├── 000029.png
│       │   │   │   ├── 000030.png
│       │   │   │   ├── 000031.png
│       │   │   │   ├── 000032.png
│       │   │   │   ├── 000033.png
│       │   │   │   ├── 000034.png
│       │   │   │   ├── 000035.png
│       │   │   │   ├── 000036.png
│       │   │   │   ├── 000037.png
│       │   │   │   ├── 000038.png
│       │   │   │   ├── 000039.png
│       │   │   │   ├── 000040.png
│       │   │   │   ├── 000041.png
│       │   │   │   ├── 000042.png
│       │   │   │   ├── 000043.png
│       │   │   │   ├── 000044.png
│       │   │   │   ├── 000045.png
│       │   │   │   ├── 000046.png
│       │   │   │   ├── 000047.png
│       │   │   │   ├── 000048.png
│       │   │   │   ├── 000049.png
│       │   │   │   ├── 000050.png
│       │   │   │   ├── 000051.png
│       │   │   │   ├── 000052.png
│       │   │   │   ├── 000053.png
│       │   │   │   ├── 000054.png
│       │   │   │   ├── 000055.png
│       │   │   │   ├── 000056.png
│       │   │   │   ├── 000057.png
│       │   │   │   ├── 000058.png
│       │   │   │   ├── 000059.png
│       │   │   │   ├── 000060.png
│       │   │   │   ├── 000061.png
│       │   │   │   ├── 000062.png
│       │   │   │   ├── 000063.png
│       │   │   │   ├── 000064.png
│       │   │   │   ├── 000065.png
│       │   │   │   ├── 000066.png
│       │   │   │   ├── 000067.png
│       │   │   │   ├── 000068.png
│       │   │   │   ├── 000069.png
│       │   │   │   ├── 000070.png
│       │   │   │   ├── 000071.png
│       │   │   │   ├── 000072.png
│       │   │   │   ├── 000073.png
│       │   │   │   ├── 000074.png
│       │   │   │   ├── 000075.png
│       │   │   │   ├── 000076.png
│       │   │   │   └── 000077.png
│       │   │   └── 0012_1
│       │   │       ├── 000000.png
│       │   │       ├── 000001.png
│       │   │       ├── 000002.png
│       │   │       ├── 000003.png
│       │   │       ├── 000004.png
│       │   │       ├── 000005.png
│       │   │       ├── 000006.png
│       │   │       ├── 000007.png
│       │   │       ├── 000008.png
│       │   │       ├── 000009.png
│       │   │       ├── 000010.png
│       │   │       ├── 000011.png
│       │   │       ├── 000012.png
│       │   │       ├── 000013.png
│       │   │       ├── 000014.png
│       │   │       ├── 000015.png
│       │   │       ├── 000016.png
│       │   │       ├── 000017.png
│       │   │       ├── 000018.png
│       │   │       ├── 000019.png
│       │   │       ├── 000020.png
│       │   │       ├── 000021.png
│       │   │       ├── 000022.png
│       │   │       ├── 000023.png
│       │   │       ├── 000024.png
│       │   │       ├── 000025.png
│       │   │       ├── 000026.png
│       │   │       ├── 000027.png
│       │   │       ├── 000028.png
│       │   │       ├── 000029.png
│       │   │       ├── 000030.png
│       │   │       ├── 000031.png
│       │   │       ├── 000032.png
│       │   │       ├── 000033.png
│       │   │       ├── 000034.png
│       │   │       ├── 000035.png
│       │   │       ├── 000036.png
│       │   │       ├── 000037.png
│       │   │       ├── 000038.png
│       │   │       ├── 000039.png
│       │   │       ├── 000040.png
│       │   │       ├── 000041.png
│       │   │       ├── 000042.png
│       │   │       ├── 000043.png
│       │   │       ├── 000044.png
│       │   │       ├── 000045.png
│       │   │       ├── 000046.png
│       │   │       ├── 000047.png
│       │   │       ├── 000048.png
│       │   │       ├── 000049.png
│       │   │       ├── 000050.png
│       │   │       ├── 000051.png
│       │   │       ├── 000052.png
│       │   │       ├── 000053.png
│       │   │       ├── 000054.png
│       │   │       ├── 000055.png
│       │   │       ├── 000056.png
│       │   │       ├── 000057.png
│       │   │       ├── 000058.png
│       │   │       ├── 000059.png
│       │   │       ├── 000060.png
│       │   │       ├── 000061.png
│       │   │       ├── 000062.png
│       │   │       ├── 000063.png
│       │   │       ├── 000064.png
│       │   │       ├── 000065.png
│       │   │       ├── 000066.png
│       │   │       ├── 000067.png
│       │   │       ├── 000068.png
│       │   │       ├── 000069.png
│       │   │       ├── 000070.png
│       │   │       ├── 000071.png
│       │   │       ├── 000072.png
│       │   │       ├── 000073.png
│       │   │       ├── 000074.png
│       │   │       ├── 000075.png
│       │   │       ├── 000076.png
│       │   │       └── 000077.png
│       │   └── labels
│       │       ├── 0000.txt
│       │       ├── 0001.txt
│       │       ├── 0002.txt
│       │       ├── 0003.txt
│       │       ├── 0004.txt
│       │       ├── 0005.txt
│       │       ├── 0006.txt
│       │       ├── 0007.txt
│       │       ├── 0008.txt
│       │       ├── 0009.txt
│       │       ├── 0010.txt
│       │       ├── 0011.txt
│       │       ├── 0012_1.txt
│       │       ├── 0012.txt
│       │       ├── 0013.txt
│       │       ├── 0014.txt
│       │       ├── 0015.txt
│       │       ├── 0016.txt
│       │       ├── 0017.txt
│       │       ├── 0018.txt
│       │       ├── 0019.txt
│       │       └── 0020.txt
│       └── objtrack1
│           ├── images
│           │   ├── 0012
│           │   │   ├── 000000.png
│           │   │   ├── 000001.png
│           │   │   ├── 000002.png
│           │   │   ├── 000003.png
│           │   │   ├── 000004.png
│           │   │   ├── 000005.png
│           │   │   ├── 000006.png
│           │   │   ├── 000007.png
│           │   │   ├── 000008.png
│           │   │   ├── 000009.png
│           │   │   ├── 000010.png
│           │   │   ├── 000011.png
│           │   │   ├── 000012.png
│           │   │   ├── 000013.png
│           │   │   ├── 000014.png
│           │   │   ├── 000015.png
│           │   │   ├── 000016.png
│           │   │   ├── 000017.png
│           │   │   ├── 000018.png
│           │   │   ├── 000019.png
│           │   │   ├── 000020.png
│           │   │   ├── 000021.png
│           │   │   ├── 000022.png
│           │   │   ├── 000023.png
│           │   │   ├── 000024.png
│           │   │   ├── 000025.png
│           │   │   ├── 000026.png
│           │   │   ├── 000027.png
│           │   │   ├── 000028.png
│           │   │   ├── 000029.png
│           │   │   ├── 000030.png
│           │   │   ├── 000031.png
│           │   │   ├── 000032.png
│           │   │   ├── 000033.png
│           │   │   ├── 000034.png
│           │   │   ├── 000035.png
│           │   │   ├── 000036.png
│           │   │   ├── 000037.png
│           │   │   ├── 000038.png
│           │   │   ├── 000039.png
│           │   │   ├── 000040.png
│           │   │   ├── 000041.png
│           │   │   ├── 000042.png
│           │   │   ├── 000043.png
│           │   │   ├── 000044.png
│           │   │   ├── 000045.png
│           │   │   ├── 000046.png
│           │   │   ├── 000047.png
│           │   │   ├── 000048.png
│           │   │   ├── 000049.png
│           │   │   ├── 000050.png
│           │   │   ├── 000051.png
│           │   │   ├── 000052.png
│           │   │   ├── 000053.png
│           │   │   ├── 000054.png
│           │   │   ├── 000055.png
│           │   │   ├── 000056.png
│           │   │   ├── 000057.png
│           │   │   ├── 000058.png
│           │   │   ├── 000059.png
│           │   │   ├── 000060.png
│           │   │   ├── 000061.png
│           │   │   ├── 000062.png
│           │   │   ├── 000063.png
│           │   │   ├── 000064.png
│           │   │   ├── 000065.png
│           │   │   ├── 000066.png
│           │   │   ├── 000067.png
│           │   │   ├── 000068.png
│           │   │   ├── 000069.png
│           │   │   ├── 000070.png
│           │   │   ├── 000071.png
│           │   │   ├── 000072.png
│           │   │   ├── 000073.png
│           │   │   ├── 000074.png
│           │   │   ├── 000075.png
│           │   │   ├── 000076.png
│           │   │   └── 000077.png
│           │   └── 0012_1
│           │       ├── 000000.png
│           │       ├── 000001.png
│           │       ├── 000002.png
│           │       ├── 000003.png
│           │       ├── 000004.png
│           │       ├── 000005.png
│           │       ├── 000006.png
│           │       ├── 000007.png
│           │       ├── 000008.png
│           │       ├── 000009.png
│           │       ├── 000010.png
│           │       ├── 000011.png
│           │       ├── 000012.png
│           │       ├── 000013.png
│           │       ├── 000014.png
│           │       ├── 000015.png
│           │       ├── 000016.png
│           │       ├── 000017.png
│           │       ├── 000018.png
│           │       ├── 000019.png
│           │       ├── 000020.png
│           │       ├── 000021.png
│           │       ├── 000022.png
│           │       ├── 000023.png
│           │       ├── 000024.png
│           │       ├── 000025.png
│           │       ├── 000026.png
│           │       ├── 000027.png
│           │       ├── 000028.png
│           │       ├── 000029.png
│           │       ├── 000030.png
│           │       ├── 000031.png
│           │       ├── 000032.png
│           │       ├── 000033.png
│           │       ├── 000034.png
│           │       ├── 000035.png
│           │       ├── 000036.png
│           │       ├── 000037.png
│           │       ├── 000038.png
│           │       ├── 000039.png
│           │       ├── 000040.png
│           │       ├── 000041.png
│           │       ├── 000042.png
│           │       ├── 000043.png
│           │       ├── 000044.png
│           │       ├── 000045.png
│           │       ├── 000046.png
│           │       ├── 000047.png
│           │       ├── 000048.png
│           │       ├── 000049.png
│           │       ├── 000050.png
│           │       ├── 000051.png
│           │       ├── 000052.png
│           │       ├── 000053.png
│           │       ├── 000054.png
│           │       ├── 000055.png
│           │       ├── 000056.png
│           │       ├── 000057.png
│           │       ├── 000058.png
│           │       ├── 000059.png
│           │       ├── 000060.png
│           │       ├── 000061.png
│           │       ├── 000062.png
│           │       ├── 000063.png
│           │       ├── 000064.png
│           │       ├── 000065.png
│           │       ├── 000066.png
│           │       ├── 000067.png
│           │       ├── 000068.png
│           │       ├── 000069.png
│           │       ├── 000070.png
│           │       ├── 000071.png
│           │       ├── 000072.png
│           │       ├── 000073.png
│           │       ├── 000074.png
│           │       ├── 000075.png
│           │       ├── 000076.png
│           │       └── 000077.png
│           └── labels
│               ├── 0000.txt
│               ├── 0001.txt
│               ├── 0002.txt
│               ├── 0003.txt
│               ├── 0004.txt
│               ├── 0005.txt
│               ├── 0006.txt
│               ├── 0007.txt
│               ├── 0008.txt
│               ├── 0009.txt
│               ├── 0010.txt
│               ├── 0011.txt
│               ├── 0012_1.txt
│               ├── 0012.txt
│               ├── 0013.txt
│               ├── 0014.txt
│               ├── 0015.txt
│               ├── 0016.txt
│               ├── 0017.txt
│               ├── 0018.txt
│               ├── 0019.txt
│               └── 0020.txt
├── datasets
│   ├── crop_gt_box.py
│   ├── data_collect.py
│   ├── FacenetTripletDataset.py
│   ├── __init__.py
│   ├── ObjtrackDataset.py
│   ├── __pycache__
│   │   ├── data_collect.cpython-37.pyc
│   │   ├── FacenetTripletDataset.cpython-37.pyc
│   │   ├── __init__.cpython-37.pyc
│   │   ├── ObjtrackDataset.cpython-37.pyc
│   │   └── transform.cpython-37.pyc
│   └── transform.py
├── detection
│   ├── crop_det.py
│   ├── detect.sh
│   ├── eval_det.py
│   ├── heads
│   │   ├── fcos_head.py
│   │   ├── __init__.py
│   │   └── __pycache__
│   │       ├── fcos_head.cpython-37.pyc
│   │       └── __init__.cpython-37.pyc
│   ├── _init_paths.py
│   ├── __init__.py
│   ├── loss
│   │   ├── fcos_loss.py
│   │   ├── __init__.py
│   │   └── __pycache__
│   │       └── fcos_loss.cpython-37.pyc
│   ├── models
│   │   ├── fcos.py
│   │   ├── __init__.py
│   │   └── __pycache__
│   │       └── fcos.cpython-37.pyc
│   ├── necks
│   │   ├── FPN.py
│   │   ├── __init__.py
│   │   └── __pycache__
│   │       ├── FPN.cpython-37.pyc
│   │       └── __init__.cpython-37.pyc
│   ├── __pycache__
│   │   ├── crop_det.cpython-37.pyc
│   │   ├── __init__.cpython-37.pyc
│   │   └── _init_paths.cpython-37.pyc
│   ├── test_det.py
│   ├── train_det.py
│   ├── trainer
│   │   ├── FCOStrainer.py
│   │   ├── __init__.py
│   │   └── __pycache__
│   │       ├── FCOStrainer.cpython-37.pyc
│   │       ├── __init__.cpython-37.pyc
│   │       └── trainer.cpython-37.pyc
│   └── utils
│       ├── box_utils.py
│       ├── __init__.py
│       ├── metrics.py
│       └── __pycache__
│           ├── box_utils.cpython-37.pyc
│           ├── __init__.cpython-37.pyc
│           └── metrics.cpython-37.pyc
├── evaluate.py
├── extensions
│   └── __init__.py
├── inference.py
├── metrics
│   └── __init__.py
├── __pycache__
│   └── _init_paths.cpython-37.pyc
├── README.md
├── requirements.txt
├── TC_tracker.py
├── TNT
│   ├── clusters
│   │   ├── cluster_optimize
│   │   │   ├── cluster_assign.py
│   │   │   ├── cluster_break.py
│   │   │   ├── cluster_merge.py
│   │   │   ├── cluster_split.py
│   │   │   ├── cluster_switch.py
│   │   │   └── __init__.py
│   │   ├── generate_optimal_cluster.py
│   │   ├── init_cluster.py
│   │   ├── __init__.py
│   │   └── utils
│   │       └── __init__.py
│   ├── __init__.py
│   ├── tracklets
│   │   ├── eval_pair_fushion.py
│   │   ├── generate_tracklets.py
│   │   ├── __init__.py
│   │   ├── loss
│   │   │   ├── appearance_change_loss.py
│   │   │   ├── __init__.py
│   │   │   ├── interval_loss.py
│   │   │   ├── smoothness_loss.py
│   │   │   └── velocity_loss.py
│   │   ├── pair_fushion
│   │   │   └── __init__.py
│   │   ├── train_pair_fushion.py
│   │   └── utils
│   │       └── __init__.py
│   └── utils
│       └── __init__.py
├── tree.md
├── units
│   ├── losses.py
│   ├── __pycache__
│   │   ├── losses.cpython-37.pyc
│   │   ├── trainer.cpython-37.pyc
│   │   └── units.cpython-37.pyc
│   ├── trainer.py
│   └── units.py
├── utils
│   ├── __init__.py
│   ├── pred_loc.py
│   ├── __pycache__
│   │   ├── __init__.cpython-37.pyc
│   │   ├── registry.cpython-37.pyc
│   │   └── utils.cpython-37.pyc
│   ├── registry.py
│   └── utils.py
└── work_dirs
    ├── facenet_triplet_appearance
    │   ├── 2020-02-22-22-10
    │   │   └── train_2020-02-22-22-10_rank0.log
    │   ├── 2020-02-22-22-19
    │   │   └── train_2020-02-22-22-19_rank0.log
    │   ├── 2020-02-22-22-24
    │   │   └── train_2020-02-22-22-24_rank0.log
    │   ├── 2020-02-22-22-26
    │   │   └── train_2020-02-22-22-26_rank0.log
    │   ├── 2020-02-22-22-27
    │   │   └── train_2020-02-22-22-27_rank0.log
    │   ├── 2020-02-22-22-29
    │   │   └── train_2020-02-22-22-29_rank0.log
    │   ├── 2020-02-22-22-30
    │   │   └── train_2020-02-22-22-30_rank0.log
    │   ├── 2020-02-22-22-34
    │   │   └── train_2020-02-22-22-34_rank0.log
    │   ├── 2020-02-22-22-35
    │   │   └── train_2020-02-22-22-35_rank0.log
    │   ├── 2020-02-22-22-38
    │   │   └── train_2020-02-22-22-38_rank0.log
    │   ├── 2020-02-22-22-41
    │   │   └── train_2020-02-22-22-41_rank0.log
    │   ├── 2020-02-22-22-44
    │   │   └── train_2020-02-22-22-44_rank0.log
    │   ├── 2020-02-24-00-04
    │   │   └── train_2020-02-24-00-04_rank0.log
    │   ├── 2020-02-24-00-09
    │   │   ├── events.out.tfevents.1582474161.n224-022-185
    │   │   └── train_2020-02-24-00-09_rank0.log
    │   ├── 2020-02-24-00-11
    │   │   ├── events.out.tfevents.1582474269.n224-022-185
    │   │   ├── InceptionResnetV1_epoch000_iter000010_checkpoint.pth
    │   │   ├── InceptionResnetV1_epoch000_iter000020_checkpoint.pth
    │   │   ├── model_best.pth
    │   │   └── train_2020-02-24-00-11_rank0.log
    │   ├── 2020-02-24-00-14
    │   │   └── train_2020-02-24-00-14_rank0.log
    │   ├── 2020-02-24-00-15
    │   │   └── train_2020-02-24-00-15_rank0.log
    │   ├── 2020-02-24-00-16
    │   │   └── train_2020-02-24-00-16_rank0.log
    │   ├── 2020-02-24-00-17
    │   │   ├── events.out.tfevents.1582474670.n224-022-185
    │   │   ├── InceptionResnetV1_epoch000_iter000010_checkpoint.pth
    │   │   ├── InceptionResnetV1_epoch000_iter000020_checkpoint.pth
    │   │   ├── model_best.pth
    │   │   └── train_2020-02-24-00-17_rank0.log
    │   └── 2020-02-24-00-21
    │       ├── events.out.tfevents.1582474873.n224-022-185
    │       ├── events.out.tfevents.1582474893.n224-022-185
    │       └── train_2020-02-24-00-21_rank0.log
    └── fcos_detector
        └── 2020-02-21-22-54
            ├── events.out.tfevents.1582296891.n224-022-185
            ├── FCOS_epoch111_iter001000_checkpoint.pth
            ├── FCOS_epoch222_iter002000_checkpoint.pth
            └── train_2020-02-21-22-54_rank0.log

119 directories, 2194 files
