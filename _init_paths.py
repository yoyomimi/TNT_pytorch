import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

project_path = osp.dirname(this_dir)
model_path = osp.join(project_path, 'detection', 'models')
dataset_path = osp.join(project_path, 'detection', 'datasets')
det_loss_path = osp.join(project_path, 'detection', 'losses')
trainer_path = osp.join(project_path, 'detection', 'trainer')

add_path(project_path)
add_path(model_path)
add_path(dataset_path)
add_path(det_loss_path)
add_path(trainer_path)