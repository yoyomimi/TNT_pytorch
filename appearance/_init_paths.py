import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

project_path = osp.dirname(this_dir)
dataset_path = osp.join(project_path, 'datasets')
loss_path = osp.join(project_path, 'appearance', 'loss')
trainer_path = osp.join(project_path, 'appearance', 'trainer')

add_path(project_path)
add_path(dataset_path)
add_path(loss_path)
add_path(trainer_path)
