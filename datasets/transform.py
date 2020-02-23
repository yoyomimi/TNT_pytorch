import cv2
import numpy as np
from numpy import random

from scipy import misc

import torch

class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None, no_object=False):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None, no_object=False):
        return image.astype(np.float32), boxes, labels


class Crop(object):
    def __init__(self, random_crop=True, image_size=182):
        self.random_crop = random_crop
        self.image_size = image_size

    def __call__(self, image, boxes=None, labels=None, no_object=False):
        if image.shape[1] > self.image_size:
            sz1 = int(image.shape[1]//2)
            sz2 = int(image_size//2)
            if self.random_crop:
                diff = sz1 - sz2
                (h, v) = (np.random.randint(-diff, diff+1), np.random.randint(-diff, diff+1))
            else:
                (h, v) = (0,0)
            image = image[(sz1-sz2+v):(sz1+sz2+v),(sz1-sz2+h):(sz1+sz2+h),:]
        return image, boxes, labels
  

class Flip(object):
    def __init__(self, random_flip=True):
        self.random_flip = random_flip

    def __call__(self, image, boxes=None, labels=None, no_object=False):
        if self.random_flip and np.random.choice([True, False]):
            image = np.fliplr(image)
        return image, boxes, labels


class Random_rotate_image(object):
    def __call__(self, image, boxes=None, labels=None, no_object=False):
        angle = np.random.uniform(low=-10.0, high=10.0)
        return misc.imrotate(image, angle, 'bicubic'), boxes, labels

class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None, no_object=False):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None, no_object=False):
        if not no_object:
            height, width, _ = image.shape
            if boxes.dtype != np.float32:
                boxes = boxes.astype(np.float32)
            boxes[:, 0:4:2] *= width
            boxes[:, 1:4:2] *= height
        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None, no_object=False):
        if not no_object:
            height, width, _ = image.shape
            if boxes.dtype != np.float32:
                boxes = boxes.astype(np.float32)
            boxes[:, 0:4:2] /= width
            boxes[:, 1:4:2] /= height
        return image, boxes, labels


class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, boxes=None, labels=None, no_object=False):
        size = (self.max_size, self.min_size)
        image_new = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        return image_new, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class ToTensor(object):
    def __call__(self, img, boxes=None, labels=None, no_object=False):
        return torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1), boxes, labels


class RandomMirror(object):
    def __call__(self, image, boxes, classes, no_object=False):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            if no_object:
                return image, boxes, classes
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes


class TrainTransform(object):
    def __init__(self, size=[788, 1400], mean=(102.9801, 115.9465, 122.7717), std=(1.0, 1.0, 1.0), flip_prob=0.5):
        self.mean = mean
        self.std = std
        self.min_size, self.max_size = size
        self.flip_prob = flip_prob

        self.augment = Compose([
            ConvertFromInts(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.min_size, self.max_size),
            ToAbsoluteCoords(),  
            SubtractMeans(mean),
            ToTensor(),
        ])

    def __call__(self, img, boxes, labels, no_object=False):
        return self.augment(img, boxes, labels, no_object)


class EvalTransform(object):
    def __init__(self, size=[788, 1400], mean=(102.9801, 115.9465, 122.7717), std=(1.0, 1.0, 1.0), flip_prob=0):
        # BGR
        self.mean = mean
        self.std = std
        self.min_size, self.max_size = size
        self.flip_prob = flip_prob

        self.augment = Compose([
            ConvertFromInts(),
            ToPercentCoords(),
            Resize(self.min_size, self.max_size),
            ToAbsoluteCoords(),  
            SubtractMeans(mean),
            ToTensor(),
        ])
    
    def __call__(self, img, boxes, labels, no_object=False):
        return self.augment(img, boxes, labels, no_object)


class PredictionTransform(object):
    def __init__(self, size=[800, 1333], mean=(102.9801, 115.9465, 122.7717), std=(1.0, 1.0, 1.0), flip_prob=0):
        # BGR
        self.mean = mean
        self.std = std
        self.min_size, self.max_size = size
        self.flip_prob = flip_prob

        self.transform = Compose([
            ConvertFromInts(),
            Resize(self.min_size, self.max_size),
            SubtractMeans(mean),
            ToTensor(),
        ])

    def __call__(self, img):
        img, _, _ = self.transform(img)
        return img


class FacenetTransform(object):
    def __init__(self, size=[182, 182], mean=(102.9801, 115.9465, 122.7717), std=(1.0, 1.0, 1.0)):
        # BGR
        self.mean = mean
        self.std = std
        self.min_size, self.max_size = size

        self.transform = Compose([
            ConvertFromInts(),
            Random_rotate_image(),
            Crop(image_size=self.max_size),
            Flip(),
            Resize(self.min_size, self.max_size),
            SubtractMeans(mean),
            ToTensor(),
        ])

    def __call__(self, img):
        img, _, _ = self.transform(img)
        return img

if __name__ == '__main__':
    import os
    import cv2

    from ObjtrackDataset import ObjtrackDataset
    
    def write_transform_result(index, frame_id, img, boxes, labels, track_ids, dir, is_train=True):
        img = img.permute(1, 2, 0).numpy()
        class_dict = {
            -1: 'DontCare',
            0: 'Pedestrian',
            1: 'Car',
            2: 'Cyclist',
        }
        boxes = boxes.type(torch.LongTensor)
        for i in range(boxes.shape[0]):
            # boxes: abs coordinate
            box = boxes[i]
            label = class_dict[labels[i]] + '_' + str(track_ids[i])
            cv2.rectangle(img, (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])), (0, 0, 255), 1)
            cv2.putText(
                img,
                label,
                (int(box[0]) + 10, int(box[1]) + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,  # font scale
                (0, 255, 9),
                1)   # line type
        if is_train:
            cv2.imwrite(dir + '/train_' + str(index) + '_' + str(frame_id) + '.jpg', img)
        else:
            cv2.imwrite(dir + '/eval_' + str(index)  + '_' + str(frame_id) + '.jpg', img)

    train_dataset = ObjtrackDataset('/Users/chenmingfei/Downloads/Hwang_Papers/TNT_pytorch/data/training',
                                TrainTransform([352, 1216]))
    img_dir = 'test_trans_image'
    if os.path.exists(img_dir):
        import shutil
        shutil.rmtree(img_dir)
    os.makedirs(img_dir)
    for i in range(len(train_dataset)):
        img, target, frame_id, index = train_dataset[i]  
        write_transform_result(i, frame_id, img, target[:, :4], target[:, 4], target[:, 5], img_dir, is_train=False)
    

