#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""
import copy

import albumentations as A
import cv2
import numpy as np

from yolox.utils import xyxy2cxcywh


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):
    # box1(4,n), box2(4,n)
    # Compute candidate boxes which include follwing 5 things:
    # box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (
            (w2 > wh_thr)
            & (h2 > wh_thr)
            & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr)
            & (ar < ar_thr)
    )  # candidates


def xyxy2xywh(boxes):
    return np.array([[x1, y1, x2 - x1, y2 - y1]
                     for x1, y1, x2, y2 in boxes])


def xywh2xyxy(boxes):
    return np.array([[x1, y1, x1 + w, y1 + h]
                     for x1, y1, w, h in boxes])


class TrainTransform:
    def __init__(self, image_size, p=0.5, rgb_means=None, std=None, max_labels=50):
        self.image_size = image_size
        self.means = rgb_means
        self.std = std
        self.p = p
        self.max_labels = max_labels
        self.transform = A.Compose([
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.HorizontalFlip(p=0.4),
            A.VerticalFlip(p=0.4),
            A.RandomCrop(height=self.image_size[0], width=self.image_size[1],
                         always_apply=False, p=0.1),
            A.Resize(height=self.image_size[0], width=self.image_size[1],
                     interpolation=cv2.INTER_CUBIC, always_apply=True),
            A.ToFloat(max_value=255, always_apply=True),
            # A.Normalize(mean=rgb_means, std=std)
        ],
            bbox_params=A.BboxParams(format='coco', min_visibility=0.5, label_fields=['class_labels']))
        self.truncated_transform = A.Compose([
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.HorizontalFlip(p=0.4),
            A.VerticalFlip(p=0.4),
            A.Resize(height=self.image_size[0], width=self.image_size[1],
                     interpolation=cv2.INTER_CUBIC, always_apply=True),
            A.ToFloat(max_value=255, always_apply=True),
            # A.Normalize(mean=rgb_means, std=std, always_apply=True)
        ],
            bbox_params=A.BboxParams(format='coco', min_visibility=0.5, label_fields=['class_labels']))
        self.swap = (2, 0, 1)

    def __call__(self, image, targets, input_dim):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image[image != 255] = 0
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV)
        image = np.stack([image] * 3, axis=2).astype(np.uint8)
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        boxes = xyxy2xywh(boxes)
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            transformed = self.transform(image=image,
                                         bboxes=boxes,
                                         class_labels=labels)
            image = transformed['image']
            image = image.transpose(self.swap)
            image = np.ascontiguousarray(image, dtype=np.float32)
            return image, targets

        image_t = copy.deepcopy(image)
        height, width, _ = image_t.shape
        boxes = np.array([
            [x, y, w if x + w <= width else width - x, h if y + h <= height else height - y]
            for
            x, y, w, h in boxes])
        real_boxes = copy.deepcopy(boxes)
        real_labels = copy.deepcopy(labels)
        padded_labels = np.zeros((self.max_labels, 5))
        transformed = self.transform(image=image_t,
                                     bboxes=boxes,
                                     class_labels=labels)
        image_t = transformed['image']
        boxes = np.array(transformed['bboxes'])
        labels = np.array(transformed['class_labels'])
        if len(boxes) == 0:
            image_t = copy.deepcopy(image)
            boxes = copy.deepcopy(real_boxes)
            labels = copy.deepcopy(real_labels)
            transformed = self.truncated_transform(image=image_t,
                                                   bboxes=boxes,
                                                   class_labels=labels)
            image_t = transformed['image']
            boxes = np.array(transformed['bboxes'])
            labels = np.array(transformed['class_labels'])

        boxes = xywh2xyxy(boxes)
        boxes = np.int64(boxes)
        boxes = xyxy2cxcywh(boxes)
        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 8
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]

        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((labels_t, boxes_t))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[: self.max_labels]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        # image_t += np.min(image_t)
        # image_t /= (np.max(image_t) - np.min(image_t))
        # image = copy.deepcopy(image_t)
        # image *= 255
        # image = np.array(image).astype(np.uint8)
        # cv2.imwrite('image.png', image)
        image_t = image_t.transpose(self.swap)
        image_t = np.ascontiguousarray(image_t, dtype=np.float32)
        return image_t, padded_labels


class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, image_size, rgb_means=None, std=None, swap=(2, 0, 1)):
        self.image_size = image_size
        self.means = rgb_means
        self.swap = swap
        self.std = std
        self.transform = A.Compose([
            A.Resize(height=image_size[0], width=image_size[1],
                     interpolation=cv2.INTER_CUBIC, always_apply=True),
            A.ToFloat(max_value=255, always_apply=True),
            A.Normalize(mean=rgb_means, std=std, always_apply=True)
        ])

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        transformed = self.transform(image=img)
        img = transformed['image']
        img = img.transpose(self.swap)
        img = np.ascontiguousarray(img, dtype=np.float32)
        return img, np.zeros((1, 5))
