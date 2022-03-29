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
import random

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


def _distort(image):
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):
        _convert(image, beta=random.uniform(-32, 32))

    if random.randrange(2):
        _convert(image, alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if random.randrange(2):
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp

    if random.randrange(2):
        _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def _mirror(image, boxes):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def xyxy2xywh(boxes):
    return np.array([[x1, y1, x2 - x1, y2 - y1]
                     for x1, y1, x2, y2 in boxes])


def xywh2xyxy(boxes):
    return np.array([[x1, y1, x1 + w, y1 + h]
                     for x1, y1, w, h in boxes])


class TrainTransform:
    def __init__(self, p=0.5, rgb_means=None, std=None, max_labels=50):
        self.means = rgb_means
        self.std = std
        self.p = p
        self.max_labels = max_labels
        self.transform = A.Compose([
            A.GaussianBlur(blur_limit=(3, 7), p=0.6),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomCrop(height=2048, width=2048,
                         always_apply=False, p=0.0),
            A.Resize(height=2048, width=2048,
                     interpolation=cv2.INTER_CUBIC, always_apply=True),
            A.ToFloat(max_value=255, always_apply=True),
            # A.Normalize(mean=rgb_means, std=std)
        ],
            bbox_params=A.BboxParams(format='coco', min_visibility=0.5, label_fields=['class_labels']))

    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        boxes = xyxy2xywh(boxes)

        immm = copy.deepcopy(image)
        boxes1 = np.int64(boxes)
        for x, y, w, h in boxes1:
            cv2.rectangle(immm, (x, y), (x + w, y + h), (255, 0, 0), 5)
        cv2.imwrite('original.png', immm)
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            transformed = self.transform(image=image,
                                         bboxes=boxes,
                                         class_labels=labels)
            image = transformed['image']
            # boxes = transformed['bboxes']
            # labels = transformed['class_labels']
            image = np.ascontiguousarray(image, dtype=np.float32)
            return image, targets

        # image_o = image.copy()
        # targets_o = targets.copy()
        # height_o, width_o, _ = image_o.shape
        # boxes_o = targets_o[:, :4]
        # labels_o = targets_o[:, 4]
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        # boxes_o = xyxy2cxcywh(boxes)

        # cv2.imwrite('original.png', image)
        image_t = copy.deepcopy(image)
        # image_t = _distort(image)
        # cv2.imwrite('_distort.png', image_t)
        # image_t, boxes = _mirror(image_t, boxes)
        # cv2.imwrite('_mirror.png', image_t)
        height, width, _ = image_t.shape
        boxes = np.array([
            [x, y, w if x + w <= width else width - x, h if y + h <= height else height - y]
            for
            x, y, w, h in boxes])
        try:
            transformed = self.transform(image=image_t,
                                         bboxes=boxes,
                                         class_labels=labels)
            image_t = transformed['image']
            boxes = np.array(transformed['bboxes'])
            labels = np.array(transformed['class_labels'])
            image_t *= 255
            immm = np.array(image_t).astype(np.uint8)
            boxes1 = np.int64(boxes)
            for x, y, w, h in boxes1:
                cv2.rectangle(immm, (x, y), (x + w, y + h), (255, 0, 0), 5)
            cv2.imwrite('image_t.png', immm)
            boxes = xywh2xyxy(boxes)
            boxes = np.int64(boxes)
            boxes = xyxy2cxcywh(boxes)
            mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 8
            boxes_t = boxes[mask_b]
            labels_t = labels[mask_b]
            # if len(boxes_t) == 0:
            #     # image_t, r_o = preproc(image_o, input_dim, self.means, self.std)
            #     # boxes_o *= r_o
            #     # boxes_t = boxes_o
            #     # labels_t = labels_o
            #     transformed = self.transform(image=image_o,
            #                                  bboxes=boxes_o,
            #                                  class_labels=labels)
            #     image_t = transformed['image']
            #     boxes_t = transformed['bboxes']
            #     labels_t = transformed['class_labels']
            labels_t = np.expand_dims(labels_t, 1)
            targets_t = np.hstack((labels_t, boxes_t))
            padded_labels = np.zeros((self.max_labels, 5))
            padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[: self.max_labels]
            padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
            image_t = np.ascontiguousarray(image_t, dtype=np.float32)
        except Exception as ex:
            print(ex)
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

    def __init__(self, rgb_means=None, std=None, swap=(2, 0, 1)):
        self.means = rgb_means
        self.swap = swap
        self.std = std
        self.transform = A.Compose([
            A.Resize(height=2048, width=2048,
                     interpolation=cv2.INTER_CUBIC, always_apply=True),
            A.ToFloat(max_value=255, always_apply=True),
            A.Normalize(mean=rgb_means, std=std, always_apply=True)
        ])

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        transformed = self.transform(image=img)
        img = transformed['image']
        return img, np.zeros((1, 5))