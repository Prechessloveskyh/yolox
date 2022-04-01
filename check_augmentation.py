import copy
import os
import random

import albumentations as A
import cv2
import numpy as np
import json


def main1():
    class TrainTransform:
        def __init__(self, p=0.5, rgb_means=None, std=None, max_labels=50):
            self.means = rgb_means
            self.std = std
            self.p = p
            self.max_labels = max_labels
            self.transform = A.Compose([
                A.GaussianBlur(blur_limit=(5, 7), p=0.6),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomCrop(height=2048, width=2048,
                             always_apply=False, p=0.5),
                A.Resize(height=2048, width=2048,
                         interpolation=cv2.INTER_CUBIC, always_apply=True),
                A.ToFloat(max_value=255, always_apply=True),
                A.Normalize(mean=rgb_means, std=std)
            ],
                bbox_params=A.BboxParams(format='coco', min_visibility=0.5, label_fields=['class_labels']))

        def __call__(self, image, targets, input_dim=0):
            boxes = targets[:, :4].copy()
            labels = targets[:, 4].copy()
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

            image_o = image.copy()
            targets_o = targets.copy()
            height_o, width_o, _ = image_o.shape
            boxes_o = targets_o[:, :4]
            labels_o = targets_o[:, 4]
            # bbox_o: [xyxy] to [c_x,c_y,w,h]
            # boxes_o = xyxy2cxcywh(boxes_o)

            cv2.imwrite('original.png', image)
            image_t = copy.deepcopy(image)
            # image_t = _distort(image)
            # cv2.imwrite('_distort.png', image_t)
            # image_t, boxes = _mirror(image_t, boxes)
            # cv2.imwrite('_mirror.png', image_t)
            height, width, _ = image_t.shape
            boxes = np.array([
                [x1, y1, w if x1 + w <= width_o else width_o - w, h if y1 + h <= height_o else height_o - h]
                for
                x1, y1, w, h in boxes])
            try:
                transformed = self.transform(image=image_t,
                                             bboxes=boxes,
                                             class_labels=labels)

                image_t = transformed['image']
                boxes = transformed['bboxes']
                labels = transformed['class_labels']
                # image_t *= 255
                # immm = np.array(image_t).astype(np.uint8)
                # boxes1 = np.int64(boxes)
                # for x, y, w, h in boxes1:
                #     cv2.rectangle(immm, (x, y), (x + w, y + h), (255, 0, 0), 5)
                # cv2.imwrite('image_t.png', immm)
            except Exception as ex:
                print(ex)
            # # boxes = xyxy2cxcywh(boxes)
            # mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 8
            # boxes_t = boxes[mask_b]
            # labels_t = labels[mask_b]
            #
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
            # immm = copy.deepcopy(image_t)
            # immm = np.array(immm * 255).astype(np.unt8)
            # cv2.imwrite('image_t.png', immm)
            # labels_t = np.expand_dims(labels_t, 1)
            #
            # targets_t = np.hstack((labels_t, boxes_t))
            # padded_labels = np.zeros((self.max_labels, 5))
            # padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            #                                                           : self.max_labels
            #                                                           ]
            #
            # padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
            # image_t = np.ascontiguousarray(image_t, dtype=np.float32)
            # return image_t, padded_labels

    preproc = TrainTransform(
        rgb_means=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_labels=50,
    )
    with open(os.path.join('data', 'annotations', '3.json')) as ff:
        data = json.loads(ff.read())
    images = data['images']
    annotations = data['annotations']
    print(len(images))
    for image in images:
        id = image['id']
        file_name = image['file_name']
        file_path = os.path.join('data', 'images', '3', file_name)
        image = cv2.imread(file_path)
        # if not os.path.exists(file_path):
        #     print(file_path)
        # if image is None:
        #     print(file_path)
        bbox = np.array([ann['bbox'] for ann in annotations if ann['image_id'] == id])
        category_id = np.array([ann['category_id'] for ann in annotations if ann['image_id'] == id])
        # category_id=np.expand_dims(category_id, axis=1)
        target = np.array([[x, y, w, h, t] for (x, y, w, h), t in zip(bbox, category_id)])
        # target = np.stack([bbox, category_id], axis=0)
        preproc(image, target)


def create_dataset():
    path_to_data = os.path.join('..', '1', 'yolo5', 'datav5', 'data', 'zumen')
    path_to_train_image = os.path.join(path_to_data, 'train', 'images')
    path_to_train_label = os.path.join(path_to_data, 'train', 'labels')
    path_to_test_image = os.path.join(path_to_data, 'test', 'images')
    path_to_test_label = os.path.join(path_to_data, 'test', 'labels')

    # with open(os.path.join('data', 'annotations', '7_12_14_15_16_18_10.json'), 'r') as ff:
    #     data_7_12_14_15_16_18_10 = json.loads(ff.read())
    # with open(os.path.join('data', 'annotations', '1234569_11_13.json'), 'r') as ff:
    #     data_1234569_11_13 = json.loads(ff.read())
    # images1 = copy.deepcopy(data_1234569_11_13['images'])
    # images2 = copy.deepcopy(data_7_12_14_15_16_18_10['images'])
    # n = len(images2)
    # image1_changed = []
    # for image in images1:
    #     image['id'] = image['id'] + n
    #     image1_changed.append(image)
    # # images2.extend(images1)
    # images = []
    # images.extend(images2)
    # images.extend(image1_changed)
    # # [images1.append(x) for x in images2]
    # annotations1 = copy.deepcopy(data_1234569_11_13['annotations'])
    # annotations2 = copy.deepcopy(data_7_12_14_15_16_18_10['annotations'])
    # annotation1_changed = []
    # an = len(annotations2)
    # for ann in annotations1:
    #     ann['id'] = ann['id'] + an
    #     ann['image_id'] = ann['image_id'] + n
    #     annotation1_changed.append(ann)
    # annotations2.extend(annotation1_changed)
    # all_data = {
    #     'info': data_1234569_11_13['info'],
    #     'licenses': data_1234569_11_13['licenses'],
    #     'categories': data_1234569_11_13['categories'],
    #     'images': images,
    #     'annotations': annotations1,
    #
    # }
    # with open(os.path.join('data', 'annotations', 'all.json'), 'w') as ff:
    #     ff.write(json.dumps(all_data, indent=3))

    with open(os.path.join('data', 'annotations', 'all.json'), 'r') as ff:
        data = json.loads(ff.read())
    train_names = os.listdir(path_to_train_image)
    test_names = os.listdir(path_to_test_image)
    train_data = {
        'info': data['info'],
        'licenses': data['licenses'],
        'categories': data['categories'],
    }
    test_data = {
        'info': data['info'],
        'licenses': data['licenses'],
        'categories': data['categories'],

    }
    images = data['images']
    annotations = data['annotations']
    print(len(images))
    train_image = []
    train_labels = []
    test_image = []
    test_labels = []
    im_train_name = []
    im_test_name = []
    for image in images:
        id = image['id']
        file_name = image['file_name']
        if file_name in train_names and file_name not in im_train_name:
            train_image.append(image)
            im_train_name.append(file_name)
            train_labels.extend([ann for ann in annotations if ann['image_id'] == id])

        elif file_name in test_names and file_name not in im_test_name:
            test_image.append(image)
            im_test_name.append(file_name)
            test_labels.extend([ann for ann in annotations if ann['image_id'] == id])
        else:
            print(file_name)
    # unique_im = np.unique(train_image)
    train_data['images'] = train_image
    train_data['annotations'] = train_labels
    with open(os.path.join('data', 'annotations', 'train_data.json'), 'w') as ff:
        ff.write(json.dumps(train_data, indent=3))
    test_data['images'] = test_image
    test_data['annotations'] = test_labels
    with open(os.path.join('data', 'annotations', 'test_data.json'), 'w') as ff:
        ff.write(json.dumps(test_data, indent=3))


create_dataset()
