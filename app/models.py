from django.db import models
import os
import sys
import math
import numpy as np
from skimage import io
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
import COCO.coco
import tensorflow as tf
import cv2
import base64

ROOT_DIR = os.path.abspath("./")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

graph = tf.get_default_graph()
class InferenceConfig(COCO.coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
config = InferenceConfig()


class Photo(models.Model):
    image = models.ImageField(upload_to='media')
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    def predict(self):
        global graph
        with graph.as_default():
            img_data = io.imread(self.image)
            result = self.model.detect([img_data], verbose=1)
            r = result[0]
            N = r['rois'].shape[0]
            result_image = img_data.copy()
            colors = visualize.random_colors(N)
            for i in range(N):
                color = colors[i]
                rgb = (255, 255, 255)
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = self.class_names[r['class_ids'][i]]
                result_image = cv2.putText(result_image, text, (r['rois'][i][1],r['rois'][i][0]),
                                font, 0.8, rgb, 1, cv2.LINE_AA)
                mask = r['masks'][:,:,i]
                result_image = visualize.apply_mask(result_image, mask, color)

            return result_image

    def img_src(self):
        with self.image.open() as img:
            base64_img = base64.b64encode(img.read()).decode()
            return 'data:' + img.file.content_type + ';base64,' + base64_img













