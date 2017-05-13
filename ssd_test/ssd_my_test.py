import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim

import matplotlib.image as mpimg
import sys
sys.path.append('../')

from ssd_nets import ssd_vgg_300, np_methods
from ssd_preprocessing import ssd_vgg_preprocessing
from ssd_test import visualization

sess = tf.Session()

# Input placeholder.
net_shape = (300, 300)
num_classes = 2
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)# WARP_RESIZE
image_4d = tf.expand_dims(image_pre, 0)
print("image_pre",ssd_vgg_300.tensor_shape(image_pre))

print("bboxes_pre",ssd_vgg_300.tensor_shape(bboxes_pre))
print("bbox_img",ssd_vgg_300.tensor_shape(bbox_img))
print("image_4d",ssd_vgg_300.tensor_shape(image_4d))

#Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_params = ssd_vgg_300.SSDNet.default_params._replace(num_classes=num_classes)
ssd_net = ssd_vgg_300.SSDNet(ssd_params)

with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False)#, reuse=reuse)
    #return predictions, localisations, logits, end_points
    print("predictions", len(predictions))
    print("localisations", len(localisations))

# Restore SSD model.
# ckpt_filename = '../synthText/model.ckpt-12225'
# init_fn  = slim.assign_from_checkpoint_fn(ckpt_filename, None)
# init_fn(sess)

checkpoint_dir = '../synthText/'
saver = tf.train.Saver()
dir = tf.train.latest_checkpoint(checkpoint_dir)
saver.restore(sess, dir)


# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)


# Main image processing routine.
def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = sess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=2, decode=True)#原来是num_class = 21

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes

# Test on some demo image and visualize output.
path = '../demo/'
image_names = sorted(os.listdir(path))

img = mpimg.imread(path + image_names[-1])
rclasses, rscores, rbboxes =  process_image(img)

# visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
visualization.plt_bboxes(img, rclasses, rscores, rbboxes)



sess.close()