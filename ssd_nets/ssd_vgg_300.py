import tensorflow as tf
from ssd_nets import custom_layers
import numpy as np
from collections import namedtuple
import math
from ssd_nets import ssd_common
import ssd_tfrecords as tfe
slim = tf.contrib.slim

# =========================================================================== #
# SSD class definition.
# =========================================================================== #
SSDParams = namedtuple('SSDParameters', ['img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'
                                         ])


class SSDNet(object):
    """Implementation of the SSD VGG-based 300 network.

    The default features layers with 300x300 image input are:
      conv4 ==> 38 x 38
      conv7 ==> 19 x 19
      conv8 ==> 10 x 10
      conv9 ==> 5 x 5
      conv10 ==> 3 x 3
      conv11 ==> 1 x 1
    The default image size used to train this network is 300x300.
    """
    default_params = SSDParams(
        img_shape=(300, 300),
        num_classes=2,#Aimee 21
        no_annotation_label=2,#Aimee 21
        feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11'],
        feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],#特征图大小
        #anchor_size_bounds=[0.15, 0.90],
        anchor_size_bounds=[0.15, 0.90],#默认框最大最小值
        # anchor_sizes=[(21., 45.),
        #               (45., 99.),
        #               (99., 153.),
        #               (153., 207.),
        #               (207., 261.),
        #               (261., 315.)],
        anchor_sizes=[(30.,114),
                      (60., 114.),
                      (114., 168.),
                      (168., 222.),
                      (222., 276.),
                      (276., 330.)],
        # anchor_ratios=[[2, .5],
        #                [2, .5, 3, 1./3],
        #                [2, .5, 3, 1./3],
        #                [2, .5, 3, 1./3],
        #                [2, .5],
        #                [2, .5]],
        anchor_ratios=[[2, 3, 5, 7, 10],
                       [2, 3, 5, 7, 10],
                       [2, 3, 5, 7, 10],
                       [2, 3, 5, 7, 10],
                       [2, 3, 5, 7, 10],
                       [2, 3, 5, 7, 10]],
        anchor_steps=[8, 16, 32, 64, 100, 300],#imageshape/steps ~ feat_shapes
        anchor_offset=0.5,
        normalizations=[20, -1, -1, -1, -1, -1],
        prior_scaling=[0.1, 0.1, 0.2, 0.2]
        )

    def __init__(self, params=None):
        """Init the SSD net with some parameters. Use the default ones
        if none provided.
        """
        if isinstance(params, SSDParams):
            self.params = params
        else:
            self.params = SSDNet.default_params

    # ======================================================================= #
    def net(self, inputs,
            is_training=True,
            update_feat_shapes=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='ssd_300_vgg'):
        """SSD network definition.
        """
        r = ssd_net(inputs,
                    num_classes=self.params.num_classes,
                    feat_layers=self.params.feat_layers,
                    anchor_sizes=self.params.anchor_sizes,
                    anchor_ratios=self.params.anchor_ratios,
                    normalizations=self.params.normalizations,
                    is_training=is_training,
                    dropout_keep_prob=dropout_keep_prob,
                    prediction_fn=prediction_fn,
                    reuse=reuse,
                    scope=scope)
        # Update feature shapes (try at least!)
        if update_feat_shapes:
            shapes = ssd_feat_shapes_from_net(r[0], self.params.feat_shapes)#shape is [[38,38,6],[19,19,7],[10,10,7],[3,3,7],[1,1,7]]
            self.params = self.params._replace(feat_shapes=shapes)
        return r

    def arg_scope(self, weight_decay=0.0005, data_format='NHWC'):
        """Network arg_scope.
        """
        return ssd_arg_scope(weight_decay, data_format=data_format)

    # def arg_scope_caffe(self, caffe_scope):
    #     """Caffe arg_scope used for weights importing.
    #     """
    #     return ssd_arg_scope_caffe(caffe_scope)

    # ======================================================================= #
    def update_feature_shapes(self, predictions):
        """Update feature shapes from predictions collection (Tensor or Numpy
        array).
        """
        shapes = ssd_feat_shapes_from_net(predictions, self.params.feat_shapes)
        self.params = self.params._replace(feat_shapes=shapes)

    def anchors(self, img_shape, dtype=np.float32):
        """Compute the default anchor boxes, given an image shape.
        """
        return ssd_anchors_all_layers(img_shape,
                                      self.params.feat_shapes,
                                      self.params.anchor_sizes,
                                      self.params.anchor_ratios,
                                      self.params.anchor_steps,
                                      self.params.anchor_offset,
                                      dtype)

    def bboxes_encode(self, labels, bboxes, anchors,
                      scope=None):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_encode(
            labels, bboxes, anchors,
            self.params.num_classes,
            self.params.no_annotation_label,
            ignore_threshold=0.5,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def bboxes_decode(self, feat_localizations, anchors,
                      scope='ssd_bboxes_decode'):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_decode(
            feat_localizations, anchors,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def detected_bboxes(self, predictions, localisations,
                        select_threshold=None, nms_threshold=0.5,
                        clipping_bbox=None, top_k=400, keep_top_k=200):
        """Get the detected bounding boxes from the SSD network output.
        """
        # Select top_k bboxes from predictions, and clip
        rscores, rbboxes = \
            ssd_common.tf_ssd_bboxes_select(predictions, localisations,
                                            select_threshold=select_threshold,
                                            num_classes=self.params.num_classes)
        rscores, rbboxes = \
            tfe.bboxes_sort(rscores, rbboxes, top_k=top_k)
        # Apply NMS algorithm.
        rscores, rbboxes = \
            tfe.bboxes_nms_batch(rscores, rbboxes,
                                 nms_threshold=nms_threshold,
                                 keep_top_k=keep_top_k)
        if clipping_bbox is not None:
            rbboxes = tfe.bboxes_clip(clipping_bbox, rbboxes)
        return rscores, rbboxes

    def losses(self, logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               scope='ssd_losses'):
        """Define the SSD network losses.
        """
        return ssd_losses(logits, localisations,
                          gclasses, glocalisations, gscores,
                          match_threshold=match_threshold,
                          negative_ratio=negative_ratio,
                          alpha=alpha,
                          label_smoothing=label_smoothing,
                          scope=scope)



    #############################Aimee#############################
    def losses_my(self, logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               scope='ssd_losses'):
        """Define the SSD network losses.
        """
        return ssd_losses_my(logits, localisations,
                          gclasses, glocalisations, gscores,
                          match_threshold=match_threshold,
                          negative_ratio=negative_ratio,
                          alpha=alpha,
                          label_smoothing=label_smoothing,
                          scope=scope)


# =========================================================================== #
# SSD tools...
# =========================================================================== #

#Aimee 有两个选项，”NHWC”（默认）, “NCHW”。指明输入数据和输出数据的格式，
    # 对于”NHWC”，数据存储的格式[batch, in_height, in_width, in_channels]
    # 对于”NCHW”, 数据存储顺序为: [batch, in_channels, in_height, in_width].
def ssd_arg_scope(weight_decay=0.0005, data_format='NHWC'):
    """Defines the VGG arg scope.

    Args:
      weight_decay: The l2 regularization coefficient.

    Returns:
      An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),#Aimee 为了使得网络中信息更好的流动，每一层输出的方差应该尽量相等,所以用这个处事方法
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            data_format=data_format):
            with slim.arg_scope([custom_layers.pad2d,
                                 custom_layers.l2_normalization,
                                 custom_layers.channel_to_last],
                                data_format=data_format) as sc:
                return sc

def ssd_feat_shapes_from_net(predictions, default_shapes=None):
    """Try to obtain the feature shapes from the prediction layers. The latter
    can be either a Tensor or Numpy ndarray.

    Return:
      list of feature shapes. Default values if predictions shape not fully
      determined.
    """
    feat_shapes = []
    for l in predictions:
        # Get the shape, from either a np array or a tensor.
        if isinstance(l, np.ndarray):
            shape = l.shape
        else:
            shape = l.get_shape().as_list()
        shape = shape[1:4]
        # Problem: undetermined shape...
        if None in shape:
            return default_shapes
        else:
            feat_shapes.append(shape)
    return feat_shapes


def ssd_anchor_one_layer(img_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         step,
                         offset=0.5,
                         dtype=np.float32):
    """
    Aimee返回某一层的anchor的配置信息
    Computer SSD default anchor boxes for one feature layer.

    Determine the relative position grid of the centers, and the relative
    width and height.

    Arguments:
      feat_shape: Feature shape, used for computing relative position grids;
      size: Absolute reference sizes;
      ratios: Ratios to use on these features;
      img_shape: Image shape, used for computing height, width relatively to the
        former;
      offset: Grid offset.

    Return:
      y, x, h, w: Relative x and y grids, and height and width.
    """
    # Compute the position grid: simple way.
    # y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    # y = (y.astype(dtype) + offset) / feat_shape[0]
    # x = (x.astype(dtype) + offset) / feat_shape[1]
    # Weird SSD-Caffe computation using steps values...
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = len(sizes) + len(ratios)
    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)
    # Add first anchor boxes with ratio=1.
    h[0] = sizes[0] / img_shape[0]
    w[0] = sizes[0] / img_shape[1]
    di = 1
    if len(sizes) > 1:
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        di += 1
    for i, r in enumerate(ratios):
        h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
    return y, x, h, w


def ssd_anchors_all_layers(img_shape,
                           layers_shape,
                           anchor_sizes,
                           anchor_ratios,
                           anchor_steps,
                           offset=0.5,
                           dtype=np.float32):
    """Compute anchor boxes for all feature layers.
    """
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = ssd_anchor_one_layer(img_shape, s,
                                             anchor_sizes[i],
                                             anchor_ratios[i],
                                             anchor_steps[i],
                                             offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors

# =========================================================================== #
# Functional definition of VGG-based SSD 300.
# =========================================================================== #
def tensor_shape(x, rank=3):
    """Returns the dimensions of a tensor.
    Args:
      image: A N-D Tensor of shape.
    Returns:
      A list of dimensions. Dimensions that are statically known are python
        integers,otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]



def ssd_multibox_layer(inputs,
                       num_classes,
                       sizes,
                       ratios=[1],
                       normalization=-1,
                       bn_normalization=False):
    """Construct a multibox layer, return a class and localization predictions.
    """
    net = inputs
    if normalization > 0:
        net = custom_layers.l2_normalization(net, scaling=True)
    # Number of anchors.
    num_anchors = len(sizes) + len(ratios)

    # Location.
    num_loc_pred = num_anchors * 4
    # block4_box num_loc_pred 48 56 56 56 56 56
    #

    ###############Aimee 新加的一句代码#####################
    net = custom_layers.pad2d(net,pad=(0,2))
    print("net :",tensor_shape(net))
    loc_pred = slim.conv2d(net, num_loc_pred,[1, 5],padding = 'VALID' ,activation_fn=None,#Aimee [3,3]改成了[1,5]
                           scope='conv_loc')

    print("loc_predshape:",tensor_shape(loc_pred))

    loc_pred = custom_layers.channel_to_last(loc_pred)

    #Aimee 把输入矩阵 n * c * h * w 转换为向量n * (c*h*w) * 1 * 1.
    loc_pred = tf.reshape(loc_pred,
                          tensor_shape(loc_pred, 4)[:-1]+[num_anchors, 4])

    # Class prediction.
    num_cls_pred = num_anchors * num_classes
    #block4_box num_cls_pred = 24 28 28 28 28 28
    #

    cls_pred = slim.conv2d(net, num_cls_pred, [1, 5],padding = 'VALID', activation_fn=None,#Aimee [3,3]改成了[1,5]
                           scope='conv_cls')

    print("cls_preshape",tensor_shape(cls_pred))

    cls_pred = custom_layers.channel_to_last(cls_pred)
    cls_pred = tf.reshape(cls_pred,
                          tensor_shape(cls_pred, 4)[:-1]+[num_anchors, num_classes])

    #######################Aimee######################################
    print("after loc_predshape:",tensor_shape(loc_pred))
    print("after cls_preshape",tensor_shape(cls_pred))

    return cls_pred, loc_pred


def ssd_net(inputs,
            num_classes=SSDNet.default_params.num_classes,
            feat_layers=SSDNet.default_params.feat_layers,
            anchor_sizes=SSDNet.default_params.anchor_sizes,
            anchor_ratios=SSDNet.default_params.anchor_ratios,
            normalizations=SSDNet.default_params.normalizations,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='ssd_300_vgg'):
    """SSD net definition.
    """
    # if data_format == 'NCHW':
    #     inputs = tf.transpose(inputs, perm=(0, 3, 1, 2))

    # End_points collect relevant activations for external use.
    end_points = {}
    with tf.variable_scope(scope, 'ssd_300_vgg', [inputs], reuse=reuse):
        # Original VGG-16 blocks.
        print("input:shape", (input))
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        end_points['block1'] = net
        print("block1:shape", tensor_shape(net))
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        # Block 2.

        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        end_points['block2'] = net
        print("block2:shape", tensor_shape(net))
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        # Block 3.

        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        end_points['block3'] = net
        print("block3:shape", tensor_shape(net))
        net = slim.max_pool2d(net, [2, 2], scope='pool3')

        # Block 4.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        end_points['block4'] = net
        print("block4:shape", tensor_shape(net))
        net = slim.max_pool2d(net, [2, 2], scope='pool4')


        # Block 5.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        end_points['block5'] = net
        net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool5')
        print("block5:shape", tensor_shape(net))

        # Additional SSD blocks.
        # Block 6: let's dilate the hell out of it!
        net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')#Aimee padding = 6 新放入
        end_points['block6'] = net
        print("block6:shape", tensor_shape(net))

        # Block 7: 1x1 conv. Because the fuck.
        net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
        end_points['block7'] = net
        print("block7:shape", tensor_shape(net))

        # Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
        end_point = 'block8'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')#Aimee padding='VALID'
            net = custom_layers.pad2d(net, pad=(1, 1))
            net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding = 'VALID')#Aimee padding = "VALID
        end_points[end_point] = net
        print("block8:shape", tensor_shape(net))

        end_point = 'block9'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = custom_layers.pad2d(net, pad=(1, 1))
            net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        end_points[end_point] = net

        print("block9:shape", tensor_shape(net))

        end_point = 'block10'
        #Aimee block10:shape[1, 256, 3, 3]
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
        end_points[end_point] = net
        print("block10:shape", tensor_shape(net))

        end_point = 'block11'
        #Aimee block11:shape [1, 127, 1, 3]
        with tf.variable_scope(end_point):
            # net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            # net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
            net = slim.avg_pool2d(net,[3,3],scope='pool6')
            print("block11:shape", tensor_shape(net))
        end_points[end_point] = net

        # Prediction and localisations layers.
        predictions = []
        logits = []
        localisations = []
        for i, layer in enumerate(feat_layers):
            with tf.variable_scope(layer + '_box'):
                p, l = ssd_multibox_layer(end_points[layer],
                                          num_classes,
                                          anchor_sizes[i],
                                          anchor_ratios[i],
                                          normalizations[i])
            predictions.append(prediction_fn(p))
            logits.append(p)
            localisations.append(l)

        return predictions, localisations, logits, end_points
ssd_net.default_image_size = 300


def ssd_losses(logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,#和groundtruth bbox jaccard重叠阈值超过0.5为正样本
               negative_ratio=3.,#Aimee 保持正负样本1:3的比例
               alpha=1.,
               label_smoothing=0.,
               scope=None):
    """Loss functions for training the SSD 300 VGG network.

    This function defines the different loss components of the SSD, and
    adds them to the TF loss collection.

    Arguments:
      logits: (list of) predictions logits Tensors;
      localisations: (list of) localisations Tensors;
      gclasses: (list of) groundtruth labels Tensors;
      glocalisations: (list of) groundtruth localisations Tensors;
      gscores: (list of) groundtruth score Tensors;
    """
    with tf.name_scope(scope, 'ssd_losses'):
        l_cross_pos = []
        l_cross_neg = []
        l_loc = []
        l_total =[]
        for i in range(len(logits)):#len(logits) = 6
            dtype = logits[i].dtype
            with tf.name_scope('block_%i' % i):
                # Determine weights Tensor.
                pmask = gscores[i] > match_threshold
                fpmask = tf.cast(pmask, dtype)
                n_positives = tf.reduce_sum(fpmask)#Aimee 正样本个数

                # Select some random negative entries.
                # n_entries = np.prod(gclasses[i].get_shape().as_list())
                # r_positive = n_positives / n_entries
                # r_negative = negative_ratio * n_positives / (n_entries - n_positives)

                # Negative mask.
                no_classes = tf.cast(pmask, tf.int32)
                predictions = slim.softmax(logits[i])
                nmask = tf.logical_and(tf.logical_not(pmask),
                                       gscores[i] > -0.5)
                fnmask = tf.cast(nmask, dtype)

                nvalues = tf.where(nmask,
                                   predictions[:, :, :, :, 0],
                                   1. - fnmask)

                nvalues_flat = tf.reshape(nvalues, [-1])
                # Number of negative entries to select.
                n_neg = tf.cast(negative_ratio * n_positives, tf.int32)#正负比例1:3
                n_neg = tf.maximum(n_neg, tf.size(nvalues_flat) // 8)#为什么是8
                n_neg = tf.maximum(n_neg, tf.shape(nvalues)[0] * 4)#大于20
                max_neg_entries = 1 + tf.cast(tf.reduce_sum(fnmask), tf.int32)
                n_neg = tf.minimum(n_neg, max_neg_entries)

                val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
                minval = val[-1]
                # Final negative mask.
                nmask = tf.logical_and(nmask, -nvalues > minval)
                fnmask = tf.cast(nmask, dtype)

                # Add cross-entropy loss.
                with tf.name_scope('cross_entropy_pos'):
                    loss_cross_pos_get = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],
                                                                          labels=gclasses[i])
                    loss_cross_pos_get = tf.losses.compute_weighted_loss(loss_cross_pos_get, fpmask)
                    l_cross_pos.append(loss_cross_pos_get)


                with tf.name_scope('cross_entropy_neg'):
                    loss_cross_neg_get = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],
                                                                          labels=no_classes)
                    loss_cross_neg_get = tf.losses.compute_weighted_loss(loss_cross_neg_get, fnmask)#loss_collection="loss_cross_neg"
                    l_cross_neg.append(loss_cross_neg_get)


                # Add localization loss: smooth L1, L2, ...
                with tf.name_scope('localization'):
                    # Weights Tensor: positive mask + random negative.
                    weights = tf.expand_dims(alpha * fpmask, axis=-1)
                    loss_loc_get = custom_layers.abs_smooth(localisations[i] - glocalisations[i])
                    loss_loc_get = tf.losses.compute_weighted_loss(loss_loc_get, weights)
                    l_loc.append(loss_loc_get)
                # with tf.name_scope('total'):
                #     loss = loss_cross_pos_get + loss_cross_neg_get + loss_loc_get
                #     #mask = n_positives > 0
                #     #loss = tf.where(mask,
                #     #                tf.div(loss, n_positives),
                #     #                tf.cast(mask, dtype))
                #     #loss = tf.where(tf.is_nan(loss), tf.cast(mask, tf.float32), loss)
                #     tf.losses.add_loss(loss)
                #     l_total.append(loss)



        # Additional total losses...
        with tf.name_scope('total'):
            total_cross_pos = tf.add_n(l_cross_pos, 'cross_entropy_pos')
            total_cross_neg = tf.add_n(l_cross_neg, 'cross_entropy_neg')
            total_cross = tf.add(total_cross_pos, total_cross_neg, 'cross_entropy')
            total_loc = tf.add_n(l_loc, 'localization')




            # Add to EXTRA LOSSES TF.collection
            tf.add_to_collection('EXTRA_LOSSES', total_cross_pos)
            tf.add_to_collection('EXTRA_LOSSES', total_cross_neg)
            tf.add_to_collection('EXTRA_LOSSES', total_cross)
            tf.add_to_collection('EXTRA_LOSSES', total_loc)



############################################Aimee##############################################

def ssd_losses_my(logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,  # 和groundtruth bbox jaccard重叠阈值超过0.5为正样本
               negative_ratio=3.,  # Aimee 保持正负样本1:3的比例
               alpha=1.,
               label_smoothing=0.,
               scope=None):
    """Loss functions for training the SSD 300 VGG network.

    This function defines the different loss components of the SSD, and
    adds them to the TF loss collection.

    Arguments:
      logits: (list of) predictions logits Tensors;
      localisations: (list of) localisations Tensors;
      gclasses: (list of) groundtruth labels Tensors;
      glocalisations: (list of) groundtruth localisations Tensors;
      gscores: (list of) groundtruth score Tensors;
    """
    # Some debugging...
    # for i in range(len(gclasses)):
    #     print(localisations[i].get_shape())
    #     print(logits[i].get_shape())
    #     print(gclasses[i].get_shape())
    #     print(glocalisations[i].get_shape())
    #     print()
    with tf.name_scope(scope):
        l_cross_pos = []
        l_cross_neg = []
        l_loc = []
        l_total =[]
        for i in range(len(logits)):#len(logits) = 6
            dtype = logits[i].dtype
            with tf.name_scope('block_%i' % i):
                # Determine weights Tensor.
                pmask = gscores[i] > match_threshold
                fpmask = tf.cast(pmask, dtype)
                n_positives = tf.reduce_sum(fpmask)#Aimee 正样本个数

                # Select some random negative entries.
                # n_entries = np.prod(gclasses[i].get_shape().as_list())
                # r_positive = n_positives / n_entries
                # r_negative = negative_ratio * n_positives / (n_entries - n_positives)

                # Negative mask.
                no_classes = tf.cast(pmask, tf.int32)
                predictions = slim.softmax(logits[i])
                nmask = tf.logical_and(tf.logical_not(pmask),
                                       gscores[i] > -0.5)
                fnmask = tf.cast(nmask, dtype)

                nvalues = tf.where(nmask,
                                   predictions[:, :, :, :, 0],
                                   1. - fnmask)

                nvalues_flat = tf.reshape(nvalues, [-1])
                # Number of negative entries to select.
                n_neg = tf.cast(negative_ratio * n_positives, tf.int32)#正负比例1:3
                n_neg = tf.maximum(n_neg, tf.size(nvalues_flat) // 8)#为什么是8
                n_neg = tf.maximum(n_neg, tf.shape(nvalues)[0] * 4)#大于20
                max_neg_entries = 1 + tf.cast(tf.reduce_sum(fnmask), tf.int32)
                n_neg = tf.minimum(n_neg, max_neg_entries)

                #######################Aimee#####################################

                print("n_neg:shape", tensor_shape(n_neg))
                print("tf.size(nvalues_flat):",tensor_shape(tf.size(nvalues_flat)))

                val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
                minval = val[-1]
                # Final negative mask.
                nmask = tf.logical_and(nmask, -nvalues > minval)
                fnmask = tf.cast(nmask, dtype)

                # Add cross-entropy loss.
                with tf.name_scope('cross_entropy_pos'):
                    loss_cross_pos_get = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],
                                                                          labels=gclasses[i])
                    loss_cross_pos_get = tf.losses.compute_weighted_loss(loss_cross_pos_get, fpmask,loss_collection="loss_cross")
                    l_cross_pos.append(loss_cross_pos_get)
                    print("loss_cross_pos_get shape", tensor_shape(loss_cross_pos_get))

                with tf.name_scope('cross_entropy_neg'):
                    loss_cross_neg_get = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],
                                                                          labels=no_classes)
                    loss_cross_neg_get = tf.losses.compute_weighted_loss(loss_cross_neg_get, fnmask,loss_collection="loss_cross")
                    l_cross_neg.append(loss_cross_neg_get)
                print("loss_cross_neg_get shape", tensor_shape(loss_cross_neg_get))

                # Add localization loss: smooth L1, L2, ...
                with tf.name_scope('localization'):
                    # Weights Tensor: positive mask + random negative.
                    weights = tf.expand_dims(alpha * fpmask, axis=-1)
                    loss_loc_get = custom_layers.abs_smooth(localisations[i] - glocalisations[i])
                    loss_loc_get = tf.losses.compute_weighted_loss(loss_loc_get, weights,loss_collection="loss_loc")
                    l_loc.append(loss_loc_get)
                    print("loss_loc_get shape", tensor_shape(loss_loc_get))

                with tf.name_scope('total'):
                    #los_cross_neg_get >0 7.3140 ;loss_cross_pos_get 5.4568,4.0713;#loss_loc_get = 2.774,5.7814
                    #n_positives 36.5199,104.5119,11.5199
                    # Weights Tensor: positive mask + random negative.
                    loss_total = loss_loc_get + loss_cross_neg_get +loss_cross_pos_get
                    print("loss_total shape",tensor_shape(loss_total))
                    mask = n_positives>0
                    loss = tf.where(mask,
                             tf.div(loss_total, n_positives),
                                    tf.cast(mask, dtype))
                    loss = tf.where(tf.is_nan(loss),tf.cast(mask, tf.float32),loss)
                    tf.losses.add_loss(loss)
                    l_total.append(loss)

        # Additional total losses...
        with tf.name_scope('total'):
            total_cross_pos = tf.add_n(l_cross_pos, 'cross_entropy_pos')
            total_cross_neg = tf.add_n(l_cross_neg, 'cross_entropy_neg')
            total_cross = tf.add(total_cross_pos, total_cross_neg, 'cross_entropy')
            total_loc = tf.add_n(l_loc, 'localization')




            # Add to EXTRA LOSSES TF.collection
            tf.add_to_collection('EXTRA_LOSSES', total_cross_pos)
            tf.add_to_collection('EXTRA_LOSSES', total_cross_neg)
            tf.add_to_collection('EXTRA_LOSSES', total_cross)
            tf.add_to_collection('EXTRA_LOSSES', total_loc)

