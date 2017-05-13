# import tensorflow as tf
# from ssd_preprocessing import preprocessing_factory
#
# import tf_utils
# from ssd_datasets import dataset_factory
# from ssd_nets import nets_factory
#
# slim = tf.contrib.slim
#
# dataset_name = "pascalvoc_2007"
# dataset_split_name = "train"
# dataset_dir = "./ssd_tfrecords"
# model_name = "ssd_300_vgg"
# num_classes = 2
# logdir = './crnn_tmp/my/ssd_logs'
#
#
# preprocessing_name = None
#
# num_readers = 4
# batch_size = 5
#
# num_preprocessing_threads = 1
#
# weight_decay = 0.0005
# match_threshold = 0.5
# negative_ratio = 3.
# loss_alpha =1
# label_smoothing = 0
# tf.logging.set_verbosity(tf.logging.DEBUG)#设置显示的log的阈值
#
# with tf.Graph().as_default():
#     global_step = slim.create_global_step()
#
#     # Select the crnn_dataset.Aimee返回dataset元组   在pascalvoc_2007里面告之class_num
#     dataset = dataset_factory.get_dataset(
#         dataset_name, dataset_split_name, dataset_dir)
#
#     # Get the SSD network and its anchors.
#     ssd_class = nets_factory.get_network(model_name)
#     ssd_params = ssd_class.default_params._replace(num_classes=num_classes)
#     ssd_net = ssd_class(ssd_params)
#     ssd_shape = ssd_net.params.img_shape
#     ssd_anchors = ssd_net.anchors(ssd_shape)
#
#     # Select the ssd_preprocessing function.
#     preprocessing_name = preprocessing_name or model_name
#     image_preprocessing_fn = preprocessing_factory.get_preprocessing(
#         preprocessing_name, is_training=True)
#
#     # =================================================================== #
#     # Create a crnn_dataset provider and batches.
#     # =================================================================== #
#
#     with tf.name_scope(dataset_name + '_data_provider'):
#         provider = slim.dataset_data_provider.DatasetDataProvider(
#             dataset,
#             num_readers=num_readers,
#             common_queue_capacity=20 * batch_size,
#             common_queue_min=10 * batch_size,
#             shuffle=True)
#     # Get for SSD network: image, labels, bboxes.
#     [image, shape, glabels, gbboxes] = provider.get(['image', 'shape',
#                                                      'object/label',
#                                                      'object/bbox'])
#
#     # Pre-processing image, labels and bboxes.
#     image, glabels, gbboxes = \
#         image_preprocessing_fn(image, glabels, gbboxes, ssd_shape)
#     # Encode groundtruth labels and bboxes.
#     gclasses, glocalisations, gscores = \
#         ssd_net.bboxes_encode(glabels, gbboxes, ssd_anchors)
#     batch_shape = [1] + [len(ssd_anchors)] * 3  # batch_shape is list [1,6,6,6]
#     # gclasses,gscores 是一个list ，里面放着：tensor(shape(38,38,6)),(19,19,7),(10,10,7),(5,5,7),(3,3,7),(1,1,7)
#     # Training batches and queue.
#     r = tf.train.batch(
#         crnn_tf_utils.reshape_list([image, gclasses, glocalisations, gscores]),
#         batch_size=batch_size,
#         num_threads=num_preprocessing_threads,
#         capacity=5 * batch_size)  # The maximum number of elements in the queue.
#     # r is list (5,300,300,3)(5,38,38,6)(5,19,19,7)(5,10,10,7)(5,5,5,7)(5,3,3,7)(5,1,1,7)(5,38,38,6,4)(5,19,19,7,4)(5,10,10,7,4)(5,5,5,7,4)(5,3,3,7,4)(5,1,1,7,4)(5,38,38,6)(5,19,19,7)(5,10,10,7)(5,5,5,7)(5,1,1,7)
#     b_image, b_gclasses, b_glocalisations, b_gscores = \
#         crnn_tf_utils.reshape_list(r, batch_shape)  # Aimee 就是把r中的内容对应的分开
#
#
#     def setNet():
#         # Construct SSD network.
#         arg_scope = ssd_net.arg_scope(weight_decay= weight_decay)
#         with slim.arg_scope(arg_scope):
#             predictions, localisations, logits, end_points = \
#                 ssd_net.net(b_image, is_training=True)
#         # Add loss function.
#         return ssd_net.losses_my(logits, localisations,
#                        b_gclasses, b_glocalisations, b_gscores,
#                        match_threshold= match_threshold,
#                        negative_ratio= negative_ratio,
#                        alpha= loss_alpha,
#                        label_smoothing= label_smoothing)
#
#
#
#
#     # Gather initial summaries.
#     summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
#
#     # Add summaries for end_points.
#     setNet()
#     total_loss = tf.get_collection(tf.GraphKeys.LOSSES)
#     all_losses=[]
#     if total_loss:
#         clone_loss = tf.add_n(total_loss, name='clone_loss')
#     all_losses.append(clone_loss)
#     total_loss = tf.add_n(all_losses)
#
#
#     # Add summaries for variables.
#     for variable in slim.get_model_variables():
#         summaries.add(tf.summary.histogram(variable.op.name, variable))
#
#
#     # =================================================================== #
#     # Configure the moving averages.
#     # =================================================================== #
#
#
#     # =================================================================== #
#     # Configure the optimization procedure.
#     # =================================================================== #
#
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
#
#
#     saver = tf.train.Saver(max_to_keep=5,  # 最多保存多少ckpt
#                            keep_checkpoint_every_n_hours=1.0,
#                            # How often to keep checkpoints. Defaults to 10,000 hours.
#                            write_version=2,
#                            pad_step_number=False)
#
#     train_op = slim.learning.create_train_op(total_loss, optimizer)
#
#     slim.learning.train(
#         train_op,
#         logdir,
#         number_of_steps=1000,
#         save_summaries_secs=300,
#         save_interval_secs=60)
#
#
