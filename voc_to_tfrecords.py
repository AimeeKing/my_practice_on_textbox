"""将VOC转为TFrecords
JPEG文件在JPEGImages,同理，Annotation里放着位置信息
 training 和evalution data 分别由1024 和 128个TFrecord文件组成
 validation 包含500个records，每个training TF包含1000个，每一个TFrecord文件是序列化的，包含内容有：
image/encoded: string containing JPEG encoded image in RGB colorspace
    image/height: integer, image height in pixels
    image/width: integer, image width in pixels
    image/channels: integer, specifying the number of channels, always 3
    image/format: string, specifying the format, always'JPEG'


    image/object/bbox/xmin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/xmax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/label: list of integer specifying the classification index.
    image/object/bbox/label_text: list of string descriptions.
"""
import os
import random
import sys
import xml.etree.ElementTree as ET

import tensorflow as tf
from ssd_datasets.dataset_utils import int64_feature,float_feature,bytes_feature

from ssd_datasets.dataset_common import DIRECTORY_ANNOTATIONS,DIRECTORY_IMAGES,VOC_LABELS


def _process_image(directory, name):
    """Process a image and annotation file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    filename = directory+'\\' + DIRECTORY_IMAGES + name + '.jpg'
    print("image filename :"+filename)
    image_data = tf.gfile.FastGFile(filename, 'r').read()

    # Read the XML annotation file.
    filename = os.path.join(directory, DIRECTORY_ANNOTATIONS, name + '.xml')
    print("xml filename:"+filename)
    tree = ET.parse(filename)
    root = tree.getroot()

    # Image shape.
    size = root.find('size')

    shape = [int(size.find('width').text),#Aimee 这里我字打错了，应该是height,我弄成了hight
             int(size.find('height').text),#Aimee 这里两个我位置换过了
             int(size.find('depth').text)]#应该是depth
    # Find annotations.
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(int(VOC_LABELS[label][0]))
        labels_text.append(label.encode('ascii'))

        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)
        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)

        bbox = obj.find('bndbox')

        bboxes.append((float(bbox.find('ymin').text) / shape[0],
                       float(bbox.find('xmin').text) / shape[1],
                       float(bbox.find('ymax').text) / shape[0],
                       float(bbox.find('xmax').text) / shape[1]
                       ))
    ############################text############################
    print("labels_text")
    print(labels_text)
    print("labels")
    print(labels)

    return image_data, shape, bboxes, labels, labels_text, difficult, truncated


def _convert_to_example(image_data, labels, labels_text, bboxes, shape,
                        difficult, truncated):
    """Build an Example proto for an image example.

    Args:
      image_data: string, JPEG encoding of RGB image;
      labels: list of integers, identifier for the ground truth;
      labels_text: list of strings, human-readable labels;
      bboxes: list of bounding boxes; each box is a list of integers;
          specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
          to the same label as the image label.
      shape: 3 integers, image shapes in pixels.
    Returns:
      Example proto
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([xmin, ymin, xmax, ymax], b)]
        # pylint: enable=expression-not-assigned

    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/shape': int64_feature(shape),
            'image/object/bbox/xmin': float_feature(xmin),
            'image/object/bbox/xmax': float_feature(xmax),
            'image/object/bbox/ymin': float_feature(ymin),
            'image/object/bbox/ymax': float_feature(ymax),
            'image/object/bbox/label': int64_feature(labels),
            'image/object/bbox/label_text': bytes_feature(labels_text),
            'image/object/bbox/difficult': int64_feature(difficult),
            'image/object/bbox/truncated': int64_feature(truncated),
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data)}))
    return example


def _add_to_tfrecord(dataset_dir, name, tfrecord_writer):
    """Loads ssd_data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    image_data, shape, bboxes, labels, labels_text, difficult, truncated = \
        _process_image(dataset_dir, name)
    example = _convert_to_example(image_data, labels, labels_text,
                                  bboxes, shape, difficult, truncated)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name):
    return '%s/%s.tfrecord' % (output_dir, name)


def run(dataset_dir, output_dir, name='voc_train', shuffling=False):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    tf_filename = _get_output_filename(output_dir, name)
    if tf.gfile.Exists(tf_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return
    # Dataset filenames, and shuffling.
    path = os.path.join(dataset_dir, DIRECTORY_ANNOTATIONS)
    print("annotation path:"+path)
    filenames = sorted(os.listdir(path))
    if shuffling:
        random.seed(12345)
        random.shuffle(filenames)

    # Process dataset files.

    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for i, filename in enumerate(filenames):
            sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(filenames)))
            sys.stdout.flush()

            name = filename[:-4]
            _add_to_tfrecord(dataset_dir, name, tfrecord_writer)

    # Finally, write the labels file:
    #labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    #write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the Pascal VOC dataset!')
