"""Provides data for the Pascal VOC Dataset (images + annotations).
"""
import tensorflow as tf

from ssd_datasets import pascalvoc_common

slim = tf.contrib.slim

FILE_PATTERN = 'voc_2007_%s.tfrecord'#Aimee 需要的tf的文件名是这里确定的
ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'shape': 'Shape of the image',
    'object/bbox': 'A list of bounding boxes, one per each object.',
    'object/label': 'A list of labels, one per each object.',
}
SPLITS_TO_SIZES = {#The number of samples in the dataset
    'train': 229,#Aimee 5011
    'ssd_test': 233,#Aimee 4952
}
NUM_CLASSES = 1#Aimee 原来是20类，我改成1类就够了


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):#file_pattern tf文件名的正则表达，确定去读取那个tf文件
    """Gets a dataset tuple with instructions for reading ImageNet.

    Args:
      split_name: A train/ssd_test split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
        ValueError: if `split_name` is not a valid train/ssd_test split.
    """
    if not file_pattern:
        file_pattern = FILE_PATTERN
    return pascalvoc_common.get_split(split_name, dataset_dir,  #Aimee slite_name确定划分的量
                                      file_pattern, reader,
                                      SPLITS_TO_SIZES,
                                      ITEMS_TO_DESCRIPTIONS,  #描述信息
                                      NUM_CLASSES)
