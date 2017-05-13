"""A factory-pattern class which returns classification image/label pairs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ssd_datasets import pascalvoc_2007

from ssd_datasets import synthText_datset

#以后好添加新的东西
datasets_map = {
    'pascalvoc_2007': pascalvoc_2007,
    'synthText': synthText_datset
}


def get_dataset(name, split_name, dataset_dir, file_pattern=None, reader=None):
    """Given a dataset name and a split_name returns a Dataset.

    Args:
        name: String, the name of the dataset.
        split_name: A train/ssd_test split name.
        dataset_dir: The directory where the dataset files are stored.
        file_pattern: The file pattern to use for matching the dataset source files.
        reader: The subclass of tf.ReaderBase. If left as `None`, then the default
            reader defined by each dataset is used.
    Returns:
        A `Dataset` class.
    Raises:
        ValueError: If the dataset `name` is unknown.
    """
    if name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % name)
    return datasets_map[name].get_split(split_name,
                                        dataset_dir,
                                        file_pattern,
                                        reader)
def get_synthText_dataset(name, dataset_dir, reader=None):
    SPLITS_TO_SIZES = {  # The number of samples in the dataset
        'train': 229,  # Aimee 5011
        'ssd_test': 233,  # Aimee 4952
    }
    ITEMS_TO_DESCRIPTIONS = {
        'image': 'A color image of varying height and width.',
        'shape': 'Shape of the image',
        'object/bbox': 'A list of bounding boxes, one per each object.',
        'object/label': 'A list of labels, one per each object.',
    }
    NUM_CLASSES = 1  # Aimee 原来是20类，我改成1类就够了
    return datasets_map[name].get_split( dataset_dir,#Aimee slite_name确定划分的量
                                       reader,
                                      ITEMS_TO_DESCRIPTIONS,#描述信息
                                      NUM_CLASSES)