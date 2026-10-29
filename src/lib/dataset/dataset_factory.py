from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .datasets.custom_dataset import CustomDataset

dataset_factory = {
  'custom': CustomDataset,
}


def get_dataset(dataset):
  return dataset_factory[dataset]
