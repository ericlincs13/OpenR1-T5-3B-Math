import json
import os
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset

DATASET_ROOT = 'datasets'


class HendrycksDatasetLoader(object):
    def __init__(self):
        self.data_root = DATASET_ROOT
        self.dataset_name = 'hendrycks_math'
        self.source_dataset_name = 'hendrycks_math'
        self.dataset_version = None
        self.has_valid = False
        self.split_map = {
            'train': 'train',
            'test': 'test',
        }

        self.batch_size = 16
        self.train_batch_idxs = range(2)
        self.test_batch_idxs = range(1)
        self.valid_batch_idxs = None

        assert self.split_map is not None

    def load_from_source(self, category=None):
        if category is not None:
            with open(
                    f'{self.data_root}/{self.dataset_name}/train_{category}_.json'
            ) as f:
                original_train_dataset = json.load(f)
            with open(
                    f'{self.data_root}/{self.dataset_name}/test_{category}_.json'
            ) as f:
                original_test_dataset = json.load(f)
        else:
            original_train_dataset = []
            original_test_dataset = []
            for name in [
                    f for f in os.listdir(
                        f'{self.data_root}/{self.dataset_name}')
                    if f.startswith('train_')
            ]:
                with open(f'{self.data_root}/{self.dataset_name}/{name}') as f:
                    original_train_dataset.extend(json.load(f))
            for name in [
                    f for f in os.listdir(
                        f'{self.data_root}/{self.dataset_name}')
                    if f.startswith('test_')
            ]:
                with open(f'{self.data_root}/{self.dataset_name}/{name}') as f:
                    original_test_dataset.extend(json.load(f))

        dataset = list()
        for data in original_train_dataset:
            dataset.append({
                'input': data["input"],
                'process': data["process"],
                'label': data["label"],
            })

        train_idxs = np.random.RandomState(seed=0).permutation(
            len(original_train_dataset))
        test_idxs = np.random.RandomState(seed=0).permutation(
            len(original_test_dataset))

        train_dataset = Dataset.from_list(
            np.array(original_train_dataset)[train_idxs].tolist())
        test_dataset = Dataset.from_list(
            np.array(original_test_dataset)[test_idxs].tolist())

        datasets = DatasetDict({'train': train_dataset, 'test': test_dataset})

        return datasets
