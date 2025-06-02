import json
import re
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset

DATASET_ROOT = 'datasets'


class DatasetLoader(object):
    def __init__(self,
                 dataset_name,
                 source_dataset_name,
                 dataset_version,
                 has_valid,
                 split_map,
                 batch_size,
                 train_batch_idxs,
                 test_batch_idxs,
                 valid_batch_idxs=None):
        self.data_root = DATASET_ROOT
        self.dataset_name = dataset_name
        self.source_dataset_name = source_dataset_name
        self.dataset_version = dataset_version
        self.has_valid = has_valid
        self.split_map = split_map

        self.batch_size = batch_size
        self.train_batch_idxs = train_batch_idxs
        self.test_batch_idxs = test_batch_idxs
        self.valid_batch_idxs = valid_batch_idxs

        assert self.split_map is not None

    def load_from_source(self):
        if self.source_dataset_name is None:
            self.source_dataset_name = self.dataset_name
        if self.dataset_version is None:
            datasets = load_dataset(self.source_dataset_name)
        else:
            datasets = load_dataset(self.source_dataset_name,
                                    self.dataset_version)
        return datasets

    def to_json(self, datasets):
        for k, v in self.split_map.items():
            datasets[v].to_json(
                f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_{k}.json'
            )

    def load_from_json(self):
        data_files = {
            'train':
            f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_train.json',
            'test':
            f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_test.json',
        }

        if self.has_valid:
            data_files.update({
                'valid':
                f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_valid.json',
            })

        datasets = load_dataset('json', data_files=data_files)
        datasets = self._post_process(datasets)

        # subsample training dataset if needed
        num_train = len(datasets['train'])
        idxs = list()
        for idx in self.train_batch_idxs:
            idxs += range(idx * self.batch_size, (idx + 1) * self.batch_size)
        datasets['train'] = Dataset.from_dict(
            datasets['train'][[idx for idx in idxs if idx < num_train]])

        return datasets

    def load_llm_preds(self, split):
        labels = list()
        rationales = list()
        for idx in getattr(self, f'{split}_batch_idxs'):
            with open(
                    f'{self.data_root}/{self.dataset_name}/llm/{split}_CoT_{idx}.json'
            ) as f:
                outputs = json.load(f)

            for output in outputs:
                rationale, label = self._parse_llm_output(output)

                rationales.append(rationale)
                labels.append(label)

        return rationales, labels

    def load_gpt_preds(self, split):
        labels = list()
        rationales = list()

        with open(f'{self.data_root}/gpt-neox/{self.dataset_name}/{split}.json'
                  ) as f:
            outputs = json.load(f)

        for output in outputs:
            rationale, label = self._parse_gpt_output(output)

            rationales.append(rationale)
            labels.append(label)

        return rationales, labels

    def _post_process(self, datasets):
        raise NotImplementedError

    def _parse_llm_output(self, output):
        raise NotImplementedError

    def _parse_gpt_output(self, output):
        raise NotImplementedError


class SVAMPDatasetLoader(DatasetLoader):
    def __init__(self):
        dataset_name = 'svamp'
        source_dataset_name = 'svamp'
        dataset_version = None
        has_valid = False
        split_map = {
            'train': 'train',
            'test': 'test',
        }
        batch_size = 500
        train_batch_idxs = range(2)
        test_batch_idxs = range(1)

        super().__init__(dataset_name,
                         source_dataset_name,
                         dataset_version,
                         has_valid,
                         split_map,
                         batch_size,
                         train_batch_idxs,
                         test_batch_idxs,
                         valid_batch_idxs=None)

    def load_from_source(self):
        with open(f'{self.data_root}/{self.dataset_name}/SVAMP.json') as f:
            original_dataset = json.load(f)

        dataset = list()
        for data in original_dataset:
            input = f'{data["Body"]}\n{data["Question"]}'
            equation = data["Equation"]

            dataset.append({
                'input': input,
                'label': equation,
            })

        idxs = np.random.RandomState(seed=0).permutation(len(dataset))
        train_idxs = idxs[:800]
        test_idxs = idxs[800:]

        train_dataset = Dataset.from_list(
            np.array(dataset)[train_idxs].tolist())
        test_dataset = Dataset.from_list(np.array(dataset)[test_idxs].tolist())

        datasets = DatasetDict({'train': train_dataset, 'test': test_dataset})

        return datasets

    def _post_process(self, datasets):
        return datasets

    def _parse_llm_output(self, output):
        rationale_label = output.split('Q:')[0]
        rationale_label = rationale_label.rstrip()
        try:
            rationale, label = rationale_label.split('The answer is')
        except:
            rationale = ' '
            label = ' '
            return rationale, label

        rationale = rationale.rstrip()
        try:
            parsed = re.search(r'\(.*\)', label)
            if parsed:
                label = parsed.group(0)
            else:
                label = ' '
        except:
            label = ' '

        return rationale, label

    def _parse_gpt_output(self, output):
        rationale_label = output.split('Q:')[0]
        rationale_label = rationale_label.rstrip().lstrip()
        try:
            rationale, label = rationale_label.split('The answer is')
        except:
            rationale = ' '
            label = ' '
            return rationale, label

        rationale = rationale.rstrip()
        try:
            parsed = re.search(r'\(.*\)', label)
            if parsed:
                label = parsed.group(0)
            else:
                label = ' '
        except:
            label = ' '

        return rationale, label
