from typing import Union, List, Tuple, Sequence, Dict, Any
from functools import lru_cache

from pathlib import Path
import pickle as pkl

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.distance import pdist, squareform

from fairseq.data import FairseqDataset, plasma_utils

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
import contextlib
import itertools
import logging
import os
import warnings
from typing import Optional, Tuple

from .tokenizers import TAPETokenizer

logger = logging.getLogger(__name__)


class SSADataset(FairseqDataset):

    def __init__(self, path, pad_idx=0, nums=None):  # , append_eos=True, reverse_order=False):
        # load the contact maps first
        self.ssa_matrix = plasma_utils.PlasmaArray(np.load(path, allow_pickle=True))

        self.size = len(self.ssa_matrix.array)
        self.sizes = 1  # only one float num for ordinal regression
        self.pad_idx = pad_idx

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        return i

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.sizes

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.sizes

    def collate_tokens(self, values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False,
                       pad_to_length=None):
        """Convert a list of 1d tensors into a padded 2d tensor.
        samples, self.pad_idx, left_pad
        """
        assert len(values) % 2 == 0, "error with SSA's paired scheme, set --required-batch-size-multiple 2"
        half_batch_len = len(values) // 2
        res = torch.as_tensor(
            [self.ssa_matrix.array[values[i], values[i + half_batch_len]] for i in range(half_batch_len)])

        return res

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        return self.collate_tokens(samples, self.pad_idx)


class PSSMDataset(FairseqDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path, pad_idx=0, num=None):  # , append_eos=True, reverse_order=False):
        # load the contact maps first
        totaldata = []
        # if path == 'None':
        for i in range(len(path)):
            p = path[i]
            if p == 'None':
                tmp = [np.array([[-1]*20])] * num[i]
                # tmp = np.array(tmp, dtype=object)
                totaldata.extend(tmp)
            else:
                totaldata.extend(np.load(p, allow_pickle=True))

        self.pad_idx = pad_idx
        self.pssms = plasma_utils.PlasmaArray(np.array(totaldata, dtype=object))
        self.size = len(self.pssms.array)
        self.sizes = [len(line) for line in self.pssms.array]

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        return self.pssms.array[i]

    # def get_original_text(self, i):
    #     return self.pssms[i][0]

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    def collate_tokens(self, values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False,
                       pad_to_length=None):
        """Convert a list of 1d tensors into a padded 2d tensor.
        samples, self.pad_idx, left_pad
        """
        size = max(len(v) for v in values) + 2  # for padding start and end token
        # size = size if pad_to_length is None else max(size, pad_to_length)
        res = [
            np.pad(value, ((1, size - len(value) - 1), (0, 0)), mode='constant',
                   constant_values=pad_idx) for value in values]
        res = torch.as_tensor(res)

        return res

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        return self.collate_tokens(samples, self.pad_idx)


class ContactMapsDataset(FairseqDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path, label_pad_idx=-1, nums=None):  # , append_eos=True, reverse_order=False):
        # load the contact maps first
        self.cmaps = plasma_utils.PlasmaArray(np.load(path, allow_pickle=True))
        self.size = len(self.cmaps.array)
        self.sizes = [len(cmap) for cmap in self.cmaps.array]
        self.label_pad_idx = label_pad_idx

        # self.append_eos = append_eos
        # self.reverse_order = reverse_order

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        return self.cmaps.array[i]

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    def collate_tokens(self, values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False,
                       pad_to_length=None):
        """Convert a list of 1d tensors into a padded 2d tensor.
        samples, self.pad_idx, left_pad
        """
        size = max(len(v) for v in values)  # + 2  # no need to pad extra 2 tokens, for padding start and end token
        # size = size if pad_to_length is None else max(size, pad_to_length)
        res = [
            np.pad(value, ((0, size - len(value)), (0, size - len(value))), mode='constant', constant_values=0)
            for value in values]
        res = torch.as_tensor(res)

        return res

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        return self.collate_tokens(samples, self.label_pad_idx)


def dataset_factory(data_file: Union[str, Path], *args, **kwargs) -> Dataset:
    data_file = Path(data_file)
    if not data_file.exists():
        raise FileNotFoundError(data_file)
    if data_file.suffix == '.lmdb':
        return LMDBDataset(data_file, *args, **kwargs)
    elif data_file.suffix in {'.fasta', '.fna', '.ffn', '.faa', '.frn'}:
        return FastaDataset(data_file, *args, **kwargs)
    elif data_file.suffix == '.json':
        return JSONDataset(data_file, *args, **kwargs)
    elif data_file.is_dir():
        return NPZDataset(data_file, *args, **kwargs)
    else:
        raise ValueError(f"Unrecognized datafile type {data_file.suffix}")


def pad_sequences(sequences: Sequence, constant_value=0, dtype=None) -> np.ndarray:
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array


class JSONDataset(Dataset):
    """Creates a dataset from a json file. Assumes that data is
       a JSON serialized list of record, where each record is
       a dictionary.
    Args:
        data_file (Union[str, Path]): Path to json file.
        in_memory (bool): Dummy variable to match API of other datasets
    """

    def __init__(self, data_file: Union[str, Path], in_memory: bool = True):
        import json
        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)
        records = json.loads(data_file.read_text())

        if not isinstance(records, list):
            raise TypeError(f"TAPE JSONDataset requires a json serialized list, "
                            f"received {type(records)}")
        self._records = records
        self._num_examples = len(records)

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        item = self._records[index]
        if not isinstance(item, dict):
            raise TypeError(f"Expected dataset to contain a list of dictionary "
                            f"records, received record of type {type(item)}")
        if 'id' not in item:
            item['id'] = str(index)
        return item


class LMDBDataset(Dataset):
    """Creates a dataset from an lmdb file.
    Args:
        data_file (Union[str, Path]): Path to lmdb file.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                 data_file: Union[str, Path],
                 in_memory: bool = False):

        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)

        env = lmdb.open(str(data_file), max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)

        with env.begin(write=False) as txn:
            num_examples = pkl.loads(txn.get(b'num_examples'))

        if in_memory:
            cache = [None] * num_examples
            self._cache = cache

        self._env = env
        self._in_memory = in_memory
        self._num_examples = num_examples

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        if self._in_memory and self._cache[index] is not None:
            item = self._cache[index]
        else:
            with self._env.begin(write=False) as txn:
                item = pkl.loads(txn.get(str(index).encode()))
                if 'id' not in item:
                    item['id'] = str(index)
                if self._in_memory:
                    self._cache[index] = item
        return item


class FluorescenceDataset(FairseqDataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'uniprot21',
                 in_memory: bool = False,
                 pad_idx=0,
                 label_pad_idx=-1):

        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. "
                             f"Must be one of ['train', 'valid', 'test']")
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'fluorescence/fluorescence_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file, in_memory)

        self.sizes = np.asarray([int(item['protein_length']) for item in self.data]) + 2

        self.pad_idx = pad_idx
        self.label_pad_idx = label_pad_idx

    def __len__(self) -> int:
        return len(self.data)

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    def __getitem__(self, index: int):
        item = self.data[index]
        token_ids = self.tokenizer.encode(item['primary'])
        protein_length = len(item['primary'])
        input_mask = np.ones_like(token_ids)
        return token_ids, input_mask, float(item['log_fluorescence'][0]), protein_length

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        try:
            input_ids, input_mask, fluorescence_true_value, protein_length = tuple(zip(*batch))
        except:
            return None
        input_ids = torch.from_numpy(pad_sequences(input_ids, self.pad_idx))
        input_mask = torch.from_numpy(pad_sequences(input_mask, self.pad_idx))
        fluorescence_true_value = torch.FloatTensor(fluorescence_true_value)  # type: ignore
        fluorescence_true_value = fluorescence_true_value.unsqueeze(1)
        protein_length = torch.LongTensor(protein_length)  # type: ignore

        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'targets': fluorescence_true_value,
                'protein_length': protein_length}

    def collater(self, samples):
        return self.collate_fn(samples)


class StabilityDataset(FairseqDataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'uniprot21',
                 in_memory: bool = False,
                 pad_idx=0,
                 label_pad_idx=-1):

        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. "
                             f"Must be one of ['train', 'valid', 'test']")
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'stability/stability_{split}.lmdb'

        self.data = dataset_factory(data_path / data_file, in_memory)

        self.sizes = np.asarray([int(item['protein_length']) for item in self.data]) + 2

        self.pad_idx = pad_idx
        self.label_pad_idx = label_pad_idx

    def __len__(self) -> int:
        return len(self.data)

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    def __getitem__(self, index: int):
        item = self.data[index]
        token_ids = self.tokenizer.encode(item['primary'])
        protein_length = len(item['primary'])
        input_mask = np.ones_like(token_ids)
        return token_ids, input_mask, float(item['stability_score'][0]), protein_length

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        try:
            input_ids, input_mask, stability_true_value, protein_length = tuple(zip(*batch))
        except:
            return None

        input_ids = torch.from_numpy(pad_sequences(input_ids, self.pad_idx))
        input_mask = torch.from_numpy(pad_sequences(input_mask, self.pad_idx))
        stability_true_value = torch.FloatTensor(stability_true_value)  # type: ignore
        stability_true_value = stability_true_value.unsqueeze(1)
        protein_length = torch.LongTensor(protein_length)  # type: ignore

        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'targets': stability_true_value,
                'protein_length': protein_length}

    def collater(self, samples):
        return self.collate_fn(samples)


class RemoteHomologyDataset(FairseqDataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'uniprot21',
                 in_memory: bool = False,
                 pad_idx=0,
                 label_pad_idx=-1):

        if split not in ('train', 'valid', 'test_fold_holdout',
                         'test_family_holdout', 'test_superfamily_holdout'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'valid', 'test_fold_holdout', "
                             f"'test_family_holdout', 'test_superfamily_holdout']")
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'remote_homology/remote_homology_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file, in_memory)
        self.sizes = np.asarray([int(item['protein_length']) for item in self.data]) + 2

        self.pad_idx = pad_idx
        self.label_pad_idx = label_pad_idx

    def __len__(self) -> int:
        return len(self.data)

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    def __getitem__(self, index: int):
        item = self.data[index]
        token_ids = self.tokenizer.encode(item['primary'])
        protein_length = len(item['primary'])
        input_mask = np.ones_like(token_ids)
        return token_ids, input_mask, item['fold_label'], protein_length

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        try:
            input_ids, input_mask, fold_label, protein_length = tuple(zip(*batch))
        except:
            return None
        input_ids = torch.from_numpy(pad_sequences(input_ids, self.pad_idx))
        input_mask = torch.from_numpy(pad_sequences(input_mask, self.pad_idx))
        fold_label = torch.LongTensor(fold_label)  # type: ignore
        protein_length = torch.LongTensor(protein_length)  # type: ignore

        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'targets': fold_label,
                'protein_length': protein_length}

    def collater(self, samples):
        return self.collate_fn(samples)


class ProteinnetDataset(FairseqDataset):
    """Dataset for TAPE's contact prediction evaluation
    """

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'uniprot21',
                 in_memory: bool = False,
                 pad_idx=0,
                 label_pad_idx=-1,
                 contact_esm=False):

        if split not in ('train', 'train_unfiltered', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'train_unfiltered', 'valid', 'test']")

        # super().__init__(data_path, split, tokenizer, in_memory, pad_idx, label_pad_idx)
        super(ProteinnetDataset).__init__()
        self.contact_esm = contact_esm

        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        if self.contact_esm: 
            data_file = f'esm_contact/pdb25-7267-{split}.json'
        else:
            data_file = f'proteinnet/proteinnet_{split}.lmdb'
        print(data_file)
        self.data = dataset_factory(data_path / data_file, in_memory)

        self.sizes = np.asarray([int(item['protein_length']) for item in self.data]) + 2

        self.pad_idx = pad_idx
        self.label_pad_idx = label_pad_idx

    @lru_cache(maxsize=8)
    def __getitem__(self, index: int):
        item = self.data[index]
        protein_length = len(item['primary'])
        token_ids = self.tokenizer.encode(item['primary'])
        input_mask = np.ones_like(token_ids)

        if self.contact_esm:
            contact_map = np.array(item['contact_map']).astype(np.int64)
            yind, xind = np.indices(contact_map.shape)
            contact_map[np.abs(yind - xind) < 6] = -1
        else:
            valid_mask = item['valid_mask']
            contact_map = np.less(squareform(pdist(item['tertiary'])), 8.0).astype(np.int64)

            yind, xind = np.indices(contact_map.shape)
            invalid_mask = ~(valid_mask[:, None] & valid_mask[None, :])
            invalid_mask |= np.abs(yind - xind) < 6
            contact_map[invalid_mask] = -1

        return token_ids, input_mask, contact_map, protein_length

    def __len__(self) -> int:
        return len(self.data)

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        try:
            input_ids, input_mask, contact_labels, protein_length = tuple(zip(*batch))
        except:
            return None
        input_ids = torch.from_numpy(pad_sequences(input_ids, self.pad_idx))
        input_mask = torch.from_numpy(pad_sequences(input_mask, self.pad_idx))
        contact_labels = torch.from_numpy(pad_sequences(contact_labels, self.label_pad_idx))
        protein_length = torch.LongTensor(protein_length)  # type: ignore

        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'targets': contact_labels,
                'protein_length': protein_length}

    def collater(self, samples):
        return self.collate_fn(samples)


class ContactMediumLongRangeDataset(FairseqDataset):
    """Dataset for TAPE's contact prediction evaluation
    """

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'uniprot21',
                 in_memory: bool = False,
                 pad_idx=0,
                 label_pad_idx=-1,
                 contact_esm=False):

        if split not in ('train', 'train_unfiltered', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'train_unfiltered', 'valid', 'test']")

        # super().__init__(data_path, split, tokenizer, in_memory, pad_idx, label_pad_idx)
        super(ContactMediumLongRangeDataset).__init__()
        self.contact_esm = contact_esm

        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        if self.contact_esm: 
            data_file = f'esm_contact/pdb25-7267-{split}.json'
        else:
            data_file = f'proteinnet/proteinnet_{split}.lmdb'
        print(data_file)
        self.data = dataset_factory(data_path / data_file, in_memory)

        self.sizes = np.asarray([int(item['protein_length']) for item in self.data]) + 2

        self.pad_idx = pad_idx
        self.label_pad_idx = label_pad_idx

    @lru_cache(maxsize=8)
    def __getitem__(self, index: int):
        item = self.data[index]
        protein_length = len(item['primary'])
        token_ids = self.tokenizer.encode(item['primary'])
        input_mask = np.ones_like(token_ids)

        if self.contact_esm:
            contact_map = np.array(item['contact_map']).astype(np.int64)

            yind, xind = np.indices(contact_map.shape)
            contact_map[np.abs(yind - xind) < 12] = -1
        else:
            valid_mask = item['valid_mask']
            contact_map = np.less(squareform(pdist(item['tertiary'])), 8.0).astype(np.int64)

            yind, xind = np.indices(contact_map.shape)
            invalid_mask = ~(valid_mask[:, None] & valid_mask[None, :])
            invalid_mask |= np.abs(yind - xind) < 12
            contact_map[invalid_mask] = -1

        return token_ids, input_mask, contact_map, protein_length

    def __len__(self) -> int:
        return len(self.data)

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        try:
            input_ids, input_mask, contact_labels, protein_length = tuple(zip(*batch))
        except:
            return None
        input_ids = torch.from_numpy(pad_sequences(input_ids, self.pad_idx))
        input_mask = torch.from_numpy(pad_sequences(input_mask, self.pad_idx))
        contact_labels = torch.from_numpy(pad_sequences(contact_labels, self.label_pad_idx))
        protein_length = torch.LongTensor(protein_length)  # type: ignore

        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'targets': contact_labels,
                'protein_length': protein_length}

    def collater(self, samples):
        return self.collate_fn(samples)


class SecondaryStructureDataset(FairseqDataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'uniprot21',
                 in_memory: bool = False,
                 pad_idx=0,
                 label_pad_idx=-1):

        if split not in ('train', 'valid', 'casp12', 'ts115', 'cb513'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'valid', 'casp12', "
                             f"'ts115', 'cb513']")
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'secondary_structure/secondary_structure_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file, in_memory)

        # not right way to get data's len
        self.sizes = np.asarray([int(item['protein_length']) for item in self.data]) + 2  # for begin & stop tokens

        self.pad_idx = pad_idx
        self.label_pad_idx = label_pad_idx

    def __len__(self) -> int:
        return len(self.data)

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    def __getitem__(self, index: int):
        item = self.data[index]
        token_ids = self.tokenizer.encode(item['primary'])
        protein_length = len(item['primary'])
        input_mask = np.ones_like(token_ids)

        # pad with -1s because of cls/sep tokens
        labels = np.asarray(item['ss3'], np.int64)
        labels = np.pad(labels, (1, 1), 'constant', constant_values=self.label_pad_idx)

        return token_ids, input_mask, labels, protein_length

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        try:
            input_ids, input_mask, ss_label, protein_length = tuple(zip(*batch))
        except:
            return None
        input_ids = torch.from_numpy(pad_sequences(input_ids, self.pad_idx))
        input_mask = torch.from_numpy(pad_sequences(input_mask, self.pad_idx))
        ss_label = torch.from_numpy(pad_sequences(ss_label, self.label_pad_idx))
        protein_length = torch.LongTensor(protein_length)  # type: ignore

        output = {'input_ids': input_ids,
                  'input_mask': input_mask,
                  'targets': ss_label,
                  'protein_length': protein_length}

        return output

    def collater(self, samples):
        return self.collate_fn(samples)


def load_indexed_dataset(
    path, dictionary=None, dataset_impl=None, combine=False, default="cached"
):
    """A helper function for loading indexed datasets.

    Args:
        path (str): path to indexed dataset (e.g., 'data-bin/train')
        dictionary (~fairseq.data.Dictionary): data dictionary
        dataset_impl (str, optional): which dataset implementation to use. If
            not provided, it will be inferred automatically. For legacy indexed
            data we use the 'cached' implementation by default.
        combine (bool, optional): automatically load and combine multiple
            datasets. For example, if *path* is 'data-bin/train', then we will
            combine 'data-bin/train', 'data-bin/train1', ... and return a
            single ConcatDataset instance.
    """
    from fairseq.data.concat_dataset import ConcatDataset
    import fairseq.data.indexed_dataset as indexed_dataset

    datasets = []
    numslen = []
    for p in path:
        for k in itertools.count():
            path_k = p + (str(k) if k > 0 else "")
            path_k = indexed_dataset.get_indexed_dataset_to_local(path_k)

            dataset_impl_k = dataset_impl
            if dataset_impl_k is None:
                dataset_impl_k = indexed_dataset.infer_dataset_impl(path_k)
            dataset = indexed_dataset.make_dataset(
                path_k,
                impl=dataset_impl_k or default,
                fix_lua_indexing=True,
                dictionary=dictionary,
            )
            if dataset is None:
                break
            logger.info("loaded {} examples from: {}".format(len(dataset), path_k))
            datasets.append(dataset)
            numslen.append(len(dataset))
            if not combine:
                break
    if len(datasets) == 0:
        return None, None
    elif len(datasets) == 1:
        return datasets[0], numslen
    else:
        return ConcatDataset(datasets), numslen
