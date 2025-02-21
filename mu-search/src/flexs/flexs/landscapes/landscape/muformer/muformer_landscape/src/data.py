import os
import warnings
from functools import lru_cache
import numpy as np
import pandas as pd
from Bio import SeqIO
import torch.utils.data
from sklearn.model_selection import KFold, ShuffleSplit

from torch.utils.data import Dataset

from .ccmpred import CCMPredEncoder

def index_encoding(sequences):
    '''
    Modified from https://github.com/openvax/mhcflurry/blob/master/mhcflurry/amino_acid.py#L110-L130

    Parameters
    ----------
    sequences: list of equal-length sequences

    Returns
    -------
    np.array with shape (#sequences, length of sequences)
    '''
    df = pd.DataFrame(iter(s) for s in sequences)
    encoding = df.replace(vocab.AMINO_ACID_INDEX)
    encoding = encoding.values.astype(np.int)
    return encoding

def _pad_sequences(sequences, constant_value=1,length=0):
    batch_size = len(sequences)
    if not length:
        shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()
    else:
        shape = [batch_size] + [length]
    array = np.zeros(shape, sequences[0].dtype) + constant_value

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array

class FairseqLMEncoder(object):
    def __init__(self,
        task,
        batch_size: int = 2,            # Cuda Out of Memory need to modify
        full_sequence_embed: bool = True,
        num_workers: int = 4,
        progress_bar: bool = True,
        ):
        """
        Parameters
        ----------
        seq_df: pd.DataFrame object. Two columns are requried `ID` and `sequence`.
        """

        self.tokenizer = task.source_dictionary
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.full_sequence_embed = full_sequence_embed
        self.progress_bar = progress_bar

    def encode(self, sequences,length=0) -> np.ndarray:
        """
        Parameters
        ----------
        sequences: list of equal-length sequences

        Returns
        -------
        np array with shape (#sequences, length of sequences, embedding dim)
        """

        encoding = []
        for item in sequences:
            temp = '<s> '+' '.join(item)
            token_ids = self.tokenizer.encode_line(temp).numpy()  # seq -> <cls> <seq> encoder
            encoding.append(token_ids)
        # return torch.from_numpy(_pad_sequences(np.array(encoding))).long()
        res_comb = {}
        mask = [np.ones(i.shape[0]) for i in encoding]
        if len(encoding) > 1:
            res = torch.from_numpy(_pad_sequences(np.array(encoding,dtype=object),length=length).astype(int)).long()
            res_mask = torch.from_numpy(_pad_sequences(np.array(mask,dtype=object),length=length).astype(int)).long()
        else:
            res = torch.from_numpy(_pad_sequences(np.array(encoding),length=length).astype(int)).long()
            res_mask = torch.from_numpy(_pad_sequences(np.array(mask),length=length).astype(int)).long()
        res_comb['glob_feat'] = res
        res_comb['glob_mask'] = res_mask

        return res_comb


class MetagenesisData(Dataset):
    def __init__(self, data, mode,
        native_sample,
        fasta=None,
        ccmpred_encoder=None,
        label_oper="minus",
        msa_feat_used=True,
        aug_ratio=0.0,
        cross_protein=False,):

        self.data = data
        self.fasta = fasta
        self.mode = mode
        self.native_sample = native_sample
        if self.fasta is not None:
            self.native_sequence = self._read_native_sequence()
        self.msa_feat_used = msa_feat_used
        if self.msa_feat_used:
            self.ccmpred_encoder = ccmpred_encoder
        else:
            self.ccmpred_encoder = None
        self.label_oper = label_oper
        self.aug_ratio = aug_ratio
        self.data_augment = self.aug_ratio > 0
        self.cross_protein = cross_protein

    def __len__(self):
        if (self.mode == 'test') or (not self.data_augment):
        # if (self.mode != 'train') or (not self.data_augment):
            return len(self.data)
        else:
            return int(len(self.data) * (1 + self.aug_ratio))

    def _read_native_sequence(self):
        fasta = SeqIO.read(self.fasta, 'fasta')
        native_sequence = str(fasta.seq)
        return native_sequence

    def encode_loc_feat(self, sequences):
        feat = self.ccmpred_encoder.encode_pair(sequences)
        feat = torch.from_numpy(feat).float()
        return feat

    @lru_cache(maxsize=32)
    def __getitem__(self, index):
        data_tmp = self.data[index%len(self.data)].copy()

        if self.msa_feat_used and type(data_tmp['loc_feat']) == str:
            data_tmp['loc_feat'] = self.encode_loc_feat(np.array([data_tmp['loc_feat']]))[0]

            # if (self.mode != 'train') or (not self.data_augment) or (index<len(self.data)):
            if (self.mode == 'test') or (not self.data_augment) or (index<len(self.data)):
                rand_index = -1
            else:
                rand_index = torch.randint(0,len(self.data),(1,))[0]
                while rand_index == index%len(self.data):
                    rand_index = torch.randint(0,len(self.data),(1,))[0]
            if rand_index < 0:
                data_tmp2 = self.native_sample
            else:
                data_tmp2 = self.data[rand_index].copy()
                data_tmp2['loc_feat'] = self.encode_loc_feat(np.array([data_tmp2['loc_feat']]))[0]
        elif self.cross_protein:
            data_tmp2 = self.native_sample[data_tmp['flag']]
        else:
            # if (self.mode != 'train') or (not self.data_augment) or (index<len(self.data)):
            if (self.mode == 'test') or (not self.data_augment) or (index<len(self.data)):
                rand_index = -1
            else:
                rand_index = torch.randint(0,len(self.data),(1,))[0]
                while rand_index == index % len(self.data):
                    rand_index = torch.randint(0,len(self.data),(1,))[0]
            if rand_index < 0:
                data_tmp2 = self.native_sample
            else:
                data_tmp2 = self.data[rand_index]

        if self.label_oper == "divide":
            data_tmp['label_af'] = data_tmp['label'] / data_tmp2['label']
        elif self.label_oper == "minus":
            data_tmp['label_af'] = data_tmp['label'] - data_tmp2['label']
        return (data_tmp,data_tmp2)

class PairedDataset(object):
    def __init__(self,
            train_tsv=None, valid_tsv=None, test_tsv=None,
            fasta=None, ccmpred_output=None,
            split_ratio=[0.9, 0.1],
            random_seed=42,
            label_oper="minus",
            aug_ratio=-1,
            task=None,
            msa_feat_used=True,
            locfeat_batch_encode=False,
            flip_train_ratio=1,
            ):
        """
        split_ratio: [train, valid] or [train, valid, test]
        """

        self.train_tsv = train_tsv
        self.valid_tsv = valid_tsv
        self.test_tsv = test_tsv
        self.fasta = fasta
        self.split_ratio = split_ratio
        self.rng = np.random.RandomState(random_seed)
        self.ccmpred_output = ccmpred_output
        self.label_oper = label_oper
        self.aug_ratio = aug_ratio
        self.msa_feat_used = msa_feat_used
        self.locfeat_batch_encode = locfeat_batch_encode
        self.lm_encoder = FairseqLMEncoder(task)
        self.flip_train_ratio = flip_train_ratio

        if self.train_tsv and self.valid_tsv and self.test_tsv:
            self.cross_protein = True
        else:
            self.cross_protein = False
            
        if 'Flip' in self.train_tsv:
            self.flip = True
        else:
            self.flip = False

        if 'Tranception' in self.train_tsv:
            self.tranception = True
        else:
            self.tranception = False
        
        if 'one_vs_multi' in self.train_tsv:
            self.tran_ovm = True
        else:
            self.tran_ovm = False

        if self.fasta is not None:
            self.native_sequence = self._read_native_sequence()
        if train_tsv is not None and not self.cross_protein:
            self.full_df = self._read_mutation_df(train_tsv)
        else:
            self.full_df = None

        if test_tsv is None:
            if self.flip:
                groups = self.full_df.groupby(self.full_df.set)
                self.train_df = groups.get_group("train")
                if self.flip_train_ratio<1:
                    self.train_df = self._sample_df(self.train_df,self.flip_train_ratio)
                self.valid_df=groups.get_group("valid")
                self.test_df=groups.get_group("test")
            elif self.tran_ovm:
                groups = self.full_df.groupby(self.full_df.set)
                self.train_df = groups.get_group("train")
                if len(split_ratio) != 2:
                    split_ratio = [0.9, 0.1]
                    warnings.warn("\nsplit_ratio should have 2 elements if test_tsv is provided. " + \
                        f"Changing split_ratio to {split_ratio}. " + \
                        "Set to other values using --split_ratio.")
                self.train_df, self.valid_df, _ = self._split_dataset_df(self.train_df, split_ratio)
                self.test_df=groups.get_group("test")
            else:
                if len(split_ratio) != 3:
                    split_ratio = [0.7, 0.1, 0.2]
                    warnings.warn("\nsplit_ratio should have 3 elements if test_tsv is None." + \
                        f"Changing split_ratio to {split_ratio}. " + \
                        "Set to other values using --split_ratio.")
                self.train_df, self.valid_df, self.test_df = \
                    self._split_dataset_df(self.full_df, split_ratio)
        else:
            if self.cross_protein:
                self.train_df, train_native = self._read_multi_tsv(self.train_tsv)
                if self.valid_tsv[0].endswith('split'):
                    valid_native = train_native.copy()
                    self.train_df,self.valid_df = self._split_multi_df(self.train_df)
                else:
                    self.valid_df, valid_native = self._read_multi_tsv(self.valid_tsv)
                self.test_df, test_native = self._read_multi_tsv(self.test_tsv)
                self.split_native_sequence = {
                    'train':train_native,
                    'valid':valid_native,
                    'test':test_native,
                }
            else:
                if len(split_ratio) != 2:
                    split_ratio = [0.9, 0.1]
                    warnings.warn("\nsplit_ratio should have 2 elements if test_tsv is provided. " + \
                        f"Changing split_ratio to {split_ratio}. " + \
                        "Set to other values using --split_ratio.")
                self.test_df = self._read_mutation_df(test_tsv)
                if self.full_df is not None:
                    self.train_df, self.valid_df, _ = \
                        self._split_dataset_df(self.full_df, split_ratio)

        if self.full_df is not None and not self.flip:
            self.train_valid_df = pd.concat(
                [self.train_df, self.valid_df]).reset_index(drop=True)

        if msa_feat_used:
            self.ccmpred_encoder = CCMPredEncoder(
                ccmpred_output=ccmpred_output, seq_len=len(self.native_sequence))
        else:
            self.ccmpred_encoder = None

    def _read_native_sequence(self):
        fasta = SeqIO.read(self.fasta, 'fasta')
        native_sequence = str(fasta.seq)
        return native_sequence

    def _read_multi_tsv(self, tsvfilelist):
        df = None
        sequences = []
        for flag_cnt,tsvfile in enumerate(tsvfilelist):
            tmp_df = self._read_mutation_df(tsvfile+'.tsv')
            tmp_list = []
            for key in tmp_df.keys():
                if key == 'target':
                    tmp_df['score'] = tmp_df['target']
                if key not in ['sequence','score']:
                    tmp_list.append(key)
                tmp_df['flag'] = flag_cnt
            tmp_df = tmp_df.drop(columns=tmp_list)
            assert len(tmp_df.keys()) == 3
            df = tmp_df if df is None else pd.concat([df,tmp_df],ignore_index=True)
            fasta = SeqIO.read(tsvfile+'.fasta', 'fasta')
            sequences.append(str(fasta.seq))
        return df,sequences
    
    def _split_multi_df(self, df, split_ratio=[0.9,0.1]):
        df = df.copy()
        df = df.sample(frac=1, random_state=self.rng).reset_index(drop=True)
        N = len(df)
        train_ratio, valid_ratio = split_ratio
        train_len = int(round(train_ratio * N))
        valid_len = int(round(valid_ratio * N)) if int(round(valid_ratio * N)) < N-train_len else N-train_len

        train_df = df.iloc[:train_len].reset_index(drop=True)
        valid_df = df.iloc[train_len:train_len+valid_len].reset_index(drop=True)

        return train_df, valid_df
    
    def _sample_df(self,df,flip_train_ratio):
        df_tmp = df.copy()
        df_tmp = df_tmp.sample(frac=1, random_state=self.rng).reset_index(drop=True)
        N = len(df)
        sample_len = int(round(flip_train_ratio * N))
        train_df = df_tmp.iloc[:sample_len].reset_index(drop=True)
        return train_df
    
    def _check_split_ratio(self, split_ratio):
        """
        Modified from: https://github.com/pytorch/text/blob/3d28b1b7c1fb2ddac4adc771207318b0a0f4e4f9/torchtext/data/dataset.py#L284-L311
        """
        test_ratio = 0.
        if isinstance(split_ratio, float):
            assert 0. < split_ratio < 1., (
                "Split ratio {} not between 0 and 1".format(split_ratio))
            valid_ratio = 1. - split_ratio
            return (split_ratio, valid_ratio, test_ratio)
        elif isinstance(split_ratio, list):
            length = len(split_ratio)
            assert length == 2 or length == 3, (
                "Length of split ratio list should be 2 or 3, got {}".format(split_ratio))
            ratio_sum = sum(split_ratio)
            if not ratio_sum == 1.:
                if ratio_sum > 1:
                    split_ratio = [float(ratio) / ratio_sum for ratio in split_ratio]
            if length == 2:
                return tuple(split_ratio + [test_ratio])
            return tuple(split_ratio)
        else:
            raise ValueError('Split ratio must be float or a list, got {}'
                            .format(type(split_ratio)))


    def _split_dataset_df(self, input_df, split_ratio, resample_split=False):
        """
        Modified from:
        https://github.com/pytorch/text/blob/3d28b1b7c1fb2ddac4adc771207318b0a0f4e4f9/torchtext/data/dataset.py#L86-L136
        """
        _rng = self.rng.randint(512) if resample_split else self.rng
        df = input_df.copy()
        df = df.sample(frac=1, random_state=_rng).reset_index(drop=True)
        N = len(df)
        train_ratio, valid_ratio, test_ratio = self._check_split_ratio(split_ratio)
        train_len = int(round(train_ratio * N))
        valid_len = int(round(valid_ratio * N)) if int(round(valid_ratio * N)) < N-train_len else N-train_len
        test_len = int(round(test_ratio * N)) if int(round(test_ratio * N)) < N-train_len-valid_len else N-train_len-valid_len

        train_df = df.iloc[:train_len].reset_index(drop=True)
        valid_df = df.iloc[-1-test_len:-1-test_len-valid_len:-1].reset_index(drop=True)
        test_df = df.iloc[-1:-1-test_len:-1].reset_index(drop=True)

        return train_df, valid_df, test_df


    def _mutation_to_sequence(self, mutation):
        '''
        Parameters
        ----------
        mutation: ';'.join(WiM) (wide-type W at position i mutated to M)
        '''
        sequence = self.native_sequence
        for mut in mutation.split(';'):
            wt_aa = mut[0]
            mt_aa = mut[-1]
            pos = int(mut[1:-1])
            assert wt_aa == sequence[pos - 1],\
                    "%s: %s->%s (fasta WT: %s)"%(pos, wt_aa, mt_aa, sequence[pos - 1])
            sequence = sequence[:(pos - 1)] + mt_aa + sequence[pos:]
        return sequence


    def _mutations_to_sequences(self, mutations):
        return [self._mutation_to_sequence(m) for m in mutations]


    def _drop_invalid_mutation(self, df):
        '''
        Drop mutations WiM where
        - W is incosistent with the i-th AA in native_sequence
        - M is ambiguous, e.g., 'X'
        '''
        flags = []
        for mutation in df['mutation'].values:
            for mut in mutation.split(';'):
                wt_aa = mut[0]
                mt_aa = mut[-1]
                pos = int(mut[1:-1])
                valid = True if wt_aa == self.native_sequence[pos - 1] else False
                valid = valid and (mt_aa not in ['X'])
            flags.append(valid)
        df = df[flags].reset_index(drop=True)
        return df

    def _read_mutation_df(self, tsv):
        if self.flip or self.tranception or self.cross_protein:
            df = pd.read_table(tsv)
            return df
        else:
            df = pd.read_table(tsv)
            df = self._drop_invalid_mutation(df)
            df['sequence'] = self._mutations_to_sequences(df['mutation'].values)
            return df


    def encode_seq_enc(self, sequences):
        seq_enc = index_encoding(sequences)
        seq_enc = torch.from_numpy(seq_enc.astype(np.int))
        return seq_enc

    def encode_loc_feat(self, sequences):
        feat = self.ccmpred_encoder.encode_pair(sequences)
        feat = torch.from_numpy(feat).float()
        return feat

    def encode_sequence(self, sequences,length=0):
        if self.flip or self.tranception:
            feat = self.lm_encoder.encode(sequences,length)
        else:
            feat = self.lm_encoder.encode(sequences)
        return feat

    def build_data(self, mode, return_df=False):
        if mode == 'train':
            df = self.train_df.copy()
        elif mode == 'valid':
            df = self.valid_df.copy()
        elif mode == 'test':
            df = self.test_df.copy()
        elif mode == 'zeroshot':
            df = self.full_df.copy()
        else:
            raise NotImplementedError

        sequences = df['sequence'].values
        # seq_enc = self.encode_seq_enc(sequences)
        if self.cross_protein:
            glob = self.encode_sequence(sequences)
        else:
            maxlen_native = len(self.native_sequence)
            data_maxlen = np.max([len(seq) for seq in sequences], 0)
            length = data_maxlen+2 if maxlen_native<=data_maxlen else maxlen_native+2
            glob = self.encode_sequence(sequences,length=length)

        if self.msa_feat_used:
            if self.locfeat_batch_encode:
                loc_feat = sequences
            else:
                loc_feat = self.encode_loc_feat(sequences)
        else:
            loc_feat = None

        if self.flip:
            labels = df['target'].values
        else:
            labels = df['score'].values
        labels = torch.from_numpy(labels.astype(np.float32))
        if self.cross_protein:
            flag = torch.from_numpy(df['flag'].values)
        samples = []
        for i in range(len(df)):
            sample = {
                'sequence':sequences[i],
                'label':labels[i],
                # 'seq_enc': seq_enc[i],
            }
            if self.cross_protein:
                sample['flag'] = flag[i]
            if self.msa_feat_used:
                sample['loc_feat'] = loc_feat[i]
            sample['glob_feat'] = glob['glob_feat'][i]
            sample['glob_mask'] = glob['glob_mask'][i]
            samples.append(sample)
        if self.cross_protein:
            native_sequences = self.split_native_sequence[mode]
            native_globs = self.encode_sequence(native_sequences)   #,length=glob['glob_feat'].shape[1])
            native_sample=[]
            for i in range(len(native_sequences)):
                native_tmp = {
                    'sequence': native_sequences[i],
                    'glob_feat': native_globs['glob_feat'][i],
                    'glob_mask': native_globs['glob_mask'][i],
                }
                if self.label_oper=="divide":
                    native_tmp['label'] = torch.tensor(1.0)
                elif self.label_oper =="minus":
                    native_tmp['label'] = torch.tensor(0.0)
                native_sample.append(native_tmp)
        else:
            sequence = self.native_sequence
            glob = self.encode_sequence(np.array([self.native_sequence]),length=glob['glob_feat'].shape[1])
            native_sample = {
                'sequence':sequence,
                # 'seq_enc': self.encode_seq_enc(np.array([self.native_sequence]))[0],
                'glob_feat': glob['glob_feat'][0],
                }
            if self.msa_feat_used:
                loc_feat = self.encode_loc_feat(np.array([self.native_sequence]))
                native_sample['loc_feat'] = loc_feat[0]

            if self.label_oper=="divide":
                native_sample['label'] = torch.tensor(1.0)
            elif self.label_oper =="minus":
                native_sample['label'] = torch.tensor(0.0)

            native_sample['glob_mask'] = glob['glob_mask'][0]

        data = MetagenesisData(samples,mode,
            native_sample,
            fasta=self.fasta,
            ccmpred_encoder=self.ccmpred_encoder,
            label_oper=self.label_oper,
            msa_feat_used=self.msa_feat_used,
            aug_ratio=self.aug_ratio,
            cross_protein=self.cross_protein,)

        if return_df:
            return data, df
        else:
            return data

    def get_dataloader(self, mode, batch_size=128,
            return_df=False, resample_train_valid=False,num_workers=1):
        if os.cpu_count() < num_workers:
            num_workers = 2
        else:
            num_workers = 4

        if resample_train_valid:
            self.train_df, self.valid_df, _ = \
                self._split_dataset_df(
                    self.train_valid_df, self.split_ratio[:2], resample_split=True)

        if mode == 'train_valid':
            train_data, train_df = self.build_data('train', return_df=True)
            valid_data, valid_df = self.build_data('valid', return_df=True)
            if self.flip and self.flip_train_ratio<1:
                train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,num_workers=num_workers,pin_memory=False)
            else:
                train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,num_workers=num_workers,pin_memory=False,drop_last=True)
            valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size,num_workers=num_workers,pin_memory=False)
            if return_df:
                return (train_loader, train_df), (valid_loader, valid_df)
            else:
                return train_loader, valid_loader
        elif mode == 'test':
            test_data, test_df = self.build_data('test', return_df=True)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,num_workers=num_workers,pin_memory=False)
            if return_df:
                return test_loader, test_df
            else:
                return test_loader
        elif mode == 'zeroshot':
            if self.flip:
                test_data, test_df = self.build_data('test', return_df=True)
            else:
                test_data, test_df = self.build_data('zeroshot', return_df=True)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,num_workers=num_workers,pin_memory=False)
            if return_df:
                return test_loader, test_df
            else:
                return test_loader
        else:
            raise NotImplementedError

