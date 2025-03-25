
from torch.utils.data import Dataset
from data.sequence import Sequence
import random
import csv
import pickle

class HashedProteinDataset(Dataset):
    """
    Firstly, pre-hash your master dataset:
    > HashedProteinDataset.create_hashed_data("massive_file.csv", "massive_file.hash")

    Later, use hashed file to generate the dataset with hashed reference:
    dataset = HashedProteinDataset("massive_file.csv", "massive_file.hash")
    """
    def __init__(self, csv_file, hashed_data_path):
        with open(hashed_data_path, 'rb') as f:
            data = pickle.load(f)
        self.header = data['header']
        self.line_offsets = data['line_offsets']
        self.lengths = data['lengths']
        self.taxon_ids = data['taxon_ids']
        self.csv_file = csv_file
        self._file_handle = None

    def __len__(self):
        return len(self.line_offsets)

    def _get_file_handle(self):
        if self._file_handle is None:
            self._file_handle = open(self.csv_file, 'r', newline='', encoding='utf-8')
        return self._file_handle

    def __getitem__(self, index):
        f = self._get_file_handle()
        f.seek(self.line_offsets[index])
        line = f.readline().strip()
        row = line.split(',')
        return Sequence(row[3], row[2], row[1], row[0])


    @staticmethod
    def create_hashed_data(csv_file, hashed_data_prefix, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):

        offsets_train = []
        lengths_train = []
        taxon_ids_train = []
        offsets_val = []
        lengths_val = []
        taxon_ids_val = []
        offsets_test = []
        lengths_test = []
        taxon_ids_test = []

        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
            
            # header from CSV file
            header = next(csv.reader([f.readline()]))
            pos = f.tell()
            line = f.readline()
            
            # validate ratio input
            total = train_ratio + val_ratio + test_ratio
            if total == 1:
                raise ValueError("All ratios must equal one!")
            
            while line:

                # compute line offset
                r = random.random() * total
                if r < train_ratio:
                    offsets_train.append(pos)
                elif r < train_ratio + val_ratio:
                    offsets_val.append(pos)
                else:
                    offsets_test.append(pos)
                row = line.strip().split(',')
                d = dict(zip(header, row))

                # append data to respective lists based on the same random decision
                if r < train_ratio:
                    lengths_train.append(int(d['sequence_length']))
                    taxon_ids_train.append(int(d['taxon_id']))
                elif r < train_ratio + val_ratio:
                    lengths_val.append(int(d['sequence_length']))
                    taxon_ids_val.append(int(d['taxon_id']))
                else:
                    lengths_test.append(int(d['sequence_length']))
                    taxon_ids_test.append(int(d['taxon_id']))
                pos = f.tell()
                line = f.readline()

        # data dictionaries for sets
        data_train = {'header': header, 'line_offsets': offsets_train, 'lengths': lengths_train, 'taxon_ids': taxon_ids_train}
        data_val = {'header': header, 'line_offsets': offsets_val, 'lengths': lengths_val, 'taxon_ids': taxon_ids_val}
        data_test = {'header': header, 'line_offsets': offsets_test, 'lengths': lengths_test, 'taxon_ids': taxon_ids_test}

        # save hashed data
        with open(hashed_data_prefix + "_train.hash", 'wb') as f:
            pickle.dump(data_train, f)
        with open(hashed_data_prefix + "_val.hash", 'wb') as f:
            pickle.dump(data_val, f)
        with open(hashed_data_prefix + "_test.hash", 'wb') as f:
            pickle.dump(data_test, f)
