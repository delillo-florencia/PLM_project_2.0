import csv
import pickle
from torch.utils.data import Dataset
import datetime
import os
import sys


class Sequence:
    
    def __init__(self, sequence, length, taxon_id):
        self.sequence = str(sequence)
        self.length = int(length)
        self.taxon_id = int(taxon_id)


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
        return Sequence(row[3], row[2], row[1])

    @staticmethod
    def create_hashed_data(csv_file, hashed_data_path):
        offsets = []
        lengths = []
        taxon_ids = []
        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
            header = next(csv.reader([f.readline()]))
            pos = f.tell()
            line = f.readline()
            while line:
                offsets.append(pos)
                row = line.strip().split(',')
                d = dict(zip(header, row))
                lengths.append(int(d['sequence_length']))
                taxon_ids.append(d['taxon_id'])
                pos = f.tell()
                line = f.readline()
        data = {'header': header, 'line_offsets': offsets, 'lengths': lengths, 'taxon_ids': taxon_ids}
        with open(hashed_data_path, 'wb') as f:
            pickle.dump(data, f)

CSV_FILE = str(sys.argv[1])
HASH_FILE = str(sys.argv[2])

#conda activate proteusAI
#python run_hashing CSV_FILE HASH_FILE

if not os.path.exists(HASH_FILE):
    print("hashing start", datetime.datetime.now())
    HashedProteinDataset.create_hashed_data(CSV_FILE, HASH_FILE)
    print("hashing done", datetime.datetime.now())
else:
    print("hash file exists")
    
    
