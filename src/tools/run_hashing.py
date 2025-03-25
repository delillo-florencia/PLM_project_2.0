from data.dataset import HashedProteinDataset
import datetime
import os
import sys

CSV_FILE = str(sys.argv[1])
HASH_PREFIX = str(sys.argv[2])

if not os.path.exists(HASH_PREFIX):
    print("Hashing start", datetime.datetime.now())
    HashedProteinDataset.create_hashed_data(CSV_FILE, HASH_PREFIX, 
                                            train_ratio=0.7, 
                                            val_ratio=0.2, 
                                            test_ratio=0.1)
    print("Hashing done", datetime.datetime.now())
else:
    print("Hash file exists")