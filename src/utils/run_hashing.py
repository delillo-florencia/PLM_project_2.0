from PLM.utils.data_utils import HashedProteinDataset
import datetime
import os
import sys

CSV_FILE = str(sys.argv[1])
HASH_FILE = str(sys.argv[2])

if not os.path.exists(HASH_FILE):
    print("Hashing start", datetime.datetime.now())
    HashedProteinDataset.create_hashed_data(CSV_FILE, HASH_FILE)
    print("Hashing done", datetime.datetime.now())
else:
    print("Hash file exists")