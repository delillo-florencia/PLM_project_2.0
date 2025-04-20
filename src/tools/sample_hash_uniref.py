#!/usr/bin/env python3
import argparse
import os
import csv
import gzip
import pickle
import sqlite3
import hashlib
from tqdm import tqdm
import xml.sax

# Split proportions for train/val/test
PROBS = (0.7, 0.2, 0.1)


def build_uniref50_map_sqlite(xml_gz_path, db_path):
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM mapping LIMIT 1")
            conn.close()
            print(f"[✓] Reusing existing SQLite map: {db_path}")
            return
        except Exception:
            os.remove(db_path)

    print(f"[+] Building SQLite cluster map at: {db_path}")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        CREATE TABLE IF NOT EXISTS mapping (id TEXT PRIMARY KEY, rep TEXT);
        CREATE INDEX IF NOT EXISTS idx_id ON mapping(id);
        """
    )

    class UniRefHandler(xml.sax.ContentHandler):
        def __init__(self, cursor):
            super().__init__()
            self.cur = cursor
            self.in_entry = False
            self.rep_id = None

        def startElement(self, name, attrs):
            if name == 'entry':
                self.in_entry = True
                self.rep_id = None
            elif name == 'representativeMember' and self.in_entry:
                self.rep_id = attrs.get('id')
                self.cur.execute(
                    "INSERT OR IGNORE INTO mapping(id, rep) VALUES(?, ?)",
                    (self.rep_id, self.rep_id)
                )
            elif name == 'member' and self.rep_id:
                mem_id = attrs.get('id')
                self.cur.execute(
                    "INSERT OR IGNORE INTO mapping(id, rep) VALUES(?, ?)",
                    (mem_id, self.rep_id)
                )

        def endElement(self, name):
            if name == 'entry':
                self.in_entry = False
                self.rep_id = None

    parser = xml.sax.make_parser()
    parser.setContentHandler(UniRefHandler(cur))
    with gzip.open(xml_gz_path, 'rb') as f:
        parser.parse(f)

    conn.commit()
    conn.close()
    print(f"[✓] Cluster map built")


class ClusterMap:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cur = self.conn.cursor()

    def get(self, key, default=None):
        self.cur.execute("SELECT rep FROM mapping WHERE id = ?", (key,))
        row = self.cur.fetchone()
        return row[0] if row else default

    def close(self):
        self.conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Hash-based UniRef100 splitter & hasher (single-pass)'
    )
    parser.add_argument('--input_csv', required=True, help='UniRef100 CSV file')
    parser.add_argument('--hash_prefix', required=True, help='Output filename prefix')
    parser.add_argument('--xml', required=True, help='UniRef50 XML .gz path')
    parser.add_argument('--subset', choices=['full', 'half', 'quarter'], default='full')
    parser.add_argument('--subset_index', type=int, default=0)
    args = parser.parse_args()

    # Build or load cluster map
    db_path = args.xml + '.sqlite'
    build_uniref50_map_sqlite(args.xml, db_path)
    cmap = ClusterMap(db_path)

    # Read header to infer columns
    with open(args.input_csv, newline='', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            seq_idx = header.index('sequence_length')
            tax_idx = header.index('taxon_id')
        except ValueError:
            raise RuntimeError("CSV missing 'sequence_length' or 'taxon_id' columns")

    # Prepare CSV writers
    writers = {}
    for lbl in ('train', 'val', 'test'):
        path = f"{args.hash_prefix}_{lbl}.csv"
        f_out = open(path, 'w', newline='', encoding='utf-8')
        w = csv.writer(f_out)
        w.writerow(header)
        writers[lbl] = (f_out, w)

    # Single-pass: assign each row by hashing its rep ID
    with open(args.input_csv, 'r', encoding='utf-8', errors='ignore') as f:
        # Skip header line
        f.readline()
        for line in tqdm(f, desc='Splitting rows'):
            row = line.strip().split(',')
            rep_id = cmap.get(row[0], row[0])
            # subset filtering if requested
            if args.subset != 'full':
                parts = {'half': 2, 'quarter': 4}[args.subset]
                if int(hashlib.md5(rep_id.encode()).hexdigest(), 16) % parts != args.subset_index:
                    continue
            # determine split fraction
            hval = int(hashlib.md5(rep_id.encode()).hexdigest(), 16) / float(1 << 128)
            if hval < PROBS[0]:
                split = 'train'
            elif hval < PROBS[0] + PROBS[1]:
                split = 'val'
            else:
                split = 'test'
            writers[split][1].writerow(row)

    # Close writers and SQLite
    for f_out, _ in writers.values():
        f_out.close()
    cmap.close()

    # Hashing: collect offsets, lengths, tax_ids
    for lbl in ('train', 'val', 'test'):
        csv_file = f"{args.hash_prefix}_{lbl}.csv"
        hash_file = f"{args.hash_prefix}_{lbl}.hash"
        offsets, lengths, tax_ids = [], [], []
        with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
            # Skip header line for manual offset tracking
            f.readline()
            pos = f.tell()
            line = f.readline()
            while line:
                parts = line.rstrip('\n').split(',')
                if len(parts) > max(seq_idx, tax_idx):
                    tax = parts[tax_idx].strip()
                    if tax and tax != 'Unknown':
                        try:
                            offsets.append(pos)
                            lengths.append(int(parts[seq_idx]))
                            tax_ids.append(tax)
                        except ValueError:
                            pass
                pos = f.tell()
                line = f.readline()
        with open(hash_file, 'wb') as hf:
            pickle.dump({'header': header,
                         'offsets': offsets,
                         'lengths': lengths,
                         'tax_ids': tax_ids}, hf)
        print(f"Wrote {hash_file}")
