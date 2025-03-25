#!/bin/bash

# 1. Download UniRef100 FASTA file (compressed)
echo "Downloading UniRef100 dataset..."
wget -q ftp://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref100/uniref100.fasta.gz -O uniref100.fasta.gz

# Ensure output directory exists
mkdir -p taxon_output

# Process the FASTA file, extract required information, and group by taxon_id
gzip -dc uniref100.fasta.gz | awk '
    BEGIN { OFS="," }
    /^>/ {
        if (id) {
            print id, taxon, length(seq), seq;
        }
        split($0, a, " ");
        id = substr(a[1], 2);  # Extract sequence_id (remove ">")

        taxon = "Unknown";
        if (match($0, /TaxID=[0-9]+/)) {
            taxon = substr($0, RSTART+6, RLENGTH-6);
        }

        seq = "";
        next;
    }
    /^[^>]/ {
        seq = seq $0;
    }
    END {
        if (id) {
            print id, taxon, length(seq), seq;
        }
    }
' > temp_file.csv

# Sort by taxon_id (second column)
sort -t, -k2,2 -S 50% --temporary-directory=/home/developer/PLM_project_2.0/data/raw/uniref100/temp_dir -o sorted_file.csv temp_file.csv

# Add header to the final file and move the result to the desired output location
echo "sequence_id,taxon_id,sequence_length,sequence" | cat - sorted_file.csv > taxon_output/all_taxons_sequences.csv

echo "Processing complete! Results in 'taxon_output/all_taxons_sequences.csv'"
