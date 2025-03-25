class Sequence:
    
    def __init__(self, sequence, length, taxon_id, seq_id):
        self.sequence = str(sequence)
        self.length = int(length)
        self.taxon_id = int(taxon_id)
        self.seq_id = str(seq_id)
        self.masked_seq = ""

    def add_masking(self, seq):
        self.masked_seq = seq
