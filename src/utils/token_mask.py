import numpy as np
from torch.nn.utils.rnn import pad_sequence


def extract_masked_logits(logits, masked_pos):
    extracted_logits = [logits[i][mask] for i, mask in enumerate(masked_pos)]
    padded_logits = pad_sequence(extracted_logits, batch_first=True,padding_value=0.0)
    return padded_logits


def mask_batch(batch, batch_number, epoch):
    """
    Efficiently masks a whole batch of Sequence objects.
    
    Parameters:
        batch (list[Sequence]): List of Sequence objects.
        batch_number (int): The current batch number.
        epoch (int): The current epoch number.
        
    Returns:
        List of tuples (masked_seq, mask_indices) for each sequence.
    """

    vocab = np.array(['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O'])
    
    results = []
    batch_size = len(batch)
    
    for i, seq_obj in enumerate(batch):

        # reproducibility across all sequences
        seed = epoch * 1000000 + batch_number * batch_size + i
        rng = np.random.default_rng(seed)
        
        seq = seq_obj.sequence
        chain = np.array(list(seq), dtype='<U10')
        seq_len = len(chain)
        num_to_mask = max(1, int(round(0.15 * seq_len)))
        
        #  unique indices to mask
        mask_indices = rng.choice(seq_len, size=num_to_mask, replace=False)
        
        # counts for actions based on ratios
        num_mask    = int(round(0.8 * num_to_mask))
        num_random  = int(round(0.1 * num_to_mask))
        num_unchanged = num_to_mask - num_mask - num_random
        
        # adjust counts to sum exactly to num_to_mask
        while num_mask + num_random + num_unchanged > num_to_mask:
            num_mask -= 1
        while num_mask + num_random + num_unchanged < num_to_mask:
            num_mask += 1
        
        # array of action labels
        actions = np.array(['mask'] * num_mask + 
                           ['random'] * num_random + 
                           ['unchanged'] * num_unchanged)
        rng.shuffle(actions)
        
        # actions for each chosen index
        for idx, action in zip(mask_indices, actions):
            if action == 'unchanged':
                continue
            elif action == 'mask':
                chain[idx] = '<mask>'
            elif action == 'random':
                # pick a letter from vocab that isn't the original
                orig = chain[idx]
                allowed = vocab[vocab != orig]
                chain[idx] = rng.choice(allowed)
        
        seq_obj.add_masking(''.join(chain.tolist()))
        results.append((seq_obj, tuple(mask_indices.tolist())))
    
    return results


