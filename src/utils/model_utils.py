import os

def map_new_to_original(new_rank: int,
                        new_batch_idx: int,
                        orig_num_replicas: int,
                        new_num_replicas: int):
    """
    Given:
      - new_rank:       0..new_num_replicas-1  (your current global_rank)
      - new_batch_idx:  index of this batch *on this GPU* (batch_idx in training_step)
      - orig_num_replicas:  how many GPUs you used at precompute time (e.g. 50)
      - new_num_replicas:   how many GPUs you are using now (e.g. 2)
    Returns:
      - orig_rank:  which GPU originally produced this file
      - orig_batch: which batch‑idx that GPU wrote out
    """
    # global position in the *original* flattened batch list:
    global_idx = new_rank + new_batch_idx * new_num_replicas
    # invert the original sampler’s `batches[orig_rank::orig_num_replicas]`
    orig_rank  = global_idx % orig_num_replicas
    orig_batch = global_idx // orig_num_replicas
    return orig_rank, orig_batch


def get_seq_rep(results, batch_lens):
    """
    Get sequence representations from esm_compute
    """
    token_representations = results["last_hidden_state"]

    # Generate per-sequence representations via averaging
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))

    return sequence_representations


def get_logits(results):
    """
    Extracts logits from esm_compute
    """
    logits = results["logits"]  

    return logits


def int_env(*vars, default):
    """
    Try each name in order and return its int value if present and valid;
    otherwise return default.
    """
    for v in vars:
        val = os.environ.get(v)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                pass
    return default


