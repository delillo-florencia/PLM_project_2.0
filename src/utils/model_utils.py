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


