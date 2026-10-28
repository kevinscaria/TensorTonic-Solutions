import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    N = len(seqs)
    if max_len is not None:
        L = max_len
    else:
        L = max(len(seq) for seq in seqs) if N > 0 else 0

    result = np.full((N, L), fill_value=pad_value, dtype=int)
    for i, seq in enumerate(seqs):
        truncated = seq[:L]
        result[i, :len(truncated)] = truncated
    return result

    
    