import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    pe = np.zeros((seq_len, d_model))

    # for pos in range(seq_len):
    #     for i in range(d_model):
    #         angle = pos / np.power(base, (2 * (i // 2)) / d_model)
    #         if i % 2 == 0:
    #             pe[pos, i] = np.sin(angle)
    #         else:
    #             pe[pos, i] = np.cos(angle)

    pos = np.arange(seq_len)[:, np.newaxis]
    dim = np.arange(d_model)[np.newaxis, :]
    angles = pos / np.power(base, (2 * (dim // 2)) / d_model)

    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe