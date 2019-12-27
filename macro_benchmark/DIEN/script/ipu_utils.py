import numpy as np

# padding sequence to maxlen
def prepare_data(input, target, dtype, maxlen = None, return_neg = False):
    # x: a list of sentences
    lengths_x = [len(s[4]) for s in input]
    seqs_mid = [inp[3] for inp in input]
    seqs_cat = [inp[4] for inp in input]
    noclk_seqs_mid = [inp[5] for inp in input]
    noclk_seqs_cat = [inp[6] for inp in input]

    if maxlen is not None:
        new_seqs_mid = []
        new_seqs_cat = []
        new_noclk_seqs_mid = []
        new_noclk_seqs_cat = []
        new_lengths_x = []
        for l_x, inp in zip(lengths_x, input):
            if l_x > maxlen:
                new_seqs_mid.append(inp[3][l_x - maxlen:])
                new_seqs_cat.append(inp[4][l_x - maxlen:])
                new_noclk_seqs_mid.append(inp[5][l_x - maxlen:])
                new_noclk_seqs_cat.append(inp[6][l_x - maxlen:])
                new_lengths_x.append(maxlen)
            else:
                new_seqs_mid.append(inp[3] + [0] * (maxlen - len(inp[3])))
                new_seqs_cat.append(inp[4] + [0] * (maxlen - len(inp[4])))
                list_0 = []
                for i in range(maxlen - len(inp[5])):
                    list_0.append([0,0,0,0,0])
                new_noclk_seqs_mid.append(inp[5] + list_0)

                list_1 = []
                for i in range(maxlen - len(inp[6])):
                    list_1.append([0,0,0,0,0])
                new_noclk_seqs_cat.append(inp[6] + list_1)
                new_lengths_x.append(maxlen)

        lengths_x = new_lengths_x
        seqs_mid = new_seqs_mid
        seqs_cat = new_seqs_cat
        noclk_seqs_mid = new_noclk_seqs_mid
        noclk_seqs_cat = new_noclk_seqs_cat

        if len(lengths_x) < 1:
            return None, None, None, None

    n_samples = len(seqs_mid)
    maxlen_x = np.max(lengths_x)
    neg_samples = len(noclk_seqs_mid[0][0])

    mid_his = np.zeros((n_samples, maxlen_x)).astype('int32')
    cat_his = np.zeros((n_samples, maxlen_x)).astype('int32')
    noclk_mid_his = np.zeros((n_samples, maxlen_x, neg_samples)).astype('int32')
    noclk_cat_his = np.zeros((n_samples, maxlen_x, neg_samples)).astype('int32')

    if dtype == 'FP32':
        data_type = 'float32'
    elif dtype == 'FP16':
        data_type = 'float16'
    else:
        raise ValueError("Invalid model data type: %s" % dtype)
    mid_mask = np.zeros((n_samples, maxlen_x)).astype(data_type)
    for idx, [s_x, s_y, no_sx, no_sy] in enumerate(zip(seqs_mid, seqs_cat, noclk_seqs_mid, noclk_seqs_cat)):
        mid_mask[idx, :lengths_x[idx]] = 1.
        mid_his[idx, :lengths_x[idx]] = s_x
        cat_his[idx, :lengths_x[idx]] = s_y
        noclk_mid_his[idx, :lengths_x[idx], :] = no_sx
        noclk_cat_his[idx, :lengths_x[idx], :] = no_sy

    uids = np.array([inp[0] for inp in input])
    mids = np.array([inp[1] for inp in input])
    cats = np.array([inp[2] for inp in input])

    if return_neg:
        return uids, mids, cats, mid_his, cat_his, mid_mask, np.array(target), np.array(lengths_x), noclk_mid_his, noclk_cat_his

    else:
        return uids, mids, cats, mid_his, cat_his, mid_mask, np.array(target), np.array(lengths_x)

