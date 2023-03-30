


def parse_bitmask(mask, nbits):
    bits = [bit for bit in range(0, nbits) if (mask & 1 << bit) > 0]
    return bits

def satisfies_val(masks, val, nbits):
    idxs = []
    for j in range(len(masks)):
        bits = parse_bitmask(masks[j], nbits)
        if (val >= 0) & (val in bits):
            idxs.append(j)
    return idxs
