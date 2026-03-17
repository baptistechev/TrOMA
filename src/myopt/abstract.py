def compute_marginals_pair(spectrum_bin, spectrum_val, bit1, bit2, bit_v1, bit_v2):
    filtered_vals = [v for s, v in zip(spectrum_bin, spectrum_val) if s[bit1] == bit_v1 and s[bit2] == bit_v2]

    return float(np.sum(filtered_vals))

def compute_marginals_triplets(spectrum_bin, spectrum_val, bit1, bit2, bit3, bit_v1, bit_v2, bit_v3):
    filtered_vals = [v for s, v in zip(spectrum_bin, spectrum_val) if s[bit1] == bit_v1 and s[bit2] == bit_v2 and s[bit3] == bit_v3]

    return float(np.sum(filtered_vals))

def compute_marginals_quadruplets(spectrum_bin, spectrum_val, bit1, bit2, bit3, bit4, bit_v1, bit_v2, bit_v3, bit_v4):
    filtered_vals = [v for s, v in zip(spectrum_bin, spectrum_val) if s[bit1] == bit_v1 and s[bit2] == bit_v2 and s[bit3] == bit_v3 and s[bit4] == bit_v4]

    return float(np.sum(filtered_vals))