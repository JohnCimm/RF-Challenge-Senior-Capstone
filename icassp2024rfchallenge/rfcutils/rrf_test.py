import numpy as np

from rrc_helper_fn import get_psf
from rrc_helper_fn import matched_filter
from rrc_helper_fn import get_psf1
from rrc_helper_fn import matched_filter1
samples_per_symbol = 8
span_in_symbols = 6
beta = 0.25
sig = np.random.randn(1000)  # Random input signal



#rrc_filter = get_psf(samples_per_symbol, span_in_symbols, beta)
#print('original')
#print(rrc_filter)


# Apply matched filter
#filtered_signal = matched_filter(sig, samples_per_symbol, span_in_symbols, beta)
#print('original')
#print(filtered_signal)

filtered_signal = matched_filter1(sig, samples_per_symbol, span_in_symbols, beta)
print('new')
print(filtered_signal)
rrc_filter = get_psf1(samples_per_symbol, span_in_symbols, beta)
print('new')
print(rrc_filter)