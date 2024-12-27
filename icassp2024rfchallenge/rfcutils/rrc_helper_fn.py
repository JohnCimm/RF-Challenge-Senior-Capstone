import sionna as sn
import numpy as np
import tensorflow as tf
import numpy as np
from scipy.signal import fir_filter_design as ffd

def get_psf1(samples_per_symbol, span_in_symbols, beta):
    """
    Generates a Root Raised Cosine (RRC) filter.

    Args:
        samples_per_symbol (int): Oversampling factor (samples per symbol).
        span_in_symbols (int): Number of symbols the filter spans.
        beta (float): Roll-off factor of the RRC filter.

    Returns:
        np.ndarray: Root Raised Cosine filter coefficients.
    """
    num_taps = span_in_symbols * samples_per_symbol + 1  # Number of filter coefficients
    t = np.linspace(-span_in_symbols / 2, span_in_symbols / 2, num_taps)

    # Handle the RRC filter calculation to avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        numerator = np.sin(np.pi * t * (1 - beta)) + 4 * beta * t * np.cos(np.pi * t * (1 + beta))
        denominator = np.pi * t * (1 - (4 * beta * t) ** 2)
        rrcf = np.divide(numerator, denominator, out=np.zeros_like(t), where=denominator != 0)

    # Fill the central tap manually to avoid NaNs from the division by zero
    central_idx = len(t) // 2
    rrcf[central_idx] = 1 - beta + (4 * beta / np.pi)

    # Normalize the filter coefficients to ensure unity gain
    rrcf /= np.sqrt(np.sum(rrcf ** 2))
    return rrcf


def get_psf(samples_per_symbol, span_in_symbols, beta):
    # samples_per_symbol: Number of samples per symbol, i.e., the oversampling factor
    # beta: Roll-off factor
    # span_in_symbols: Filter span in symbold
    rrcf = sn.signal.RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta)
    return rrcf

def matched_filter(sig, samples_per_symbol, span_in_symbols, beta):
    rrcf = get_psf(samples_per_symbol, span_in_symbols, beta)
    x_mf = rrcf(sig, padding="same")
    return x_mf
from scipy.signal import convolve

def matched_filter1(sig, samples_per_symbol, span_in_symbols, beta):
    """
    Applies a matched filter to the input signal using a Root Raised Cosine (RRC) filter.

    Args:
        sig (np.ndarray): Input signal to be filtered.
        samples_per_symbol (int): Oversampling factor (samples per symbol).
        span_in_symbols (int): Number of symbols the filter spans.
        beta (float): Roll-off factor of the RRC filter.

    Returns:
        np.ndarray: Filtered signal after applying the matched filter.
    """
    # Get the RRC filter coefficients
    rrcf = get_psf(samples_per_symbol, span_in_symbols, beta)
    
    # Apply convolution (matched filtering)
    x_mf = convolve(sig, rrcf, mode='same')
    
    return x_mf