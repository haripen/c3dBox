import numpy as np
from scipy.signal import butter, filtfilt, firwin


def butter_bandpass_filter(signal, fs, low, high, order=4):
    nyquist = 0.5 * fs
    low = max(1e-6, float(low))
    high = min(float(high), nyquist * 0.95)
    if high <= low:
        raise ValueError("Invalid bandpass bounds after adjustment.")
    b, a = butter(order, [low / nyquist, high / nyquist], btype="band")
    return filtfilt(b, a, signal)


def apply_lowpass_filter(signal, cutoff_frequency, frame_rate, order=2, package="scipy", filter_type="FIR"):
    """
    Mirror of harvestC3Ds4opensim.py lowpass (FIR by default).
    """
    if (package == "scipy") and (filter_type == "butter"):
        nyquist = 0.5 * frame_rate
        corrected_cutoff = cutoff_frequency / 0.802
        normal_cutoff = corrected_cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        non_zero_indices = np.nonzero(signal)[0]
        if len(non_zero_indices) > 3 * (max([len(a), len(b)])) - 1:
            start_idx = non_zero_indices[0]
            end_idx = non_zero_indices[-1]
            filtered_segment = filtfilt(b, a, signal[start_idx:end_idx + 1])
            filt_signal = np.zeros_like(signal)
            filt_signal[start_idx:end_idx + 1] = filtered_segment
            if start_idx > 0:
                filt_signal[:start_idx] = signal[:start_idx]
            if end_idx < len(signal) - 1:
                filt_signal[end_idx + 1:] = signal[end_idx + 1:]
        else:
            filt_signal = signal
    elif (package == "scipy") and (filter_type == "FIR"):
        numtaps = 13
        pad_width = numtaps
        signal = np.pad(signal, pad_width, mode="reflect")
        window = "hamming"
        fir_coefficients = firwin(numtaps, cutoff_frequency, pass_zero="lowpass", fs=frame_rate, window=window)
        filtered_signal = np.convolve(signal, fir_coefficients, mode="same")
        filt_signal = filtered_signal[pad_width:-pad_width]
    else:
        raise ValueError("Unsupported lowpass settings. Use scipy + FIR or butter.")
    return filt_signal


def process_semg_signal(
    raw,
    fs,
    bp_low=10.0,
    bp_high=490.0,
    lp_cut=15.0,
    demean=True,
    rectify=True,
    non_negative=True,
):
    x = np.asarray(raw, dtype=float)
    if x.size < 5 or not np.isfinite(x).any():
        return None
    if demean:
        x = x - np.nanmean(x)
    x = butter_bandpass_filter(x, fs, bp_low, bp_high, order=4)
    if rectify:
        x = np.abs(x)
    # Match harvestC3Ds4opensim.py: scipy butter lowpass
    x = apply_lowpass_filter(x, lp_cut, fs, order=2, package="scipy", filter_type="butter")
    if non_negative:
        x[x < 0] = 0.0
    return x
