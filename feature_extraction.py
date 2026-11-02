
import numpy as np
from scipy.stats import skew, kurtosis

BEAT_SYMBOLS = {
    "N", "L", "R", "A", "a", "J", "S",
    "V", "E", "F", "e", "j", "/", "f", "Q"
}


def extract_features(record, annotation):
    signal = record.p_signal[:, 0]
    fs = record.fs

    features = []
    labels = []

    beats = [
        (sample, symbol)
        for sample, symbol in zip(annotation.sample, annotation.symbol)
        if symbol in BEAT_SYMBOLS
    ]

    for i in range(1, len(beats) - 1):
        sample, symbol = beats[i]

        previous_sample = beats[i - 1][0]
        next_sample = beats[i + 1][0]

        rr_previous = (sample - previous_sample) / fs
        rr_next = (next_sample - sample) / fs

        rr_ratio = rr_previous / rr_next if rr_next > 0 else 0
        heart_rate = 60 / rr_previous if rr_previous > 0 else 0

        start = max(0, sample - int(0.2 * fs))
        end = min(len(signal), sample + int(0.2 * fs))
        segment = signal[start:end]

        if len(segment) == 0:
            continue

        amplitude_range = np.max(segment) - np.min(segment)
        mean_amplitude = np.mean(segment)
        std_amplitude = np.std(segment)
        signal_energy = np.mean(segment ** 2)

        if std_amplitude > 1e-10:
            signal_skewness = skew(segment)
            signal_kurtosis = kurtosis(segment)
        else:
            signal_skewness = 0
            signal_kurtosis = 0

        beat_features = [
            rr_previous,
            rr_next,
            rr_ratio,
            heart_rate,
            amplitude_range,
            mean_amplitude,
            std_amplitude,
            signal_energy,
            signal_skewness,
            signal_kurtosis
        ]

        if not np.all(np.isfinite(beat_features)):
            continue

        features.append(beat_features)
        labels.append(symbol)

    return np.array(features), np.array(labels)
