
import os
import matplotlib.pyplot as plt
from data_loader import load_ecg_record


def visualize_ecg(record_id="100", start_seconds=0, duration=10):
    # Load ECG signal and heartbeat annotations
    record, annotation = load_ecg_record(record_id)

    fs = record.fs
    signal = record.p_signal[:, 0]

    start = int(start_seconds * fs)
    end = min(int((start_seconds + duration) * fs), len(signal))

    time = [i / fs for i in range(start, end)]
    ecg_segment = signal[start:end]

    # Plot ECG waveform
    plt.figure(figsize=(12, 4))
    plt.plot(time, ecg_segment, color="blue", linewidth=1)

    # Mark annotated heartbeat locations
    for sample, symbol in zip(annotation.sample, annotation.symbol):
        if start <= sample < end:
            plt.axvline(
                x=sample / fs,
                color="red",
                linestyle="--",
                alpha=0.4
            )

    plt.title(f"ECG Signal - MIT-BIH Record {record_id}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (mV)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/ecg_signal.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    visualize_ecg()
