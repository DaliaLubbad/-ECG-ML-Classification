
import wfdb


def load_ecg_record(record_id="100"):
    """
    Load an ECG recording and its annotations
    from the MIT-BIH Arrhythmia Database.
    """
    record = wfdb.rdrecord(record_id, pn_dir="mitdb")
    annotation = wfdb.rdann(record_id, "atr", pn_dir="mitdb")

    return record, annotation
