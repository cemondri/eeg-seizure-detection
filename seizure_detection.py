"""
EEG-based epileptic seizure detection on the CHB-MIT dataset.

Compares patient-specific vs patient-independent performance to show that
seizure patterns are largely patient-specific: a model trained on some
patients generalizes poorly to unseen patients, while a model evaluated
within the same patient performs much better.

Features: band power, line length, inter-channel synchronization.
Model: random forest with patient-level cross-validation to avoid leakage.

Data: https://physionet.org/content/chbmit/1.0.0/
(Not included in this repo. Download the patient folders you want to use.)
"""

import numpy as np
import mne
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score

# --- Configuration ---
# Folder containing the downloaded CHB-MIT patient folders (chb01/, chb02/, ...)
DATA_DIR = 'chbmit_data'

# Seizure onset/offset times (in seconds) per file, grouped by patient.
# Taken from each patient's summary file (e.g. chb01-summary.txt).
patient_seizures = {
    'chb01': {
        'chb01_03': (2996, 3036),
        'chb01_04': (1467, 1494),
        'chb01_15': (1732, 1772),
        'chb01_16': (1015, 1066),
        'chb01_18': (1720, 1810),
        'chb01_21': (327, 420),
        'chb01_26': (1862, 1963),
    },
    'chb02': {
        'chb02_16': (130, 212),
        'chb02_16+': (2972, 3053),
        'chb02_19': (3369, 3378),
    },
    'chb03': {
        'chb03_01': (362, 414),
        'chb03_02': (731, 796),
        'chb03_03': (432, 501),
        'chb03_04': (2162, 2214),
        'chb03_34': (1982, 2029),
        'chb03_35': (2592, 2656),
        'chb03_36': (1725, 1778),
    },
    'chb05': {
        'chb05_06': (417, 532),
        'chb05_13': (1086, 1196),
        'chb05_16': (2317, 2413),
        'chb05_17': (2451, 2571),
        'chb05_22': (2348, 2465),
    },
    'chb08': {
        'chb08_02': (2670, 2841),
        'chb08_05': (2856, 3046),
        'chb08_11': (2988, 3122),
        'chb08_13': (2417, 2577),
        'chb08_21': (2083, 2347),
    },
}

# Standard EEG frequency bands (Hz)
bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 40),
}

EPOCH_SECONDS = 4.0


def process_file(file_name, seizure_start, seizure_end):
    """Read one EDF file, extract features per epoch, and label epochs.

    Returns:
        X: feature matrix, shape (n_epochs, n_features)
        y: labels, shape (n_epochs,)  -- 1 = seizure, 0 = non-seizure
    """
    # Derive the patient folder from the file name (e.g. 'chb01_03' -> 'chb01')
    patient = file_name.split('_')[0]
    path = f'{DATA_DIR}/{patient}/{file_name}.edf'

    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    raw.filter(1.0, 40.0, verbose=False)

    # Split the continuous recording into fixed-length epochs
    epochs = mne.make_fixed_length_epochs(
        raw, duration=EPOCH_SECONDS, preload=True, verbose=False
    )

    # --- Feature 1: band power (per channel, per band) ---
    band_features = []
    for name, (fmin, fmax) in bands.items():
        psd = epochs.compute_psd(fmin=fmin, fmax=fmax, verbose=False)
        power = psd.get_data().mean(axis=2)   # average over frequency -> (epochs, channels)
        band_features.append(power)
    band_X = np.log(np.concatenate(band_features, axis=1))   # (epochs, channels * bands)

    # Raw epoch data for time-domain features: (epochs, channels, samples)
    data = epochs.get_data()

    # --- Feature 2: line length (per channel) ---
    # Sum of absolute differences between consecutive samples.
    line_length = np.sum(np.abs(np.diff(data, axis=2)), axis=2)   # (epochs, channels)
    line_length = np.log(line_length)

    # --- Feature 3: inter-channel synchronization (one value per epoch) ---
    # Mean absolute correlation across all channel pairs.
    sync = []
    for epoch in data:                              # epoch: (channels, samples)
        corr = np.corrcoef(epoch)                   # (channels, channels)
        upper = corr[np.triu_indices_from(corr, k=1)]   # unique pairs, exclude diagonal
        sync.append(np.mean(np.abs(upper)))
    sync = np.array(sync).reshape(-1, 1)            # (epochs, 1)

    # Combine all features
    X = np.concatenate([band_X, line_length, sync], axis=1)

    # --- Labels ---
    # An epoch is labeled seizure (1) if it overlaps the seizure interval.
    epoch_times = epochs.events[:, 0] / raw.info['sfreq']
    y = np.zeros(len(epochs), dtype=int)
    for i, t in enumerate(epoch_times):
        if t < seizure_end and (t + EPOCH_SECONDS) > seizure_start:
            y[i] = 1

    return X, y


def build_dataset():
    """Process all seizure files and assemble the full dataset."""
    X_list, y_list, group_list = [], [], []

    for patient, files in patient_seizures.items():
        for file_name, (start, end) in files.items():
            try:
                X_f, y_f = process_file(file_name, start, end)
                X_list.append(X_f)
                y_list.append(y_f)
                # Tag every epoch of this file with its patient (for patient-level split)
                group_list.append(np.full(len(y_f), patient))
                print(f"{file_name}: {X_f.shape[0]} epochs, {np.sum(y_f)} seizure")
            except FileNotFoundError:
                print(f"{file_name}: SKIPPED (file not found)")

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    groups = np.concatenate(group_list)
    return X, y, groups


def evaluate_patient_independent(X, y, groups):
    """Train on some patients, test on unseen patients (GroupKFold)."""
    model = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced'
    )
    gkf = GroupKFold(n_splits=5)
    recall = cross_val_score(model, X, y, groups=groups, cv=gkf, scoring='recall')
    return recall


def evaluate_patient_specific(X, y, groups):
    """Evaluate within each patient separately (train/test on the same patient).

    NOTE: this uses a plain 5-fold split, not a seizure-level split, so
    epochs from the same seizure may appear in both train and test.
    This can slightly inflate the reported recall.
    """
    results = {}
    for patient in np.unique(groups):
        mask = (groups == patient)
        X_p, y_p = X[mask], y[mask]
        n_seizure = np.sum(y_p == 1)
        if n_seizure < 5:
            print(f"{patient}: too few seizure epochs ({n_seizure}), skipped")
            continue
        model = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced'
        )
        recall = cross_val_score(model, X_p, y_p, cv=5, scoring='recall')
        results[patient] = recall.mean()
        print(f"{patient}: recall {recall.mean():.3f}  (seizure epochs: {n_seizure})")
    return results


def main():
    print("Building dataset...")
    X, y, groups = build_dataset()
    print(f"\nTotal: {X.shape[0]} epochs, {X.shape[1]} features")
    print(f"Seizure: {np.sum(y == 1)} / Non-seizure: {np.sum(y == 0)}")
    print(f"Patients: {np.unique(groups)}\n")

    print("--- Patient-independent (unseen patients) ---")
    recall_indep = evaluate_patient_independent(X, y, groups)
    print(f"Per-fold recall: {recall_indep}")
    print(f"Mean recall: {recall_indep.mean():.3f}\n")

    print("--- Patient-specific (same patient) ---")
    results = evaluate_patient_specific(X, y, groups)
    print(f"\nMean patient-specific recall: {np.mean(list(results.values())):.3f}")


if __name__ == '__main__':
    main()
