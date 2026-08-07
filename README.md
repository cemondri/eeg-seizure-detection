# CHB-MIT EEG Seizure Detection

This repository contains a seizure detection pipeline built on the CHB-MIT dataset. The primary goal of this project is not to chase artificially high accuracy scores, but to measure **patient-independent generalization** using an honest evaluation methodology.

> **Note:** This is an exploratory and methodological project. It is not a real-time seizure prediction or clinical warning system.

## Overview
Seizure detection means identifying when an epileptic seizure occurs from EEG. It has practical uses, such as helping clinicians review long EEG recordings faster. But this project focuses less on detection itself and more on a methodological question: **does the model actually work on a completely new patient?** 

Many studies report very high scores, but they often mix patient data across training and test sets. I built this to see the real performance on unseen patients.

## Dataset
This project uses the [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/). 

* A subset of **5 patients** (`chb01`, `chb02`, `chb03`, `chb05`, `chb08`) is used for this analysis to keep the scope manageable.
* **Data not included:** The raw EDF files are huge and are not tracked in this repository. You will need to download the specific patient folders directly from PhysioNet.

## Methodology
* **Preprocessing:** EEG signals are filtered (1-40 Hz) and divided into 4-second windows (epochs).
* **Features:** Three main features are extracted from the signals:
  * *Band power:* Captures the energy in different brain wave frequencies (delta, theta, alpha, beta, gamma).
  * *Line length:* Catches sudden jumps and signal complexity.
  * *Inter-channel synchronization:* Measures how synchronized the different EEG channels are with each other.
* **Model:** A `RandomForestClassifier` with `class_weight='balanced'` is used, as seizures are extremely rare compared to non-seizure brain activity.
* **Validation:** To properly test on unseen patients, the evaluation uses a patient-level split (`GroupKFold`).

## Results & The Leakage Problem
The core findings of the project highlight the difficulty of generalization:

* **Patient-independent testing (AUC vs. Recall):** When tested on an unseen patient, the average recall is only **~28%**. However, the **ROC-AUC is ~0.865**. This means the model separates seizure epochs from non-seizure epochs quite well. The low recall comes from the default 0.5 threshold, not from a lack of discriminative power.
* **The threshold trade-off:** Lowering the threshold (e.g., to 0.3) pushes the recall up to ~37%, but the false alarms (false positives) also increase. This is the classic precision-recall trade-off.
* **Patient-specific testing:** When the model is evaluated on a patient it trained on, the average recall jumps to **~84.5%**.
* **Huge variance between patients:** In the patient-independent test, the recall across different folds ranges from 0% to ~60%. The model completely misses some patients. This makes the "seizures are highly patient-specific" finding very concrete.
* **The Accuracy Trap:** The model achieves **~95% accuracy**, but it misses over 70% of the seizures. Accuracy is completely misleading in highly imbalanced datasets like this, which is why the focus here is on Recall and AUC.
* **Feature impact:** Adding line length and synchronization did not improve patient-independent recall. This suggests the bottleneck is the biological variability between patients, not the features.
* **Why does this happen? (Data Leakage):** If you put data from the same patient into both the training and testing sets, the model learns to recognize that specific patient's EEG, including their particular seizure pattern, rather than a general seizure signature. This is data leakage, and it makes random splits look artificially successful.

## Honesty & Limitations
* **AUC Skepticism (The PR Curve):** In highly imbalanced datasets, the ROC/AUC score can be overly optimistic because of the massive number of True Negatives (non-seizure brain activity). **While ROC-AUC is 0.865, the PR average precision is only 0.186 (vs. a 0.027 random baseline).** This confirms that ROC is overly optimistic on imbalanced data, while PR reflects the low reliability of the minority-class predictions.

<img width="700" height="500" alt="pr_curve" src="https://github.com/user-attachments/assets/1bbb06de-cd4f-46a3-bb9d-3d8e743aa6fe" />

* In the `evaluate_patient_specific` function, I used a plain 5-fold split instead of a strict seizure-level split. This means epochs from the same seizure can appear in both the train and test sets, which probably inflates the ~84.5% score slightly.
* The pipeline uses basic features and limits the scope to 5 patients. The main point of this repo is not to build a perfect, state-of-the-art model, but to demonstrate why honest testing and patient-level splitting are absolute requirements in clinical machine learning.

## How to Run
1. Clone this repository.
2. Install the necessary dependencies: pip install mne scikit-learn numpy matplotlib
3. Download the patient data from PhysioNet. Create a directory named chbmit_data/ in the same folder as the script, and place the patient folders (chb01/, chb02/, etc.) inside it.
4. Run the detection script (this will also generate and save the pr_curve.png plot): python seizure_detection.py

## Future Work
Preictal phase: Expanding the scope to catch the pre-seizure phase (seizure prediction) rather than just detection.
Scaling: Optimizing the feature extraction to run efficiently across all patients in the CHB-MIT dataset.
