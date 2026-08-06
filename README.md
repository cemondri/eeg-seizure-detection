# EEG Seizure Detection — Patient-Independent vs Patient-Specific

## 1. Summary

**What I did:** I built a seizure detection model using the CHB-MIT dataset. My goal was not to get high scores, but to measure patient-independent generalization with an honest methodology.

**What I didn't do:** This is not a real-time seizure prediction or warning system.

## 2. Overview / Motivation

Seizure detection means identifying when an epileptic seizure occurs from EEG. It has practical uses, such as helping clinicians review long EEG recordings faster. But this project focuses less on detection itself and more on a methodological question: does the model actually work on a completely new patient?

Many studies show very high scores, but they mix the data. I wanted to build this to see the real performance on unseen patients.

## 3. Dataset

This project uses the CHB-MIT Scalp EEG Database.

* I used 5 patients (chb01, chb02, chb03, chb05, chb08) for this analysis to keep things manageable.
* **Data not included:** Because the raw EDF files are huge, I did not put them in this repo. You can download the specific patient folders from the PhysioNet CHB-MIT page.

## 4. Method

* **Preprocessing:** The EEG signals are filtered (1-40 Hz) and cut into 4-second windows (epochs).
* **Features:** I extracted three main features from the signals:
  * **Band power:** Captures the energy in different brain wave frequencies (delta, theta, alpha, beta, gamma).
  * **Line length:** Catches sudden jumps and signal complexity.
  * **Inter-channel synchronization:** Measures how synchronized the different EEG channels are with each other.
* **Model:** I used a RandomForestClassifier. I also added `class_weight='balanced'` because seizures are very rare compared to normal brain activity.
* **Validation:** To test on unseen patients, I used a patient-level split (GroupKFold).

## 5. Results & The Leakage Problem

This is the core finding of the project:

* **Patient-independent testing:** When the model is tested on a completely new patient it has never seen, the average recall drops to ~29%.
* **Patient-specific testing:** When the model is tested on a patient it already trained on, the average recall is ~84%.
* **Feature impact:** Adding line length and synchronization did not improve patient-independent recall. This suggests the bottleneck is the biological variability between patients, not the features.
* **Why does this happen? (Data Leakage):** If you put data from the same patient into both the training and testing sets, the model just learns to recognize that specific patient's normal brain wave patterns. It memorizes the patient, not the actual seizure. This is data leakage, and it makes random splits look artificially successful.
* **Takeaway:** Seizure patterns are highly unique to each person. A general model performs poorly on new patients.

## 6. Honesty & Limitations

I want to be completely transparent about this code:

* In the `evaluate_patient_specific` function, I used a plain 5-fold split instead of a strict seizure-level split. This means epochs from the same seizure can appear in both the train and test sets, which probably inflates the ~84% score a little bit.
* I only used basic features and 5 patients.
* The main point of this repo is not to build a perfect, state-of-the-art model. The goal is to show why honest testing and patient-level splitting are absolute requirements in clinical machine learning.

## 7. How to Run

1. Clone this repository.
2. Install the necessary libraries (you will need `mne`, `scikit-learn`, and `numpy`).
3. Download the patient data from PhysioNet. Create a folder named `chbmit_data/` in the same directory as the script, and put the patient folders (`chb01/`, `chb02/`, etc.) inside it.
4. Run the script: seizure_detection.py

## 8. Future Work

* **Preictal phase:** Trying to catch the pre-seizure phase to do seizure prediction instead of detection.
* **More data:** Expanding the setup to run on all patients in the CHB-MIT dataset.
