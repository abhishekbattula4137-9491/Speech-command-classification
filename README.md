# Speech-command-classification
# Speech Command Classification under Noisy Conditions

MSc Data Science research project, University of Hertfordshire (2024 to 2025)

## Overview
This project builds and evaluates machine learning models that classify short spoken commands in noisy, real-world conditions. It compares neural networks of different depths to understand how model complexity affects performance.

## What I did
- **Data pipeline:** built a reusable Python pipeline (pandas, NumPy, scikit-learn) to load, clean, validate, label and split several thousand raw audio records
- **Noise augmentation:** added noise to the training data to simulate unpredictable real-world conditions and improve robustness
- **Model comparison:** designed and trained 3, 5 and 7-layer neural networks in TensorFlow and Keras on an 80/20 train and test split
- **Evaluation:** compared validation accuracy and training behaviour across all three models

## Key results
| Model | Validation accuracy | Notes |
|---|---|---|
| 3-layer | **71%** | Best performing model |
| 5-layer | Lower | Failed to converge |
| 7-layer | Lower | Failed to converge |

The simplest model performed best. I investigated why the deeper models underperformed and identified likely causes:
- input representation of the audio features
- missing normalisation
- limited training time

## Lessons learned
1. Clean, validated data matters more than model complexity
2. Deeper networks need careful normalisation and tuning to converge
3. Understanding why a model fails is often as valuable as the result itself

## Next steps
- Use MFCC or spectrogram features as input
- Add batch normalisation and learning-rate tuning
- Train for longer with early stopping
- Try convolutional neural networks (CNNs), which suit audio spectrograms

## Tools
Python · pandas · NumPy · scikit-learn · TensorFlow · Keras · matplotlib · Jupyter Notebook · Git

## How to run
1. Clone this repository
2. Install the dependencies: `pip install pandas numpy scikit-learn tensorflow matplotlib`
3. Open `speech_command_classification.ipynb` in Jupyter Notebook and run all cells

## Author
**Abhishek Battula**, MSc Data Science
[LinkedIn](https://www.linkedin.com/in/abhishek-battula-a25454273)
