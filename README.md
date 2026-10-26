# Understanding the Dynamics of Audio Classification in Command Recognition

MSc Data Science research project, University of Hertfordshire (2024 to 2025)

## Overview
This project trains and evaluates convolutional neural network (CNN) models that classify spoken commands such as "bed", "cat", "tree" and "house". It focuses on speech features such as waveforms and Mel spectrograms, and compares CNNs of different depths to understand how model complexity affects performance.

## Dataset
The project uses the **Synthetic Speech Commands Dataset (augmented version)**, which contains WAV recordings of several spoken words, organised in subdirectories by command. Audio is sampled at 16,000 Hz and resampled to 8,000 Hz before training.

## Project structure
- **Data exploration:** analysed the distribution of files and recording durations, and visualised waveforms and Mel spectrograms
- **Preprocessing:** built a Python pipeline to load, clean, validate, label and split several thousand audio records, padding or trimming all samples to a consistent length and extracting Mel spectrogram features
- **Model architecture:** built CNNs in Keras with convolutional layers, dense layers and dropout regularisation
- **Training and evaluation:** compared 3, 5 and 7-layer CNNs on an 80/20 train and validation split, plotting training and validation loss and accuracy

## Key results
| Model | Validation accuracy | Notes |
|---|---|---|
| 3-layer CNN | **71%** | Best performing model |
| 5-layer CNN | Lower | Failed to converge |
| 7-layer CNN | Lower | Failed to converge |

The simplest model performed best. Likely reasons the deeper models underperformed:
- the input representation of the audio features
- missing normalisation
- limited training time

## Lessons learned
1. Clean, consistent input data matters more than model complexity
2. Deeper networks need careful normalisation and tuning to converge
3. Understanding why a model fails is often as valuable as the result itself

## Next steps
- Add batch normalisation and tune the learning rate
- Train for longer with early stopping
- Compare Mel spectrograms with MFCC features

## Tools
Python · TensorFlow · Keras · NumPy · pandas · scikit-learn · matplotlib · Jupyter Notebook

## How to run
1. Clone this repository
2. Install the dependencies: `pip install tensorflow numpy pandas scikit-learn matplotlib`
3. Open `Final Code_NEW.ipynb` in Jupyter Notebook and run all cells

## Author
**Abhishek Battula**, MSc Data Science
[LinkedIn](https://www.linkedin.com/in/abhishek-battula-a25454273)
