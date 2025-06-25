## Perceptron Language Classifier

This Python script implements a simple single-layer perceptron to classify text into languages based on letter frequency.

### Features

- Reads normalized letter frequency vectors from a CSV file.
- Trains one perceptron per language (e.g., English, Polish, Spanish, German).
- Predicts the most likely language for a given input vector.

### How to Use

1. Prepare a CSV file with letter frequency vectors and language labels.
2. Run the script with Python 3:

```bash
python main.py
