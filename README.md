# MNIST Neural Network from Scratch

This project starts with a simple binary classification problem (distinguishing between digits 3 and 8) and then extends the model to a 10-class classifier (digits 0–9).

## Step 1: Binary Classification (3 vs 8)

- File: `mnist_binary_38.py`
- Implemented a single-layer neural network with a sigmoid output.
- Task: distinguish between digit 3 and digit 8.
- Loss: binary cross-entropy.

## Step 2: Multi-class Classification (0–9)

- File: `mnist_multiclass_0_9.py`
- Extended the model to handle all 10 digits.
- Changed the output layer to 10 neurons and used softmax.
- Loss: categorical cross-entropy with one-hot encoded labels.

## Future Work

- Add a hidden layer (two-layer neural network).
- Compare performance with tree-based models (Decision Tree, Random Forest).
