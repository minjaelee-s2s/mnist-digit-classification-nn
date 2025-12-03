# STEP 1. Download MNIST
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X = mnist['data'] # (70000, 784)
y = mnist['target'].astype(int) # (70000,)


# STEP 2. Scaling + Train/Test split
X = X / 255.0  # 0~255 → 0~1

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# STEP 3. Single-Layer NN: input 784 → output 10 (softmax)
N, D = X_train.shape # N: number of samples, D: number of features (784)
K = 10 # number of classes (0~9)

rng = np.random.default_rng(0)

# W: (D, K), b: (K,)
W = 0.01 * rng.standard_normal((D, K))
b = np.zeros(K)

# Activation: softmax
def softmax(z):
    # z : (N, K)
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Loss: categorical cross-entropy
def categorical_cross_entropy(y_true_onehot, y_prob, eps=1e-12):
    # y_true_onehot : (N, K) one-hot labels
    # y_prob : (N, K) softmax probabilities
    y_prob = np.clip(y_prob, eps, 1 - eps)
    # for each sample: -sum(y_true * log(y_prob))
    loss_per_sample = -np.sum(y_true_onehot * np.log(y_prob), axis=1)
    return np.mean(loss_per_sample)

# Convert integer labels to one-hot encoded vectors
def to_one_hot(y, num_classes):
    N = y.shape[0]
    one_hot = np.zeros((N, num_classes))
    one_hot[np.arange(N), y] = 1
    return one_hot

Y_train = to_one_hot(y_train, K) # (N, 10)

lr = 0.005
epochs = 500

for epoch in range(epochs):
    # Forward
    z = np.dot(X_train, W) + b # (N, K)
    y_hat = softmax(z) # (N, K)

    loss = categorical_cross_entropy(Y_train, y_hat)

    # Backprop (gradient of softmax + cross-entropy)
    dz = (y_hat - Y_train) / N # (N, K)
    dW = np.dot(X_train.T, dz) # (D, K)
    db = np.sum(dz, axis=0) # (K,)

    # Gradient descent
    W -= lr * dW
    b -= lr * db

    if (epoch + 1) % 50 == 0:
        print(f"epoch {epoch+1:4d}, loss = {loss:.4f}")

# STEP 4. Accuracy calculation
def predict(X, W, b):
    z = np.dot(X, W) + b # (N, K)
    y_prob = softmax(z) # (N, K)
    y_pred = np.argmax(y_prob, axis=1)
    return y_pred

y_train_pred = predict(X_train, W, b)
train_acc = np.mean(y_train_pred == y_train)

y_test_pred = predict(X_test, W, b)
test_acc = np.mean(y_test_pred == y_test)

print(f"Train accuracy: {train_acc:.4f}")
print(f"Test  accuracy: {test_acc:.4f}")
