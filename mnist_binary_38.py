# STEP 1. Download MNIST and filter 3 & 8
import numpy as np
from sklearn.datasets import fetch_openml # to download data from OpenML website

mnist = fetch_openml('mnist_784', version=1, as_frame=False) # downloading MNIST

X = mnist['data'] # X is a feature vector, which contains the info of the image
y = mnist['target'].astype(int) # answer vector

# print(X.shape, y.shape) # (70000, 784) (70000,)

mask = (y == 3) | (y == 8) # boolean indexing, used to subtract rows which match the condition
X_38 = X[mask]
y_38 = y[mask]

# print(X_38.shape, y_38.shape) # 13966 of 3&8s among 70000 image sets
# print(np.unique(y_38)) # check whether it filtered correctly by removing duplicates

y_bin = (y_38 == 8).astype(int) # turn into binary, 3 -> 0 (false) 8 -> 1 (true)

# print(np.unique(y_bin)) [0, 1]

# STEP 2. Scaling, Divide train and test sets
from sklearn.model_selection import train_test_split

X_38 = X_38 / 255.0 # pixel scale(gray scale) 0 ~ 255 -> 0 ~ 1
# going to use sigmoid function as an activation function, which is stable on range (0, 1)

X_train, X_test, y_train, y_test = train_test_split(X_38, y_bin, test_size=0.2, random_state=42, stratify=y_bin)
# 1. check the length of input datas (same length)
# 2. mix data set, maintaining index
# 3. divide train and test sets (20% in my code)
# 4. return as tuple of length 4
# cf) random_state = num; 이거는 섞는 방식을 고정하라는 뜻이고, stratify = 이거는 3과 8의 비율을 맞춰주는 역할이라는데,,, 사실 잘 모르겠
# print(X_train.shape, X_test.shape) # (11172, 784) (2794, 784)

# STEP 3. Designing Single-Layer NN
N, D = X_train.shape
# print(N, D) # 11172 784

rng = np.random.default_rng(0) # random number generator with fixed seed 0
# Build z = XW + b
W = 0.01 * rng.standard_normal(D) # 가중치를 왜 이렇게 만들까?
b = 0.0 # 바이어스를 왜 보통 0으로 시작할까?
# 둘은 각각 초기 추측값, 기본값이라고 해요.

# Build an activation function
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

# Build a loss function; decreases when closer to answer (supervised leaning)
def binary_cross_entropy(y_true, y_prob, eps=1e-12): # 수업 때 배운 것과 다른 부분
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -np.mean(
        y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)
    )

lr = 0.005
epochs = 5000 # epoch: 전체 데이터를 학습한 횟수

for epoch in range(epochs):
    z = np.dot(X_train, W) + b
    y_hat = sigmoid(z)

    loss = binary_cross_entropy(y_train, y_hat)

    dz = (y_hat - y_train) / N
    dW = np.dot(X_train.T, dz)
    db = np.sum(dz)

    W -= lr * dW
    b -= lr * db

    if (epoch + 1) % 20 == 0:
        print(f"epoch {epoch+1 : 3d}, loss = {loss:.4f}")

# STEP 4: Check Accuracy
y_train_pred_prob = sigmoid(np.dot(X_train, W) + b)
y_train_pred = (y_train_pred_prob >= 0.5).astype(int)
train_acc = np.mean(y_train_pred == y_train)

y_test_pred_prob = sigmoid(np.dot(X_test, W) + b)
y_test_pred = (y_test_pred_prob >= 0.5).astype(int)
test_acc = np.mean(y_test_pred == y_test)

print(f"Train accuracy: {train_acc:.4f}")
print(f"Test accuracy : {test_acc:.4f}")