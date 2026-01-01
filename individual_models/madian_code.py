import os
import numpy as np
from PIL import Image


# ============================
# 1) 
# ============================
dataset_dir = "SmallDataset"

X_data, y_data = [], []

for folder_name in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, folder_name)
    label = int(folder_name)  # 
    for file in os.listdir(folder_path):
        img = Image.open(os.path.join(folder_path, file)).convert("L").resize((28, 28))
        arr = np.array(img).astype(np.float32).flatten() / 255.0
        X_data.append(arr)
        y_data.append(label)

X_data = np.array(X_data)
y_data = np.array(y_data, dtype=np.int32)

# ============================
# 2) 
# ============================
train_idx, test_idx = [], []
for c in np.unique(y_data):
    idxs = np.where(y_data == c)[0]
    np.random.shuffle(idxs)
    split = int(0.85 * len(idxs))
    train_idx.extend(idxs[:split])
    test_idx.extend(idxs[split:])

train_idx, test_idx = np.array(train_idx), np.array(test_idx)
np.random.shuffle(train_idx)
np.random.shuffle(test_idx)

X_train, y_train = X_data[train_idx], y_data[train_idx]
X_test, y_test = X_data[test_idx], y_data[test_idx]

# ============================
# 3) 
# ============================
def relu(x): return np.maximum(0, x)
def relu_grad(x): return (x > 0).astype(float)

def softmax(z):
    z -= np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-9, 1-1e-9)
    return -np.mean(np.log(y_pred[np.arange(len(y_true)), y_true]))

# ============================
# 4) 
# ============================
class DenseLayer:
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2. / in_dim)
        self.b = np.zeros((1, out_dim))

    def forward(self, X):
        return X @ self.W + self.b

# 
n_inputs = 28 * 28
hidden1, hidden2, outputs = 200, 100, 10

L1 = DenseLayer(n_inputs, hidden1)
L2 = DenseLayer(hidden1, hidden2)
L3 = DenseLayer(hidden2, outputs)

# ============================
# 5) 
# ============================
epochs = 15
lr = 0.001

for ep in range(1, epochs+1):
    total_loss, correct = 0, 0

    for x, lbl in zip(X_train, y_train):
        x = x.reshape(1, -1)

        # forward
        z1 = L1.forward(x); a1 = relu(z1)
        z2 = L2.forward(a1); a2 = relu(z2)
        z3 = L3.forward(a2); out = softmax(z3)

        # loss
        loss = cross_entropy([lbl], out)
        total_loss += loss
        correct += (np.argmax(out, axis=1)[0] == lbl)

        # backward
        y_onehot = np.eye(outputs)[lbl].reshape(1, -1)
        dz3 = out - y_onehot
        dW3, db3 = a2.T @ dz3, dz3

        da2 = dz3 @ L3.W.T
        dz2 = da2 * relu_grad(z2)
        dW2, db2 = a1.T @ dz2, dz2

        da1 = dz2 @ L2.W.T
        dz1 = da1 * relu_grad(z1)
        dW1, db1 = x.T @ dz1, dz1

        # update
        L3.W -= lr * dW3; L3.b -= lr * db3
        L2.W -= lr * dW2; L2.b -= lr * db2
        L1.W -= lr * dW1; L1.b -= lr * db1

    train_acc = correct / len(X_train)
    avg_loss = total_loss / len(X_train)
    print(f"Epoch {ep}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.2f}%")

# ============================
# 6) 
# ============================
z1 = L1.forward(X_test); a1 = relu(z1)
z2 = L2.forward(a1); a2 = relu(z2)
z3 = L3.forward(a2); preds = softmax(z3)

test_loss = cross_entropy(y_test, preds)
test_acc = np.mean(np.argmax(preds, axis=1) == y_test)
print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%")


# save model as madian_model.npz
np.savez("madian_model.npz", 
        Weights1=L1.W, Biases1=L1.b,
        Weights2=L2.W, Biases2=L2.b,
        Weights3=L3.W, Biases3=L3.b)
print("Model saved as madian_model.npz")
