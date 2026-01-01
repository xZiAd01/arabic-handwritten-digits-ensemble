import numpy as np
from PIL import Image

# ------------------------
# Dense Layer
# ------------------------
class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))

    def get_z(self, X):
        return np.dot(X, self.weights) + self.biases

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def softmax(z):
        z = z - np.max(z, axis=1, keepdims=True)  # stability
        exp_vals = np.exp(z)
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)


# ------------------------
# Multi-Layer Perceptron
# ------------------------
class MLP:
    def __init__(self, layer_sizes):  # e.g. [784, 256, 128, 10]
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1]))

    def load_from_npz(self, path):
        data = np.load(path)
        for i, layer in enumerate(self.layers):
            layer.weights = data[f"Weights{i+1}"]
            layer.biases = data[f"Biases{i+1}"]

    def predict_proba(self, X):
        out = X
        for i, layer in enumerate(self.layers):
            z = layer.get_z(out)
            if i < len(self.layers) - 1:  # hidden
                out = Layer.relu(z)
            else:  # last layer → softmax
                out = Layer.softmax(z)
        return out

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
    
# ------------------------
# Utility: preprocess image
# ------------------------
def preprocess(img_path):
    img = Image.open(img_path).convert('L').resize((28,28))
    arr = np.array(img, dtype=np.float32).reshape(1, -1) / 255.0
    return arr

# ------------------------
# Ensemble (Voting System)
# ------------------------
def hard_voting(X, models):
    preds = np.stack([m.predict(X) for m in models], axis=0) 
    M, B = preds.shape # (M,B = number of models, batch size)
    out = np.empty(B, dtype=int)
    votes = np.empty(B, dtype=int)  # store how many models voted for winner

    for i in range(B):
        vals, counts = np.unique(preds[:, i], return_counts=True)
        winner_idx = np.argmax(counts)
        out[i] = vals[winner_idx]
        votes[i] = counts[winner_idx]   # how many voted for winner

    return out, votes, M   # predictions, votes per sample, total models
    

# --------------------
# Load models
# --------------------

zeyad_model = MLP([28*28, 256, 128, 10])
zeyad_model.load_from_npz("zeyad_model.npz")
print("Zeyad model loaded.")

iyas_model = MLP([28*28, 384, 10])
iyas_model.load_from_npz("eyas_model.npz")
print("Iyas model loaded.")

omar_model = MLP([28*28, 256, 128, 10])
omar_model.load_from_npz("omar_model.npz")
print("Omar model loaded.")

asiri_model = MLP([28*28, 256, 128, 10])
asiri_model.load_from_npz("asiri_model.npz")
print("Asiri model loaded.")

madian_model = MLP([28*28, 200, 100, 10])
madian_model.load_from_npz("madian_model.npz")
print("Madian model loaded.")

# --------------------
# Load and preprocess data
# --------------------
img_path = input("Enter the path of the image: ")
X = preprocess(img_path)

# --------------------
# Individual model predictions
# --------------------
all_probs = []  # store probabilities from each model

for model, name in [
    (zeyad_model, "Zeyad"),
    (iyas_model, "Iyas"),
    (omar_model, "Omar"),
    (asiri_model, "Asiri"),
    (madian_model, "Madian")
]:
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    all_probs.append(proba)

    print(f"\n{name} model prediction: {pred}")
    print("Class probabilities:")
    for cls, prob in enumerate(proba):
        print(f"Class {cls}: {prob*100:.2f}%")

# ---- average probabilities ----
avg_proba = np.mean(all_probs, axis=0)
avg_pred = np.argmax(avg_proba)

print("\n===== Ensemble Average Probabilities (Soft-Vote) =====")
print(f"Ensemble prediction: {avg_pred}")
print("Average class probabilities:")
for cls, prob in enumerate(avg_proba):
    print(f"Class {cls}: {prob*100:.2f}%")

# ---- hard voting ----
models = [zeyad_model, omar_model, asiri_model, madian_model, iyas_model]

pred_hard, votes, total = hard_voting(X, models)

print("\n===== Hard-vote prediction =====")
print(f"Ensemble hard-vote prediction: {pred_hard[0]} with {votes[0]}/{total} votes")

if input("\nenter 'y' if you want to calculate the accuracy of 'merged-model': ") == 'y':
    print("\n==== Getting the Accuracy on SmallDataset ====")
    import os
    folder = "SmallDataset"
    X, y = [], []
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        label = int(subfolder.split('/')[-1])   # safer if folders are 0–9
        for filename in os.listdir(subfolder_path):
            img = Image.open(os.path.join(subfolder_path, filename)).convert('L').resize((28,28))
            arr = np.array(img, dtype=np.float32).reshape(-1) / 255.0
            X.append(arr)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    # Predictions in batch
    probs = [m.predict_proba(X) for m in models]     # list of (B,C)
    avg_proba = np.mean(probs, axis=0)               # (B,C)
    soft_preds = np.argmax(avg_proba, axis=1)

    pred_hard, votes, total = hard_voting(X, models)

    # Accuracy
    soft_acc = np.mean(soft_preds == y)
    hard_acc = np.mean(pred_hard == y)
    print(f"Soft-vote accuracy: {soft_acc*100:.2f}%")
    print(f"Hard-vote accuracy: {hard_acc*100:.2f}%")
