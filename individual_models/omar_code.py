import numpy as np
from PIL import Image
import os
from model import Model

folder = 'SmallDataset'

x,y = [],[]

for subfolder in os.listdir(folder):
    subfolder_path = os.path.join(folder, subfolder)
    print(subfolder_path)

    label = subfolder.split('/')[-1]

    for filename in os.listdir(subfolder_path):
        img_path = os.path.join(subfolder_path, filename)
        img = Image.open(img_path).convert("L")  # grayscale
        img = img.resize((28, 28))  # resize all images to same shape
        flattend = np.array(img).flatten().tolist()
        x.append(flattend)
        y.append(int(label))



print(f"Total images: {len(x)}")

# print example
print("\n--------------------------------\n")

X = np.array(x) / 255.0   # normalize pixel values (0–1 instead of 0–255)
y = np.array(y)           # labels

input_size = X.shape[1]   # number of pixels (28*28 = 784)
hidden1_size = 256    # Example hidden layer size
hidden2_size = 128     # Example hidden layer size
output_size = 10      # Example output size (e.g., 10 classes for classification)

indices = np.arange(len(X))
np.random.shuffle(indices)
split_idx = int(0.8 * len(X))
train_indices = indices[:split_idx]
test_indices = indices[split_idx:]
X_train, y_train = X[train_indices], y[train_indices] 
X_test, y_test = X[test_indices], y[test_indices]

# Create layers
model = Model(input_size, hidden1_size, hidden2_size, output_size, learning_rate=0.01)



model.train(X_train, y_train, epochs=100)


print(f"\nFinal Training Accuracy: {model.acc:.2f}%")

pred = model.forward(X_test)
test_loss = model.calcLoss(pred, y_test)
test_acc = model.calcAccuracy(pred, y_test)

print(f"\nFinal testing Accuracy: {test_acc:.2f}%")

np.savez("Omar_model.npz",
Weights1 = model.layer1.weights, Biases1 = model.layer1.biases,
Weights2 = model.layer2.weights, Biases2 = model.layer2.biases,
Weights3 = model.layer3.weights, Biases3 = model.layer3.biases 
)

print("model saved to Omar_model.npz")

#print(f"Layer1 weight: {model.layer1.weights}, Layer1 weight: {model.layer1.weights}," )
