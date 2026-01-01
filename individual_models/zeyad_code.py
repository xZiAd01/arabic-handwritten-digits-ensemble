import numpy as np
from PIL import Image
import os


folder = 'SmallDataset'

X, y = [], []

for subfolder in os.listdir(folder):
    subfolder_path = os.path.join(folder, subfolder)
    #print(subfolder_path)

    label = subfolder.split('/')[-1]

    first_images = os.listdir(subfolder_path)
    for filename in first_images:
        img = Image.open(os.path.join(subfolder_path, filename))
        flattend = np.array(img).flatten().tolist()
        X.append(flattend)
        y.append(label)

X = np.array(X) / 255.0 # Normalize to [0, 1]
y = np.array(y).astype(np.int32)




# Stratified split into training and testing sets
train_indices, test_indices = [], []
classes = np.unique(y)

for cls in classes:
    # indices for this class
    cls_indices = np.where(y == cls)[0]
    np.random.shuffle(cls_indices)

    # split 90% train / 10% test within this class
    split_point = int(0.9 * len(cls_indices))
    train_indices.extend(cls_indices[:split_point])
    test_indices.extend(cls_indices[split_point:])

# Convert to arrays
train_indices = np.array(train_indices)
test_indices = np.array(test_indices)

# Shuffle overall (so not grouped by class anymore)
np.random.shuffle(train_indices)
np.random.shuffle(test_indices)

# Final datasets
X_train, y_train = X[train_indices], y[train_indices]
X_test, y_test = X[test_indices], y[test_indices]


# Activation functions
def relu(z):
        return np.maximum(0, z)
    
def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_vals = np.exp(z)
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

# Loss function
def cross_entropy_loss(y_true, y_pred):
    probs = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(np.log(probs[np.arange(len(y_true)), y_true]))



# Network architecture
n_inputs = 28*28
n_hidden1 = 256
n_hidden2 = 128
n_outputs = 10

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((1, output_size))

    def get_z(self, X):
        return np.dot(X, self.weights) + self.biases
    

layer1 = Layer(n_inputs, n_hidden1)
layer2 = Layer(n_hidden1, n_hidden2)
layer3 = Layer(n_hidden2, n_outputs)



# Training parameters
num_epochs = 12
learning_rate = 1e-3

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0
    correct = 0

    for img_array, label in zip(X_train, y_train):
        
        img_array = img_array.reshape(1, -1)  # Reshape to (1, n_inputs)
        
        # Forward pass
        z1 = layer1.get_z(img_array)
        a1 = relu(z1)
        z2 = layer2.get_z(a1)
        a2 = relu(z2) 
        z3 = layer3.get_z(a2)
        y_pred = softmax(z3)

        # Backward pass
        dZ3 = y_pred - np.eye(n_outputs)[label].reshape(1, -1)
        dW3 = a2.T @ dZ3 
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        # Backprop into Layer 2
        dA2 = dZ3 @ layer3.weights.T          # gradient wrt activations of layer2
        dZ2 = dA2 * (z2 > 0)                   # apply ReLU derivative
        dW2 = a1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Backprop into Layer 1
        dA1 = dZ2 @ layer2.weights.T
        dZ1 = dA1 * (z1 > 0)                   # ReLU derivative
        dW1 = img_array.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Update weights
        layer3.weights -= learning_rate * dW3
        layer3.biases  -= learning_rate * db3
        layer2.weights -= learning_rate * dW2
        layer2.biases  -= learning_rate * db2
        layer1.weights -= learning_rate * dW1
        layer1.biases  -= learning_rate * db1

        # Compute loss and accuracy
        loss = cross_entropy_loss(np.array([label]), y_pred)
        epoch_loss += loss
        correct += (np.argmax(y_pred, axis=1) == label).sum()

    epoch_loss /= len(X_train)
    accuracy = correct / len(X_train)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy*100:.2f}%")



# Evaluate on test data
z1_test = layer1.get_z(X_test)
a1_test = relu(z1_test)
z2_test = layer2.get_z(a1_test)
a2_test = relu(z2_test)
z3_test = layer3.get_z(a2_test)
y_test_pred = softmax(z3_test)

test_loss = cross_entropy_loss(y_test, y_test_pred)
test_accuracy = np.mean(np.argmax(y_test_pred, axis=1) == y_test)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy*100:.2f}%")


# get a prediction for a single image
while True:
    img_path = input("Enter the path of the image you want to predict, Enter '0' to quit: ")
    if img_path.strip() == '0':
        print("Exiting prediction loop.")
        break

    try:
        # Load and preprocess image
        img = Image.open(img_path).convert('L').resize((28, 28))  # grayscale + resize
        img_array = np.array(img).flatten().astype(np.float32) / 255.0
        img_array = img_array.reshape(1, -1)  # (1, n_inputs)

        # Forward pass
        z1_img = layer1.get_z(img_array)
        a1_img = relu(z1_img)
        z2_img = layer2.get_z(a1_img)
        a2_img = relu(z2_img)
        z3_img = layer3.get_z(a2_img)
        y_img_pred = softmax(z3_img)  # softmax on final logits

        # Predicted class
        predicted_label = np.argmax(y_img_pred, axis=1)[0]
        print(f"\nPredicted label: {predicted_label}")

        # Print probabilities
        print("Class probabilities:")
        for cls, prob in enumerate(y_img_pred[0]):
            print(f"Class {cls}: {prob*100:.2f}%")

    except Exception as e:
        print(f"Error processing {img_path}: {e}")


import json
# best weights and biases
if input("Do you want to save the best weights and biases? (y/n): ").lower() == 'y':
    data = {
        "test_accuracy": f"{test_accuracy*100:.2f}%",
        "Weights1": layer1.weights.tolist(),
        "Biases1": layer1.biases.tolist(),
        "Weights2": layer2.weights.tolist(),
        "Biases2": layer2.biases.tolist(),
        "Weights3": layer3.weights.tolist(),
        "Biases3": layer3.biases.tolist()
    }

    with open("best_weights_biases.json", "w") as f:
        json.dump(data, f, indent=4)

    print("Weights and biases saved to best_weights_biases.json")


# save model in .npz format
if input("Do you want to save the model in .npz format? (y/n): ").lower() == 'y':
    np.savez("model.npz",
             Weights1=layer1.weights, Biases1=layer1.biases,
             Weights2=layer2.weights, Biases2=layer2.biases,
             Weights3=layer3.weights, Biases3=layer3.biases)
    print("Model saved to model.npz")
