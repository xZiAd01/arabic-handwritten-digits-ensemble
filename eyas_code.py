import numpy as np
from PIL import Image
import os
import time

folder = 'SmallDataset'

X, y = [], []

for subfolder in sorted(os.listdir(folder)):
    if subfolder.startswith('.'):
        continue
        
    subfolder_path = os.path.join(folder, subfolder)
    if not os.path.isdir(subfolder_path):
        continue
        
    label = int(subfolder)
    images = [f for f in os.listdir(subfolder_path) if not f.startswith('.')]
    
    for filename in images:
        try:
            img = Image.open(os.path.join(subfolder_path, filename))
            flattened = np.array(img).flatten()
            X.append(flattened)
            y.append(label)
        except:
            continue

X = np.array(X, dtype=np.float32) / 255.0  # Used float32 for speed
y = np.array(y, dtype=np.int32)

# Stratified split
train_indices, test_indices = [], []
classes = np.unique(y)

for cls in classes:
    cls_indices = np.where(y == cls)[0]
    np.random.shuffle(cls_indices)
    split_point = int(0.9 * len(cls_indices))
    train_indices.extend(cls_indices[:split_point])
    test_indices.extend(cls_indices[split_point:])

train_indices = np.array(train_indices)
test_indices = np.array(test_indices)


X_train, y_train = X[train_indices], y[train_indices]
X_test, y_test = X[test_indices], y[test_indices]

# activation functions
def relu(z):
    return np.maximum(0, z) 

def softmax(z):
    # Numerically stable softmax
    z_max = np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    # Clip to prevent log(0)
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(np.log(y_pred_clipped[np.arange(len(y_true)), y_true]))

# Network Architecture
n_inputs = 784
n_hidden = 384  
n_outputs = 10


class FastLayer:
    def __init__(self, input_size, output_size):
        # He initialization
        self.W = np.random.randn(input_size, output_size).astype(np.float32) * np.sqrt(2.0 / input_size)
        self.b = np.zeros((1, output_size), dtype=np.float32)
        
    def forward(self, X):
        return X @ self.W + self.b

# Initialize layers
layer1 = FastLayer(n_inputs, n_hidden)
layer2 = FastLayer(n_hidden, n_outputs)

# Optimized training parameters
batch_size = 32  
num_epochs = 8   
learning_rate = 0.005  




for epoch in range(num_epochs):
    epoch_loss = 0.0
    correct = 0
    epoch_start = time.time()
    
    # Shuffle training data
    indices = np.random.permutation(len(X_train))
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]
    
    # Batch processing for SPEED
    for i in range(0, len(X_train), batch_size):
        end_idx = min(i + batch_size, len(X_train))
        X_batch = X_shuffled[i:end_idx]
        y_batch = y_shuffled[i:end_idx]
        
        # Forward pass (vectorized)
        z1 = layer1.forward(X_batch)
        a1 = relu(z1.copy())  # ReLU activation
        z2 = layer2.forward(a1)
        y_pred = softmax(z2)
        
        # Loss and accuracy
        batch_loss = cross_entropy_loss(y_batch, y_pred)
        epoch_loss += batch_loss * len(X_batch)
        correct += np.sum(np.argmax(y_pred, axis=1) == y_batch)
        
        # Backward pass
        m = len(X_batch)
        
        # Output layer gradients
        dZ2 = y_pred.copy()
        dZ2[np.arange(m), y_batch] -= 1  # Efficient one-hot subtraction
        dZ2 /= m
        
        dW2 = a1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        dA1 = dZ2 @ layer2.W.T
        dZ1 = dA1 * (z1 > 0)  # ReLU derivative
        
        dW1 = X_batch.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        # Update weights
        layer2.W -= learning_rate * dW2
        layer2.b -= learning_rate * db2
        layer1.W -= learning_rate * dW1
        layer1.b -= learning_rate * db1
    
    # Epoch statistics
    epoch_time = time.time() - epoch_start
    epoch_loss /= len(X_train)
    accuracy = correct / len(X_train)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy*100:.2f}%")

# Evaluate on test data
z1_test = layer1.forward(X_test)
a1_test = relu(z1_test.copy())
z2_test = layer2.forward(a1_test)
y_test_pred = softmax(z2_test)

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
        img = Image.open(img_path).convert('L').resize((28, 28))
        img_array = np.array(img).flatten().astype(np.float32) / 255.0
        img_array = img_array.reshape(1, -1)
        
        # Forward pass
        z1_img = layer1.forward(img_array)
        a1_img = relu(z1_img.copy())
        z2_img = layer2.forward(a1_img)
        y_img_pred = softmax(z2_img)
        
        # Predicted class
        predicted_label = np.argmax(y_img_pred, axis=1)[0]
        print(f"\nPredicted label: {predicted_label}")
        
        # Print probabilities
        print("Class probabilities:")
        for cls, prob in enumerate(y_img_pred[0]):
            print(f"Class {cls}: {prob*100:.2f}%")
            
    except Exception as e:
        print(f"Error processing {img_path}: {e}")



# Save model as NPZ
if input("Do you want to save the model as NPZ? (y/n): ").lower() == 'y':
    np.savez("model.npz",
             Weights1=layer1.W, Biases1=layer1.b,
             Weights2=layer2.W, Biases2=layer2.b)
    print("Model saved to model.npz")
