#Full Program: MNIST Digit Classifier with TensorFlow & Keras
# Step 1: Import Required Libraries
# TensorFlow & Keras for model building, NumPy for data handling, Matplotlib for visualization
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Load the MNIST Dataset
# Keras provides MNIST as part of its datasets module
(x_train, y_train), (x_test, y_test) =mnist.load_data()

# Step 3: Preprocess the Data
# Normalize pixel values to the range [0, 1]
# Flatten 28x28 images into 784-dimensional vectors
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype("float32") / 255
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype("float32") / 255


# One-hot encode the target labels
y_train =to_categorical(y_train)
y_test = to_categorical(y_test)

# Step 4: Build the Neural Network Model
# Sequential model with one hidden layer and one output layer
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # Convolutional layer
    layers.MaxPooling2D((2, 2)),  # Max pooling layer
    layers.Conv2D(64, (3, 3), activation='relu'),  # Convolutional layer
    layers.MaxPooling2D((2, 2)),  # Max pooling layer
    layers.Flatten(),
    layers.Dense(64, activation='relu'),  # Hidden layer with ReLU
    layers.Dense(10, activation='softmax')  # Output layer with softmax for 10 classes
])

# Step 5: Compile the Model
# Define optimizer, loss function, and metrics to monitor
# Step 6: Train the Model
# Train the model with training data for 5 epochs
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)


# Step 7: Evaluate the Model
# Check model accuracy on test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.2f}")
#print("Test accuracy:", test_acc)

# Step 8 : Make predictions
predictions = model.predict(x_test)
print(predictions[0])

#Step 9 : Show prediction for one image
plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {np.argmax(predictions[0])}")
plt.show()