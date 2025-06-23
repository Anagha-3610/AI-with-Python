#Full Program: MNIST Digit Classifier with TensorFlow & Keras
# Step 1: Import Required Libraries
# TensorFlow & Keras for model building, NumPy for data handling, Matplotlib for visualization
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Load the MNIST Dataset
# Keras provides MNIST as part of its datasets module
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Step 3: Preprocess the Data
# Normalize pixel values to the range [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten 28x28 images into 784-dimensional vectors
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

# Step 4: Build the Neural Network Model
# Sequential model with one hidden layer and one output layer
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),  # Hidden layer with ReLU
    layers.Dense(10, activation='softmax')  # Output layer with softmax for 10 classes
])

# Step 5: Compile the Model
# Define optimizer, loss function, and metrics to monitor
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 6: Train the Model
# Train the model with training data for 5 epochs
model.fit(x_train, y_train, epochs=5)

# Step 7: Evaluate the Model
# Check model accuracy on test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

# Step 8 : Make predictions
predictions = model.predict(x_test)
print(predictions[9])

#Step 9 : Show prediction for one image
plt.imshow(x_test[9].reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {np.argmax(predictions[9])}")
plt.show()