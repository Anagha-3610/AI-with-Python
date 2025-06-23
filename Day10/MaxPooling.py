import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype("float32") / 255
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype("float32") / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # Convolution layer
    layers.MaxPooling2D((2, 2)),                                            # Pooling layer
    layers.Conv2D(64, (3, 3), activation='relu'),                          # Another Conv layer
    layers.MaxPooling2D((2, 2)),                                           # Pooling
    layers.Flatten(),
    layers.Dense(64, activation='relu'),                                   # Fully connected layer
    layers.Dense(10, activation='softmax')                                 # Output for 10 classes
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.2f}")
