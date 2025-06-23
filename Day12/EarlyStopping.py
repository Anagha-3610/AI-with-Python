from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical

early_stop = EarlyStopping(
    monitor='val_loss',    # Could be 'val_accuracy'
    patience=3,            # Stop after 3 epochs of no improvement
    restore_best_weights=True
)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data: flatten and normalize
x_train = x_train.reshape(-1, 784).astype('float32') / 255
x_test = x_test.reshape(-1, 784).astype('float32') / 255


# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

model = Sequential([
    Input(shape=(784,)),
    Dense(128, activation='relu'),    # hidden layer
    Dense(10, activation='softmax')
])   

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_split=0.2, callbacks=[early_stop])

print("Test loss and test accuracy:=>",model.evaluate(x_test, y_test))
