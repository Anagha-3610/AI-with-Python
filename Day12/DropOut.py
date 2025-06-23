#1. Dropout
#What it does: Randomly disables a fraction of neurons during training to prevent overfitting.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dropout(0.5),  # Drop 50% of neurons during training
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])
print(model.summary())