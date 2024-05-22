# Network Architecture:
# Input layer: 2 neurons (for two input features)
# Hidden layer: 2 neurons
# Output layer: 1 neuron (outputting the result of the AND operation)
import tensorflow as tf
import numpy as np

# Data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [0], [0], [1]])

# The architecture of the neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=2, input_shape=(2,), activation='sigmoid'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='binary_crossentropy', metrics=['accuracy'])

# Training
model.fit(X, Y, epochs=1000, verbose=0)

# Testing
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = model.predict(test_data)

# Output the predictions
print("Input  Predicted Output")
for i in range(len(test_data)):
    print(test_data[i], "   ", round(predictions[i][0]))

