import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.metrics import accuracy_score

data = load_breast_cancer()

# Constructing feature matrix X and target vector y
X = data.data
y = data.target

# 80/20 train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Standardize the input features
standard_scaler = StandardScaler()
X_train = standard_scaler.fit_transform(X_train)
X_test = standard_scaler.transform(X_test)

# Set random seed
tf.random.set_seed(42)

# Build the neural network model
neural_network_model = Sequential()

# Input layer (30 features)
neural_network_model.add(InputLayer(input_shape=(30,)))

# Hidden layer
neural_network_model.add(Dense(8, activation='relu'))

# Output layer (binary classification)
neural_network_model.add(Dense(1, activation='sigmoid'))

# Compile the model
neural_network_model.compile(loss='binary_crossentropy')

# Train the model
neural_network_model.fit(X_train, y_train, epochs=10)

# Make predictions on training data
train_predictions = neural_network_model.predict(X_train)
train_predicted_classes = (train_predictions > 0.5).astype(int).ravel()

# Make predictions on testing data
test_predictions = neural_network_model.predict(X_test)
test_predicted_classes = (test_predictions > 0.5).astype(int).ravel()

# Calculate accuracy
train_accuracy_nn = accuracy_score(y_train, train_predicted_classes)
test_accuracy_nn = accuracy_score(y_test, test_predicted_classes)

print("Neural Network Training Accuracy:", train_accuracy_nn)
print("Neural Network Test Accuracy:", test_accuracy_nn)

"""
Feature scaling is important for neural networks because it keeps all the input features
on a similar range. If one feature has really big values and another is small,
the model might focus more on the bigger one, which messes up learning.
Scaling helps the model learn faster and more consistently.

An epoch is one full pass through the entire training dataset.
So if you train for 10 epochs, the model goes through all the data 10 times.
Each time, it adjusts and (hopefully) gets better at making predictions.
"""