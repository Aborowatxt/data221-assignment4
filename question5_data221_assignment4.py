import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer


# Load the dataset
data = load_breast_cancer()

# Constructing feature matrix X and target vector y
X = data.data
y = data.target

# 80/20 train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, stratify=y, random_state=42)


# Constrained Decision Tree
decision_tree_classifier_constrained = DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=42)
decision_tree_classifier_constrained.fit(X_train, y_train)

decision_tree_train_predictions_constrained = decision_tree_classifier_constrained.predict(X_train)
decision_tree_test_predictions_constrained = decision_tree_classifier_constrained.predict(X_test)

decision_tree_training_accuracy_constrained = accuracy_score(y_train, decision_tree_train_predictions_constrained)
decision_tree_test_accuracy_constrained = accuracy_score(y_test, decision_tree_test_predictions_constrained)

print("Constrained Decision Tree Training Accuracy:", decision_tree_training_accuracy_constrained)
print("Constrained Decision Tree Test Accuracy:", decision_tree_test_accuracy_constrained)

# Confusion matrix for constrained decision tree
decision_tree_confusion_matrix = confusion_matrix(y_test,decision_tree_test_predictions_constrained)

print("\nConstrained Decision Tree Confusion Matrix:")
print(decision_tree_confusion_matrix)

decision_tree_display = ConfusionMatrixDisplay(confusion_matrix=decision_tree_confusion_matrix,display_labels=data.target_names)
decision_tree_display.plot()
plt.title("Constrained Decision Tree Confusion Matrix")
plt.show()

# Neural Network

# Standardize the input features for the neural network
standard_scaler = StandardScaler()
X_train_scaled = standard_scaler.fit_transform(X_train)
X_test_scaled = standard_scaler.transform(X_test)

# Set random seed
tf.random.set_seed(42)

# Build the neural network model
neural_network_model = Sequential()

# Input layer (30 features)
neural_network_model.add(InputLayer(input_shape=(30,)))

# Hidden layer
neural_network_model.add(Dense(8, activation='relu'))

# Output layer for binary classification
neural_network_model.add(Dense(1, activation='sigmoid'))

# Compile the model
neural_network_model.compile(loss='binary_crossentropy')

# Train the model
neural_network_model.fit(X_train_scaled, y_train, epochs=10, verbose=0)

# Make predictions on training data
neural_network_train_predictions = neural_network_model.predict(X_train_scaled)
neural_network_train_predicted_classes = (
    neural_network_train_predictions > 0.5
).astype(int).ravel()

# Make predictions on testing data
neural_network_test_predictions = neural_network_model.predict(X_test_scaled)
neural_network_test_predicted_classes = (
    neural_network_test_predictions > 0.5
).astype(int).ravel()

# Accuracy for neural network
neural_network_training_accuracy = accuracy_score(
    y_train,
    neural_network_train_predicted_classes
)
neural_network_test_accuracy = accuracy_score(
    y_test,
    neural_network_test_predicted_classes
)

print("\nNeural Network Training Accuracy:", neural_network_training_accuracy)
print("Neural Network Test Accuracy:", neural_network_test_accuracy)

# Confusion matrix for neural network
neural_network_confusion_matrix = confusion_matrix(
    y_test,
    neural_network_test_predicted_classes
)

print("\nNeural Network Confusion Matrix:")
print(neural_network_confusion_matrix)

neural_network_display = ConfusionMatrixDisplay(
    confusion_matrix=neural_network_confusion_matrix,
    display_labels=data.target_names
)
neural_network_display.plot()
plt.title("Neural Network Confusion Matrix")
plt.show()


"""
Discussion:
I’d go with the constrained decision tree for this. Both models do solid, but the decision tree has better
test accuracy (around 94.7% vs 91.2% for the neural network). It also makes fewer bad mistakes.

From the confusion matrix, it only has 4 false negatives, meaning only 4 malignant tumors were predicted as
benign. That’s really important because that’s the worst type of mistake in this case.

The decision tree is also nice because it’s easy to understand. You can actually see how it’s making decisions
and what it’s using. The downside is it might miss more complex patterns.

The neural network can pick up more complex stuff and has slightly better training accuracy, but it doesn’t 
do as well on the test set. Plus, it’s way harder to understand, so you don’t really know why it’s making 
certain predictions.
"""
