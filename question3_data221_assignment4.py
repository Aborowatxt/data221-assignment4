from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

data = load_breast_cancer()
# Constructing feature matrix X and target vector y
X = data.data
y = data.target

# 80/20 train–test split with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Constrained decision tree model (added max_depth)
decision_tree_classifier_constrained = DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=42)
decision_tree_classifier_constrained.fit(X_train, y_train)

# Model predictions
train_predictions_constrained = decision_tree_classifier_constrained.predict(X_train)
test_predictions_constrained = decision_tree_classifier_constrained.predict(X_test)

# Training and Testing accuracy
train_accuracy_constrained = accuracy_score(y_train, train_predictions_constrained)
test_accuracy_constrained = accuracy_score(y_test, test_predictions_constrained)

print("Constrained Training Accuracy:", train_accuracy_constrained)
print("Constrained Test Accuracy:", test_accuracy_constrained)

# Feature importance
feature_importance = decision_tree_classifier_constrained.feature_importances_

# Sort features from most important to the least important
sorted_indices = np.argsort(feature_importance)[::-1]
feature_names = data.feature_names
print("Top 5 most important features:")
for i in range(5):
    print(f"{feature_names[sorted_indices[i]]}: {feature_importance[sorted_indices[i]]}")

# Discussion:
# By adding a constraint like max_depth, the decision tree is forced to stay simpler instead of growing too deep.
# This helps prevent the model from just memorizing the training data, which reduces overfitting.
# Because of this, the training accuracy might drop a bit, but the test accuracy usually stays similar or improves,
# which means the model is generalizing better to new data.

# Feature importance shows how much each feature helps the model make decisions by reducing entropy.
# Features with higher importance values have a bigger impact on the predictions.
# This makes decision trees easier to understand, since we can clearly see which features are most important
# in determining whether a tumor is malignant or benign.