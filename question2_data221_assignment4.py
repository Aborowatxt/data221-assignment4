from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = load_breast_cancer()

#Constructing feature matrix X and target vector y
X = data.data
y = data.target

# 80/20 train–test split with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#Decision tree model
decision_tree_classifier = DecisionTreeClassifier(criterion='entropy')
decision_tree_classifier.fit(X_train, y_train)

#Model predictions
train_predictions = decision_tree_classifier.predict(X_train)
test_predictions = decision_tree_classifier.predict(X_test)

#Training and Testing accuracy
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

"""
Entropy represents how well a split separates the classes. Low entropy is an indicator of a good split/ a pure node.
These results are suggesting overfitting, because the model memorized the data to get a 100% accuracy score,
but the test, the score slightly drops, probably because it doesn't generalize to unseen data.
"""