from sklearn.datasets import load_breast_cancer
import numpy as np

data = load_breast_cancer()

#Constructing feature matrix X and target vector y
X = data.data
y = data.target

#Shape of X and y
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

print("Target names:", data.target_names)

# Report the number of samples belonging to each class
#use np.unique() to find how many times each class appears
unique_classes, class_counts = np.unique(y, return_counts=True)

#zip() pairs element together
for label, count in zip(unique_classes, class_counts):
    print(f"Class {label}: {count} samples")

"""
The dataset contains 569 samples with 30 features each, and a target vector of size 569.
There are 212 samples labeled as malignant and 357 samples labeled as benign.

The dataset is slightly imbalanced because the benign class has more samples than the malignant class
Class imbalance is important in classification because a model may become biased toward the majority class,
leading to misleading accuracy.
"""