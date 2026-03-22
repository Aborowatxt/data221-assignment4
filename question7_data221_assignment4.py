import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize pixel values to the range [0, 1]
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

# Reshape images
train_images = train_images[..., None]
test_images = test_images[..., None]

# Build the CNN model
cnn_model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(16, 3, padding="same", activation="relu"),
    layers.MaxPool2D(),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax")
])

# Compile the model
cnn_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the CNN model
cnn_model.fit(
    train_images,
    train_labels,
    validation_split=0.1,
    epochs=15,
    batch_size=64,
    verbose=1
)

# Fashion MNIST class names
fashion_class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# Generate predictions on the test set
predicted_probabilities = cnn_model.predict(test_images, verbose=0)
predicted_labels = np.argmax(predicted_probabilities, axis=1)

# Compute and display the confusion matrix
cnn_confusion_matrix = confusion_matrix(test_labels, predicted_labels)
confusion_matrix_display = ConfusionMatrixDisplay(
    confusion_matrix=cnn_confusion_matrix,
    display_labels=fashion_class_names
)

plt.figure(figsize=(10, 10))
confusion_matrix_display.plot(cmap="Blues", xticks_rotation=45)
plt.title("CNN Confusion Matrix - Fashion MNIST")
plt.show()

# Identify misclassified images
misclassified_indices = np.where(predicted_labels != test_labels)[0]

# Display three misclassified images
for image_position in range(3):
    misclassified_index = misclassified_indices[image_position]

    plt.figure(figsize=(4, 4))
    plt.imshow(test_images[misclassified_index].squeeze(), cmap="gray")
    plt.title(
        f"True Label: {fashion_class_names[test_labels[misclassified_index]]}\n"
        f"Predicted Label: {fashion_class_names[predicted_labels[misclassified_index]]}"
    )
    plt.axis("off")
    plt.show()

# One pattern I observe in the misclassifications is that the model sometimes
# confuses clothing items that look similar, like shirts, pullovers, coats,
# and T-shirts/tops. Since the images are grayscale and only 28x28 pixels,
# some categories can look really alike.

# One realistic way to improve the CNN would be to tune the model more,
# like adding another convolution layer, training a bit longer, or using
# data augmentation so the model can generalize better.