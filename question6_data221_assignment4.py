import tensorflow as tf
from tensorflow.keras import layers, models

# Load the Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize pixel values to the range [0, 1]
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

# Reshape images to include the channel dimension
train_images = train_images[..., None]
test_images = test_images[..., None]

print("Training image shape:", train_images.shape)
print("Testing image shape:", test_images.shape)

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

# Show model summary
cnn_model.summary()

# Compile the model
cnn_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model for at least 15 epochs
training_history = cnn_model.fit(
    train_images,
    train_labels,
    validation_split=0.1,
    epochs=15,
    batch_size=64,
    verbose=1
)

# Evaluate the model on the test set
test_loss, test_accuracy = cnn_model.evaluate(test_images, test_labels, verbose=0)

print("CNN Test Loss:", test_loss)
print("CNN Test Accuracy:", test_accuracy)

"""
CNNs are usually better than fully connected networks for image data because they keep the spatial structure 
of the image. They can learn local patterns like edges, textures, and shapes, instead of flattening 
everything into one long vector and losing that information.

In this task, the convolution layer is learning useful visual patterns from the clothing images, like edges,
curves, textures, and shapes. Early layers learn simple features, while deeper layers combine them into more
meaningful patterns for classification.
"""