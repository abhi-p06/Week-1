# 🧩 1️⃣ Import Libraries

import tensorflow as tf
from tensorflow.keras import datasets, layers, models # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt

# Check TensorFlow version
print("✅ TensorFlow version:", tf.__version__)

# loading the Dataset

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Show dataset shapes
print("Training images:", train_images.shape)
print("Test images:", test_images.shape)

# Display a few images
plt.figure(figsize=(8, 4))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(train_images[i], cmap='gray')
    plt.title(f"Label: {train_labels[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# ==========================================================
# 🧩 3️⃣ Preprocess Data
# ==========================================================
# CNNs expect images with 3D shape (height, width, channels)
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Normalize pixel values (0–255 → 0–1)
train_images, test_images = train_images / 255.0, test_images / 255.0

# ==========================================================
# 🧩 4️⃣ Build the CNN Model
# ==========================================================
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Model summary
model.summary()

# ==========================================================
# 🧩 5️⃣ Compile the Model
# ==========================================================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================================================
# 🧩 6️⃣ Train the Model
# ==========================================================
history = model.fit(train_images, train_labels, epochs=5, 
                    validation_data=(test_images, test_labels))

# ==========================================================
# 🧩 7️⃣ Evaluate the Model
# ==========================================================
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\n✅ Test Accuracy:", test_acc)

# ==========================================================
# 🧩 8️⃣ Visualize Training Progress
# ==========================================================
plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('📊 Model Accuracy over Epochs')
plt.show()
