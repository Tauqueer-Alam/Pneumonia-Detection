import tensorflow as tf
from tensorflow.keras import layers, models
import os
import matplotlib.pyplot as plt

base_dir = r"d:\CODING\Project - Pneumonia Detection\chest_xray"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

print("Loading dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(180, 180),
    batch_size=32
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(180, 180),
    batch_size=32
)

num_classes = 2

model = models.Sequential([
    tf.keras.Input(shape=(180, 180, 3)),
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

model.summary()

model.compile(optimizer="adam", 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
              metrics=['accuracy'])

print("Starting training...")
result = model.fit(train_ds, validation_data=test_ds, epochs=10)

print("Evaluating final model...")
loss, accuracy = model.evaluate(test_ds)
print(f"Test accuracy: {accuracy*100:.2f}%")

model.save('pneumonia_model_custom.keras')
print("Training completed. Model saved as 'pneumonia_model_custom.keras'.")
