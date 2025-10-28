import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# ----------------------------
# 1. Dataset paths (relative to this script)
# ----------------------------
base_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
train_dir = os.path.join(base_dir, "train")
val_dir   = os.path.join(base_dir, "validation")  # validation folder

# ----------------------------
# 2. Verify dataset folders
# ----------------------------
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"❌ Train directory not found: {train_dir}")
if not os.path.exists(val_dir):
    raise FileNotFoundError(f"❌ Validation directory not found: {val_dir}")
print(f"✅ Dataset found! Train: {train_dir}, Validation: {val_dir}")

# ----------------------------
# 3. Image data generators
# ----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# ----------------------------
# 4. Build CNN model
# ----------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_gen.num_classes, activation='softmax')
])

# ----------------------------
# 5. Compile model
# ----------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ----------------------------
# 6. Train model
# ----------------------------
history = model.fit(
    train_gen,
    epochs=15,
    validation_data=val_gen
)

# ----------------------------
# 7. Save model
# ----------------------------
model_dir = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "rice_leaf_model.h5")
model.save(model_path)
print(f"✅ Model saved at {model_path}")

# ----------------------------
# 8. Plot training accuracy & loss
# ----------------------------
plt.figure(figsize=(10, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
chart_path = os.path.join(os.path.dirname(__file__), "training_performance.png")
plt.savefig(chart_path)
plt.show()

print(f"✅ Training completed and performance chart saved as '{chart_path}'")
