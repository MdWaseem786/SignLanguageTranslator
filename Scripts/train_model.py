import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout  # Added Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # Added Callbacks

# Paths
train_dir = r"C:\signtranslator\Data\Train"
validation_dir = r"C:\signtranslator\Data\Validation"
model_save_path = r"C:\signtranslator\Model\trained_model.h5"

# Parameters
img_size = (128, 128)
batch_size = 32  # Increased batch size (experiment with different values)
epochs = 20  # Increased epochs (but using EarlyStopping)
num_classes = len(os.listdir(train_dir))

# Data augmentation (more aggressive)
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,  # Increased rotation
    width_shift_range=0.3,  # Increased shift
    height_shift_range=0.3,  # Increased shift
    shear_range=0.3,  # Increased shear
    zoom_range=0.3,  # Increased zoom
    horizontal_flip=True,
    fill_mode='nearest'  # Added fill_mode
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

# Load pre-trained MobileNetV2
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(img_size[0], img_size[1], 3))

# Freeze the base model (initially)
base_model.trainable = False

# Add custom layers on top (with Dropout)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)  # Increased Dense layer size
x = Dropout(0.5)(x)  # Added Dropout for regularization
x = Dense(256, activation="relu")(x)  # Added another Dense layer
x = Dropout(0.3)(x)  # Added Dropout
output_layer = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output_layer)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"]) # Reduced learning rate

# Callbacks for better training control
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  # Early Stopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001) # Learning rate reduction

# Train the model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr]  # Added callbacks
)

# Unfreeze some layers of the base model for fine-tuning (after initial training)
base_model.trainable = True
for layer in base_model.layers[:100]:  # Freeze the earlier layers (experiment with this number)
    layer.trainable = False

# Recompile the model with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.00001), loss="categorical_crossentropy", metrics=["accuracy"]) # Lower learning rate for fine-tuning

# Continue training (fine-tuning)
fine_tuning_epochs = 10 # Fine tuning epochs

history_fine = model.fit(
    train_generator,
    epochs=fine_tuning_epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr]
)


# Save the trained model
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)
print("Model saved!")

# Plot training results (combined initial and fine-tuning)
def plot_training(history, history_fine):
    # ... (rest of the plotting code - combine the history objects)
    acc = history.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss = history.history['loss'] + history_fine.history['loss']
    val_loss = history.history['val_loss'] + history_fine.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.show()

plot_training(history, history_fine)