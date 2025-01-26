import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load VGG16 pre-trained model
base_model_vgg16 = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# Fine-tune the top layers of VGG16
for layer in base_model_vgg16.layers[:-4]:  # Freeze all layers except the last 4
    layer.trainable = False

# Build the custom model
model = models.Sequential([
    base_model_vgg16,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),  # Increased capacity
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')  # Assuming 4 emotion classes
])

# Compile the model with a lower learning rate
model.compile(optimizer=optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Data augmentation for training and validation sets
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=10,  # Reduced for overfitting
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Directories for dataset
train_generator = train_datagen.flow_from_directory(
    'emotion_dataset_split/train',  # Update with your train dataset path
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'emotion_dataset_split/test',  # Update with your validation dataset path
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)

# Train the model
history = model.fit(
    train_generator,
    epochs=100,  # Allow more epochs for overfitting
    validation_data=val_generator,
    callbacks=[early_stopping, checkpoint, reduce_lr],
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator)
)

# Plot training history (accuracy and loss)
def plot_training_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax[0].plot(history.history['accuracy'], label='Train Accuracy')
    ax[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    ax[0].set_title('Model Accuracy')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()

    # Loss
    ax[1].plot(history.history['loss'], label='Train Loss')
    ax[1].plot(history.history['val_loss'], label='Val Loss')
    ax[1].set_title('Model Loss')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].legend()

    plt.show()

# Visualize the training process
plot_training_history(history)

# Save the final model
model.save('3-emotion_recognition_model.h5')

print("Model training complete. Saved as '3-emotion_recognition_model.h5'.")
