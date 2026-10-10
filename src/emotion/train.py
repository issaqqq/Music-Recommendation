import os
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

import sys
from pathlib import Path
# Add project root to sys.path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import EMOTION_DATASET_PATH, EMOTION_MODEL_PATH, IMG_SHAPE, BATCH_SIZE, EPOCHS, EMOTIONS

def build_model(input_shape, num_classes):
    """Builds VGG19 Transfer Learning model."""
    print("Building VGG19 model...")
    # Load VGG19 without the top dense layers
    base_model = VGG19(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape))
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def plot_history(history):
    """Plots training/validation accuracy and loss."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax[0].plot(history.history['accuracy'], label='Train Acc')
    ax[0].plot(history.history['val_accuracy'], label='Val Acc')
    ax[0].set_title('Model Accuracy')
    ax[0].legend()
    
    # Loss
    ax[1].plot(history.history['loss'], label='Train Loss')
    ax[1].plot(history.history['val_loss'], label='Val Loss')
    ax[1].set_title('Model Loss')
    ax[1].legend()
    
    plt.savefig('emotion_training_history.png')
    print("Saved training history plot as emotion_training_history.png")

def main():
    train_dir = os.path.join(EMOTION_DATASET_PATH, 'train')
    val_dir = os.path.join(EMOTION_DATASET_PATH, 'test')
    
    if not os.path.exists(train_dir):
        print(f"Error: Dataset not found at {train_dir}")
        return

    # 1. Data Generators with Augmentation
    print("Setting up Data Generators...")
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    # Note: color_mode='rgb' will duplicate grayscale channels to make it (48, 48, 3) for VGG19
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SHAPE[0], IMG_SHAPE[1]),
        batch_size=BATCH_SIZE,
        color_mode='rgb', 
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_SHAPE[0], IMG_SHAPE[1]),
        batch_size=BATCH_SIZE,
        color_mode='rgb',
        class_mode='categorical'
    )

    # 2. Build Model
    model = build_model(IMG_SHAPE, len(EMOTIONS))
    
    # 3. Callbacks
    checkpoint = ModelCheckpoint(
        filepath=str(EMOTION_MODEL_PATH),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

    # 4. Train Model
    print("Starting training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )

    plot_history(history)
    print("Phase 2 Complete: Emotion model trained and saved.")

if __name__ == "__main__":
    main()