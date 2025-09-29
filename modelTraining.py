# modelTraining.py
# This is the main script to train the CNN model.
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Kaggle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, auc
# Import our custom modules
import keys
import loadData

# --- Configuration ---
SAMPLE_SIZE_PER_CLASS = 20000
EPOCHS = 30  # Increased for better convergence
BATCH_SIZE = 256
MODEL_OUTPUT_PATH = "QuickDraw_CNN_35_classes_best.h5"

def split_train_val_test(x_data, y_data, train_ratio=0.8, val_ratio=0.1):
    """
    Splits data into train, validation, and test sets.
    Default: 80% train, 10% validation, 10% test
    """
    total_samples = len(x_data)
    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))
    
    # Shuffle indices
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    # Split the data
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    x_train = x_data[train_idx]
    y_train = y_data[train_idx]
    x_val = x_data[val_idx]
    y_val = y_data[val_idx]
    x_test = x_data[test_idx]
    y_test = y_data[test_idx]
    
    return x_train, y_train, x_val, y_val, x_test, y_test

def build_model(num_classes):
    """Builds an improved CNN model with better feature extraction."""
    model = models.Sequential()
    
    # Input Block - Enhanced feature extraction
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Convolutional Block 2 - Deeper feature learning
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Convolutional Block 3 - More complex patterns
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    
    # Additional Block for better discrimination
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.GlobalAveragePooling2D())
    
    # Dense Layers with higher capacity
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))
    
    # Output Layer with label smoothing for better generalization
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Use label smoothing to prevent overconfidence
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    return model

def create_data_augmentation():
    """
    Creates data augmentation generator to help model learn invariant features.
    This helps distinguish similar shapes by learning robust representations.
    """
    datagen = ImageDataGenerator(
        rotation_range=15,        # Random rotations
        width_shift_range=0.1,    # Horizontal shifts
        height_shift_range=0.1,   # Vertical shifts
        zoom_range=0.1,           # Random zoom
        shear_range=0.1,          # Shear transformations
        fill_mode='nearest'       # Fill mode for new pixels
    )
    return datagen

def plot_history(history):
    """Plots the training and validation accuracy and loss."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot training & validation accuracy values
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')
    ax1.grid(True)
    
    # Plot training & validation loss values
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')
    ax2.grid(True)
    
    plt.savefig("training_history.png")
    plt.close()
    print("Training history plot saved to 'training_history.png'")

def calculate_convergence_rate(history, window=3):
    """
    Calculate convergence rate based on validation accuracy improvement.
    Returns the average improvement per epoch over the last 'window' epochs.
    """
    val_acc = history.history['val_accuracy']
    if len(val_acc) < window + 1:
        window = len(val_acc) - 1
    
    if window < 1:
        return 0.0
    
    # Calculate average change over the last 'window' epochs
    recent_changes = []
    for i in range(len(val_acc) - window, len(val_acc)):
        change = val_acc[i] - val_acc[i-1]
        recent_changes.append(change)
    
    convergence_rate = np.mean(recent_changes)
    return convergence_rate

def plot_roc_curves(model, x_test, y_test_cat, categories, num_classes=10):
    """
    Plot ROC curves for the first 'num_classes' categories.
    """
    print("Generating predictions for ROC curves...")
    # Get predictions
    y_pred_proba = model.predict(x_test, verbose=0)
    
    plt.figure(figsize=(12, 8))
    
    # Plot ROC curve for each of the first num_classes
    for i in range(min(num_classes, len(categories))):
        fpr, tpr, _ = roc_curve(y_test_cat[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, 
                label=f'{categories[i]} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - First 10 Categories', fontsize=14)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    
    plt.savefig("roc_curves_first_10.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("ROC curves plot saved")

def main():
    """Main function to run the training process."""
    print("="*60)
    print("Starting QuickDraw CNN Training")
    print("="*60)
    
    # 1. Load data
    print("\n[1/12] Loading data...")
    categories = keys.get_keys()
    num_classes = len(categories)
    x_data, y_data, _, _ = loadData.load_data(categories, SAMPLE_SIZE_PER_CLASS)
    print(f"Total data loaded: {len(x_data)} samples across {num_classes} classes")
    
    # 2. Split into train/val/test (80/10/10)
    print("\n[2/12] Splitting data into train/val/test (80/10/10)...")
    x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(x_data, y_data)
    print(f"Training set: {len(x_train)} samples")
    print(f"Validation set: {len(x_val)} samples")
    print(f"Test set: {len(x_test)} samples")
    
    # 3. One-hot encode the labels
    print("\n[3/12] One-hot encoding labels...")
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    print("Labels encoded successfully")
    
    # 4. Create data augmentation generator
    print("\n[4/12] Setting up data augmentation...")
    datagen = create_data_augmentation()
    datagen.fit(x_train)
    print("Data augmentation configured")
    
    # 5. Build the improved model
    print("\n[5/12] Building improved CNN model...")
    model = build_model(num_classes)
    model.summary()
    
    # 6. Setup callbacks
    print("\n[6/12] Setting up training callbacks...")
    checkpoint = ModelCheckpoint(
        MODEL_OUTPUT_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Reduce learning rate when validation accuracy plateaus
    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
    print("Callbacks configured: ModelCheckpoint, ReduceLROnPlateau, EarlyStopping")
    
    # 7. Train the model with data augmentation
    print("\n" + "="*60)
    print("[7/12] Starting Model Training with Data Augmentation")
    print("="*60)
    history = model.fit(
        datagen.flow(x_train, y_train_cat, batch_size=BATCH_SIZE),
        steps_per_epoch=len(x_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_val, y_val_cat),
        callbacks=[checkpoint, reduce_lr, early_stop],
        verbose=1
    )
    print("\n" + "="*60)
    print("Model Training Complete!")
    print("="*60)
    
    # 8. Load the best model and evaluate
    print(f"\n[8/12] Loading best model from {MODEL_OUTPUT_PATH}...")
    best_model = models.load_model(MODEL_OUTPUT_PATH)
    
    # Evaluate on validation set
    print("\n[9/12] Evaluating on validation set...")
    val_loss, val_acc = best_model.evaluate(x_val, y_val_cat, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f} ({val_acc:.2%})")
    
    # Evaluate on test set
    print("\n[10/12] Evaluating on test set...")
    test_loss, test_acc = best_model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc:.2%})")
    
    # 9. Calculate and print convergence rate
    print("\n" + "="*60)
    print("[11/12] Convergence Analysis")
    print("="*60)
    conv_rate = calculate_convergence_rate(history, window=3)
    print(f"Convergence Rate (avg change over last 3 epochs): {conv_rate:.6f}")
    if abs(conv_rate) < 0.001:
        print("✓ Model has converged (minimal improvement in recent epochs)")
    else:
        print(f"→ Model is {'improving' if conv_rate > 0 else 'degrading'} at {abs(conv_rate):.4%} per epoch")
    
    # 10. Plot ROC curves for first 10 categories
    print("\n[12/12] Generating ROC Curves for first 10 categories...")
    plot_roc_curves(best_model, x_test, y_test_cat, categories, num_classes=10)
    print("✓ ROC curves saved to 'roc_curves_first_10.png'")
    
    # 11. Save the test data for the prediction script
    print("\nSaving test data for predict.py...")
    np.save('test_images.npy', x_test)
    np.save('test_labels.npy', y_test)
    print("✓ Test data saved")
    
    # 12. Plot history
    print("\nGenerating training history plots...")
    plot_history(history)
    
    print("\n" + "="*60)
    print("ALL TASKS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nFinal Results:")
    print(f"  - Best Validation Accuracy: {val_acc:.2%}")
    print(f"  - Test Accuracy: {test_acc:.2%}")
    print(f"  - Model saved: {MODEL_OUTPUT_PATH}")
    print(f"  - Plots saved: training_history.png, roc_curves_first_10.png")

if __name__ == "__main__":
    main()
