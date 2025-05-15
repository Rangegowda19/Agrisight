import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
import os

# Paths to your dataset (replace with your actual dataset paths)
train_dir = r"C:\Users\RangeGowda.GP\Downloads\dataset\datasets1\train"
valid_dir = r"C:\Users\RangeGowda.GP\Downloads\dataset\datasets1\valid"
test_dir = r"C:\Users\RangeGowda.GP\Downloads\dataset\datasets1\test"

# Step 1: Create ImageDataGenerators for Data Augmentation and Rescaling
train_datagen = ImageDataGenerator(
    rescale=1.0/255,         # Normalize pixel values to [0, 1]
    rotation_range=30,       # Randomly rotate images by up to 30 degrees
    width_shift_range=0.2,   # Randomly shift images horizontally
    height_shift_range=0.2,  # Randomly shift images vertically
    shear_range=0.2,         # Apply random shearing transformations
    zoom_range=0.2,          # Randomly zoom images
    horizontal_flip=True,    # Randomly flip images horizontally
    fill_mode="nearest"      # Fill pixels when transformations occur
)

valid_datagen = ImageDataGenerator(rescale=1.0/255)  # Only normalize
test_datagen = ImageDataGenerator(rescale=1.0/255)   # Only normalize

# Step 2: Load the Data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Resize images to 224x224
    batch_size=32,          # Batch size for training
    class_mode='categorical' # Multi-class classification
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Step 3: Define the Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')  # Output layer
])

# Step 4: Compile the Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Step 5: Train the Model
history = model.fit(
    train_generator,
    epochs=10,                # Set the number of epochs
    validation_data=valid_generator
)

# Step 6: Save the Model
model.save("plant_model.keras")
print("Model saved as 'plant_model.keras'")

# Step 7: Save the Class Indices
with open("class_indices.pkl", "wb") as f:
    pickle.dump(train_generator.class_indices, f)
print("Class indices saved as 'class_indices.pkl'")

# Step 8: Evaluate the Model (Optional)
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
