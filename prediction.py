import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
import os

# Paths to your dataset (replace with your actual dataset paths)
train_dir = r"C:\Users\RangeGowda.GP\Downloads\dataset\datasets1\train"
valid_dir = r"C:\Users\RangeGowda.GP\Downloads\dataset\datasets1\valid"
test_dir = r"C:\Users\RangeGowda.GP\Downloads\dataset\datasets1\test"

train_datagen = ImageDataGenerator(
    rescale=1.0/255,         
    rotation_range=30,       
    width_shift_range=0.2,   
    height_shift_range=0.2,  
    shear_range=0.2,         
    zoom_range=0.2,          
    horizontal_flip=True,    
    fill_mode="nearest"      
)

valid_datagen = ImageDataGenerator(rescale=1.0/255)  
test_datagen = ImageDataGenerator(rescale=1.0/255)  


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  
    batch_size=32,          
    class_mode='categorical' 
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


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')  
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    epochs=10,               
    validation_data=valid_generator
)

model.save("plant_model.keras")
print("Model saved as 'plant_model.keras'")


with open("class_indices.pkl", "wb") as f:
    pickle.dump(train_generator.class_indices, f)
print("Class indices saved as 'class_indices.pkl'")


test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
