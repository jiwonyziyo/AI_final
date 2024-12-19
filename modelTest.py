from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


model1 = load_model('updated_emotion_model.h5')  
model2 = load_model('expression_model_sara.h5')  

model1.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

model2.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

test_generator_model1 = test_datagen.flow_from_directory(
    'data/test',
    target_size=(64, 64),  
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

test_generator_model2 = test_datagen.flow_from_directory(
    'data/test',
    target_size=(48, 48),  
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

results_model1 = model1.evaluate(test_generator_model1)
results_model2 = model2.evaluate(test_generator_model2)

print(f"Model 1 (64x64) - Loss: {results_model1[0]:.4f}, Accuracy: {results_model1[1]:.4f}")
print(f"Model 2 (48x48) - Loss: {results_model2[0]:.4f}, Accuracy: {results_model2[1]:.4f}")