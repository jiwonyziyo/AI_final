from keras.models import load_model
from tensorflow.keras.models import save_model


model = load_model('emotion_model.hdf5', compile=False)


save_model(model, 'updated_emotion_model.h5')
