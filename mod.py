from keras.models import load_model
import tensorflow as tf
model=load_model('BrainTumor40Epochs_Categorical.h5')

comverter=tf.lite.TFLiteConverter.from_keras_model(model)
tf_lite_model=comverter.convert()

with open('model.tflite','wb') as f:
    f.write(tf_lite_model)