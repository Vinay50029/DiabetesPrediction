import tensorflow as tf

# Load your Keras model
model = tf.keras.models.load_model('model/densenet_weights.hdf5')

# Convert it to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open('model/densenetw.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Conversion successful. File saved as 'densenetw.tflite'")
