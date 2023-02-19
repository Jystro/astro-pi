import tflite_runtime.interpreter as tflite
import os
import numpy as np

img_height, img_width = (180, 180)

class_names = ['Clouds', 'Land', 'Night', 'Sea']

TF_MODEL_FILE_PATH = 'model.tflite' # The default path to the saved TensorFlow Lite model

interpreter = tflite.Interpreter(model_path=TF_MODEL_FILE_PATH)

print(interpreter.get_signature_list())

classify_lite = interpreter.get_signature_runner('serving_default')

img = tf.keras.utils.load_img(
	os.path.abspath("photo_00000.jpg"), target_size=(img_height, img_width)
)

img_array = tf.expand_dims(tf.keras.utils.img_to_array(img), 0)

predictions_lite = classify_lite(sequential_1_input=img_array)['outputs']
score_lite = tf.nn.softmax(predictions_lite)

print("This image most likely belongs to {} with a {:.2f} percent confidence."
.format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite)))

print("Other perc:")
print(class_names)
print(np.array2string(score_lite.numpy() * 100, precision=2, suppress_small=True))
