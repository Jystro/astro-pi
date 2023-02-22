import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image

class_names = ['Clouds', 'Land', 'Night', 'Sea']

model_path = 'model.tflite' # The default path to the saved TensorFlow Lite model


interpreter = tflite.Interpreter(model_path=model_path,
	experimental_delegates=[
		tflite.load_delegate('libedgetpu.so.1')
	]
)

shape = interpreter.get_input_details()[0]['shape']
shape = (shape[1], shape[2])

interpreter.allocate_tensors()


img = Image.open('img_14_06_45.jpg').convert('RGB').resize(shape)


interpreter.tensor(interpreter.get_input_details()[0]['index'])()[0][:, :] = img

interpreter.invoke()

scores = np.squeeze(interpreter.tensor(interpreter.get_output_details()[0]['index'])())

print(scores)

print("This image most likely belongs to {}."
.format(class_names[int(np.where(scores == max(scores))[0])]))