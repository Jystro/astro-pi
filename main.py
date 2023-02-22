import time

running_time = time.time() + 60 * 180

import os
from picamera import PiCamera
from io import BytesIO
from PIL import Image

path = os.path.realpath(os.path.dirname(__file__))

#camera
camera = PiCamera(resolution=(2592, 1944))
camera.start_preview()

time.sleep(2)

camera.iso = 120

camera.shutter_speed = camera.exposure_speed
camera.exposure_mode = 'off'

tmp = camera.awb_gains
camera.awb_mode = 'off'
camera.awb_gains = tmp

#model
import tflite_runtime.interpreter as tflite
import numpy as np

class_names = ['Clouds', 'Land', 'Night', 'Sea']

model_path = 'model.tflite'

interpreter = tflite.Interpreter(model_path=model_path,
	experimental_delegates=[
		tflite.load_delegate('libedgetpu.so.1')
	]
)

shape = interpreter.get_input_details()[0]['shape']
shape = (shape[1], shape[2])

interpreter.allocate_tensors()




while time.time() < running_time:

	#check space
	size = 0
	for path, dirs, files in os.walk(path):
		for f in files:
			fp = os.path.join(path, f)
			size += os.path.getsize(fp)

	if size >= 3_000_000_000:
		print('Reached max size')
		break

	#take photo
	stream = BytesIO()

	camera.capture(stream, format='jpeg')#img_{}.jpg'.format(time.strftime('%H_%M_%S', time.gmtime()))

	stream.seek(0)
	image = Image.open(stream)

	#classify
	model_img = image.convert('RGB').resize(shape)

	interpreter.tensor(interpreter.get_input_details()[0]['index'])()[0][:, :] = model_img

	interpreter.invoke()

	scores = np.squeeze(interpreter.get_tensor(interpreter.get_output_details()[0]['index']))

	print(scores)
	print("This image most likely belongs to {}."
	.format(class_names[int(np.where(scores == max(scores))[0])]))

	#wait
	time.sleep(5)