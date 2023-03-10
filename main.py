import time
from logzero import logger, logfile

running_time = time.time() + 60 * 180

import os

base = os.path.realpath(os.path.dirname(__file__))
images_path = os.path.join(base, 'images')

logfile(os.path.join(base, 'logs.log'))
logger.info('Started timer')

from picamera import PiCamera
from PIL import Image
from orbit import ISS
import tflite_runtime.interpreter as tflite
import numpy as np


def convert(angle):
		sign, degrees, minutes, seconds = angle.signed_dms()
		exif_angle = f'{degrees:.0f}/1,{minutes:.0f}/1,{seconds*10:.0f}/10'
		return sign < 0, exif_angle


#camera
logger.info('Starting camera...')
camera = PiCamera(resolution=(2592, 1944))

time.sleep(1)
logger.info('Done')

camera.shutter_speed = camera.exposure_speed
camera.exposure_mode = 'off'

ai = False
try:
	#load model
	logger.info('Loading model...')

	class_names = ['Clouds', 'Land', 'Night', 'Sea']

	model_path = os.path.join(base, 'model.tflite')

	interpreter = tflite.Interpreter(model_path=model_path,
		experimental_delegates=[
			tflite.load_delegate('libedgetpu.so.1')
		]
	)

	shape = interpreter.get_input_details()[0]['shape']
	shape = (shape[1], shape[2])

	interpreter.allocate_tensors()
except BaseException:
	logger.warning('Failed to load model. Retrying in 5 secs')
	time.sleep(5)
	try:
		#load model
		logger.info('Loading model...')

		class_names = ['Clouds', 'Land', 'Night', 'Sea']

		model_path = os.path.join(base, 'model.tflite')

		interpreter = tflite.Interpreter(model_path=model_path,
			experimental_delegates=[
				tflite.load_delegate('libedgetpu.so.1')
			]
		)

		shape = interpreter.get_input_details()[0]['shape']
		shape = (shape[1], shape[2])

		interpreter.allocate_tensors()
	except BaseException:
		logger.warning('Failed to load model')
	else:
		logger.info('Done')
		ai = True
else:
	logger.info('Done')
	ai = True
finally:
	logger.info(f'AI is {"on" if ai else "off"}')

#check if images folder exists
if not os.path.exists(images_path):
	logger.info('Creating images dir...')
	try:
		os.makedirs(images_path)
	except BaseException:
		logger.warning('Failed to create images directory. Saving in root folder')
		images_path = base
	else:
		logger.info('Done')



logger.info('Starting experiment')
while time.time() < running_time:

	try:
		#check space
		size = 0
		for path, dirs, files in os.walk(base):
			for f in files:
				fp = os.path.join(path, f)
				size += os.path.getsize(fp)

		if size >= 2_950_000_000:
			logger.warning('Reached max size')
			break
	except KeyboardInterrupt:
		break
	except BaseException:
		logger.warning('Failed to check size')


	try:
		#geo locations
		point = ISS.coordinates()

		south, exif_latitude = convert(point.latitude)
		west, exif_longitude = convert(point.longitude)

		camera.exif_tags['GPS.GPSLatitude'] = exif_latitude
		camera.exif_tags['GPS.GPSLatitudeRef'] = "S" if south else "N"
		camera.exif_tags['GPS.GPSLongitude'] = exif_longitude
		camera.exif_tags['GPS.GPSLongitudeRef'] = "W" if west else "E"
	except KeyboardInterrupt:
		break
	except BaseException:
		logger.warning('Failed to retrieve GPS data')


	try:
		#take photo
		name = 'img_{}'.format(time.strftime('%H_%M_%S', time.gmtime()))

		camera.capture(os.path.join(images_path, name + '.jpg'))
	except KeyboardInterrupt:
		break
	except BaseException:
		logger.warning('Failed to capture')
	else:

		if ai:
			try:
				#classify
				model_img = Image.open(os.path.join(images_path, name + '.jpg')).convert('RGB').resize(shape)

				interpreter.tensor(interpreter.get_input_details()[0]['index'])()[0][:, :] = model_img

				interpreter.invoke()

				scores = np.squeeze(interpreter.get_tensor(interpreter.get_output_details()[0]['index']))

				klass = class_names[int(np.where(scores == max(scores))[0])]


				logger.info(klass)
			except BaseException:
				logger.warning('Failed to classify')
			else:


				try:
					#check
					if klass == 'Sea':
						logger.info(f'Saved {name}')
						time.sleep(1)
					elif klass == 'Clouds':
						logger.info(f'Saved {name}')
						time.sleep(2)
					elif klass == 'Night':
						os.remove(os.path.join(images_path, name + '.jpg'))
						time.sleep(10)
					else:
						os.remove(os.path.join(images_path, name + '.jpg'))
						time.sleep(2)
				except BaseException:
					logger.warning('Failed to classify')

logger.info('Time\'s over')