import time

running_time = time.time() + 60 * 180

import os
from picamera import PiCamera

camera = PiCamera(resolution=(2592, 1944))
camera.start_preview()

time.sleep(2)

camera.iso = 120

print(camera.exposure_speed)
camera.shutter_speed = camera.exposure_speed
camera.exposure_mode = 'off'

tmp = camera.awb_gains
camera.awb_mode = 'off'
camera.awb_gains = tmp

path = os.path.realpath(os.path.dirname(__file__))

camera.capture('test.jpg')

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
	#classify