import time, os

running_time =time.time() + 60 * 180

path = os.path.realpath(os.path.dirname(__file__))

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
	pass