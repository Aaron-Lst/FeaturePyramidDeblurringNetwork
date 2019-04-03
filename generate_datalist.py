import os

input_path = '/home/opt603/sda5/GOPRO_Large/test'

filesName = sorted(os.listdir(input_path))

f = open('test_datalist.txt', 'a+')

for fileName in filesName:
	path_blur = os.path.join(input_path, fileName, 'blur')
	path_sharp = os.path.join(input_path, fileName, 'sharp')
	imagesName = sorted(os.listdir(path_blur))
	for imageName in imagesName:
		f.write(os.path.join(path_blur, imageName))
		f.write(' ')
		f.write(os.path.join(path_sharp, imageName))
		f.write('\n')
f.close()
