from matplotlib import pyplot as plt
from src.image_preproc import extract_nail
from src.utils import list_images, distribute_paths
import argparse

def plot_images(images):
	'''
	Plot the images to show the whole process of extracting each of the images
	@params:
		images: dict containing key -> value :: title -> image
	'''
	rows = 2
	cols = 5

	idx = 0
	for title, image in images.items():
		plt.subplot(rows,cols,idx+1),plt.imshow(image)
		plt.title(title)
		plt.xticks([]),plt.yticks([])
		idx += 1
	plt.show()


def main():
	# Construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--path", default='nailgun', help="path to nailgun folder")
	ap.add_argument("-cs", "--csize", default=80, help="paramter to crop the image around the nailgun")
	ap.add_argument("-ex", "--ext", type=str, default='.jpeg', help="extension of the images")
	args = vars(ap.parse_args())

	# Load paramters
	crop_size = args['csize']
	path_to_images = args['path']
	ext = args['ext']

	# List all of the images
	paths, labels = list_images(path_to_images, ext)

	# Get paths correctly distibuted good/bad
	n_paths = distribute_paths(paths)

	# Iterate over the images with debug =  True
	for path in n_paths:
		_, images = extract_nail(path, debug=True)
		try:
			plot_images(images)
		except KeyboardInterrupt:
			pass

if __name__ == '__main__':
	main()