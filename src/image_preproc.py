import cv2, imutils, pdb
import numpy as np

def extract_nail(image_path, crop_size=80, debug=False):
	'''
	Process the input image, detect the nail in the image, find its contour
	@params:
		image_path: path to the location of the image
		(optional) debug: If it is True, it will show images of the whole process
	@return: 
		ret: 1D vector containing values of the pixels and height and width.
	'''

	# Load the image
	if debug:
		images = {}

	full_image = cv2.imread(image_path)

	if debug:
		images['full image'] = full_image
	# Define image size
	width = np.size(full_image, 1)
	height = np.size(full_image, 0)

	# Crop the image to get rid of the upper white bar
	image = full_image[100:height, 0:width] 

	# Resize the image to process
	image = imutils.resize(image, height = 300)

	if debug:
		images['resize and remove white block'] = image

	# Convert to gray, blur the image and find edges in the image
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	inverted = invert_img(gray)
	smoothed = cv2.bilateralFilter(inverted, 11, 17, 17)
	edged = cv2.Canny(smoothed, 30, 200)

	if debug:
		images['inverted'] = inverted
		images['smoothed'] = smoothed
		images['edged'] = edged

	# Perform dilatation with a 2x2 Kernel
	kernel = np.ones((2,2),np.uint8)
	dilatation = cv2.dilate(edged,kernel,iterations = 1)

	if debug:
		images['dilated'] = dilatation

	# Find contours of the image, and sort them by area's size
	contours = cv2.findContours(dilatation.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours = imutils.grab_contours(contours)
	contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
	screenCnt = None

	if debug:
		img = image.copy()
		cv2.drawContours(img, contours[0], -1, (255,255,0))
		images['nail detected'] = img

	# Gets the perimeter of the maximum contour
	peri = cv2.arcLength(contours[0], True)

	# Approximate to the most likely shape to remove noise on the nail's shape
	screenCnt = cv2.approxPolyDP(contours[0], 0.015 * peri, True)

	if debug:
		cv2.drawContours(img, contours[0], -1, (255,0,0))
		images['bounding rectangle'] = img

	# Approximate the rotated rectangule where the nail fits
	rect = cv2.minAreaRect(screenCnt)

	# Crop the nail and paste it into a black image
	im_r = crop_to_black(image, contours[0], [crop_size,crop_size])

	if debug:
		images['crop to black'] = im_r

	# Rotate the image to fit the nail
	img_crop, img_rot = rotate_rect(im_r, rect)
	
	# Convert to grayscale
	img_c_gray = cv2.cvtColor(img_rot, cv2.COLOR_RGB2GRAY)

	# Add width and height of the nailgun. This will help with the bent nails
	[width, height] = img_crop.shape[:2]
	
	# Check if the nail is horizontal or vertical and turn it
	if width > height:

		width, height = height, width
		ret = img_c_gray

	else:
		ret = np.transpose(img_c_gray)

	if debug:
		images['rotated'] = ret

	# Convert the 2D image into a row vector
	ret = np.reshape(ret, (1, -1))

	# Attach width and heigth to the feature vector
	h_w = np.array([height,width])
	ret = np.append(ret, h_w).astype('uint8')

	if debug:
		return ret, images
	return ret

def crop_to_black(img, pts, pads):
	'''
	Function to crop a window given by 'pads', fill it with 0 pixel value and paste
	the cropped image (img) accorindg to the points (pts)
	@params:
		img: image where to crop the contours
		pts: the contours to crop and paste on the black image
		pads: size of the window [pad_1, pad_2]
	@return:
		cropped_image: cropped image with size 'pads'
	'''

	rect = cv2.boundingRect(pts)
	x,y,w,h = rect
	croped = img[y:y+h, x:x+w].copy()

	pts = pts - pts.min(axis=0)

	mask = np.zeros(croped.shape[:2], np.uint8)
	cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

	dst = cv2.bitwise_and(croped, croped, mask=mask)

	cropped_image = np.zeros((pads[0], pads[1], 3), dtype='uint8')

	pad_width = int((pads[0] - dst.shape[0]) / 2)
	pad_height = int((pads[1] - dst.shape[1]) / 2)

	cropped_image[pad_width:pad_width+dst.shape[0], pad_height:pad_height+dst.shape[1]] = dst

	return cropped_image

def rotate_rect(img, rect):
	'''
	Rotated a given image to fit the rectangle's angle
	@params:
		img: image to rotate
		rect: rectangle  containing the rotation angle
	@return:
		img_crop: cropped image -> size of the nail
		img_rot: rotated image
	'''
	# get the parameter of the small rectangle
	height, width = img.shape[0], img.shape[1]

	center, size, angle = (height//2, width//2), rect[1], rect[2]
	center, size = tuple(map(int, center)), tuple(map(int, size))

	# calculate the rotation matrix
	M = cv2.getRotationMatrix2D(center, angle, 1)

	# rotate the original image
	img_rot = cv2.warpAffine(img, M, (width, height))

	# now rotated rectangle becomes vertical and we crop it
	img_crop = cv2.getRectSubPix(img_rot, size, center)

	return img_crop, img_rot

def invert_img(img):
	'''
	Inverts the image
	@params:
		img: 2D image. Type has to be uint8
	@return:
		final_img: inverted image
	'''
	assert img.dtype == 'uint8'
	final_img = img.copy()
	for r in range(img.shape[0]):
		for c in range(img.shape[1]):
			final_img[r,c] = 255 - img[r,c]
	return final_img