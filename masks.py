import cv2
import numpy as np
import glob
import argparse
import os

def process_label_to_mask():

	pass


def process_folder(folder_path):
	pass
	folder_path = folder_path.rstrip('/')
	# band_regex = f"{folder_path}/*_B0X.tif"
	mask_regex = f"{folder_path}/*_LBL.tif"
	# band_paths = glob.glob(band_regex,root_dir=folder_path)
	mask_paths = glob.glob(mask_regex,root_dir=folder_path)

	for file in mask_paths:
		linear_transform(f"{folder_path}/{file}")


def three_class_mask():
	pass


def linear_transform(img_path):

	# PARAMETERS
	dist_type = cv2.DIST_L2
	# dist_type = cv2.DIST_L1
	# mask_size = cv2.DIST_MASK_5
	mask_size = cv2.DIST_MASK_PRECISE #L2 only

	# READ
	img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)

	# INVERT
	img = cv2.bitwise_not(img)

	# LINEAR TRANSFORM
	dist_transform = cv2.distanceTransform(img, dist_type,mask_size)
	# dist_normalized = cv2.normalize(dist_transform,None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)
	dist_normalized = np.zeros(dist_transform.shape, dtype=np.float32)
	cv2.normalize(dist_transform, dist_normalized, 0.0, 1.0, cv2.NORM_MINMAX) # 0-1 ? 

	# PLOT TO CHECK
	# cv2.imshow('Original Binary',img)
	cv2.imshow('Linear Transform',dist_normalized)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return dist_normalized

def calculate_exponential(normalized_dist):
	# alpha = 10.0 #try multiple
	# alpha = 2.0
	# alpha = 10.0
	# alpha = 1.0 #linear

	# EXP TO BOUNDARY
	# exp_dist = (np.exp(alpha * normalized_dist) - 1) / (np.exp(alpha) - 1)
	# exp_dist = exp_dist.astype(np.float32)

	# OR SIGMOID? -- need two sided
	k  = 20.0 #edge drop
	d0 = 0.1 #shift
	exp_dist = 1/(1+np.exp(-k * (normalized_dist - d0)))
	exp_dist = np.clip(exp_dist,0.0,1.0).astype(np.float32)

	# PLT TO CHECK
	display = (exp_dist * 255).astype(np.uint8)
	# cv2.imwrite('exp_dist_map.png',display)
	cv2.imshow('Exp Distance Transform',display)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def parse_args():

	# DEFINE AND READ
	parser = argparse.ArgumentParser()
	parser.add_argument('--chip-dir',required=True,default=None,help='Dataset (chip) directory.')
	args = parser.parse_args()

	# CHECK VALUES
	assert os.path.isdir(args.chip_dir), f"Chip dir {args.chip_dir} not found."
	args.chip_dir = args.chip_dir.rstrip('/')

	# RETURN
	return args


################################################################################
# MAIN
################################################################################
if __name__ == '__main__':

	args = parse_args()

	labels = sorted(glob.glob(f"{args.chip_dir}/validation/*_LBL.tif"))
	print(labels[0])

	# linear_transform(labels[0])
	calculate_exponential(linear_transform(labels[0]))
	pass

