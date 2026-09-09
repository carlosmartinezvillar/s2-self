import cv2
import numpy as np
import glob
import argparse
import os


def three_class_mask(img_path,debug=False):

	# PARAMETERS
	kernel = np.ones((3,3),np.uint8)

	# READ (values 0 / 255)
	img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)

	# SHORELINE = pixels where dilation and erosion disagree, i.e. touching the opposite class
	shoreline = cv2.subtract(cv2.dilate(img,kernel), cv2.erode(img,kernel))

	# RECLASSIFY THOSE PIXELS AS THE NEW SHORELINE CLASS
	three_class = img.copy()
	three_class[shoreline > 0] = 127

	# PLOT TO CHECK
	if debug:
		cv2.imshow('Three-Class Mask',three_class)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	return three_class


def count_class_pixels(folder_path,suffix='_3CL.tif'):
	folder_path = folder_path.rstrip('/')
	paths = sorted(glob.glob(f"{folder_path}/*{suffix}"))

	counts = {}
	for path in paths:
		img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
		values,pixel_counts = np.unique(img,return_counts=True)
		for value,count in zip(values.tolist(),pixel_counts.tolist()):
			counts[value] = counts.get(value,0) + count

	return counts


def process_folder_3class(folder_path,out_folder):
	folder_path = folder_path.rstrip('/')
	out_folder  = out_folder.rstrip('/')
	os.makedirs(out_folder,exist_ok=True)
	label_paths = sorted(glob.glob(f"{folder_path}/*_LBL.tif"))

	for i,label_path in enumerate(label_paths):
		three_class = three_class_mask(label_path,debug=False)

		basename = os.path.basename(label_path)[:-len('_LBL.tif')] + '_3CL.tif'
		out_path = f"{out_folder}/{basename}"
		cv2.imwrite(out_path,three_class)

		if i % 1000 == 0:
			print(f"{i}/{len(label_paths)} written.")
		# print(f"Wrote {out_path}")

# def calculate_exponential(normalized_dist):
# 	# alpha = 10.0 #try multiple
# 	# alpha = 2.0
# 	# alpha = 10.0
# 	# alpha = 1.0 #linear

# 	# EXP TO BOUNDARY
# 	# exp_dist = (np.exp(alpha * normalized_dist) - 1) / (np.exp(alpha) - 1)
# 	# exp_dist = exp_dist.astype(np.float32)

# 	# OR SIGMOID? -- need two sided
# 	k  = 20.0 #edge drop
# 	d0 = 0.1 #shift
# 	exp_dist = 1/(1+np.exp(-k * (normalized_dist - d0)))
# 	exp_dist = np.clip(exp_dist,0.0,1.0).astype(np.float32)

# 	# PLT TO CHECK
# 	display = (exp_dist * 255).astype(np.uint8)
# 	# cv2.imwrite('exp_dist_map.png',display)
# 	cv2.imshow('Exp Distance Transform',display)
# 	cv2.waitKey(0)
# 	cv2.destroyAllWindows()

def boundary_distance_transform(img_path,debug=False):

	# PARAMETERS
	dist_type = cv2.DIST_L2
	mask_size = cv2.DIST_MASK_PRECISE #L2 only

	# READ
	img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)

	# DISTANCE TO NEAREST OPPOSITE-CLASS PIXEL, FROM BOTH SIDES OF THE EDGE
	# dist_to_fg is 0 inside the foreground and grows in the background;
	# dist_to_bg is 0 in the background and grows inside the foreground.
	# exactly one term is nonzero per pixel, so the sum is the distance to
	# the class boundary everywhere
	dist_to_fg = cv2.distanceTransform(img, dist_type,mask_size)
	dist_to_bg = cv2.distanceTransform(cv2.bitwise_not(img), dist_type,mask_size)
	dist_transform = dist_to_fg + dist_to_bg
	dist_normalized = np.zeros(dist_transform.shape, dtype=np.float32)
	cv2.normalize(dist_transform, dist_normalized, 0.0, 1.0, cv2.NORM_MINMAX) #0-1

	# PLOT TO CHECK
	if debug:
		cv2.imshow('Distance to Boundary',dist_normalized)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	return dist_normalized


def calculate_boundary_weight(normalized_dist,debug=False):
	# alpha controls how narrow the emphasized boundary band is
	alpha = 10.0 #try multiple?
	# alpha = 2.0
	# alpha = 20.0

	# HIGH WEIGHT AT THE BOUNDARY, DECAYING WITH DISTANCE ON BOTH SIDES
	# normalized_dist is in [0,1] and alpha > 0, so this is already in (0,1] -- no clip needed
	exp_dist = np.exp(-alpha * normalized_dist).astype(np.float32)

	# PLT TO CHECK
	if debug:
		display = (exp_dist * 255).astype(np.uint8)
		cv2.imshow('Boundary Weight Map',display)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	return exp_dist


def process_folder(folder_path,out_folder):
	folder_path = folder_path.rstrip('/')
	out_folder  = out_folder.rstrip('/')
	os.makedirs(out_folder,exist_ok=True)
	label_paths = sorted(glob.glob(f"{folder_path}/*_LBL.tif"))

	for i,label_path in enumerate(label_paths):
		weight = calculate_boundary_weight(boundary_distance_transform(label_path,debug=False),debug=False)
		weight_img = (weight * 255).astype(np.uint8)

		basename = os.path.basename(label_path)[:-len('_LBL.tif')] + '_MSK.tif'
		out_path = f"{out_folder}/{basename}"
		cv2.imwrite(out_path,weight_img)

		if i % 1000 == 0:
			print(f"{i}/{len(label_paths)} written.")
		# print(f"Wrote {out_path}")


def parse_args():

	# DEFINE AND READ
	parser = argparse.ArgumentParser()
	parser.add_argument('--chip-dir',required=True,default=None,help='Dataset (chip) directory.')
	parser.add_argument('--out-folder',required=True,default=None,help='Where to write _MSK.tif files.')
	args = parser.parse_args()

	# CHECK VALUES
	assert os.path.isdir(args.chip_dir), f"Chip dir {args.chip_dir} not found."
	args.chip_dir = args.chip_dir.rstrip('/')
	assert args.out_folder is not None, f"No output folder given."
	assert os.path.isdir(args.out_folder), f"Out dir {args.out_folder} not found."
	args.out_folder = args.out_folder.rstrip('/')

	# RETURN
	return args


################################################################################
# MAIN
################################################################################
if __name__ == '__main__':

	args = parse_args()

	# QUICK VISUAL CHECK ON A SINGLE VALIDATION LABEL
	# labels = sorted(glob.glob(f"{args.chip_dir}/validation/*_LBL.tif"))
	# print(labels[0])
	# calculate_boundary_weight(boundary_distance_transform(labels[0],debug=True),debug=True)
	# three_class_mask(labels[0],debug=True)

	# PROCESS DATASET
	#################
	# GENERATE _MSK.tif BOUNDARY WEIGHT MAPS FOR THE FULL TRAINING SET
	# process_folder(f"{args.chip_dir}/training",f"{args.out_folder}/training")
	# process_folder(f"{args.chip_dir}/validation",f"{args.out_folder}/validation")

	# GENERATE _3CL.tif 3-CLASS MAPS FOR THE FULL TRAINING SET
	# process_folder_3class(f"{args.chip_dir}/training",f"{args.out_folder}/training")
	# process_folder_3class(f"{args.chip_dir}/validation",f"{args.out_folder}/validation")

	# COUNT PIXELS PER CLASS -- FOR INVERSE-FREQUENCY LOSS WEIGHTS
	#########################
	counts = count_class_pixels(f"{args.out_folder}/training")
	total = sum(counts.values())
	print("Pixel counts:",counts)
	print("Inverse-frequency weights:",{v: total/(len(counts)*c) for v,c in counts.items()})
	print({v: c/total for v,c in counts.items()})
