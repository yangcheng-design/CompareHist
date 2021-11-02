"""comparehist.py"""

from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required = True,
	help = "Path to the directory of images")
args = vars(parser.parse_args())

# initialize the index dictionary to store the image name
# and corresponding histograms and the images dictionary
# to store the images themselves
index = {}
images = {}

target = input("Enter the image filename (including .jpg): ")

# loop over the image paths
for imagePath in glob.glob(args["dataset"] + "/*.jpg"):
	# extract the image filename (assumed to be unique) and
	# load the image, updating the images dictionary
	filename = imagePath[imagePath.rfind("/") + 1:]
	image = cv2.imread(imagePath)
    # convert to RGB format for matplotlib
	images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
	# extract a 3D RGB color histogram from the image,
	# using 8 bins per channel, normalize, and update
	# the index
	hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
		[0, 256, 0, 256, 0, 256])
	hist = cv2.normalize(hist, hist).flatten()
	index[filename] = hist

# initialize the results dictionary and the sort
# direction
results = {}
reverse = False

# loop over the index
for (k, hist) in index.items():
	# compute the distance between the two histograms
	# using the method and update the results dictionary
	d = cv2.compareHist(index[target], hist, cv2.HISTCMP_BHATTACHARYYA)
	results[k] = d
# sort the results
results = sorted([(v, k) for (k, v) in results.items()], reverse = reverse)

# import pdb; pdb.set_trace()
# calculate similarity
results_dict = dict()
  
for score, img in results:
    results_dict.setdefault(score, []).append(img)

# get average similarity
total_similarity = 0
for val in results_dict.keys():
    total_similarity += val
avg_similarity = total_similarity / (len(results_dict) - 1)

print("\nAverage Similarity Score : ", avg_similarity)

if (avg_similarity < 0.6):
	print("\n\nThis image is too similar to the ones you've uploaded\n\n")
	print("Please try varying the lighting, angle of the object, background.\n\n")

elif (avg_similarity >= 0.6):
	print("You uploaded an image that is different enough. Good job!\n\n")
	print("You can continue to upload more images.\n\n")

# show the query image
fig = plt.figure("Most recently uploaded image")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(images[target])
plt.axis("off")
# initialize the results figure
fig = plt.figure("Results: %s" % ("Hellinger"))
fig.suptitle("Your Training Images", fontsize = 20)
# loop over the results
for (i, (v, k)) in enumerate(results):
	# show the result
	ax = fig.add_subplot(1, len(images), i + 1)
	ax.set_title("%s: %.2f" % (k, v))
	plt.imshow(images[k])
	plt.axis("off")

print("Printing all training images and similarity score.\n\n")
plt.show()
