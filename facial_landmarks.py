import argparse
import dlib
import cv2
from imageutils import resize

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
image = cv2.imread(args["image"])
image = resize(image, width=600)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)

# loop over the all face detected by dlib.get_frontal_face_detector
for (i, rect) in enumerate(rects):
	shape_68 = predictor(gray, rect)
	for part_index in range(shape_68.num_parts):
		cv2.circle(image, (shape_68.part(part_index).x, shape_68.part(part_index).y), 2, (0, 255, 0), 1)

# show landmarks that were circled in the last step
cv2.imshow("Output", image)
cv2.waitKey(0)

