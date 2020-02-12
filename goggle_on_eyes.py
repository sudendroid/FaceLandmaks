# python goggle_on_eyes.py --image images/salman.jpg

import argparse
import dlib
import cv2
from imageutils import overlay_transparent, resize

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
goggle = cv2.imread("images/goggle.png", cv2.IMREAD_UNCHANGED)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
offset = 8

image = resize(image, width=600)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):
    shape_68 = predictor(gray, rect)

    # calculate bounds to draw goggle
    left_x = shape_68.part(17).x - offset
    left_y = shape_68.part(19).y
    right_x = shape_68.part(26).x + offset
    resize_goggle = resize(goggle, width=right_x - left_x, inter=cv2.INTER_NEAREST)
    overlay_transparent(image, resize_goggle, left_x, left_y)

cv2.imshow("Output", image)
cv2.waitKey(0)

