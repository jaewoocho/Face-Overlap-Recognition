### OpenCV(cv2) = Image process Library
### Dlib = Image process / recognition Library
### numpy = matrix multiplication library

import cv2, dlib, sys
import numpy as np

###resizing the video by scaling
scaler = 0.3

###dlib face detector module
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

### Load video cv2
### 'girl.mp4' can be replaced with 0, which will turn on the Webcam
cap = cv2.VideoCapture('girl.mp4')

### Load overlay Image
### Cv2.IMReadUnchanged file lets us read the file imageas a BGRA type
overlay = cv2.imread('ryan_transparent.png', cv2.IMREAD_UNCHANGED)

### Overlay function ↓
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
bg_img = background_img.copy()

### convert 3 channels to 4 channels
if bg_img.shape[2] == 3:
bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

if overlay_size is not None:
img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

b, g, r, a = cv2.split(img_to_overlay_t)

mask = cv2.medianBlur(a, 5)

h, w, _ = img_to_overlay_t.shape
roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)

# convert 4 channels to 4 channels
bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

return bg_img
### Overlay function ↑

### Reading the video file by frames per second
while True:
ret, img = cap.read()
if not ret:
break
### resizing the img to dsize
img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
ori = img.copy()

### detect faces in every img
faces = detector(img)
face = faces[0]

### predictor(img, face) finds the features of the face in the img face section
dlib_shape = predictor(img, face)

###reiecves dlib shape returner for the: dlib obect conversion to numpy object(Easier calculations)
shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

### compute center of face
### np.min() find minimum value
### np.max() find max value
top_left = np.min(shape_2d, axis=0)
bottom_right = np.max(shape_2d, axis=0)

### Resizes overlay image by face image
face_size = int(max(bottom_right - top_left) * 1.8)

center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)

### Sends the ryan overlay image and resizes it
result = overlay_transparent(ori, overlay, center_x + 25, center_y - 25, overlay_size=(face_size, face_size))

### visualize face detection by rectangles
img = cv2.rectangle(img, pt1=(face.left(), face.top()), pt2=(face.right(), face.bottom()), color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

### 68 dots for face features are used to create a circle
for s in shape_2d:
cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

### Color change blue to find the top left and bottom right coordinates of the face feature
cv2.circle(img, center=tuple(top_left), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
cv2.circle(img, center=tuple(bottom_right), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

### Color change red to find the top left and bottom right coordinates of the face feature
cv2.circle(img, center=tuple((center_x, center_y)), radius=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

### showing an img on a window called 'img'
cv2.imshow('img', img)
cv2.imshow('result', result)

### waiting 1 milisecond so the video can show
cv2.waitKey(1)

#_____________________________________________________________________________________________________________________
### OpenCV(cv2) = Image process Library
### Dlib = Image process / recognition Library
### numpy = matrix multiplication library

import cv2, dlib, sys
import numpy as np

###resizing the video by scaling
scaler = 0.2

###dlib face detector module
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

### Load video cv2
### 'girl.mp4' can be replaced with 0, which will turn on the Webcam
cap = cv2.VideoCapture('eyeman.mp4')

### Load overlay Image
### Cv2.IMReadUnchanged file lets us read the file imageas a BGRA type
overlay = cv2.imread('dog.png', cv2.IMREAD_UNCHANGED)

### Overlay function + try function to adjust mask image↓
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
try:
bg_img = background_img.copy()

### convert 3 channels to 4 channels
if bg_img.shape[2] == 3:
bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

if overlay_size is not None:
img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

b, g, r, a = cv2.split(img_to_overlay_t)

mask = cv2.medianBlur(a, 5)

h, w, _ = img_to_overlay_t.shape
roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)

# convert 4 channels to 4 channels
bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)
return bg_img
except Exception : return background_img
### Overlay function ↑

### Reading the video file by frames per second
while True:
ret, img = cap.read()
if not ret:
break
### resizing the img to dsize
img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
ori = img.copy()

### detect faces in every img
faces = detector(img)
face = faces[0]

### predictor(img, face) finds the features of the face in the img face section
dlib_shape = predictor(img, face)

###reiecves dlib shape returner for the: dlib obect conversion to numpy object(Easier calculations)
shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

### compute center of face
### np.min() find minimum value
### np.max() find max value
top_left = np.min(shape_2d, axis=0)
bottom_right = np.max(shape_2d, axis=0)

### Resizes overlay image by face image
face_size = int(max(bottom_right - top_left) * 2.5)

center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)

### Sends the ryan overlay image and resizes it
result = overlay_transparent(ori, overlay, center_x - 5, center_y - 55, overlay_size=(face_size, face_size))

### visualize face detection by rectangles
img = cv2.rectangle(img, pt1=(face.left(), face.top()), pt2=(face.right(), face.bottom()), color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

### 68 dots for face features are used to create a circle
for s in shape_2d:
cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

### Color change blue to find the top left and bottom right coordinates of the face feature
cv2.circle(img, center=tuple(top_left), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
cv2.circle(img, center=tuple(bottom_right), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

### Color change red to find the top left and bottom right coordinates of the face feature
cv2.circle(img, center=tuple((center_x, center_y)), radius=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

### showing an img on a window called 'img'
cv2.imshow('img', img)
cv2.imshow('result', result)

### waiting 1 milisecond so the video can show
cv2.waitKey(1)
