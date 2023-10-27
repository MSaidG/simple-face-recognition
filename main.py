import face_recognition
import cv2
import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display

opencv_image = cv2.imread('faces.png')
color_converted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)


face_locations = face_recognition.face_locations(color_converted)
face_encodings = face_recognition.face_encodings(color_converted, face_locations)
# face_landmarks_list = face_recognition.face_landmarks(color_converted)

for (top, right, bottom, left) in face_locations:

    # Draw a box around the face
    cv2.rectangle(opencv_image, (left, top), (right, bottom), (0, 255, 255), 1)

    # Draw a label with a name below the face
    cv2.rectangle(opencv_image, (left, bottom - 35), (right, bottom), (0, 255, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(opencv_image, 'unknown', (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
cv2.imshow('faces', opencv_image)

cv2.waitKey()