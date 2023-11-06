import face_recognition
import cv2
import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display
from deepface import DeepFace

# Deepface check same or not
deepface_output = DeepFace.verify('keanu_1.jpg', 'keanu_2.jpg')
if (deepface_output['verified']):
    print('Same face')
else:
    print('Not same face')

face_objs = DeepFace.extract_faces(img_path = "faces_1.jpg", detector_backend="retinaface")

print(len(face_objs))
keys = list(face_objs[0].keys())
print(keys)
print(type(face_objs[0]))
first_face = face_objs[0].get('face')

faces = []
num = 1
for face in face_objs:


    extracted_face = face.get('face')
    converted_face = cv2.cvtColor(extracted_face , cv2.COLOR_BGR2RGB)
    faces.append(converted_face)
    cv2.imwrite("face_" + str(num) + ".jpg", 255 * converted_face)

    deepface_output = DeepFace.verify("face_1.jpg", 
        "face_" + str(num) + ".jpg", detector_backend="retinaface", enforce_detection= False)
    
    if (deepface_output['verified']):
        print(str(num) + ' Same face')
    else:
        print(str(num) + ' Not same face')
    
    num += 1


horizantally = np.concatenate(faces, axis=1)
cv2.imshow("faces", horizantally)

# im = cv2.cvtColor(face_objs[0].get('face') , cv2.COLOR_BGR2RGB)
# cv2.imshow("faces", im)
cv2.waitKey()

# Face recognition
# opencv_image = cv2.imread('faces_1.jpg')
# color_converted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

# face_locations = face_recognition.face_locations(color_converted)
# face_encodings = face_recognition.face_encodings(color_converted, face_locations)

# for (top, right, bottom, left) in face_locations:

#     # Draw a box around the face
#     cv2.rectangle(opencv_image, (left, top), (right, bottom), (0, 255, 255), 1)

#     # Draw a label with a name below the face
#     cv2.rectangle(opencv_image, (left, bottom - 35), (right, bottom), (0, 255, 255), cv2.FILLED)
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     cv2.putText(opencv_image, 'unknown', (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

# opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
# cv2.imshow('faces', opencv_image)

# cv2.waitKey()

# --------------------------------------------------------------------------------------------------------------------------------------------------

# from PIL import Image
# import face_recognition

# # Load the jpg file into a numpy array
# image = face_recognition.load_image_file("faces.png")

# # Find all the faces in the image using the default HOG-based model.
# # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
# # See also: find_faces_in_picture_cnn.py
# face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

# print("I found {} face(s) in this photograph.".format(len(face_locations)))

# for face_location in face_locations:

#     # Print the location of each face in this image
#     top, right, bottom, left = face_location
#     print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

#     # You can access the actual face itself like this:
#     face_image = image[top:bottom, left:right]
#     pil_image = Image.fromarray(face_image)
#     pil_image.show()

# --------------------------------------------------------------------------------------------------------------------------------------------------


# from PIL import Image, ImageDraw
# import face_recognition

# # Load the jpg file into a numpy array
# image = face_recognition.load_image_file("faces.png")

# # Find all facial features in all the faces in the image
# face_landmarks_list = face_recognition.face_landmarks(image)

# print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

# # Create a PIL imagedraw object so we can draw on the picture
# pil_image = Image.fromarray(image)
# d = ImageDraw.Draw(pil_image)

# for face_landmarks in face_landmarks_list:

#     # Print the location of each facial feature in this image
#     for facial_feature in face_landmarks.keys():
#         print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

#     # Let's trace out each facial feature in the image with a line!
#     for facial_feature in face_landmarks.keys():
#         d.line(face_landmarks[facial_feature], width=5)

# # Show the picture
# pil_image.show()