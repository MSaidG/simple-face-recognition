import face_recognition
import cv2
import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display
from deepface import DeepFace
import lmdb



# Deepface check same or not



# deepface_output = DeepFace.verify('./dtbs/keanu_1.jpg', './dtbs/keanu_2.jpg')
# if (deepface_output['verified']):
#     print("./dtbs/keanu_1.jpg - ./dtbs/keanu_2.jpg" + ' Same face')
# else:
#     print("./dtbs/keanu_1.jpg - ./dtbs/keanu_2.jpg" + ' Not same face')

# deepface_output = DeepFace.verify('./dtbs/keanu_1.jpg', './dtbs/keanu_3.jpg')
# if (deepface_output['verified']):
#     print("./dtbs/keanu_1.jpg - ./dtbs/keanu_3.jpg" + ' Same face')
# else:
#     print("./dtbs/keanu_1.jpg - ./dtbs/keanu_3.jpg" + ' Not same face')

# deepface_output = DeepFace.verify('./dtbs/keanu_2.jpg', './dtbs/keanu_3.jpg')
# if (deepface_output['verified']):
#     print("./dtbs/keanu_2.jpg - ./dtbs/keanu_3.jpg" + ' Same face')
# else:
#     print("./dtbs/keanu_2.jpg - ./dtbs/keanu_3.jpg" + ' Not same face')


face_objs = DeepFace.extract_faces(img_path = "./dtbs/faces_1.jpg", detector_backend="retinaface")
face_objs_mike = DeepFace.extract_faces(img_path = "./dtbs/mike_face.jpg", detector_backend="retinaface")
face_objs_me = DeepFace.extract_faces(img_path = "./dtbs/me_1.jpeg", detector_backend="retinaface")
face_objs_keanu = DeepFace.extract_faces(img_path = "./dtbs/keanu_3.jpg", detector_backend="retinaface")

# print(len(face_objs))
# keys = list(face_objs[0].keys())
# print(keys)
# print(type(face_objs[0]))
# first_face = face_objs[0].get('face')

faces = []
num = 1
for face in face_objs:


    extracted_face = face.get('face')
    converted_face = cv2.cvtColor(extracted_face , cv2.COLOR_BGR2RGB)
    faces.append(converted_face)
    cv2.imwrite("./ext_dtbs/face_" + str(num) + ".jpg", 255 * converted_face)

    # deepface_output = DeepFace.verify("./ext_dtbs/face_1.jpg", 
    #     "./ext_dtbs/face_" + str(num) + ".jpg", detector_backend="retinaface", enforce_detection= False)
    
    # if (deepface_output['verified']):
    #     print("face_1.jpg - face_" + str(num) + ".jpg" + ' Same face')
    # else:
    #     print("face_1.jpg - face_" + str(num) + ".jpg" + ' Not same face')
    
    # num += 1


num = 1
for face in face_objs_mike:

    extracted_face = face.get('face')
    converted_face = cv2.cvtColor(extracted_face , cv2.COLOR_BGR2RGB)
    faces.append(converted_face)
    cv2.imwrite("./ext_dtbs/mike_" + str(num) + ".jpg", 255 * converted_face)

    # deepface_output = DeepFace.verify("./ext_dtbs/mike_1.jpg", 
    #     "./ext_dtbs/mike_" + str(num) + ".jpg", detector_backend="retinaface", enforce_detection= False)
    
    # if (deepface_output['verified']):
    #     print("mike_1.jpg - mike_" + str(num) + ".jpg" + ' Same face')
    # else:
    #     print("mike_1.jpg - mike_" + str(num) + ".jpg" + ' Not same face')
    
    # num += 1

num = 1
for face in face_objs_me:

    extracted_face = face.get('face')
    converted_face = cv2.cvtColor(extracted_face , cv2.COLOR_BGR2RGB)
    faces.append(converted_face)
    cv2.imwrite("./ext_id/me_" + str(num) + ".jpg", 255 * converted_face)
    cv2.imwrite("./ext_dtbs/me_" + str(num) + ".jpg", 255 * converted_face)
    
    num += 1


num = 1
for face in face_objs_keanu:

    extracted_face = face.get('face')
    converted_face = cv2.cvtColor(extracted_face , cv2.COLOR_BGR2RGB)
    faces.append(converted_face)
    cv2.imwrite("./ext_id/keanu_" + str(num) + ".jpg", 255 * converted_face)
    cv2.imwrite("./ext_dtbs/keanu_" + str(num) + ".jpg", 255 * converted_face)
    
    num += 1



deepface_output = DeepFace.verify("./ext_dtbs/face_1.jpg", 
    "./ext_dtbs/keanu_1.jpg", detector_backend="retinaface", enforce_detection= False)

if (deepface_output['verified']):
    print("face_1.jpg - keanu_1.jpg" + ' Same face')
else:
    print("face_1.jpg - keanu_1.jpg" + ' Not same face')


deepface_output = DeepFace.verify("./ext_dtbs/mike_1.jpg", 
    "./ext_dtbs/keanu_1.jpg", detector_backend="retinaface", enforce_detection= False)

if (deepface_output['verified']):
    print("mike_1.jpg - keanu_1.jpg" + ' Same face')
else:
    print("mike_1.jpg - keanu_1.jpg" + ' Not same face')

    
deepface_output = DeepFace.verify("./ext_dtbs/face_1.jpg", 
    "./ext_dtbs/mike_1.jpg", detector_backend="retinaface", enforce_detection= False)

if (deepface_output['verified']):
    print("face_1.jpg - mike_1.jpg" + ' Same face')
else:
    print("face_1.jpg - mike_1.jpg" + ' Not same face')


found_images = DeepFace.find('me_2.jpeg', './ext_id', detector_backend="retinaface", enforce_detection= False)
print(found_images)
print(len(found_images))
print(type(found_images))
print(found_images[0])
print(type(found_images[0]))
print(type(found_images[0].empty))
print(found_images[0].empty)
print(found_images[0].get('identity'))

found_images = DeepFace.find('man.jpg', './ext_id', detector_backend="retinaface", enforce_detection= False)
print(found_images)
print(len(found_images))
print(type(found_images))
print(found_images[0])
print(type(found_images[0]))
print(type(found_images[0].empty))
print(found_images[0].empty)
print(found_images[0].get('identity'))

found_images = DeepFace.find('keanu_1.jpg', './ext_id', detector_backend="retinaface", enforce_detection= False)


if (found_images[0].empty):
    print("NO ID FOUND IN DATABASE")
else:
    print("ID FOUND IN DATABASE")

    # extracted_face = face.get('face')
    # converted_face = cv2.cvtColor(extracted_face , cv2.COLOR_BGR2RGB)
# faces.append(converted_face)


horizantally = np.concatenate(faces, axis=1)
cv2.imshow("faces", horizantally)

# im = cv2.cvtColor(face_objs[0].get('face') , cv2.COLOR_BGR2RGB)
# cv2.imshow("faces", im)
cv2.waitKey()

# Face recognition
# opencv_image = cv2.imread('./dtbs/faces_1.jpg')
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