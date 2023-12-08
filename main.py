import face_recognition
import cv2
import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display
from deepface import DeepFace
import lmdb

import glob
faces = []
num = 1
# for filename in glob.glob('./dtbs/*.jpg'): #assuming gif

#     ext_images = DeepFace.extract_faces(img_path = filename, detector_backend="retinaface")

#     extracted_face = ext_images[0].get('face')
#     converted_face = cv2.cvtColor(extracted_face , cv2.COLOR_BGR2RGB)
#     faces.append(converted_face)
#     cv2.imwrite("./ext_dtbs/faces_" + str(num) + ".jpg", 255 * converted_face)

#     num += 1

# Deepface check same or not

# deepface_output = DeepFace.verify('./dtbs/keanu_1.jpg', './dtbs/keanu_2.jpg')
# if (deepface_output['verified']):
#     print("./dtbs/keanu_1.jpg - ./dtbs/keanu_2.jpg" + ' Same face')
# else:
#     print("./dtbs/keanu_1.jpg - ./dtbs/keanu_2.jpg" + ' Not same face')


face_objs = DeepFace.extract_faces(img_path = "./dtbs/faces_1.jpg", detector_backend="retinaface")
face_objs_mike = DeepFace.extract_faces(img_path = "./dtbs/mike_face.jpg", detector_backend="retinaface")
face_objs_me = DeepFace.extract_faces(img_path = "./dtbs/me_1.jpeg", detector_backend="retinaface")
face_objs_keanu = DeepFace.extract_faces(img_path = "./dtbs/keanu_3.jpg", detector_backend="retinaface")

# print(len(face_objs))
# keys = list(face_objs[0].keys())
# print(keys)
# print(type(face_objs[0]))
# first_face = face_objs[0].get('face')

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

found_images = DeepFace.find('roberto.jpg', './ext_id', detector_backend="retinaface", enforce_detection= False)
print(found_images)
print(len(found_images))
print(type(found_images))
print(found_images[0])
print(type(found_images[0]))
print(type(found_images[0].empty))
print(found_images[0].empty)
print(found_images[0].get('identity'))


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

