import numpy
import cv2
from PIL import Image
import os
path=r".\Dataset_face_recognization"
list_of_files=[os.path.join(path,file) for file in os.listdir(r".\Dataset_face_recognization")]
faces = []
idd=[]

for image in list_of_files:
    img=Image.open(image).convert('L')
    
    npimage=numpy.array(img)                #each image vector
    iddd=int((os.path.split(image)[1].split('.')[1]))   #each image id
    faces.append(npimage)
    idd.append(iddd)

idd=numpy.array(idd)

model = cv2.face_LBPHFaceRecognizer.create()
model.train(faces,idd)
model.write('face_clasifier.yml')
