import cv2
import os

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the input image
img = cv2.imread( 'test.jpg')

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

print("[INFO] Found {0} Faces.".format(len(faces)))
i=0

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    i=i+1
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_color = img[y:y + h, x:x + w]
    print("[INFO] Object Found. Saving Locally.")
    
    #directory
    #directory = "Extracted_Faces"
    #parent directory path
    #parent_dir = "C:/Users/SAI/Desktop/my project/Extracted_Faces/"
    #path
    #path = os.path.join(parent_dir)
    #Creation of directory
    #os.mkdir(path)
    #print("Directory '% s' Created" % directory)
    
    cv2.imwrite('Extracted_Faces/faces{}.jpg'.format(i), roi_color)

status = cv2.imwrite('faces_detected.jpg',img)
print("[INFO] img faces_detected.jpg written to filesystem: ",status)

# Display the output
cv2.imshow('img', img)
cv2.waitKey()
