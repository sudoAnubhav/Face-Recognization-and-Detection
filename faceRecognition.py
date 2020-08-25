import cv2
import os
import numpy as np

def faceDetection(test_img):
    gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY) # as color image is not helpful 
                                                        # and inc processing time
    face_haar_cascade=cv2.CascadeClassifier('/home/anubhav/Desktop/project/FaceRecognition-master/HaarCascade/haarcascade_frontalface_default.xml')
    #location of cascade file for y=1 faces 
    
    faces=face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.32,minNeighbors=5)
    
    #multiscale return retangle on face
    #saclefactor to scale the image for ex 1.32 is reduce by 32%
    #minneighbour is one true positve for 5 nearby false positive
    return faces,gray_img
    
def labels_for_training_data(directory):   # To create labels for our training data
    faces=[]
    faceID=[]
    
    for path,subdirnames,filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):        
                print("Skipping system files")      #to skip files which are unusables
                continue
                
            id=os.path.basename(path)  #give basename of path in our case 0 or 1
            img_path=os.path.join(path,filename) # join pagth with filename
            print("img_path",img_path)
            print("id: ",id)
            test_img=cv2.imread(img_path)
            if test_img is None:
                print("image is not loaded properly")
                continue
            faces_rect,gray_img=faceDetection(test_img)
            if (len(faces_rect)!=1):
                continue    #since in training example we have to feed only one faceDetection
            
            (x,y,w,h) =faces_rect[0]
            roi_gray=gray_img[y:y+w,x:x+h]   #only useful is faces
            faces.append(roi_gray)           #all faces          
            faceID.append(int(id))           #all id's
    return faces,faceID

def train_classifier(faces,faceID):  # To train our classifier
    face_recognizer=cv2.face.LBPHFaceRecognizer_create() #local binary pattern histogram in this 3*3 pixel is taken (read from web)
    face_recognizer.train(faces,np.array(faceID))  #train on the data,this data takes as numpy array 
    return face_recognizer


def draw_rect(test_img,face):      # to draw rectangle on face
    (x,y,w,h)=face
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(136, 176, 75),thickness=2)

def put_text(test_img,text,x,y):    # to put text on image
    cv2.putText(test_img,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,1,(247, 202, 201),1)  # coordinate where we have to display ,Fonttype,size,color,thickness
                      
                    
            
