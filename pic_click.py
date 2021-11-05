import cv2 as cv
import os

def click(dir_path):
    count=0
    while True:
        ret, frame = video.read()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        detections = cascade_face.detectMultiScale(frame_gray,
                                                scaleFactor=1.3,
                                                minNeighbors=4)

        if(len(detections) > 0):
            (x,y,w,h) = detections[0]
            frame_gray = cv.rectangle(frame_gray, (x,y), (x+w,y+h), (0,255,0), thickness=3)
            print(f'Number of faces detected = {len(detections)}')
            image = str(count) + '.jpeg'
            path = os.path.join(dir_path,image)
            cv.imwrite(path, frame)
            count += 1
                
        elif (len(detections) == 0):
            print('Number of faces detected = 0')   
        
        cv.imshow('frame', frame_gray)   
        if count>=100:
            break    
    
    video.release()
    cv.destroyAllWindows()

def create_directory(dir_name):
    new_dir = dir_name
    path = os.getcwd() 
    parent_dir = path + '/Image_Database'
    if not os.path.exists(parent_dir):
       os.mkdir(parent_dir)
       print(f"New Image Database created at {parent_dir}")
    else:
        print(f"Image Database already exists at {parent_dir}")

    new_dir_path = parent_dir + '/' + new_dir
    if not os.path.exists(new_dir_path):
        os.mkdir(new_dir_path)
        print(f'New Directory Created for {new_dir}')
        click(new_dir_path)
    else:
        print(f'Directory already exists for {new_dir}')
        print('Enter new name please!')

video = cv.VideoCapture(0)
cascade_face = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

Name = input("Enter Your Name : ")
create_directory(Name)

