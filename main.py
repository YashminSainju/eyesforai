import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui

fgbg = cv2.createBackgroundSubtractorMOG2()

def draw_lines(img, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0],coords[1]), (coords[2],coords[3]), [255,255,255], 3)
    except:
        pass


def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def process_img(original_image):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    processed_img = cv2.GaussianBlur(processed_img, (3,3), 0 )
    vertices = np.array([[10,500],[10,300], [300,200], [500,200], [800,300], [800,500]], np.int32)
    processed_img = roi(processed_img, [vertices])

    #                       edges
    lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180, np.array([]), 20, 15)
    draw_lines(processed_img,lines)
    return processed_img

def bg_extract(original_image):
    fgmask = fgbg.apply(original_image)
    return fgmask

def haar_cascade(gray,img):
    drone_cascade = cv2.CascadeClassifier('dronecascade4.xml')
    warrior_cascade = cv2.CascadeClassifier('warriorcascade3.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    drones = drone_cascade.detectMultiScale(gray, 50, 50)
    warrior = warrior_cascade.detectMultiScale(gray,50,50)

    for (x,y,w,h) in drones:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        font  = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Drone',(x-w,y-h),font,0.5,(11,255,255),2,cv2.LINE_AA)
        
        #roi_gray = gray[y:y+h, x:x+w]
        #roi_color = img[y:y+h, x:x+w]

    for (x,y,w,h) in warrior:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        font  = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Drone',(x-w,y-h),font,0.5,(11,255,255),2,cv2.LINE_AA)#
        #roi_gray = gray[y:y+h, x:x+w]
        #roi_color = img[y:y+h, x:x+w]
        

def main():
    last_time = time.time()
    while(True):
        screen =  np.array(ImageGrab.grab(bbox=(0,40, 800, 640)))
        screen = cv2.cvtColor(screen,cv2.COLOR_BGR2RGB)
        new_screen = process_img(screen)
        haar_screen = haar_cascade(new_screen,screen)
        bg_screen = bg_extract(new_screen)
        #print('Loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        cv2.imshow('window', new_screen)
        cv2.imshow('sorted', screen)
        cv2.imshow('background extraction', bg_screen)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
main()
