from ultralytics import YOLO
import cv2
# import urllib.request
import serial.tools.list_ports
import serial
# import numpy as np
from time import sleep

ports = serial.tools.list_ports.comports()
serialInst = serial.Serial()

serialInst.baudrate = 9600
serialInst.port = "COM8"
serialInst.open()

model = YOLO('best.pt')
cap = cv2.VideoCapture(0)

while True:
    try:
        ret, frame = cap.read()
        
        results = model.track(frame)
        
        for box in results[0].boxes:
            class_id = int(box.cls)  # Get class ID
            class_label = results[0].names[class_id]  # Get class label from class ID
            print(f'Detected class: {class_label}')  # Print class label
        

        if "organic" in class_label:
            angle = "0"
            serialInst.write(angle.encode('utf-8'))
            sleep(5)
        elif "glass" in class_label:
            angle = "90"
            serialInst.write(angle.encode('utf-8'))
            sleep(5)
        
        # elif class_label == "paper":
        #     angle = "180"
        #     serialInst.write(angle.encode('utf-8'))
        #     sleep(5)
        
        # elif class_label == "paper":
        #     angle = "270"
        #     serialInst.write(angle.encode('utf-8'))
        #     sleep(5)


        class_label = ""

        anno = results[0].plot()
        cv2.imshow('', anno)
        
               

        cv2.waitKey(1)
    except Exception as ex:
        print(ex)