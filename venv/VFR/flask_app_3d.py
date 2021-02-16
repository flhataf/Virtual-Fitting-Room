# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 00:31:35 2021

@author: RayaBit
"""

from flask import Flask, render_template, Response
from imutils.video import VideoStream
from skeletonDetector import skeleton
import cv2
from skeleton3DDetector import Skeleton3dDetector
from visualization import Visualizer
import time

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index2.html')

def gen():

    vs = VideoStream(src=0).start()
    #vs = cv2.VideoCapture("kid3.mp4")

    # sk3d = Skeleton3dDetector(width = vs.get(3), height = vs.get(4))#width and height?
    # 320, 240
    sk3d = Skeleton3dDetector(width = 320, height = 240)#width and height?

    init_buff=[]
    for _ in range(2):
        frame = vs.read()
        while frame is None:
            frame = vs.read()
        processed_fram, x = skeleton(frame)
        init_buff.append(x)
    sk3d.fill_buff(init_buff)
    
    cv2.waitKey(0)
    #prev_frame = processed_fram
    
    visualizer = Visualizer(frame.shape[0], frame.shape[1])
    while cv2.waitKey(1) < 0: # breaks on pressing q
    
        t = time.time()
        frame = vs.read()
    
        if frame is None:
            continue
    
        prev_frame = processed_fram #curr actually
    
        processed_fram, x = skeleton(frame)
    
        kps3d = sk3d.detect(x) # buff is initialized fr3d are kps3d for prev_frame
        
        img = visualizer.draw(prev_frame,kps3d)
    
        cv2.putText(img, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .5,
                (255, 50, 0), 2, lineType=cv2.LINE_AA)

        (flag, encodedImage) = cv2.imencode(".jpg", img)

        if not flag:
            continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')
        
@app.route('/video_feed')
def video_feed():

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8087)