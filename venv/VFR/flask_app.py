from flask import Flask, render_template, Response
from imutils.video import VideoStream
from skeletonDetector import skeleton
import cv2


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index2.html')

def gen():

    vs = VideoStream(src=0).start()

    while cv2.waitKey(1) < 0:

        frame = vs.read()

        if frame is None:
            continue

        processed_fram,_ = skeleton(frame)

        (flag, encodedImage) = cv2.imencode(".jpg", processed_fram)

        if not flag:
            continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

@app.route('/video_feed')
def video_feed():

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8086)