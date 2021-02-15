
import cv2
import time
import argparse

from visualization import Visualizer
from skeleton3DDetector import Skeleton3dDetector
from skeletonDetector import skeleton


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help="path to input file", metavar='STR')
args = parser.parse_args()
input_file =args.input

vs = cv2.VideoCapture(input_file)

sk3d = Skeleton3dDetector(width = vs.get(3), height = vs.get(4))

#prev frame(s):
init_buff=[]
for _ in range(2):
    succ, frame = vs.read()
    while not succ:
        succ, frame = vs.read()
    processed_fram, x = skeleton(frame)
    init_buff.append(x)
sk3d.fill_buff(init_buff)

cv2.waitKey(0)

visualizer = Visualizer(frame.shape[0], frame.shape[1])

while cv2.waitKey(1) < 0: # breaks on pressing q
    
    t = time.time()
    succ,frame = vs.read()

    if frame is None:
        continue

    curr_frame = processed_fram

    # next frame(s):
    processed_fram, x = skeleton(frame)

    kps3d = sk3d.detect(x) # 3d kps for curr frame

    img = visualizer.draw(curr_frame,kps3d)

    cv2.putText(img, "time taken = {:.2f} sec".format(time.time() - t), (20, 20), cv2.FONT_HERSHEY_COMPLEX, .5,
            (255, 50, 0), 2, lineType=cv2.LINE_AA)
    cv2.imshow('Frame', img) 

# the video capture object 
vs.release() 
   
# Closes all the frames 
cv2.destroyAllWindows() 
