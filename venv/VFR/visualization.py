#import matplotlib
#matplotlib.use('Agg')

from matplotlib.animation import FuncAnimation, writers
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cv2
import numpy as np



skeleton_parents = [0,1,2,0,4,5,0,7,8,9,8,11,12,8,14,15] #h36m
#skeleton_parents = [0,0,1,2,0,0,5,6,7,8,0,0,11,12,13,14]

global keypoints_3d,scat3d, lines3d

global im, images

def draw_lines(sk):
    if sk.shape[-1]==3:
        xlines = [[i, j] for i,j in zip(sk[1:,0] , sk[skeleton_parents,0])]
        ylines = [[i, j] for i,j in zip(sk[1:,1] , sk[skeleton_parents,1])]
        zlines = [[i, j] for i,j in zip(sk[1:,2] , sk[skeleton_parents,2])]
        return xlines,ylines,zlines
    else:
        xlines = [[i, j] for i,j in zip(sk[1:,0] , sk[skeleton_parents,0])]
        ylines = [[i, j] for i,j in zip(sk[1:,1] , sk[skeleton_parents,1])]
        return xlines, ylines, None


def animate(i): 
    global images

    global keypoints_3d, scat3d, lines3d
    scat3d._offsets3d = (keypoints_3d[i][:,0], keypoints_3d[i][:,1], keypoints_3d[i][:,2])

    image = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB) 
    im.set_array(image)
    xlines,ylines,zlines = draw_lines(keypoints_3d[i])
    for i, (x,y,z) in enumerate(zip(xlines, ylines, zlines)):
        lines3d[i].set_data(x, y)
        lines3d[i].set_3d_properties(z)
        
 #   for i in range(NUMBER_OF_2D_KEYPOINTS):
  #      ax.text(results[0][i,0],results[0][i,1],results[0][i,2],  '%s' % (str(i)), size=11, zorder=1,  color='k') 
    artists = [scat3d, im]
    artists.extend(lines3d)
    return artists

def visualize(results, output):

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(elev=15., azim=70)
    
    radius=1.7
    ax.set_xlim3d([-radius/2, radius/2])
    ax.set_zlim3d([0, radius])
    ax.set_ylim3d([-radius/2, radius/2])
    
    global keypoints_3d, scat3d, lines3d
    keypoints_3d = results.copy()
    scat3d = ax.scatter(*keypoints_3d[0].T) #sk[:,0],sk[:,1],sk[:,2])
    lines3d = []
    xlines,ylines,zlines = draw_lines(keypoints_3d[0])
    for x,y,z in zip(xlines,ylines,zlines):
        lines3d.extend(ax.plot(x,y,z,color='red'))

    anim = FuncAnimation(fig, animate, interval=30, blit=True)

    Writer = writers['ffmpeg']
    writer = Writer(metadata={}, bitrate=3000)
    anim.save(output, writer=writer)


# input all frames + kps
# writes the results in a file specified by filename
def visualize_all(image_2d, results, filename):
    global im, images
    fig = plt.figure()

    ax_img = fig.add_subplot(1,2,1)

    image = cv2.cvtColor(image_2d[0], cv2.COLOR_BGR2RGB) 
    im = ax_img.imshow(image)
    
    images = image_2d
    
    ax = fig.add_subplot(1,2,2, projection='3d')
    ax.view_init(elev=15., azim=70)
    radius=1.7
    ax.set_xlim3d([-radius/2, radius/2])
    ax.set_zlim3d([0, radius])
    ax.set_ylim3d([-radius/2, radius/2])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    image = cv2.cvtColor(image_2d[0], cv2.COLOR_BGR2RGB) 
    im = ax_img.imshow(image)
    images = image_2d

    global keypoints_3d, scat3d, lines3d
    keypoints_3d = results.copy()
    scat3d = ax.scatter(*keypoints_3d[0].T) #sk[:,0],sk[:,1],sk[:,2])
    lines3d = []
    xlines,ylines,zlines = draw_lines(keypoints_3d[0])
    for x,y,z in zip(xlines,ylines,zlines):
        lines3d.extend(ax.plot(x,y,z,color='red'))

    anim = FuncAnimation(fig, animate,frames=len(images), interval=30, blit=True)

    Writer = writers['ffmpeg']
    writer = Writer(metadata={}, bitrate=3000)
    anim.save(filename, writer=writer)




plt.ioff()
class Visualizer():
    
    def __init__(self, frame_width, frame_height ):
        
        DPI = 96
        
        self.width  = frame_width  // DPI * 2
        self.height = frame_height // DPI
        self.fig = plt.figure(figsize=(self.width, self.height))
        plt.ion()
        
        self.ax_img = self.fig.add_subplot(1,2,1)
        self.ax = self.fig.add_subplot(1,2,2, projection='3d')
        self.ax.view_init(elev=15., azim=70)
        radius=1.7
        self.ax.set_xlim3d([-radius/2, radius/2])
        self.ax.set_zlim3d([0, radius])
        self.ax.set_ylim3d([-radius/2, radius/2])
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_zticklabels([])
        self.lines3d = []
        self.im = None;
        
    
    # input : one frame + kp3d
    # return one image
    def draw(self, frame, kps3d):

        if self.im is None:    
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            self.im = self.ax_img.imshow(frame)
            
            xlines,ylines,zlines = draw_lines(kps3d)
            for x,y,z in zip(xlines,ylines,zlines):
                self.lines3d.extend(self.ax.plot(x,y,z,color='red'))
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            self.im.set_data(frame)
    
            xlines,ylines,zlines = draw_lines(kps3d)
            for i, (x,y,z) in enumerate(zip(xlines, ylines,zlines)):
                self.lines3d[i].set_data(x, y)
                self.lines3d[i].set_3d_properties(z)
    
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img  = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    
        return img
    
    

