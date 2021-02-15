import numpy as np
import torch

from model import TemporalModel


NUMBER_OF_2D_KEYPOINTS = 17

#don't change yet
FRAMES_NUMBER = 3

coco_metadata = {
    'layout_name': 'coco',
    'num_joints': 17,
    'keypoints_symmetry': [
        [1, 3, 5, 7, 9, 11, 13, 15],
        [2, 4, 6, 8, 10, 12, 14, 16],
    ]
}

cam = {
    'id':None,
    'res_w': None,
    'res_h': None,

     # Dummy camera parameters (taken from Human3.6M), only for visualization purposes
    'azimuth': 70, # Only used for visualization
    'orientation': np.array([.14007056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088], dtype='float32'),
    'translation': np.array([1841.1070556640625, 4955.28466796875, 1563.4454345703125], dtype='float32')
}
cam['translation'] = cam['translation']/1000 # mm to meters

metadata = coco_metadata
video_metadata = {}
metadata['video_metadata'] = {}

indices = np.arange(NUMBER_OF_2D_KEYPOINTS)

joints_right = [1, 2, 3, 14, 15, 16]
joints_left  = [4, 5, 6, 11, 12, 13]
kps_left  = [1, 3, 5, 7, 9, 11, 13, 15]
kps_right = [2, 4, 6, 8, 10, 12, 14, 16]


def convert_coco(pose):

  y=[]
  y.append(pose[0])
  y.append(pose[14])
  y.append(pose[15])
  y.append(pose[16])
  y.append(pose[17])#4
  y.append(pose[2])
  y.append(pose[5])
  y.append(pose[3])
  y.append(pose[6])
  y.append(pose[4])
  y.append(pose[7])#10
  y.append(pose[8])
  y.append(pose[11])
  y.append(pose[9])
  y.append(pose[12])
  y.append(pose[10])
  y.append(pose[13])

  return y

def prepare_kp(points, kp_buf):

  points=[ p if p!=None else (np.nan, np.nan) for p in points]
  kpt = np.array(points,dtype=np.float).tolist() 

  coco_kp = convert_coco(kpt) # list 17x2

  # interp! using valid kps for last frame
  mask = np.isnan(coco_kp[:])[:,0]
  if len(indices[mask]) and len(kp_buf):
    coco_kp = np.array(coco_kp)
    coco_kp[indices[mask]] = kp_buf[-1][indices[mask]]
    coco_kp.tolist()
  
  return coco_kp


def normalize_screen_coordinates(X, w, h): 
  # assert X.shape[-1] == 2
  
  # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
  return X/w*2 - [1, h/w]

#not tested
def camera_to_world(X, rot, t):
  q= np.tile(rot, (*X.shape[:-1], 1))
  return qrot( q, X) + t

def qrot(q, v):
  assert q.shape[-1] == 4
  assert v.shape[-1] == 3
  assert q.shape[:-1] == v.shape[:-1]

  qvec = q[..., 1:]
  uv = np.cross(qvec, v, axis=len(q.shape)-1)
  uuv = np.cross(qvec, uv, axis=len(q.shape)-1)
  return (v + 2 * (q[..., :1] * uv + uuv))



class Skeleton3dDetector():

    def __init__(self, width=300, height=300, filename="camera"):

        self.frame_width  = width
        self.frame_height = height

        cam['res_w'] = self.frame_width
        cam['res_h'] = self.frame_height

        video_metadata['h'] = height
        video_metadata['w'] = width
        metadata['video_metadata'][filename] = video_metadata
        
        self.kp_buff = []

        self.checkpoint = torch.load('checkpoint/pretrained_h36m_detectron_coco.bin', map_location=lambda storage, loc: storage)
        self.model_pos = TemporalModel(NUMBER_OF_2D_KEYPOINTS, 2, 17, #num_joints
                                    filter_widths=[3,3,3,3,3], causal=False, dropout=0.25, channels=1024,
                                    dense=False)

        self.model_pos.load_state_dict(self.checkpoint['model_pos'])

        self.model_pos.eval()


    # input: prev + curr frames called once only
    def fill_buff(self,initial_points_2d):
        assert len(initial_points_2d) == FRAMES_NUMBER-1 # //2 +1

        for point in initial_points_2d:
            kp = prepare_kp(point, np.array(self.kp_buff))
            kp0 = np.nan_to_num(kp,0)
            self.kp_buff.append(kp0)

        self.kp_buff= normalize_screen_coordinates(np.array(self.kp_buff), w=cam['res_w'], h=cam['res_h']).tolist()
        assert len(self.kp_buff) == FRAMES_NUMBER-1 #//2 +1

        
    #input: next frames(2d kps)
    # returns skeleton for curr frame
    def detect(self,point2d):

        #check for dimention of input
        kp = prepare_kp(point2d, np.array(self.kp_buff))
        kpn = normalize_screen_coordinates(np.array([kp]), w=cam['res_w'], h=cam['res_h'])
        self.kp_buff.extend(kpn)


        with torch.no_grad():
            batch_2d = np.expand_dims(np.pad(self.kp_buff, ((121, 121), (0, 0), (0, 0)), 'edge'), axis=0)
            #augmenting (by flipping)
            batch_2d = np.concatenate((batch_2d, batch_2d), axis=0)
            batch_2d[1, :, :, 0] *= -1
            batch_2d[1, :, kps_left + kps_right] = batch_2d[1, :, kps_right + kps_left]

            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            predicted_3d_pos = self.model_pos(inputs_2d)
            #augmented
            predicted_3d_pos[1, :, :, 0] *= -1
            predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
            predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)

            prediction =predicted_3d_pos.squeeze(0).cpu().numpy()#nFrame x nJoints x 3d

        prediction = camera_to_world(prediction, rot= cam['orientation'], t=0)
        # We don't have the trajectory, but at least we can rebase the height
        prediction[:, :, 2] -= np.min(prediction[:, :, 2])#2 frames n kp on z is it 3?
        self.kp_buff = self.kp_buff[1:]
        
        return prediction[FRAMES_NUMBER//2]

