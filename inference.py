from utils import *
from models import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--video", help="Video path")
parser.add_argument("--out", help="BVP csv file path", default='')
parser.add_argument("--weights", help="HDF5 path of model weights", default='weights/m1.h5')

args = parser.parse_args()

model = M_1()
model.build(input_shape=(None, 450, 8, 8, 3))
model.load_weights(args.weights)

def vid(v):
    cap = cv2.VideoCapture(v)
    while 1:
        _, f = cap.read()
        if not _:
            break
        yield cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
v = args.video
bvp = []
f = []
n = 0
with mp.solutions.face_mesh.FaceMesh(max_num_faces=1) as fm:
    box = box_ = None
    for frame in vid(v):
        h, w, c = frame.shape
        if w>h:
            frame_ = frame[:, round((w-h)/2):round((w-h)/2)+h] #Crop the middle part of the widescreen to avoid detecting other people.
        else:
            frame_ = frame
            w = h
        if n%5==0:
            landmarks = fm.process(frame_).multi_face_landmarks
            if landmarks and len(landmarks): #If a face is detected, convert it to relative coordinates; otherwise, set all values to -1.
                landmark = np.array([(i.x*h/w+round((w-h)/2)/w, i.y) for i in landmarks[0].landmark])
                shape = alphashape.alphashape(landmark, 0)
                if box is None:
                    box = np.array(shape.bounds).reshape(2, 2)
                else:
                    w = 1/(1 + np.exp(-20*np.linalg.norm(np.array(shape.bounds).reshape(2, 2)-box)/np.multiply(*np.abs(box[0]-box[1]))))*2-1
                    box = np.array(shape.bounds).reshape(2, 2)*w+box*(1-w)
                if box_ is None:
                    box_ = np.clip(np.round(box*frame.shape[1::-1]).astype(int).T, a_min=0, a_max=None)
                elif np.linalg.norm(np.round(box*frame.shape[1::-1]).astype(int).T - box_) > frame.size/10**5:
                    box_ = np.clip(np.round(box*frame.shape[1::-1]).astype(int).T, a_min=0, a_max=None)
            else:
                landmark = np.full((468, 2), -1)
        n += 1
        if box_ is None:
            bvp.append(0)
        else:
            _ = cv2.resize(frame[slice(*box_[1]), slice(*box_[0])], (8, 8), interpolation=cv2.INTER_AREA)
            f.append(_)
f = np.array(f)
if f.shape[0]<450:
    exit()
_ = []
for i in range(0, len(f)-450, 150):
    _1 = np.full(f.shape[0], np.nan)
    _1[i:i+450] = model([f[i:i+450]/255])[0]
    _.append(_1)
_1 = np.full(f.shape[0], np.nan)
_1[-450:] = model([f[-450:]/255])
_.append(_1)
predict = np.nanmean(_, axis=0)
bvp = np.concatenate([bvp, predict])

out = args.out
if not out:
    out = v+'.csv'

with open(out, 'w') as f:
    f.write('BVP\n')
    for i in bvp:
        f.write(f'{i}\n')

print('\nBVP signal extraction completed, please use frames_timestamp.csv to synchronize it.')
