from config import *
import pandas as pd
import cv2
import time
import tqdm
import numpy as np
import heartpy as hp
#import warnings
#warnings.simplefilter("ignore", UserWarning)
from scipy.interpolate import UnivariateSpline
import mediapipe as mp
import alphashape
import hashlib, base64, os
from scipy.signal import find_peaks
import h5py
import json
import inspect
from concurrent.futures import ThreadPoolExecutor
import threading
from scipy.signal import welch, butter, lfilter
from scipy.sparse import spdiags
import scipy.io
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import random

tqdm = tqdm.tqdm
#tqdm = tqdm.tqdm_notebook


def get_hr(y, sr=30, min=30, max=180):
    p, q = welch(y, sr, nfft=1e5/sr, nperseg=np.min((len(y)-1, 256)))
    return p[(p>min/60)&(p<max/60)][np.argmax(q[(p>min/60)&(p<max/60)])]*60

def get_hrv(y, sr=30):
    # Please use videos longer than 5 minutes, such as the OBF dataset.
    p, q = welch(y, sr, nfft=1e6/sr, nperseg=np.min((len(y)-1, 512)))
    RF = p[(p>.04)&(p<.4)][np.argmax(q[(p>.04)&(p<.4)])]
    TP = q[(p>.04)&(p<.4)].sum()
    HF = q[(p>.15)&(p<.4)].sum()/TP
    LF = q[(p>.04)&(p<.15)].sum()/TP
    return RF, LF, HF, LF/HF

def detrend(signal, Lambda=25):
    def _detrend(signal, Lambda=Lambda):
        """detrend(signal, Lambda) -> filtered_signal
        This code is based on the following article "An advanced detrending method with application
        to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
        """
        signal_length = signal.shape[0]
        H = np.identity(signal_length)
        ones = np.ones(signal_length)
        minus_twos = -2 * np.ones(signal_length)
        diags_data = np.array([ones, minus_twos, ones])
        diags_index = np.array([0, 1, 2])
        D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
        filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
        return filtered_signal
    rst = np.zeros_like(signal)
    for i in np.arange(0, signal.shape[0], 900):
        if i<=signal.shape[0]-900:
            rst[i:i+900] = _detrend(signal[i:i+900])
        else:
            rst[i:] = _detrend(signal[-900:])[-(rst.shape[0]-i):]
    return rst

def bandpass_filter(data, lowcut=0.5, highcut=3, fs=30, order=3):
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')
    return lfilter(b, a, data)

def SNR(x, y):
    x, y = (x-np.expand_dims(np.mean(x, axis=-1), -1))/(np.expand_dims(np.std(x, axis=-1), -1)+1e-6), (y-np.expand_dims(np.mean(y, axis=-1), -1))/(np.expand_dims(np.std(y, axis=-1), -1)+1e-6)
    A_s = np.mean(np.abs(np.fft.rfft(x)), axis=-1)
    A_n = np.mean(np.abs(np.fft.rfft(y-x)), axis=-1)
    #A_n = np.mean(np.abs(np.fft.rfft(y)), axis=-1) - A_s
    return 8.685889*np.log((A_s/A_n)**2)
    #return (A_s/A_n)**2

def norm_bvp(bvp, order=1):
    order -= 1
    bvp_ = []
    _ = np.nan
    for i in bvp:
        if np.isnan(i):
            bvp_.append(_)
        else:
            bvp_.append(i)
            _ = i
    if np.isnan(bvp_[0]):
        for i in bvp_:
            if not np.isnan(i):
                _ = i
                break
        n = 0
        while 1:
            if n>=len(bvp_) or not np.isnan(bvp_[0]):
                break
            bvp_[n] = _
            n += 1
    bvp_ = np.array(bvp_)
    bvp_ = (bvp_-np.mean(bvp_))/np.std(bvp_)
    prominence=(1.5*np.std(bvp_), None)
    peaks = np.sort(np.concatenate([find_peaks(bvp_, prominence=prominence)[0], find_peaks(-bvp_, prominence=prominence)[0]]))
    bvp_ = np.concatenate((bvp_, [bvp_[-1]]))
    bvp_ = np.concatenate([((x-np.mean(x))/np.abs(x[0]-x[-1]))[:-1] for x in (bvp_[a:b] for a, b in zip(np.concatenate([(0,), peaks]), np.concatenate([peaks+1, (len(bvp_),)])))])
    if order == 0:
        return np.clip(bvp_, a_max=0.5, a_min=-0.5)
    else:
        return norm_bvp(bvp_, order)

hash = lambda x:base64.urlsafe_b64encode(hashlib.sha256(f'{x}'.encode()).digest()).decode()

class Loader:

    def __init__(self, base) -> None:
        self.base = base

class LoaderCOHFACE(Loader):

    def __call__(self, vid):
        path = f'{self.base}{vid}'
        lb = path[:-4]+'.hdf5'
        with h5py.File(lb, 'r') as f:
            bvp = f['pulse'][:]
            rr = f['respiration'][:]
            ts = f['time'][:]
            bvp_i = UnivariateSpline(ts, bvp, s=0)
            rr_i = UnivariateSpline(ts, rr, s=0)
            _ = cv2.VideoCapture(f"{self.base}{vid}")
            fps = _.get(cv2.CAP_PROP_FPS)
            n = _.get(cv2.CAP_PROP_FRAME_COUNT)
            _.release()
            ts = np.arange(n)/fps
            def _1():
                cap = cv2.VideoCapture(f"{self.base}{vid}")
                while 1:
                    _, frame = cap.read()
                    if _:
                        yield cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    else:
                        break
            class _2:
                def __iter__(self):
                    return _1()
            return _2(), bvp_i(ts), ts, {'rr':rr_i(ts)}
        
loader_cohface = LoaderCOHFACE(dataset_cohface)

class LoaderMMPD(Loader):

    def __call__(self, vid):
        path = f"{self.base}{vid}"
        f = scipy.io.loadmat(path)
        bvp = f['GT_ppg'][0]
        ts = np.arange(bvp.shape[0])/30 # 30fps
        vid = (f['video']*255).astype(np.uint8)
        # Load the entire video file directly, with slightly higher memory usage.
        return vid, bvp, ts
    
loader_mmpd = LoaderMMPD(dataset_mmpd)

class LoaderUBFCrPPG2(Loader):

    def __call__(self, vid):
        with open(f"{self.base}{'/'.join(vid.split('/')[:-1]+['ground_truth.txt'])}", 'r') as f:
            y = [float(x) for x in f.readline().split()]
            f.readline()
            x = [float(x) for x in f.readline().split()]
            x, y = pd.DataFrame([x, y]).T.drop_duplicates(subset=[0]).values.T
            f_i = UnivariateSpline(x-x[0], y, s=0)
            _ = cv2.VideoCapture(f"{self.base}{vid}")
            fps = _.get(cv2.CAP_PROP_FPS)
            n = _.get(cv2.CAP_PROP_FRAME_COUNT)
            _.release()
            ts = np.arange(0, n/fps, 1/fps)
            def _1():
                cap = cv2.VideoCapture(f"{self.base}{vid}")
                while 1:
                    _, frame = cap.read()
                    if _:
                        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        break
            class _2:
                def __iter__(self):
                    return _1()
            return _2(), f_i(ts), ts
        
loader_ubfc_rppg2 = LoaderUBFCrPPG2(dataset_ubfc_rppg2)

class LoaderUBFCPHYS(Loader):
    
    def __call__(self, vid):
        bvp = pd.read_csv(f"{self.base}{vid.replace('vid_', 'bvp_')}"[:-4]+".csv").values.T[0]
        ts = np.arange(len(bvp))/64
        f_i = UnivariateSpline(ts, bvp, s=0)
        _ = cv2.VideoCapture(f"{self.base}{vid}")
        fps = _.get(cv2.CAP_PROP_FPS)
        n = _.get(cv2.CAP_PROP_FRAME_COUNT)
        _.release()
        ts = np.arange(n)/fps
        def _1():
            cap = cv2.VideoCapture(f"{self.base}{vid}")
            while 1:
                _, frame = cap.read()
                if _:
                    yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    break
        class _2:
            def __iter__(self):
                return _1()
        return _2(), f_i(ts), ts
    
loader_ubfc_phys = LoaderUBFCPHYS(dataset_ubfc_phys)

class LoaderPURE(Loader):

    def __call__(self, vid):
        with open(f'{self.base}{vid}', 'r') as f:
            a = json.load(f)
        x, y = [i['Timestamp']/10**9 for i in a['/FullPackage']], [i['Value']['waveform'] for i in a['/FullPackage']]
        f_i = UnivariateSpline(x, y, s=0)
        ts = np.array([i['Timestamp']/10**9 for i in a['/Image']])
        bvp = f_i(ts)
        def _1():
            with ThreadPoolExecutor(8) as p:
                yield from p.map(lambda x:cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB), (f"{self.base}/{vid.replace('.json', '')}/Image{i['Timestamp']}.png" for i in a['/Image']))
        class _2:
            def __iter__(self):
                return _1()
        return _2(), bvp, ts
    
loader_pure = LoaderPURE(dataset_pure)

class LoaderCCNU(Loader):

    def __call__(self, vid):
        vid = f'{self.base}{vid}'
        frames_ts = '/'.join(vid.split('/')[:-1])+'/frames_timestamp.csv'
        bvp = '/'.join(vid.split('/')[:-1])+'/BVP.csv'
        ts = pd.read_csv(frames_ts)['timestamp'].values
        x, y = pd.read_csv(bvp).loc[:, ['timestamp', 'bvp']].drop_duplicates(subset=['timestamp']).values.T
        f_i = UnivariateSpline(x, y, s=0)
        bvp = f_i(pd.read_csv(frames_ts)['timestamp'].values)
        def _1():
            cap = cv2.VideoCapture(vid)
            while 1:
                _, frame = cap.read()
                if _:
                    yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    break
            cap.release()
        class _2:
            def __iter__(self):
                return _1()
        return _2(), bvp, ts
    
loader_ccnu = LoaderCCNU(dataset_ccnu)

class LoaderSCAMPS(Loader):

    def __call__(self, vid):
        path = f"{self.base}{vid}"
        with h5py.File(path, 'r') as f:
            bvp = f['d_ppg'][:].reshape(-1)
            ts = np.arange(0, bvp.shape[0]/30, 1/30)
        class _1:
            def __iter__(self):
                with h5py.File(path, 'r') as f:
                    frames = (f['RawFrames'][:].transpose(3, 2, 1, 0)*255).astype(np.uint8)
                return (_ for _ in frames)
            #迭代器有状态，因此有内存泄漏的隐患，要把frames放在__iter__中，令其返回一个生成器，用完即释放
        return _1(), bvp, ts

loader_scamps = LoaderSCAMPS(dataset_scamps)

def generate_vid_labels(vid, detect_per_n=5, bfill=True):
    boxes = []
    all_landmarks = []
    n = 0
    with mp.solutions.face_mesh.FaceMesh(max_num_faces=1) as fm:
        box = box_ = None
        for frame in vid:
            h, w, c = frame.shape
            if c>3:
                frame = frame[...,:3]
            elif c==2:
                frame = frame[...,:1]
            if w>h:
                frame_ = frame[:, round((w-h)/2):round((w-h)/2)+h] #Crop the middle part of the widescreen to avoid detecting other people.
            else:
                frame_ = frame
                w = h
            if n%detect_per_n==0:
                landmarks = fm.process(frame_).multi_face_landmarks
                if landmarks and len(landmarks): #If a face is detected, convert it to relative coordinates; otherwise, set all values to -1.
                    """
                    The box_ here is the cropped detection box, and it will only be updated when there is movement in the head.
                    The box is the result of continuous detection, updated using the EMA algorithm to maintain smooth motion. When its difference with box_ reaches a threshold, update box_.
                    If face detection loses the target, then set landmark to -1, and reuse the previous results for box and box_.
                    This is done to track the head while preventing noise introduced by facial detection jitter. In past datasets, the head was usually in a fixed position, so continuous detection was not required.
                    We have tried using Mediapipe for continuous monitoring directly, and the results are similar, but it lacks robustness and still requires additional processing for situations such as target loss.
                    """
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
            all_landmarks.append(landmark)
            if box_ is None:
                boxes.append(np.full((2, 2), -1))
            else:
                boxes.append(box_)
    if bfill:
        t = np.full((2, 2), -1)
        for i in range(len(boxes)):
            if (boxes[-i-1]==-1).any():
                boxes[-i-1] = t
            else:
                t = boxes[-i-1]
    return boxes, all_landmarks

def load_dataset(files, loader:Loader, threads=8):
    base = loader.base
    def load(f):
        _ = loader(f)
        if len(_) == 3:
            vid, bvp, ts = _
            ext = {}
        if len(_) == 4:
            vid, bvp, ts, ext = _
        """
        Cache the face detection, and if it is already cached, read it directly.
        """
        if not os.path.exists(f'{tmp}/{hash(base+f)}.h5'):
            with h5py.File(f'{tmp}/{hash(base+f)}.h5', 'w') as _:
                boxes, landmarks = generate_vid_labels(vid)
                _.create_dataset('boxes', data=np.array(boxes), compression=9)
                _.create_dataset('landmarks', data=np.array(landmarks), compression=9)
        else:
            try:
                with h5py.File(f'{tmp}/{hash(base+f)}.h5', 'r') as _:
                    boxes, landmarks = _['boxes'][:], _['landmarks'][:]
                if not boxes.shape[0]:
                    os.remove(f'{tmp}/{hash(base+f)}.h5')
                    return load(f)
            except:
                os.remove(f'{tmp}/{hash(base+f)}.h5')
                return load(f)
        return {'video':vid, 'bvp':bvp, 'timestamp':ts, 'boxes':boxes, 'landmarks': landmarks, 'extention':ext}
    with ThreadPoolExecutor(threads) as p:
        for i in p.map(load, files):
            yield i
                
def dump_dataset(target, files, loader, labels=None, resolution=(128, 128), threads=8, compression=9):
    if isinstance(files, pd.DataFrame):
        if not labels:
            labels = [files.loc[i].to_dict() for i in range(len(files))]
        else:
            labels = [{**files.loc[i].to_dict(), **labels[i]} for i in range(len(files))]
        files = list(files['file'])
    def resize(x, b):
        if np.min(b)<0 or np.isnan(b).any():
            return None
        b = np.int32(b)
        return cv2.resize(x[slice(*b[1]), slice(*b[0])], resolution[::-1], interpolation=cv2.INTER_AREA)
    print(f'Generating dataset {target} .....')
    with h5py.File(target, 'w') as f:
        def dump(x):
            i, j = x
            ts_, bvp_, cap, boxes, ext_ = list(j['timestamp']), list(j['bvp']), j['video'], np.round(j['boxes']), {k:list(v) for k, v in j['extention'].items()}
            ts, frames, bvp = [], [], []
            ext = {i:[] for i in ext_}
            n = 0
            for frame in map(lambda x:resize(x[0], x[1]), zip(cap, boxes)):
                if frame is not None:
                    frames.append(frame)
                    bvp.append(bvp_.pop(0))
                    ts.append(ts_.pop(0))
                    for k in ext:
                        ext[k].append(ext_[k].pop(0))
                elif frames:
                    _ = f.create_group(f'{i}_{n}')
                    _.create_dataset('video', data=np.array(frames), compression=compression)
                    _.create_dataset('bvp', data=np.array(bvp), compression=compression)
                    _.create_dataset('timestamp', data=np.array(ts), compression=compression)
                    for k in ext:
                        _.create_dataset(k, data=np.array(ext[k]), compression=compression)
                        ext[k] = []
                    n += 1
                    frames, bvp = [], []
            if not n:
                _ = f.create_group(f'{i}')
            else:
                _ = f.create_group(f'{i}_{n}')
            _.create_dataset('video', data=np.array(frames), compression=compression)
            _.create_dataset('bvp', data=np.array(bvp), compression=compression)
            _.create_dataset('timestamp', data=np.array(ts), compression=compression)
            for k in ext:
                _.create_dataset(k, data=np.array(ext[k]), compression=compression)
        with ThreadPoolExecutor(threads) as p:
            for i in p.map(dump, tqdm(enumerate(load_dataset(files, loader)), total=len(files))):
                pass
        with h5py.File(target, 'r+') as f:
            files = list(files)
            for i, j in f.items():
                n = int(i.split('_')[0])
                j.attrs['path'] = files[n]
                if labels:
                    for k, v in labels[n].items():
                        j.attrs[k] = v

def dump_datatape(dataset, datatape, shape=(32, 32, 32), extend_hr=(40, 150), ext_only=False, ext_fps=0, extend_rate=1, step=1, dtype=np.float16, compression=0, shuffle=True, buffer=32, sample=cv2.INTER_CUBIC, selector=lambda s:True, **kw):
    """
    shape: (frames, width, height)
    extend_hr: Perform data augmentation by scaling in time to generate more high heart rate and low heart rate samples.
    extend_rate: Compared to the original data, the proportion of augmented data
    step: Cutting step size, a cut will be performed every step seconds.
    dtype: By default, images are saved using float16.
    compression: Compression level, 0 for no compression, 9 for maximum compression. Compression will increase the read time by about 75% and save 40% space.
    selector: Filtering, can package only part of the data.
    """
    shape = shape[0], shape[2], shape[1]

    if not callable(sample):
        interpolation = sample
        sample = lambda x, y:cv2.resize(x, y, interpolation=interpolation)

    def resize(frames, waves, length):
        """
        Scale frames and waves in time to length frames.
        """
        p = frames.reshape(frames.shape[0], -1, frames.shape[-1])
        p = cv2.resize(p, (p.shape[1], length), interpolation=cv2.INTER_NEAREST).reshape(length, *frames.shape[1:])
        b = [UnivariateSpline(np.linspace(0, 1, i.shape[0]), i, s=0)(np.linspace(0, 1, length)) for i in waves]
        return p, b
    
    def selector_(s):
        for k, v in kw.items():
            if isinstance(v, str):
                v = [v]
            if k in s and str(s[k]) not in v:
                return False
        return selector(s)
    print(f'Generating datatape {datatape} .....')
    with h5py.File(dataset, 'r') as _, h5py.File(datatape, 'w') as f:
        f.create_dataset('vid', shape=(0, *shape, 3), maxshape=(None, *shape, 3), compression=compression, dtype=dtype)
        f.create_dataset('bvp', shape=(0, shape[0]), maxshape=(None, shape[0]), compression=compression, dtype=np.float32)
        f.create_dataset('bvp_normalized', shape=(0, shape[0]), maxshape=(None, shape[0]), compression=compression, dtype=np.float32)
        buffers = {'vid':[], 'bvp':[], 'bvp_normalized':[]}
        for __ in _.values():
            __ = __.keys()
            break
        ext_buffers = {i:[] for i in __ if i not in ('video', 'bvp', 'timestamp')}
        
        for j in ext_buffers:
            f.create_dataset(j, shape=(0, shape[0]), maxshape=(None, shape[0]), compression=compression, dtype=np.float32)
        for i in tqdm(_.values()):
            if not selector_(i.attrs):
                continue
            ts = i['timestamp'][:]
            if ts.shape[0]==0:
                continue
            sr = 1/(ts[1:]-ts[:-1]).mean()
            hr = get_hr(i['bvp'], sr)
            bvp_normalized = norm_bvp(i['bvp'])
            video = i['video'][:]
            if dtype != np.uint8:
                video = video/256
            if shape[1:] != video.shape[1:]:
                all_frames = np.array([sample(i, shape[:0:-1]) for i in video])
            else:
                video = all_frames
            n = 0
            while 1:
                frames = all_frames[n:n+shape[0]]
                if frames.shape[0]<shape[0]:
                    break
                if not ext_only:
                    buffers['vid'].append(frames)
                    buffers['bvp'].append(i['bvp'][n:n+shape[0]])
                    buffers['bvp_normalized'].append(bvp_normalized[n:n+shape[0]])
                    for j in ext_buffers:
                        ext_buffers[j].append(i[j][n:n+shape[0]])
                while extend_rate>0 and np.random.rand()<1/(1+1/extend_rate):
                    hr_target = np.log2(1+np.random.rand())*(extend_hr[1]-extend_hr[0])+extend_hr[0]
                    if ext_fps > 0:
                        hr = ext_fps / sr * hr
                    frames_target = max(round(shape[0]*hr_target/hr), 4) # Minimum 4 frames
                    if n+frames_target>all_frames.shape[0]:
                        break
                    _1, _2 = resize(all_frames[n:n+frames_target], [i['bvp'][n:n+frames_target], bvp_normalized[n:n+frames_target]]+[i[j][n:n+frames_target] for j in ext_buffers], shape[0])
                    buffers['vid'].append(_1)
                    buffers['bvp'].append(_2[0])
                    buffers['bvp_normalized'].append(_2[1])
                    for j, k in zip(ext_buffers, _2[2:]):
                        ext_buffers[j].append(k)
                if len(buffers['vid'])>=buffer:
                    _1, _2, _3 = np.stack(buffers['vid']), np.stack(buffers['bvp']), np.stack(buffers['bvp_normalized'])
                    f['vid'].resize(f['vid'].shape[0]+_1.shape[0], axis=0)
                    f['vid'][-_1.shape[0]:] = _1
                    f['bvp'].resize(f['bvp'].shape[0]+_2.shape[0], axis=0)
                    f['bvp'][-_2.shape[0]:] = _2
                    f['bvp_normalized'].resize(f['bvp_normalized'].shape[0]+_3.shape[0], axis=0)
                    f['bvp_normalized'][-_3.shape[0]:] = _3
                    for j, k in ext_buffers.items():
                        k = np.stack(k)
                        f[j].resize(f[j].shape[0]+k.shape[0], axis=0)
                        f[j][-k.shape[0]:] = k
                    for _ in buffers.values():
                        _.clear()
                    for _ in ext_buffers.values():
                        _.clear()
                n += round(sr*step)
        if buffers['vid']:
            _1, _2, _3 = np.stack(buffers['vid']), np.stack(buffers['bvp']), np.stack(buffers['bvp_normalized'])
            f['vid'].resize(f['vid'].shape[0]+_1.shape[0], axis=0)
            f['vid'][-_1.shape[0]:] = _1
            f['bvp'].resize(f['bvp'].shape[0]+_2.shape[0], axis=0)
            f['bvp'][-_2.shape[0]:] = _2
            f['bvp_normalized'].resize(f['bvp_normalized'].shape[0]+_3.shape[0], axis=0)
            f['bvp_normalized'][-_3.shape[0]:] = _3
            for j, k in ext_buffers.items():
                k = np.stack(k)
                f[j].resize(f[j].shape[0]+k.shape[0], axis=0)
                f[j][-k.shape[0]:] = k
        f.create_dataset('index', data=np.random.permutation(f['vid'].shape[0]) if shuffle else np.arange(f['vid'].shape[0]))

def dump_datatape_gray(dataset, datatape, shape=(32, 32, 32), extend_hr=(40, 150), ext_only=False, ext_fps=0, extend_rate=1, step=1, dtype=np.float16, compression=0, shuffle=True, buffer=32, sample=cv2.INTER_CUBIC, selector=lambda s:True, **kw):
    """
    shape: (frames, width, height)
    extend_hr: Perform data augmentation by scaling in time to generate more high heart rate and low heart rate samples.
    extend_rate: Compared to the original data, the proportion of augmented data
    step: Cutting step size, a cut will be performed every step seconds.
    dtype: By default, images are saved using float16.
    compression: Compression level, 0 for no compression, 9 for maximum compression. Compression will increase the read time by about 75% and save 40% space.
    selector: Filtering, can package only part of the data.
    """
    shape = shape[0], shape[2], shape[1]

    if not callable(sample):
        interpolation = sample
        sample = lambda x, y:cv2.resize(x, y, interpolation=interpolation)

    def resize(frames, waves, length):
        """
        Scale frames and waves in time to length frames.
        """
        p = frames.reshape(frames.shape[0], -1, frames.shape[-1])
        p = cv2.resize(p, (p.shape[1], length), interpolation=cv2.INTER_NEAREST).reshape(length, *frames.shape[1:])
        b = [UnivariateSpline(np.linspace(0, 1, i.shape[0]), i, s=0)(np.linspace(0, 1, length)) for i in waves]
        return p, b
    
    def selector_(s):
        for k, v in kw.items():
            if isinstance(v, str):
                v = [v]
            if k in s and str(s[k]) not in v:
                return False
        return selector(s)
    print(f'Generating datatape {datatape} .....')
    with h5py.File(dataset, 'r') as _, h5py.File(datatape, 'w') as f:
        f.create_dataset('vid', shape=(0, *shape), maxshape=(None, *shape), compression=compression, dtype=dtype)
        f.create_dataset('bvp', shape=(0, shape[0]), maxshape=(None, shape[0]), compression=compression, dtype=np.float32)
        f.create_dataset('bvp_normalized', shape=(0, shape[0]), maxshape=(None, shape[0]), compression=compression, dtype=np.float32)
        buffers = {'vid':[], 'bvp':[], 'bvp_normalized':[]}
        for __ in _.values():
            __ = __.keys()
            break
        ext_buffers = {i:[] for i in __ if i not in ('video', 'bvp', 'timestamp')}
        
        for j in ext_buffers:
            f.create_dataset(j, shape=(0, shape[0]), maxshape=(None, shape[0]), compression=compression, dtype=np.float32)
        for i in tqdm(_.values()):
            if not selector_(i.attrs):
                continue
            ts = i['timestamp'][:]
            if ts.shape[0]==0:
                continue
            sr = 1/(ts[1:]-ts[:-1]).mean()
            hr = get_hr(i['bvp'], sr)
            bvp_normalized = norm_bvp(i['bvp'])
            video = i['video'][:]
            if dtype != np.uint8:
                video = video/256
            if shape[1:] != video.shape[1:]:
                all_frames = np.array([sample(i, shape[:0:-1]) for i in video])
            else:
                video = all_frames
            n = 0
            while 1:
                frames = all_frames[n:n+shape[0]]
                if frames.shape[0]<shape[0]:
                    break
                if not ext_only:
                    buffers['vid'].append(frames)
                    buffers['bvp'].append(i['bvp'][n:n+shape[0]])
                    buffers['bvp_normalized'].append(bvp_normalized[n:n+shape[0]])
                    for j in ext_buffers:
                        ext_buffers[j].append(i[j][n:n+shape[0]])
                while extend_rate>0 and np.random.rand()<1/(1+1/extend_rate):
                    hr_target = np.log2(1+np.random.rand())*(extend_hr[1]-extend_hr[0])+extend_hr[0]
                    if ext_fps > 0:
                        hr = ext_fps / sr * hr
                    frames_target = max(round(shape[0]*hr_target/hr), 4) # Minimum 4 frames
                    if n+frames_target>all_frames.shape[0]:
                        break
                    _1, _2 = resize(all_frames[n:n+frames_target], [i['bvp'][n:n+frames_target], bvp_normalized[n:n+frames_target]]+[i[j][n:n+frames_target] for j in ext_buffers], shape[0])
                    buffers['vid'].append(_1)
                    buffers['bvp'].append(_2[0])
                    buffers['bvp_normalized'].append(_2[1])
                    for j, k in zip(ext_buffers, _2[2:]):
                        ext_buffers[j].append(k)
                if len(buffers['vid'])>=buffer:
                    _1, _2, _3 = np.stack(buffers['vid']), np.stack(buffers['bvp']), np.stack(buffers['bvp_normalized'])
                    f['vid'].resize(f['vid'].shape[0]+_1.shape[0], axis=0)
                    f['vid'][-_1.shape[0]:] = _1
                    f['bvp'].resize(f['bvp'].shape[0]+_2.shape[0], axis=0)
                    f['bvp'][-_2.shape[0]:] = _2
                    f['bvp_normalized'].resize(f['bvp_normalized'].shape[0]+_3.shape[0], axis=0)
                    f['bvp_normalized'][-_3.shape[0]:] = _3
                    for j, k in ext_buffers.items():
                        k = np.stack(k)
                        f[j].resize(f[j].shape[0]+k.shape[0], axis=0)
                        f[j][-k.shape[0]:] = k
                    for _ in buffers.values():
                        _.clear()
                    for _ in ext_buffers.values():
                        _.clear()
                n += round(sr*step)
        if buffers['vid']:
            _1, _2, _3 = np.stack(buffers['vid']), np.stack(buffers['bvp']), np.stack(buffers['bvp_normalized'])
            f['vid'].resize(f['vid'].shape[0]+_1.shape[0], axis=0)
            f['vid'][-_1.shape[0]:] = _1
            f['bvp'].resize(f['bvp'].shape[0]+_2.shape[0], axis=0)
            f['bvp'][-_2.shape[0]:] = _2
            f['bvp_normalized'].resize(f['bvp_normalized'].shape[0]+_3.shape[0], axis=0)
            f['bvp_normalized'][-_3.shape[0]:] = _3
            for j, k in ext_buffers.items():
                k = np.stack(k)
                f[j].resize(f[j].shape[0]+k.shape[0], axis=0)
                f[j][-k.shape[0]:] = k
        f.create_dataset('index', data=np.random.permutation(f['vid'].shape[0]) if shuffle else np.arange(f['vid'].shape[0]))

def load_datatape(path, shuffle=0, use_normalized_bvp=True, load_ext=False, buffer=256, gnoise=0, batch=0, dtype=np.float16):
    with h5py.File(path, 'r') as f:
        if shuffle == 0:
            index = np.arange(f['index'].shape[0])
        elif shuffle == -1:
            index = f['index'][:]
        else:
            index = np.arange(f['index'].shape[0]) 
            index_ = np.random.permutation(index[:-(index.shape[0]%buffer)].reshape(-1, buffer)).reshape(-1)
            index[:index_.shape[0]] = index_
        shape = f['vid'].shape[1:]
        if load_ext:
            ext_keys = [i for i in f.keys() if i not in ('index', 'vid', 'bvp', 'bvp_normalized')]

    def async_iter(total, buffer=buffer):
        def _1(g):
            def _2(*args, **kw):
                sp1 = threading.Semaphore(buffer)
                sp2 = threading.Semaphore(0)
                n, b = total, []
                def _3():
                    for i in g(*args, **kw):
                        sp1.acquire()
                        b.append(i)
                        sp2.release()
                threading.Thread(target=_3, daemon=True).start()
                while n:
                    sp2.acquire()
                    yield b.pop(0)
                    n -= 1
                    if len(b)<buffer:
                        sp1.release()
            return _2
        return _1

    @async_iter(total=len(index), buffer=buffer)
    def _1():
        @async_iter(total=(len(index)-1)//buffer+1, buffer=2)
        def _2():
            with h5py.File(path, 'r') as f:
                vid, bvp = f['vid'], f['bvp_normalized'] if use_normalized_bvp else f['bvp']
                if load_ext:
                    ext = {i:f[i] for i in ext_keys}
                def _3(i):
                    index_arg = np.argsort(index[i:i+buffer])
                    index_T = np.argsort(index_arg)
                    vid_, bvp_ = vid[index[i:i+buffer][index_arg]][index_T], bvp[index[i:i+buffer][index_arg]][index_T]
                    if load_ext and ext_keys:
                        ext_ = {k:v[index[i:i+buffer][index_arg]][index_T] for k, v in ext.items()}
                        return vid_, bvp_, ext_
                    return vid_, bvp_
                yield from map(_3, np.arange(0, index.shape[0], buffer))
        for _ in _2():
            if not load_ext:
                vid_, bvp_ = _
            else:
                vid_, bvp_, ext_ = _
            if vid_.dtype == np.uint8:
                vid_ = (vid_/255.).astype(dtype)
            else:
                vid_ = vid_.astype(dtype)
            if gnoise>0:
                vid_ += np.random.normal(size=vid_.shape, scale=gnoise/255).astype(dtype)
            if not load_ext:
                yield from map(lambda i:(vid_[i], (lambda x:(x-x.mean())/(x.std()+1e-6))(bvp_[i]).astype(dtype)), range(min(vid_.shape[0], buffer)))
            else:
                yield from map(lambda i:(vid_[i], (lambda x:(x-x.mean())/(x.std()+1e-6))(bvp_[i]).astype(dtype), {k:v[i] for k, v in ext_.items()}), range(min(vid_.shape[0], buffer)))
    
    def _0(n):
        def f():
            t = [], []
            for i, j in _1():
                t[0].append(i), t[1].append(j)
                if len(t[0])==n:
                    yield np.stack(t[0]), np.stack(t[1])
                    t[0].clear(), t[1].clear()
            if len(t[0]):
                yield np.stack(t[0]), np.stack(t[1])
        return f()
    
    class _:

        @property
        def shape(self):
            return shape
        
        def __len__(self):
            return index.shape[0]

        def __iter__(self):
            if batch:
                return _0(batch)
            return _1()
        
    return _()

def eval_on_dataset(dataset, model, input_frames, input_resolution, output='BVP', fps=None, step=1, save='result.h5', batch=4, cumsum=False, sample=cv2.INTER_AREA, ipt_dtype=np.float32, selector=lambda s:True, **kw):
    def selector_(s):
        for k, v in kw.items():
            if isinstance(v, str):
                v = [v]
            if k in s and str(s[k]) not in v:
                return False
        return selector(s)
    
    def resize(frames, waves, length):
        """
        Scale frames and waves in time to length frames.
        """
        p = frames.reshape(frames.shape[0], -1, frames.shape[-1])
        p = cv2.resize(p, (p.shape[1], length), interpolation=cv2.INTER_NEAREST).reshape(length, *frames.shape[1:])
        b = [UnivariateSpline(np.linspace(0, 1, i.shape[0]), i, s=0)(np.linspace(0, 1, length)) for i in waves]
        return p, b
    
    if not callable(sample):
        sample_h = hash(sample)
        interpolation = sample
        sample = lambda x, y:cv2.resize(x, y, interpolation=interpolation)
    else:
        sample_h = inspect.getsource(sample)


    with h5py.File(dataset, 'r') as fi, h5py.File(save, 'w') as fo:
        fo.attrs['model'] = model.name if hasattr(model, 'name') else ''
        fo.attrs['time'] = time.time()
        fo.attrs['dataset'] = os.path.abspath(dataset)
        fo.attrs['output'] = output
        for i, j in tqdm(fi.items()):
            attrs = {**j.attrs}
            if not selector_(attrs):
                continue
            h = hash((j.attrs['path'], dataset, input_resolution, sample_h))
            try:
                vid = np.load(f'{tmp}/{h}.npy')
            except:
                if os.path.exists(f'{tmp}/{h}.npy'):
                    os.remove(f'{tmp}/{h}.npy')
                vid = j['video'][:]
                if vid.dtype == np.uint8:
                    vid = vid/255
                if j['video'].shape[2:0:-1] != input_resolution:
                    vid = np.stack([sample(_, input_resolution[::-1]) for _ in vid]).astype(ipt_dtype)
                    np.save(f'{tmp}/{h}.npy', vid)
                    
            ofps = 1/(j['timestamp'][1:]-j['timestamp'][:-1]).mean()
            if not fps:
                fps = ofps
            else:
                vid, (bvp, ts) = resize(vid.astype(np.float32), [j['bvp'][:], j['timestamp'][:]], round(vid.shape[0]*fps/ofps))
                vid, j = vid.astype(ipt_dtype), {'bvp':bvp, 'timestamp':ts}
            
            result, ipt = [], []
            for idx in np.arange(0, vid.shape[0]-input_frames+fps*step, fps*step).astype(int):
                idx = min(idx, vid.shape[0]-input_frames)
                if idx>=0:
                    ipt.append(((idx,idx+input_frames), vid[idx:idx+input_frames]))
            if batch>0:
                for idx in np.arange(0, len(ipt), batch):
                    o = model(np.stack([i[1] for i in ipt[idx:idx+batch]]).astype(ipt_dtype))
                    n = 0
                    for (i0, i1), vid in ipt[idx:idx+batch]:
                        _ = np.full(j['bvp'].shape, np.nan)
                        _[i0:i1] = o[n]
                        result.append(_)
                        n += 1
            else:
                for (i0, i1), vid in ipt:
                    _ = np.full(j['bvp'].shape, np.nan)
                    _[i0:i1] = np.reshape(model(vid), (-1, ))
                    result.append(_)
            if not result:
                continue
            predict = np.nanmean(result, axis=0)
            if cumsum:
                predict = detrend(np.cumsum(predict))
            label = j['bvp'][:]
            _ = fo.create_group(i)
            for k in attrs:
                _.attrs[k] = attrs[k]
            if output=='BVP':
                _.attrs['SNR'] = SNR(label, predict)
            _.create_dataset('predict', data=predict, dtype=np.float32)
            _.create_dataset('label', data=label)
            _.create_dataset('timestamp', data=j['timestamp'])

mae, rmse, R = lambda r:np.mean([abs(i[0]-i[1]) for i in r]), lambda r:np.mean([(i[0]-i[1])**2 for i in r])**0.5, lambda r:np.corrcoef(np.array(r).T)[0, 1]

def get_metrics(result='result.h5', window=30, step=10, use_filter=True, selector=lambda s:True, **kw):
    def selector_(s):
        for k, v in kw.items():
            if isinstance(v, str):
                v = [v]
            if k in s and str(s[k]) not in v:
                return False
        return selector(s)
    r, r_m = [], []
    with h5py.File(result, 'r') as f:
        if 'output' in f.attrs and f.attrs['output'] == 'HR':
            #predict_hr = lambda x, **kw :np.median(x)
            predict_hr = lambda x, **kw :x.mean()
            use_filter = False
        else:
            predict_hr = get_hr
        for i, j in f.items(): 
            if not selector_(j.attrs):
                continue
            fps = 1/(j['timestamp'][1:] - j['timestamp'][:-1]).mean()
            label = j['label'][:]
            predict = j['predict'][:]
            if use_filter:
                predict = bandpass_filter(predict, fs=fps)
                #label = bandpass_filter(label, fs=fps)
            r_m.append((get_hr(label, sr=fps), predict_hr(predict, sr=fps)))
            for t in np.arange(0, label.shape[0]-(window*fps)//2, step*fps).astype(int):
                r.append((get_hr(label[t:t+round(window*fps)], sr=fps), predict_hr(predict[t:t+round(window*fps)], sr=fps)))
    return {'Sliding window': {'MAE':round(mae(r), 3), 'RMSE':round(rmse(r), 3), 'R':round(R(r), 5)}, 'Whole video': {'MAE':round(mae(r_m), 3), 'RMSE':round(rmse(r_m), 3), 'R':round(R(r_m), 5)}}

def get_metrics_HRV(result='result.h5', use_filter=True, selector=lambda s:True, **kw):
    def selector_(s):
        for k, v in kw.items():
            if isinstance(v, str):
                v = [v]
            if k in s and str(s[k]) not in v:
                return False
        return selector(s)
    SDNN = []
    with h5py.File(result, 'r') as f:
        for i, j in f.items(): 
            if not selector_(j.attrs):
                continue
            fps = 1/(j['timestamp'][1:] - j['timestamp'][:-1]).mean()
            label = j['label'][:]
            predict = j['predict'][:]
            if use_filter:
                """
                Butterworth filters can change the peak position, which may have a negative impact on HRV estimation based on peak detection.
                However, when the noise is large, this is the only choice and whether to use a filter should be decided according to needs.
                """
                predict = bandpass_filter(predict, fs=fps)
                #label = bandpass_filter(label, fs=fps)
            r1, r2 = hp.process(label, fps)[1], hp.process(predict, fps)[1]
            if np.isnan(r1['sdnn']) or np.isnan(r2['sdnn']):
                continue
            SDNN.append((r1['sdnn'], r2['sdnn']))
    return {'SDNN':{'MAE':round(mae(SDNN), 3), 'RMSE':round(rmse(SDNN), 3), 'R':round(R(SDNN), 5)},}
            

    
    