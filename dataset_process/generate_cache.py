import sys
sys.path.append("..")
from utils import *
from multiprocessing import Pool
import tqdm

cores = 8

df = pd.read_csv('../datasets/CCNU_dataset_index.csv')
files_ccnu = df[(df['codec']=='MJPG')&(df['fold']>=0)]['file']

df = pd.read_csv('../datasets/PURE_dataset_index.csv')
files_pure = df['file']

df = pd.read_csv('../datasets/UBFC_rPPG2_dataset_index.csv')
files_ubfc_rppg2 = df['file']

df = pd.read_csv('../datasets/SCAMPS_dataset_index.csv')
files_scamps = df['file']

df = pd.read_csv('../datasets/UBFC_PHYS_dataset_index.csv')
files_ubfc_phys = df['file']

df = pd.read_csv('../datasets/COHFACE_dataset_index.csv')
files_cohface = df['file']

def log(s):
    with open('cache_log.txt', 'a') as f:
        f.write(s+'\n')

def cache(f, vid, overwrite=True):
    path=f
    if overwrite or not os.path.exists(f'{tmp}/{hash(f)}.h5'):
        with h5py.File(f'{tmp}/{hash(f)}.h5', 'w') as f:
            boxes, landmarks = generate_vid_labels(vid, detect_per_n=5)
            f.create_dataset('boxes', data=np.array(boxes), compression=0)
            f.create_dataset('landmarks', data=np.array(landmarks), compression=0)
            log(f'{path} cached. {len(boxes)} frames')

def cache_ccnu(f):
    vid, bvp, ts = loader_ccnu(f)
    cache(loader_ccnu.base+f, vid)

def cache_pure(f):
    vid, bvp, ts = loader_pure(f)
    cache(loader_pure.base+f, vid)

def cache_ubfc_rppg2(f):
    vid, bvp, ts = loader_ubfc_rppg2(f)
    cache(loader_ubfc_rppg2.base+f, vid)

def cache_scamps(f):
    vid, bvp, ts = loader_scamps(f)
    cache(loader_scamps.base+f, vid)

def cache_ubfc_phys(f):
    vid, bvp, ts = loader_ubfc_phys(f)
    cache(loader_ubfc_phys.base+f, vid)

def cache_cohface(f):
    vid, bvp, ts, ext = loader_cohface(f)
    cache(loader_cohface.base+f, vid)

if __name__ == '__main__':
    with Pool(cores) as p:
        #p.map(cache_ccnu, files_ccnu)
        #p.map(cache_pure, files_pure)
        #p.map(cache_ubfc_rppg2, files_ubfc_rppg2)
        #p.map(cache_scamps, files_scamps)
        #p.map(cache_ubfc_phys, files_ubfc_phys)
        p.map(cache_cohface, files_cohface)