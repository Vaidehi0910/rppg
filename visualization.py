from utils import *
import panel as pn
import matplotlib.pyplot as plt
from PIL import Image

fig, (ax1, ax2) = plt.subplots(nrows=2)
plt.subplots_adjust(hspace=0.5)
plt.ion()
def get_plot(n, r=2.5):
    r = np.array([n-r, n+r])-max(n+r-data['ts'][-1], 0)-min(n-r, 0)
    for a in range(data['ts'].shape[0]):
        if data['ts'][a]>=r[0]:
            break
    for b in range(data['ts'].shape[0]-a):
        b = a+b
        if data['ts'][b]>=r[1]:
            break
    ts = data['ts'][a:b]
    label = data['label'][a:b]
    label = (label-label.mean())/(label.std()+1e-6)
    pre = data['predict'][a:b]
    pre = (pre-pre.mean())/(pre.std()+1e-6)
    fps = 1/(ts[1:]-ts[:-1]).mean()

    if data['ft']:
        pre = bandpass_filter(pre, lowcut=data['band'][0], highcut=data['band'][1], fs=fps)

    ax2.cla()
    p, q = welch(label, fps, nfft=1e5/fps, nperseg=len(label)-1)
    x, y = p[(p>0)&(p<3)], q[(p>0)&(p<3)]
    hr, h = x[np.argmax(y)]*60, np.max(y)
    ax2.plot(x*60, y, label='GT', color='blue')
    ax2.plot([hr], [h], 'o', color='blue')
    ax2.annotate(str(round(hr, 2)), xytext=(hr, h), xy=(hr, h), color='blue')
    p, q = welch(pre, fps, nfft=1e5/fps, nperseg=len(label)-1)
    x, y = p[(p>0)&(p<3)], q[(p>0)&(p<3)]
    hr, h = x[np.argmax(y)]*60, np.max(y)
    ax2.plot(x*60, y, label='rPPG', color='red')
    ax2.plot([hr], [h], 'o', color='red')
    ax2.annotate(str(round(hr, 2)), xy=(hr, h), color='red')
    ax2.legend(loc='upper right');
    #print(y)
    ax2.set_yticks([])
    ax2.set_xlabel('Heart Rate')
    ax2.spines['right'].set_color('none')
    ax2.spines['left'].set_color('none')
    ax2.spines['top'].set_color('none')

    ax1.cla()
    ax1.set_yticks([])
    ax1.set_xlabel('Time')
    ax1.spines['right'].set_color('none')
    ax1.spines['left'].set_color('none')
    ax1.spines['top'].set_color('none')
    ax1.plot(ts, label, label='GT', color='blue')
    ax1.plot(ts, pre, label='rPPG', color='red')
    #ax1.legend(loc='upper right');
    return fig

data = {}
ft = pn.Row(pn.widgets.Checkbox(name='Band-pass filter', margin=(20, 0, 0, 300), width=120), pn.widgets.RangeSlider(name='Band', start=0.01, end=4, value=(0.6, 2.5), step=0.01))
def show_panel():
    global video, bvp_plot
    if app[0][0].value==video[0] and app[0][1].value!=video[1]:
        video[1] = app[0][1].value
        with h5py.File(f'results/{video[0]}', 'r') as f:
            base = f.attrs['dataset']
            f = f[video[1]]
            data['vid_path'] = (base, app[0][1].value)
            data['label'] = f['label'][:]
            data['predict'] = f['predict'][:]
            data['ts'] = f['timestamp'][:]-f['timestamp'][0]
        if os.path.exists(data['vid_path'][0]):
            with h5py.File(data['vid_path'][0], 'r') as f:
                img = Image.fromarray(f[data['vid_path'][1]]['video'][0])
                '''
                data['frames'] = f[data['vid_path'][1]]['video'][:]
                if data['frames'].dtype != np.uint8:
                    data['frames'] = (data['frames']*255).astype(np.uint8)
                '''
        else:
            #data['frames'] = np.full((data['ts'].shape[0], 128, 128, 3), 255, dtype=np.uint8)
            img = Image.fromarray(np.full((128, 128, 3), 255, dtype=np.uint8))
        app[1] = pn.Column(ft, pn.Row(None, None), pn.widgets.FloatSlider(name='Time', value=0, start=0, end=data['ts'][-1]-data['ts'][0], width=960, step=0.01))
        data['t'] = app[1][2].value
        data['ft'] = app[1][0][0].value
        data['band'] = app[1][0][1].value
        bvp_plot = get_plot(0)
        app[1][1][0] = pn.pane.Matplotlib(bvp_plot, dpi=144)
        #app[1][1][1] = pn.pane.image.PNG(Image.fromarray(data['frames'][0]), width=256, height=256)
        app[1][1][1] = pn.pane.image.PNG(img, width=256, height=256)
    if 't' in data and (data['t'] != app[1][2].value or data['ft']!=app[1][0][0].value or data['band']!=app[1][0][1].value):
        data['t'] = app[1][2].value
        data['ft'] = app[1][0][0].value
        data['band'] = app[1][0][1].value
        if os.path.exists(data['vid_path'][0]):
            with h5py.File(data['vid_path'][0], 'r') as f:
                for n in range(f[data['vid_path'][1]]['video'].shape[0]):
                    if data['t']<data['ts'][n]:
                        break
                img = Image.fromarray(f[data['vid_path'][1]]['video'][n])
        else:
            img = Image.fromarray(np.full((128, 128, 3), 255, dtype=np.uint8))
        bvp_plot = get_plot(data['t'])
        app[1][1][0] = pn.pane.Matplotlib(bvp_plot, dpi=144)
        app[1][1][1] = pn.pane.image.PNG(img, width=256, height=256)
        

    

video = ['', '']
def show_videos():
    global video
    if app[0][0].value != video[0]:
        with h5py.File(f'results/{app[0][0].value}', 'r') as f:
            i = {f"{j.attrs['SNR']:>5.2f}dB\t{j.attrs['path']}":i for i, j in f.items()}
        app[0][1] = pn.widgets.Select(options=i, size=50)
        video[0] = app[0][0].value
        show_panel()

def show_results():
    global files
    if files != [i for i in os.listdir('results') if i[-3:]=='.h5']:
        app[0][0] = pn.widgets.Select(options=files)


def main():
    show_results()
    show_videos()
    show_panel()

files=[i for i in os.listdir('results') if i[-3:]=='.h5']
select = pn.widgets.Select(options=files)
app = pn.Row(pn.Column(select, None), None)
def main_loop():
    while 1:
        try:
            main()
        except Exception as e:
            print(e)
            time.sleep(1)
        time.sleep(0.1)

pn.state.schedule_task('main', main, period='0.04s')
app.show()

