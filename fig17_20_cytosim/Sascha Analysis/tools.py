import numpy as np

def load_csv(filename, clip_low_forces=True):
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    t = data[:, -1]
    dt = np.diff(t).mean()
    F = data[:, :-1].T
    if clip_low_forces:
        F[F<0] = 0
    events = np.isnan(F).argmax(axis=1)
    events[events==0] = t.shape[0]
    events -= 1
    events_t = t[events]
    return dict(F=F, events=events, dt=dt, t=t, events_t=events_t)




    
        
    