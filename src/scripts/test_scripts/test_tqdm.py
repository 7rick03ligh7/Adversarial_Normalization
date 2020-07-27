from tqdm import tqdm
from time import sleep
import numpy as np

progress_bars = []
progress_bars.append(tqdm(total=1000))
progress_bars.append(tqdm(total=1000))
progress_bars.append(tqdm(total=1000))
for i in range(1000):
    idx = int(np.random.randint(0,len(progress_bars)))
    progress_bars[idx].update()
    sleep(0.01)


