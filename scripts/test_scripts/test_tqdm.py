from tqdm import tqdm
from time import sleep
import numpy as np

# progress_bars = []
# progress_bars.append(tqdm(total=1000))
# progress_bars.append(tqdm(total=1000))
# progress_bars.append(tqdm(total=1000))
# for i in range(1000):
#     idx = int(np.random.randint(0,len(progress_bars)))
#     progress_bars[idx].update()
#     sleep(0.01)


from multiprocessing import Process, Queue

def f(q):
    q.put([42, None, 'hello'])
    sleep(3)
    q.put([111])

if __name__ == '__main__':
    q = Queue()
    p = Process(target=f, args=(q,))
    p.start()
    print(q.get())    
    print(q.get()) # prints "[42, None, 'hello']"
    p.join()