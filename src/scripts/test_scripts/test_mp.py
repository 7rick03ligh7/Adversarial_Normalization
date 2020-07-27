from multiprocessing import Process, Queue
from time import sleep

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