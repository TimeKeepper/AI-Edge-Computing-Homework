import my_Queue
import mnist_load
import threading
import time
import numpy as np
import cv2

import signal
import sys

q = my_Queue.ring_Queue(20)

data = mnist_load.mnist_load()

stop_flag = False

def image_enquene():
    cur = time.time()
    i = 0
    freq = 4
    while i < len(data):
        if stop_flag:
            break
        if time.time() - cur < 1/freq:
            continue
        q.enqueue(data[i].reshape(28, 28))
        i += 1
        cur = time.time()

image_changed = threading.Event()
image_data = np.zeros((28, 28))

def image_dequene():
    cur = time.time()
    freq = 2
    while True:
        if stop_flag:
            break
        if q.is_full() and freq == 2:
            print('speed up')
            freq = 8
        elif q.is_empty() and freq == 8:
            print('speed down')
            freq = 2
        if time.time() - cur < 1/freq:
            continue
        cur = time.time()
        global image_data
        image_data = q.dequeue()
        image_changed.set()

def image_show():
    while True:
        if not image_changed.is_set():
             continue
        image_changed.clear()
        cv2.imshow('Image', image_data)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            global stop_flag
            stop_flag = True
            cv2.destroyAllWindows()
            break
    

load_thread = threading.Thread(target=image_enquene)
release_thread = threading.Thread(target=image_dequene)
show_thread = threading.Thread(target=image_show)

if __name__ == '__main__':
    load_thread.start()
    release_thread.start()
    show_thread.start()
    # wait for the thread to finish
    load_thread.join()
    release_thread.join()
    show_thread.join()
