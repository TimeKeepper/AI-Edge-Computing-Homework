import train
import threading
import time
import numpy as np
import cv2

train_thread = threading.Thread(target=train.train)
test_thread = threading.Thread(target=train.test)
val_thread = threading.Thread(target=train.val)

if __name__ == '__main__':
    # train_thread.start()
    test_thread.start()
    # val_thread.start()

    while True:
        cv2.imshow('Image', train.image_data)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            train.stop_flag = True
            cv2.destroyAllWindows()
            break

    # train_thread.join()
    test_thread.join()
    # val_thread.join()
