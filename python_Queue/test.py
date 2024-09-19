import cv2

image_path = './data/test.jpg'
image = cv2.imread(image_path)

if image is None:
    print("Could not read the image.")
else:
    window_name = 'Image'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    cv2.imshow(window_name, image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

