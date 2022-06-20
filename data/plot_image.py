import cv2

def plot_image_opencv_fit_window(pil_image, title='Show', screen_resolution=(1920, 1080),
                                wait_key=True):
    #Define the screen resulation
    img_height, img_width = pil_image.shape[0], pil_image.shape[1]
    screen_res = screen_resolution
    scale_width = screen_res[0] / img_width
    scale_height = screen_res[1] / img_height 
    scale = min(scale_width, scale_height)
    #resized window width and height
    window_width = int(img_width * scale)
    window_height = int(img_height * scale)
    cv2.startWindowThread()
    #cv2.WINDOW_NORMAL makes the output window resizealbe
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    #resize the window according to the screen resolution
    cv2.resizeWindow(title, window_width, window_height)
    cv2.imshow(title, pil_image)
    if wait_key:
        cv2.waitKey()
