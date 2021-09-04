import cv2
import mediapipe as mp
import numpy as np
from glob import glob

HEIGHT, WIDTH = 480, 640
BG_COLOR = (255,182,193)

#load bg images
BG_PATH = 'images'
bg_images = []
path = glob(f'{BG_PATH}/*')
for image_path in path:
    image = cv2.imread(image_path)
    image = cv2.resize(image, (WIDTH, HEIGHT))
    bg_images.append(image)
i = 0
l = len(bg_images)

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
    while cap.isOpened():
        check, frame = cap.read()
        if not check:
            print('No video capture found')
            break
        bg_image = bg_images[i]
        #flip horizontally for selfie view
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame.flags.writeable = False
        results = selfie_segmentation.process(frame)
        
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.7
        if l == 0:
            bg_image = np.zeros(frame.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
        output_image = np.where(condition, frame, bg_image)
    
        cv2.imshow('MediaPipe Selfie Segmentation', output_image)
        key = cv2.waitKey(1)
        if key == ord('a'):
            i = (i-1)%l
        elif key == ord('d'):
            i = (i+1)%l
        elif key == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()