import pickle
from imutils.video import VideoStream
import imutils
import time
import face_rec
import cv2

FRAME_WIDTH = 500 # lower value means faster process but lower accuracy

trained_images_data = pickle.loads(open("encodings.pickle", "rb").read())

video_stream = VideoStream(usePiCamera=True).start()
time.sleep(2) # for the worker thread created above to finish its job
current_person_name = ""
while True:
  frame = video_stream.read()
  frame = imutils.resize(frame, width=FRAME_WIDTH)
  first_known_person_name = face_rec.get_first_known_person_name(frame, trained_images_data)
  if current_person_name != first_known_person_name:
    current_person_name = first_known_person_name
    print(current_person_name)
  cv2.imshow("Debug", frame)
  cv2.waitKey(1) # display frame for >= 1ms
