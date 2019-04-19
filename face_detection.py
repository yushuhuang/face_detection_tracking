import cv2 as cv
import numpy as np
import math

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')


def face_detection(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return faces


term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)


class meanShift:

    def __init__(self, frame, face):
        (x, y, w, h) = face
        self.track_window = (x, y, w, h)
        roi = frame[y:y+h, x:x+w]
        hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv_roi,  np.array((0., 60., 32.)),
                          np.array((180., 255., 255.)))
        self.roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv.normalize(self.roi_hist, self.roi_hist, 0, 255, cv.NORM_MINMAX)

    def meanShift(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
        ret, self.track_window = cv.meanShift(
            dst, self.track_window, term_crit)
        x, y, w, h = self.track_window
        cv.rectangle(frame, (x, y), (x + w, y + h), 255, 2)


cap = cv.VideoCapture(0)

if __name__ == '__main__':
    p = {}
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            faces = face_detection(frame)
            if len(faces) != 0:
                p = {}
                for n, face in enumerate(faces):
                    p[n] = meanShift(frame, face)
            for n in p:
                p[n].meanShift(frame)
            cv.imshow('frame', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv.destroyAllWindows()
