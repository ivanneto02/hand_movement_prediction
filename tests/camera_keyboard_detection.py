import cv2
import mediapipe as mp
from tkinter import Tk, Label
import threading
from datetime import datetime

class KeyDetector(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.start()
    
    def callback(self):
        self.root.quit()
    
    def run(self):
        self.root = Tk()
        # self.root.withdraw()
        self.root.bind('<Key>', key_pressed)
        self.root.mainloop()

def main():

    detector = KeyDetector()

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    drawing_styles = mp.solutions.drawing_styles

    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.2) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        drawing_styles.get_default_hand_landmark_style(),
                        drawing_styles.get_default_hand_connection_style())
                        
            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

def key_pressed(event):
    dt = datetime.now()
    print(event.char + ', ' + str(dt))

if __name__ == '__main__':
    main()