import cv2
import mediapipe as mp
import os

def main():

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    drawing_styles = mp.solutions.drawing_styles

    # fn = filename; Images in the img folder
    IMAGE_FILES = ['img/' + str(fn) for fn in next(os.walk('./img'))[2]]

    with mp_hands.Hands(

        static_image_mode = True, # Image input
        max_num_hands = 2, # Two hands
        min_detection_confidence = 0.5, # Probably hyperparamater
                        ) as hands:
        
        for i, file in enumerate(IMAGE_FILES):
            # Read the image, flip it around y-axis for correct handedness output.
            image = cv2.imread(file)   # Read
            image = cv2.flip(image, 1) # Flip

            # Convert the BGR image to RGB
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print handedness
            print('Handedness:', results.multi_handedness)

            if not results.multi_hand_landmarks:
                continue

            img_height, img_width, _ = image.shape

            annotated_img = image.copy()

            for hand_landmarks in results.multi_hand_landmarks:
                print('Hand_landmarks:', hand_landmarks)
                print(
                    f'Index finger tip coordinates: (',
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * img_width}, '
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * img_height})'
                )

                mp_drawing.draw_landmarks(

                    annotated_img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    drawing_styles.get_default_hand_landmark_style(),
                    drawing_styles.get_default_hand_connection_style()
                )

            # Save images
            cv2.imwrite(

                './img_result/annotated_img' + str(i) + '.png', cv2.flip(annotated_img, 1)
            )

if __name__ == '__main__':
    main()