import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Global variables
max_num_hands = 1
gesture_labels = {
    0: 'rock',
    5: 'paper',
    9: 'scissors'
}
rematch_selected = False

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Hands model
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Load pre-trained KNN model
file = np.genfromtxt('/home/pi/project_ws/data/gesture_train.csv', delimiter=',')
angle = file[:, :-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

# Open video capture
cap = cv2.VideoCapture(0)

# Variables for game logic and rematch
start_time = None
result_shown = False
countdown_started = False


# Rematch button callback function
def rematch_button_callback(event, x, y, flags, param):
    global rematch_selected
    if event == cv2.EVENT_LBUTTONDOWN:
        if 75 <= x <= 165 and 130 <= y <= 170:  # YES button clicked
            rematch_selected = True
        elif 235 <= x <= 325 and 130 <= y <= 170:  # NO button clicked
            cv2.destroyAllWindows()

# Create rematch window
rematch_window = np.zeros((200, 400, 3), dtype=np.uint8)
cv2.putText(rematch_window, text="Rematch?", org=(130, 100),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
            color=(255, 255, 255), thickness=2)
cv2.rectangle(rematch_window, (75, 130), (165, 170), (0, 255, 0), thickness=2)
cv2.putText(rematch_window, text="YES", org=(85, 160),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
            color=(0, 255, 0), thickness=2)
cv2.rectangle(rematch_window, (235, 130), (325, 170), (0, 0, 255), thickness=2)
cv2.putText(rematch_window, text="NO", org=(255, 160),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
            color=(0, 0, 255), thickness=2)

# Set mouse callback for rematch window
cv2.namedWindow('Rematch')
cv2.setMouseCallback('Rematch', rematch_button_callback)


# Main loop for capturing and processing video frames
while cap.isOpened():
    # Read frame from video capture
    ret, img = cap.read()
    if not ret:
        continue

    # Flip frame horizontally and convert color space
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Hands
    result = hands.process(img)

    # Convert color space back to BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Hand detection and gesture recognition
    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            # Extract hand landmarks
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Calculate angles between hand landmarks
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            angle = np.arccos(np.einsum('nt,nt->n',
                                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
            angle = np.degrees(angle)


            # Classify gesture using KNN model
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbors, dist = knn.findNearest(data, 3)
            user_gesture = int(results[0][0])


            # Display gesture label and hand landmarks on frame
            if user_gesture in gesture_labels.keys():
                cv2.putText(img, text=gesture_labels[user_gesture].upper(),
                            org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            # Start countdown when first hand is detected
            if start_time is None:
                start_time = time.time()
                countdown_started = True

    # Display remaining countdown time
    if countdown_started and not result_shown:
        elapsed_time = time.time() - start_time
        countdown = int(5 - elapsed_time)

        if countdown > 0:
            cv2.putText(img, text="Time: " + str(countdown), org=(20, 40),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
        else:
            # Show the game result
            result_window = np.zeros((200, 400, 3), dtype=np.uint8)
            cv2.putText(result_window, text="User: " + gesture_labels[user_gesture].upper(),
                        org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                        color=(255, 255, 255), thickness=2)
            computer_gesture = random.choice(list(gesture_labels.values()))
            cv2.putText(result_window, text="Computer: " + computer_gesture.upper(),
                        org=(50, 120), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                        color=(255, 255, 255), thickness=2)
            if user_gesture == list(gesture_labels.keys())[(list(gesture_labels.values()).index(computer_gesture) + 1) % 3]:
                cv2.putText(result_window, text="Result: User wins!",
                            org=(50, 170), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                            color=(0, 255, 0), thickness=2)
            elif user_gesture == list(gesture_labels.keys())[(list(gesture_labels.values()).index(computer_gesture) - 1) % 3]:
                cv2.putText(result_window, text="Result: Computer wins!",
                            org=(50, 170), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                            color=(0, 0, 255), thickness=2)
            else:
                cv2.putText(result_window, text="Result: It's a tie!",
                            org=(50, 170), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                            color=(255, 0, 0), thickness=2)
            cv2.imshow('Result', result_window)
            result_shown = True

    cv2.imshow('Game', img)

    # Handle rematch selection
    if result_shown and not rematch_selected:
        cv2.imshow('Rematch', rematch_window)

    if rematch_selected:
        # Reset variables for a new game
        start_time = None
        result_shown = False
        countdown_started = False
        rematch_selected = False

    if cv2.waitKey(1) == ord('q') or cv2.getWindowProperty('Game', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
