import cv2
import mediapipe as mp
import numpy as np

max_num_hands = 5  # Maximum number of hands to detect
gesture = {
    0: 'fist', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
    6: 'six', 7: 'rock', 8: 'spiderman', 9: 'yeah', 10: 'ok',
}
rps_gesture = {0: 'rock', 5: 'paper', 9: 'scissors'}

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Gesture recognition model
file = np.genfromtxt('/home/pi/project_ws/data/gesture_train.csv', delimiter=',')
angle = file[:, :-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0)

player_count = 0
players = []

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        rps_result = []  # Store the gesture result and hand positions

        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
            v = v2 - v1  # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arccos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                                       v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                       v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

            angle = np.degrees(angle)  # Convert radians to degrees

            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])

            # Draw gesture result
            if idx in rps_gesture.keys():
                org = (int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0]))
                cv2.putText(img, text=rps_gesture[idx].upper(), org=(org[0], org[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(255, 255, 255), thickness=2)

                rps_result.append({
                    'rps': rps_gesture[idx],  # Gesture result
                    'org': org  # Hand position
                })

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

        if len(rps_result) > player_count:
            player_count = len(rps_result)
            players = rps_result

        # Display player numbers
        for i, player in enumerate(players):
            cv2.putText(img, text=f"Player {i + 1}", org=(player['org'][0], player['org'][1] - 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)

        # Determine winners
        if len(players) > 1:
            winner = None
            text = ''

            if all(player['rps'] == 'rock' for player in players):
                text = 'Tie'
            elif any(player['rps'] == 'paper' for player in players) and any(player['rps'] == 'rock' for player in players):
                text = 'Paper wins'
                winner = next((i for i, player in enumerate(players) if player['rps'] == 'paper'), None)
            elif any(player['rps'] == 'scissors' for player in players) and any(
                    player['rps'] == 'rock' for player in players):
                text = 'Rock wins'
                winner = next((i for i, player in enumerate(players) if player['rps'] == 'rock'), None)
            elif any(player['rps'] == 'scissors' for player in players) and any(
                    player['rps'] == 'paper' for player in players):
                text = 'Scissors wins'
                winner = next((i for i, player in enumerate(players) if player['rps'] == 'scissors'), None)

            if winner is not None:
                cv2.putText(img, text='Winner', org=(players[winner]['org'][0], players[winner]['org'][1] + 70),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=3)
            cv2.putText(img, text=text, org=(int(img.shape[1] / 2), 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=2, color=(0, 0, 255), thickness=3)

    cv2.imshow('Game', img)
    if cv2.waitKey(1) == ord('q') or cv2.getWindowProperty('Game', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
