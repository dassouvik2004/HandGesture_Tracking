import cv2
import mediapipe as mp
import os
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

DATASET_PATH = r"C:\Users\RAJA BISWAS\OneDrive\Desktop\asl_dataset"
with open("data.csv", "w", newline="") as f:
    writer = csv.writer(f)

    for label in os.listdir(DATASET_PATH):
        folder_path = os.path.join(DATASET_PATH, label)

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    row = []

                    for lm in hand_landmarks.landmark:
                        row.append(lm.x)
                        row.append(lm.y)

                    row.append(label)
                    writer.writerow(row)

print("Dataset Created!")