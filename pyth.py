import cv2
import mediapipe as mp

# إعداد MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# دالة لحساب عدد الأصابع المرفوعة في يد واحدة
def count_fingers(hand_landmarks, hand_label):
    fingers = []
    tips_ids = [4, 8, 12, 16, 20]

    # الإبهام (يختلف حسب اليد)
    if hand_label == "Right":
        if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)
    else:  # Left
        if hand_landmarks.landmark[tips_ids[0]].x > hand_landmarks.landmark[tips_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)

    # باقي الأصابع (رأسي)
    for id in range(1, 5):
        if hand_landmarks.landmark[tips_ids[id]].y < hand_landmarks.landmark[tips_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

# فتح الكاميرا
#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#for i in range(5):
   # cap = cv2.VideoCapture(i)
   # if cap.read()[0]:
       # print(f"Camera {i} is available")
    #cap.release()
cap = cv2.VideoCapture('c:/Users/LENOVO/Desktop/n.mkv')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    total_fingers = 0

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # "Left" or "Right"
            count = count_fingers(hand_landmarks, label)
            total_fingers += count
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # عرض العدد على الصورة
    cv2.putText(img, f'Total Fingers: {total_fingers}', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

    # حفظ العدد في ملف نصي
    with open("finger_count.txt", "w") as f:
        f.write(str(total_fingers))

    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == 27:  # زر ESC للخروج
        break

cap.release()
cv2.destroyAllWindows()
