import cv2
import mediapipe as mp
import numpy as np

# Load hand template image
template = cv2.imread('hand_template.png', cv2.IMREAD_UNCHANGED)
template_h, template_w = template.shape[:2]

# Load video to play inside template
video = cv2.VideoCapture('vid.mp4')
playing_video = False

# Start webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)


def overlay_image_alpha(img, img_overlay, pos, alpha=1.0):
    """Overlay img_overlay (with alpha channel) onto img at position pos."""
    x, y = pos
    h, w = img.shape[:2]
    h_ol, w_ol = img_overlay.shape[:2]

    # Boundary checks
    if x < 0:
        img_overlay = img_overlay[:, -x:]
        w_ol = img_overlay.shape[1]
        x = 0
    if y < 0:
        img_overlay = img_overlay[-y:, :]
        h_ol = img_overlay.shape[0]
        y = 0
    if x + w_ol > w:
        img_overlay = img_overlay[:, :w - x]
        w_ol = img_overlay.shape[1]
    if y + h_ol > h:
        img_overlay = img_overlay[:h - y, :]
        h_ol = img_overlay.shape[0]

    # Blend with alpha channel
    if img_overlay.shape[2] == 4:
        alpha_overlay = (img_overlay[:, :, 3] / 255.0) * alpha
        alpha_background = 1.0 - alpha_overlay
        for c in range(3):
            img[y:y + h_ol, x:x + w_ol, c] = (
                alpha_overlay * img_overlay[:, :, c] +
                alpha_background * img[y:y + h_ol, x:x + w_ol, c]
            )
    else:
        img[y:y + h_ol, x:x + w_ol] = (
            alpha * img_overlay[:, :, :3] +
            (1 - alpha) * img[y:y + h_ol, x:x + w_ol]
        )


# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Template position (center of frame)
    center_x, center_y = w // 2 - template_w // 2, h // 2 - template_h // 2
    overlay_image_alpha(frame, template, (center_x, center_y))

    # Detect hands
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    hand_inside = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # Palm center coordinates (landmark 9)
            palm_x = int(hand_landmarks.landmark[9].x * w)
            palm_y = int(hand_landmarks.landmark[9].y * h)

            # Check if palm is inside template
            if (center_x < palm_x < center_x + template_w and
                    center_y < palm_y < center_y + template_h):
                hand_inside = True

    # Play video if hand is inside
    if hand_inside:
        if not playing_video:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            playing_video = True

        ret_vid, frame_vid = video.read()
        if ret_vid:
            frame_vid = cv2.resize(frame_vid, (template_w, template_h))
            overlay_image_alpha(frame, frame_vid, (center_x, center_y), alpha=0.8)
        else:
            playing_video = False
    else:
        playing_video = False

    # Display output
    cv2.imshow('Hand Scanner', frame)

    if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
        break

# Cleanup
cap.release()
video.release()
cv2.destroyAllWindows()
