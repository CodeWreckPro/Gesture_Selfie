import cv2
import os

# Create a folder to save the captured images
save_folder = os.path.dirname(os.path.abspath(__file__))
save_folder = os.path.join(save_folder, "captured_images")
os.makedirs(save_folder, exist_ok=True)

# OpenCV CascadeClassifier for hand detection
hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

# Initialize video capture
cap = cv2.VideoCapture(0)

# Variables for finger states
finger_state = [False, False]
num_frames_closed = 0
capture_count = 0

while True:
    # Read frame from the camera
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect hands in the frame
    hands = hand_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in hands:
        # Draw rectangle around the hand
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Check if the hand is in scissor position
        hand_roi = gray[y:y + h, x:x + w]
        contours, _ = cv2.findContours(hand_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) >= 2:
            finger_state[0] = True
            finger_state[1] = True
        else:
            finger_state[0] = False
            finger_state[1] = False

    # Check if the fingers are closed for a certain number of frames
    if all(finger_state):
        num_frames_closed += 1
        if num_frames_closed >= 10:
            # Save the captured image
            capture_count += 1
            image_name = f"captured_{capture_count}.jpg"
            image_path = os.path.join(save_folder, image_name)
            cv2.imwrite(image_path, frame)
            print(f"Image saved: {image_path}")
            num_frames_closed = 0
    else:
        num_frames_closed = 0

    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Quit the program when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
