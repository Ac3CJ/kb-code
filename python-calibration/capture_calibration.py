import cv2
import os

# Create the folder for the images
os.makedirs("calibration_images", exist_ok=True)

# Open the scrcpy v4l2 sink
cap = cv2.VideoCapture(10)

if not cap.isOpened():
    print("Error: Could not open /dev/video10. Is scrcpy running?")
    exit()

print("Press SPACE to take a photo. Press ESC to quit.")

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    cv2.imshow("Calibration Capture - Press SPACE to snap", frame)
    
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break
    elif key == 32:  # Spacebar
        filename = f"calibration_images/capture_{count:02d}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        count += 1

cap.release()
cv2.destroyAllWindows()