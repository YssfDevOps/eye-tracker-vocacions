import cv2

# Open the webcam
capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = capture.read()

    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Webcam", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
capture.release()
cv2.destroyAllWindows()
