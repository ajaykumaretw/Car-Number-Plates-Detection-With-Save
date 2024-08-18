import cv2
import os

# Path to the Haar cascade XML file for number plate detection
harcascade = r"G:\NumberPlate\model\haarcascade_russian_plate_number.xml"

# Initialize video capture from the webcam (camera index 0)
cap = cv2.VideoCapture(0)

# Set video frame width and height
cap.set(3, 640)  # width
cap.set(4, 480)  # height

# Optional: Increase brightness and exposure settings to improve visibility in low light
cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)
cap.set(cv2.CAP_PROP_EXPOSURE, -6)

# Load the Haar cascade classifier for plate detection
plate_cascade = cv2.CascadeClassifier(harcascade)

# Check if the cascade classifier loaded successfully
if plate_cascade.empty():
    raise IOError("Haar cascade xml file not found or could not be loaded.")

# Create the directory for saving images if it does not exist
if not os.path.exists("plates"):
    os.makedirs("plates")

# Minimum area to consider a detection valid (helps filter out small detections)
min_area = 500
count = 0  # Counter for saved images

while True:
    # Read a frame from the video capture
    success, img = cap.read()

    # Break the loop if there is an issue with the video capture
    if not success:
        print("Failed to capture image.")
        break

    # Convert the frame to grayscale (necessary for detection)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to enhance contrast (helps in low-light conditions)
    img_gray = cv2.equalizeHist(img_gray)

    # Optional: Apply Gaussian blur to reduce noise
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Detect number plates in the grayscale image
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    # Loop through all detected plates
    for (x, y, w, h) in plates:
        area = w * h  # Calculate the area of the detected plate

        # If the detected area is larger than the minimum area, process it
        if area > min_area:
            # Draw a rectangle around the detected plate
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Add text above the rectangle to label it as "Number Plate"
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            # Extract the region of interest (ROI) corresponding to the detected plate
            img_roi = img[y: y + h, x: x + w]
            # Display the extracted plate region
            cv2.imshow("ROI", img_roi)

    # Show the result with rectangles drawn on detected plates
    cv2.imshow("Result", img)

    # Check if the 's' key is pressed to save the detected plate image
    if cv2.waitKey(1) & 0xFF == ord('s'):
        if 'img_roi' in locals():
            # Save the image of the detected plate in the "plates" directory with a unique name
            cv2.imwrite(f"plates/scanned_img_{count}.jpg", img_roi)
            # Draw a filled rectangle and add text to indicate that the plate was saved
            cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
            # Show the updated result with the save message
            cv2.imshow("Results", img)
            cv2.waitKey(500)  # Wait for half a second before resuming
            count += 1  # Increment the image counter

            # Exit the loop and close the camera after saving
            break

# Release the video capture and close all windows when done
cap.release()
cv2.destroyAllWindows()
