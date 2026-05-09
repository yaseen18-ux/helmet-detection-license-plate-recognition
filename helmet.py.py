import cv2
import numpy as np
import imutils
import os

print("Initializing models...")

# ----------------- Print working directory -----------------
print("Python current working directory:", os.getcwd())
print("Files in this folder:", os.listdir())

# ----------------- Load Models -----------------
# Face detection using Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Person detection using HOG
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

print("Models loaded successfully!")

# ----------------- Video File -----------------
# Replace this with your full path to test.mp4
video_path = os.path.join(os.getcwd(), "test.mp4")

if not os.path.exists(video_path):
    print(f"Error: {video_path} not found!")
    exit(1)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file!")
    exit(1)

# ----------------- Prepare Video Writer -----------------
ret, frame = cap.read()
if not ret:
    print("Error: Could not read video frame!")
    exit(1)

frame = imutils.resize(frame, height=500)
height, width = frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter("output.avi", fourcc, 5.0, (width, height))

# ----------------- Colors -----------------
PERSON_COLOR = (0, 255, 0)  # Green
FACE_COLOR = (255, 0, 0)    # Blue
PLATE_COLOR = (0, 165, 255) # Orange

# ----------------- Safety Detection -----------------
def detect_safety_status(person_roi):
    """
    Detect helmet using simple edge density in head region
    """
    try:
        height, width = person_roi.shape[:2]
        head_region = person_roi[0:height//3, :]
        gray_head = cv2.cvtColor(head_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_head, 50, 150)
        edge_density = np.sum(edges) / (head_region.shape[0] * head_region.shape[1])
        if edge_density > 0.1:
            return "Helmet Detected"
        else:
            return "No Helmet"
    except:
        return "Cannot Detect"

# ----------------- Number Plate Detection -----------------
def detect_number_plates_basic(image):
    """
    Simple contour-based license plate detection
    """
    plates = []
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h
            if 2.0 <= aspect_ratio <= 6.0 and w > 60 and h > 20:
                plates.append((x, y, w, h))
    except:
        pass
    return plates

# ----------------- Video Processing -----------------
frame_count = 0
print("Starting video processing... Press ESC to quit.")

while True:
    ret, img = cap.read()
    if not ret:
        break

    frame_count += 1
    img = imutils.resize(img, height=500)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect people
    rects, weights = hog.detectMultiScale(gray, winStride=(4,4), padding=(8,8), scale=1.05)
    for (x, y, w, h) in rects:
        cv2.rectangle(img, (x, y), (x+w, y+h), PERSON_COLOR, 2)
        cv2.putText(img, "Person", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, PERSON_COLOR, 2)
        person_roi = img[y:y+h, x:x+w]
        status = detect_safety_status(person_roi)
        cv2.putText(img, status, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), FACE_COLOR, 2)
        cv2.putText(img, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, FACE_COLOR, 2)

    # Detect number plates
    plates = detect_number_plates_basic(img)
    for (x, y, w, h) in plates:
        cv2.rectangle(img, (x, y), (x+w, y+h), PLATE_COLOR, 2)
        cv2.putText(img, "Number Plate", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, PLATE_COLOR, 2)

    # Frame info
    cv2.putText(img, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(img, f"Persons: {len(rects)} | Faces: {len(faces)} | Plates: {len(plates)}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    writer.write(img)
    cv2.imshow("Helmet & Plate Detection", img)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

# Cleanup
cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"Processed {frame_count} frames. Output saved as 'output.avi'")
