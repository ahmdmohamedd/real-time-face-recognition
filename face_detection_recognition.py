import cv2
import os

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load the dataset of known faces
dataset_path = "dataset_faces"
known_faces = []
known_names = []

for name in os.listdir(dataset_path):
    for filename in os.listdir(f"{dataset_path}/{name}"):
        image = cv2.imread(f"{dataset_path}/{name}/{filename}")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
        if len(face) > 0:
            (x, y, w, h) = face[0]
            face_roi = gray_image[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (150, 150))
            known_faces.append(face_roi)
            known_names.append(name)

# Start the video capture
video_capture = cv2.VideoCapture(0)

# Run the face detection loop
while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        face_roi = cv2.resize(roi_gray, (150, 150))

        # Compare the detected face to the dataset
        match = False
        for known_face, known_name in zip(known_faces, known_names):
            similarity = cv2.compareHist(cv2.calcHist([face_roi], [0], None, [256], [0, 256]),
                                         cv2.calcHist([known_face], [0], None, [256], [0, 256]),
                                         cv2.HISTCMP_CORREL)
            if similarity > 0.8:
                match = True
                name = known_name
                break

        # Display the name of the recognized person
        if match:
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            print({name}, "detected!")
        else:
            cv2.putText(frame, "WARNING! UNKNOWN", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            print("Warning: Unknown face detected!")


    # Display the resulting frame
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and destroy the windows
video_capture.release()
cv2.destroyAllWindows()
