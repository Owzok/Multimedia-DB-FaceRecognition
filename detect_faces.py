import face_recognition     # main algorithm
import pickle               # deserialize data
import cv2                  # opencv, image processment

data = pickle.loads(open("encodings.pickle", "rb").read())

image = cv2.imread("./examples/joepedro.jpeg")
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
boxes = face_recognition.face_locations(rgb, model="hog")
encodings = face_recognition.face_encodings(rgb, boxes)
names = []

for encoding in encodings:
    matches = face_recognition.compare_faces(data["encodings"], encoding)
    name = "Unknown"
    if True in matches:
        matched_indexes = [i for (i, b) in enumerate(matches) if b]
        counts = {}

        for i in matched_indexes:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1
        
        name = max(counts, key=counts.get)
    names.append(name)

    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

cv2.imshow("Image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()