import face_recognition

print("Loading image...")
img = face_recognition.load_image_file("fyp_face_login/test.png")  # put some face image in same folder

print("Finding encodings...")
encodings = face_recognition.face_encodings(img)

print("Number of faces found:", len(encodings))
if encodings:
    print("First encoding length:", len(encodings[0]))
