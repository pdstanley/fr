import face_recognition
import cv2

# This is a demo of running face recognition on a video file and saving the results to a new video file.
#
# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Open the input movie file
input_movie = cv2.VideoCapture("media_test/person1_test4.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output movie file (make sure frame rate/size matches input video!)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc(*'H264')
# fourcc = cv2.VideoWriter_fourcc(*'X264')
# output_movie = cv2.VideoWriter('media/output_test_2.avi', fourcc, 29.97, (640, 360))
output_movie = cv2.VideoWriter('media/output_test4_a.avi', fourcc, 30.006912, (1920, 1080))

    # fps = vcap.get(cv2.cv.CV_CAP_PROP_FPS) #30.00
    # width = vcap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)   # float
    # height = vcap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT) # float

    # self._name = name + '.mp4'
    # self._cap = VideoCapture(0)
    # self._fourcc = VideoWriter_fourcc(*'MP4V')
    # self._out = VideoWriter(self._name, self._fourcc, 20.0, (640,480))


# Load some sample pictures and learn how to recognize them.
lmm_image = face_recognition.load_image_file("media_test/person1_test.jpg")
lmm_face_encoding = face_recognition.face_encodings(lmm_image)[0]

al_image = face_recognition.load_image_file("media_test/person2_test.jpg")
al_face_encoding = face_recognition.face_encodings(al_image)[0]

# ks_image = face_recognition.load_image_file("media_trump/keith_schiller.jpg")
# ks_face_encoding = face_recognition.face_encodings(al_image)[0]
#
# iv_image = face_recognition.load_image_file("media_trump/ivanka_trump.jpg")
# iv_face_encoding = face_recognition.face_encodings(iv_image)[0]

known_faces = [
    lmm_face_encoding,
    al_face_encoding
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1

    # Quit when the input video file ends
    if not ret:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

        # If you had more than 2 faces, you could make this logic a lot prettier
        # but I kept it simple for the demo
        name = None
        if match[0]:
            name = "person1 test"
        elif match[1]:
            name = "person2 test"

        face_names.append(name)

    # Label the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 60), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 15), font, 1.8, (255, 255, 255), 1)

    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

# All done!
input_movie.release()
cv2.destroyAllWindows()
