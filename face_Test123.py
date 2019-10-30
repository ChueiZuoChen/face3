import dlib
import cv2

ip_camera_url = 'http://192.168.15.2:8080/video'
cap = cv2.VideoCapture(ip_camera_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

while (cap.isOpened()):
    ret, frame = cap.read()
    face_rects, scores, idx = detector.run(frame, 0)
    for i, d in enumerate(face_rects):
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()
        text = " %2.2f ( %d )" % (scores[i], idx[i])
        print("ret:{} x1:{} y1:{} x2:{} y2:{}".format(text,x1,y1,x2,y2))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)

        cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX,
                    0.7, (255, 255, 255), 1, cv2.LINE_AA)

        landmarks_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        shape = predictor(landmarks_frame, d)
        for i in range(68):
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 3, (0, 0, 255), 2)
            cv2.putText(frame, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0),
                        1)
    cv2.imshow("Face Detection", frame)

    # print("================================")
    # print(frame.shape)
    # print("width: {} pixels".format(frame.shape[1]))
    # print("height: {} pixels".format(frame.shape[0]))
    # print("shape: {}".format(frame.shape[2]))
    # print("rate: {}".format(frame.size))
    # print("type: {}".format(frame.dtype))

    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows()