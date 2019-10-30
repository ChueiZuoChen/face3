import sys, os, dlib, glob, numpy
from skimage import io
import cv2
import os

ladygaga_acc = 0
unknow = 0
she = 0
mad = 0

def compareImg(imgp,numfiles):
    global ladygaga_acc,she,unknow,mad
    predictor_path = "shape_predictor_68_face_landmarks.dat"

    face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"

    faces_folder_path = "./image"

    detector = dlib.get_frontal_face_detector()

    sp = dlib.shape_predictor(predictor_path)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)

    descriptors = []

    candidate = []

    img_path = imgp
    print(imgp)
    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
        base = os.path.basename(f)
        candidate.append(os.path.splitext(base)[0])

        img = io.imread(f)

        dets = detector(img, 1)
        for k, d in enumerate(dets):
            shape = sp(img, d)

            face_descriptor = facerec.compute_face_descriptor(img, shape)
            v = numpy.array(face_descriptor)
            descriptors.append(v)

    try:
        img = io.imread(img_path)

        dets = detector(img, 1)
        dist = []
        for k, d in enumerate(dets):
            shape = sp(img, d)
            face_descriptor = facerec.compute_face_descriptor(img, shape)
            d_test = numpy.array(face_descriptor)

            x1 = d.left()
            y1 = d.top()
            x2 = d.right()
            y2 = d.bottom()
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)

            for i in descriptors:
                dist_ = numpy.linalg.norm(i - d_test)
                dist.append(dist_)

        c_d = dict(zip(candidate, dist))

        cd_sorted = sorted(c_d.items(), key=lambda d: d[1])

    # print(cd_sorted)

        rec_name = cd_sorted[0][0]
        print(rec_name)
        if (rec_name == "ladygaga"):
            ladygaga_acc = ladygaga_acc + 1
            # print(rec_name + ":" + str(ladygaga_acc) + "/"+ str(numfiles))
            # print("Unknow: " + str(unknow))
        elif (rec_name == "shreya"):
            she = she + 1
            # print(rec_name + ": " + str(she) + "/" + str(numfiles))
            # print("Unknow: "+str(unknow))
        elif (rec_name == "madonna"):
            mad = mad + 1
            # print(rec_name + ": " + str(mad) + "/" + str(numfiles))
            # print("Unknow: " + str(unknow))
        print("ladygaga" + ": " + str(ladygaga_acc) + "/" + str(numfiles))
        print("shreya" + ": " + str(she) + "/" + str(numfiles))
        print("madonna" + ": " + str(mad) + "/" + str(numfiles))
        print("Unknow: " + str(unknow))

    except (RuntimeError,IndexError,TypeError,ValueError):
        unknow = unknow + 1
        print("Unknow picture!!!")

#
# ip_camera_url = 'http://192.168.43.79:8080/video'
# cap = cv2.VideoCapture(ip_camera_url)
# cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
#
# while (cap.isOpened()):
#     ret, frame = cap.read()
#     face_rects, scores, idx = detector.run(frame, 0)
#     for i, d in enumerate(face_rects):
#         x1 = d.left()
#         y1 = d.top()
#         x2 = d.right()
#         y2 = d.bottom()
#         text = " %2.2f ( %d )" % (scores[i], idx[i])
#         print("ret:{} x1:{} y1:{} x2:{} y2:{}".format(text,x1,y1,x2,y2))
#
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)
#
#         cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX,
#                     0.7, (255, 255, 255), 1, cv2.LINE_AA)
#
#         landmarks_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#         shape = predictor(landmarks_frame, d)
#         for i in range(68):
#             cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 3, (0, 0, 255), 2)
#             cv2.putText(frame, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0),
#                         1)
#     cv2.imshow("Face Detection", frame)
#
#     # print("================================")
#     # print(frame.shape)
#     # print("width: {} pixels".format(frame.shape[1]))
#     # print("height: {} pixels".format(frame.shape[0]))
#     # print("shape: {}".format(frame.shape[2]))
#     # print("rate: {}".format(frame.size))
#     # print("type: {}".format(frame.dtype))
#
#     if cv2.waitKey(10) == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()



img_path = "./other_img/SHREYA"
files = os.listdir(img_path)
s = []
for img in files:
    s.append(img)

for img in s:
    compareImg(img_path + "/" + img,len(files))
