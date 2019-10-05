import sys, os, dlib, glob, numpy
from skimage import io
import cv2
import imutils

if len(sys.argv) != 2:
    print("input image name!")
    exit()

predictor_path = "shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
faces_folder_path = "./image"
img_path = "./other_img/LADY_GAGA/"+sys.argv[1]


detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)

facerec = dlib.face_recognition_model_v1(face_rec_model_path)
descriptors = []
candidate = []
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
rec_name = cd_sorted[0][0]
cv2.putText(img, rec_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
img = imutils.resize(img, width=600)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow("Face Recognition", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
