import sys, os, dlib, glob, numpy
from skimage import io
import cv2
import imutils
import os

ladygaga_acc = 0
unknow = 0
she = 0

def compareImg(imgp):
    global ladygaga_acc,she,unknow

    # 臉部68特徵點檢測器
    predictor_path = "shape_predictor_68_face_landmarks.dat"

    # 人臉辨識模型和提取特徵值
    face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"

    # 訓練圖像文件夾
    faces_folder_path = "./image"

    # 讀入模型
    # 與人臉檢測相同,使用dlib自帶的get_frontal_face_detector作為人臉檢測器
    detector = dlib.get_frontal_face_detector()

    # 使用官方提供的模型建構特徵提取器
    sp = dlib.shape_predictor(predictor_path)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)

    # 存放訓練夾人物特徵列表
    descriptors = []

    # 存放訓練夾文件的名字
    candidate = []

    # 比對的圖片路徑
    img_path = imgp
    print(imgp)
    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
        base = os.path.basename(f)
        candidate.append(os.path.splitext(base)[0])
        # 讀取所有圖片
        img = io.imread(f)

        # 人臉檢測
        # 與人臉檢測程序相同,使用detector進行人臉檢測 dets為返回的結果
        dets = detector(img, 1)
        for k, d in enumerate(dets):
            # 使用predictor 進行人臉特徵點識別,shape為返回的結果
            shape = sp(img, d)

            # 提取特徵
            face_descriptor = facerec.compute_face_descriptor(img, shape)
            v = numpy.array(face_descriptor)
            descriptors.append(v)

    # 要求的圖片路徑
    try:
        img = io.imread(img_path)

        dets = detector(img, 1)
        dist = []
        for k, d in enumerate(dets):
            shape = sp(img, d)
            face_descriptor = facerec.compute_face_descriptor(img, shape)
            d_test = numpy.array(face_descriptor)

        # x1 人臉左邊距離圖片左邊邊界的距離
        # y1 人臉上邊距離圖片上邊邊界的距離
        # x2 人臉右邊距離圖片右邊邊界的距離
        # y2 人臉下邊距離圖片下邊邊界的距離
            x1 = d.left()
            y1 = d.top()
            x2 = d.right()
            y2 = d.bottom()
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)

            for i in descriptors:
            # 計算距離
                dist_ = numpy.linalg.norm(i - d_test)
                dist.append(dist_)

    # 訓練資料夾的人物和距離組成一個字典
        c_d = dict(zip(candidate, dist))

    # 將字典的向量值由大到小排序
        cd_sorted = sorted(c_d.items(), key=lambda d: d[1])

    # print(cd_sorted)

    # 取得第一個名字,意味著比較模型之後最相似的那一張圖片的名字
        rec_name = cd_sorted[0][0]

        if (rec_name == "ladygaga"):
            ladygaga_acc = ladygaga_acc + 1
            print(rec_name + ":" + str(ladygaga_acc) + "/59")
            print("Unknow: " + str(unknow))
        if (rec_name == "She"):
            she = she + 1
            print(rec_name + ": " + str(she) + "/59")
            print("Unknow: "+str(unknow))
    except (RuntimeError,IndexError,TypeError,ValueError):
        unknow = unknow + 1
        print("Unknow picture!!!")



# 讀檔
img_path = "./other_img/LADY_GAGA"
files = os.listdir(img_path)
s = []
for img in files:
    s.append(img)

for img in s:
    compareImg(img_path + "/" + img)
