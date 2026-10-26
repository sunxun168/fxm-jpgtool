import cv2
import numpy as np
import mediapipe as mp

img_path = r"D:\图片处理工具\111.jpg"
img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
if img is None:
    raise SystemExit("图片读取失败，请确认文件存在且格式正确")
print("图片尺寸:", img.shape)

# --- MediaPipe ---
try:
    det = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = det.process(rgb)
    print("MediaPipe 原图检出数:", len(res.detections) if res.detections else 0)
    if res.detections:
        for d in res.detections:
            b = d.location_data.relative_bounding_box
            print("  bbox:", b.xmin, b.ymin, b.width, b.height, "score:", d.score)

    # 放大 2 倍再检测
    big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    res2 = det.process(cv2.cvtColor(big, cv2.COLOR_BGR2RGB))
    print("MediaPipe 2x放大检出数:", len(res2.detections) if res2.detections else 0)
except Exception as e:
    print("MediaPipe 不可用:", e)

# --- Haar ---
cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_eq = cv2.equalizeHist(gray)
for name, g in [("原灰度", gray), ("均衡化", gray_eq)]:
    faces = cascade.detectMultiScale(g, scaleFactor=1.05,
                                     minNeighbors=3, minSize=(20, 20))
    print(f"Haar({name}) 检出数:", len(faces))