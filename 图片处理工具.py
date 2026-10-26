import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import io
import sys
import os
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

# ================= 配置 =================
DEFAULT_TARGET_KB = 90
DEFAULT_SCALE_PERCENT = 80
MIN_QUALITY = 30
MAX_ITER = 12
SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# 人像截取范围默认值
DEFAULT_TOP_RATIO = 0.4
DEFAULT_BOTTOM_RATIO = 1.4
DEFAULT_SIDE_RATIO = 0.65

# 图像预处理默认值
DEFAULT_BRIGHTNESS = 5
DEFAULT_CONTRAST = 1.1
DEFAULT_DENOISE = 0
DEFAULT_SHARPEN = 0.0

# YuNet 模型名
YUNET_MODEL_NAME = "face_detection_yunet_2026may.onnx"
_face_detector = None
_face_detector_ready = False


# ================= 资源路径兼容（打包必备） =================
def get_resource_path(relative_path):
    """获取资源文件的绝对路径，兼容开发环境和 PyInstaller 打包环境"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


def _init_yunet():
    global _face_detector, _face_detector_ready
    if _face_detector_ready:
        return _face_detector
    try:
        model_path = get_resource_path(YUNET_MODEL_NAME)
        if not Path(model_path).exists():
            print(f"[警告] 未找到 YuNet 模型: {model_path}")
            _face_detector_ready = True
            return None
        _face_detector = cv2.FaceDetectorYN.create(
            model_path, "", (320, 320),
            score_threshold=0.6, nms_threshold=0.3, top_k=5000)
        _face_detector_ready = True
        print(f"[信息] YuNet 已加载: {model_path}")
        return _face_detector
    except Exception as e:
        print(f"[警告] YuNet 初始化失败: {e}")
        _face_detector_ready = True
        return None


# ================= 核心算法 =================
def preprocess_image(image_bgr, brightness, contrast, denoise, sharpen, log=print):
    """功能1：预处理"""
    img = image_bgr.copy()
    if denoise > 0:
        h_val = float(denoise)
        img = cv2.fastNlMeansDenoisingColored(img, None, h_val, h_val, 7, 21)
        log(f"  预处理：降噪强度 {denoise}")
    if brightness != 0 or abs(contrast - 1.0) > 0.01:
        img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
        log(f"  预处理：亮度 {brightness}, 对比度 {contrast}")
    if sharpen > 0:
        gaussian = cv2.GaussianBlur(img, (0, 0), 3)
        img = cv2.addWeighted(img, 1.0 + sharpen, gaussian, -sharpen, 0)
        log(f"  预处理：锐化 {sharpen}")
    return img


def order_points(pts):
    """严格按左上、右上、右下、左下顺序排列"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def find_card_quad(image_bgr):
    """最经典、最通用的四边形检测方案"""
    h, w = image_bgr.shape[:2]
    scale = 800.0 / max(h, w) if max(h, w) > 800 else 1.0
    small = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)
    # 适度膨胀，把断开的边缘线连起来
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    min_area = small.shape[0] * small.shape[1] * 0.03
    best_quad = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            if area > best_area:
                best_area = area
                best_quad = approx
                break

    if best_quad is not None:
        pts = best_quad.reshape(4, 2).astype("float32") / scale

        # 【关键修改】向外扩张 1.05 倍（即多留 5% 的边缘空间）
        center = np.mean(pts, axis=0)
        pts = center + (pts - center) * 1.05

        return order_points(pts)

    return None


def rectify_and_crop(image_bgr, quad, do_rectify, do_crop, log=print):
    """功能2+3：透视摆正与边缘裁切"""
    if quad is None:
        log("  未检测到证件边缘，降级为图像中心区域自动裁切。")
        h, w = image_bgr.shape[:2]
        margin_x = int(w * 0.05)
        margin_y = int(h * 0.05)
        return image_bgr[margin_y:h - margin_y, margin_x:w - margin_x].copy()

    (tl, tr, br, bl) = quad

    # 精准计算长宽
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    maxWidth = max(int(width_top), int(width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    maxHeight = max(int(height_left), int(height_right))

    if maxWidth < 50 or maxHeight < 50:
        log("  检测区域过小，跳过处理。")
        return image_bgr

    dst = np.array([[0, 0], [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

    if do_crop:
        M = cv2.getPerspectiveTransform(quad.astype("float32"), dst)
        warped = cv2.warpPerspective(image_bgr, M, (maxWidth, maxHeight), flags=cv2.INTER_CUBIC)
        log(f"  透视摆正 & 边缘裁切完成（已向外扩张），尺寸 {maxWidth}x{maxHeight}")
        return warped
    else:
        M = cv2.getPerspectiveTransform(quad.astype("float32"), dst)
        h, w = image_bgr.shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
        transformed_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), M).reshape(-1, 2)

        min_x, min_y = np.min(transformed_corners, axis=0)
        max_x, max_y = np.max(transformed_corners, axis=0)

        new_w = int(np.ceil(max_x - min_x))
        new_h = int(np.ceil(max_y - min_y))

        final_M = cv2.getPerspectiveTransform(quad.astype("float32"),
                                              (dst + np.array([-min_x, -min_y], dtype="float32")))

        warped = cv2.warpPerspective(image_bgr, final_M, (new_w, new_h),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(255, 255, 255))
        log(f"  透视摆正完成（保留完整背景），尺寸 {new_w}x{new_h}")
        return warped


def to_pil_rgb(image_bgr):
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def encode_jpeg(pil_img, quality):
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=quality, optimize=True)
    return buf.getvalue()


def compress_pil_to_target(pil_img, target_kb, log=print):
    """功能4：缩小文件大小"""
    target_bytes = target_kb * 1024
    lo, hi = MIN_QUALITY, 95
    best_data, best_q = None, MIN_QUALITY

    data_low = encode_jpeg(pil_img, lo)
    if len(data_low) <= target_bytes:
        best_data, best_q = data_low, lo
    else:
        for _ in range(MAX_ITER):
            mid = (lo + hi) // 2
            data = encode_jpeg(pil_img, mid)
            if len(data) <= target_bytes:
                best_data, best_q = data, mid
                lo = mid + 1
            else:
                hi = mid - 1
            if lo > hi:
                break

    if best_data is None or len(best_data) > target_bytes:
        ratio = (target_bytes / len(data_low)) ** 0.5 * 0.95
        new_w = max(10, int(pil_img.width * ratio))
        new_h = max(10, int(pil_img.height * ratio))
        resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        best_data = encode_jpeg(resized, MIN_QUALITY)
        best_q = MIN_QUALITY
        log(f"  尺寸强制缩小至 {new_w}x{new_h} 以限制大小")
    return best_data, best_q


def scale_pil_by_percent(pil_img, percent, log=print):
    """功能5：缩放尺寸"""
    ratio = max(1.0, min(percent, 500.0)) / 100.0
    new_w = max(1, int(round(pil_img.width * ratio)))
    new_h = max(1, int(round(pil_img.height * ratio)))
    resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    log(f"  尺寸缩放：{pil_img.width}x{pil_img.height} -> {new_w}x{new_h} ({percent:g}%)")
    return resized


def save_plain_jpg(pil_img, output_path, log=print):
    data = encode_jpeg(pil_img, 95)
    with open(output_path, 'wb') as f:
        f.write(data)
    log(f"  已保存 {Path(output_path).name} (质量=95, {len(data) / 1024:.1f}KB)")


def detect_face_bbox(image_bgr, log=print):
    """功能6：YuNet 检测最大人脸"""
    detector = _init_yunet()
    if detector is None:
        log("  YuNet 不可用，跳过人脸检测。")
        return None

    h, w = image_bgr.shape[:2]
    detector.setInputSize((w, h))

    try:
        _, faces = detector.detect(image_bgr)
    except Exception as e:
        log(f"  YuNet 检测异常: {e}")
        return None

    if faces is None or len(faces) == 0:
        return None

    best = None
    best_area = 0
    for face in faces:
        x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        score = float(face[14]) if len(face) > 14 else 1.0
        area = fw * fh
        if area > best_area and fw > 5 and fh > 5:
            best_area = area
            best = (x, y, fw, fh)
            log(f"  检出人脸 ({x},{y},{fw},{fh}) score={score:.2f}")
    return best


def extract_portrait(image_bgr, face_bbox, top_r, bottom_r, side_r, log=print):
    """功能6：截取人像"""
    if face_bbox is None:
        return None

    fx, fy, fw, fh = face_bbox
    h, w = image_bgr.shape[:2]
    cx = fx + fw / 2.0

    top = fy - fh * top_r
    bottom = fy + fh * bottom_r
    left = cx - fw * side_r
    right = cx + fw * side_r

    target_ratio = 3.0 / 4.0
    tw, th = right - left, bottom - top

    if tw / th > target_ratio:
        shift = (tw / target_ratio - th) / 2
        top -= shift
        bottom += shift
    else:
        shift = (th * target_ratio - tw) / 2
        left -= shift
        right += shift

    left, top = max(0, int(left)), max(0, int(top))
    right, bottom = min(w, int(right)), min(h, int(bottom))

    if right - left < 20 or bottom - top < 20:
        log("  人像区域过小，跳过。")
        return None

    portrait = image_bgr[top:bottom, left:right].copy()
    log(f"  人像截取完成，{right - left}x{bottom - top}")
    return portrait


# ================= 主处理流程 =================
def process_one(input_path, output_dir, target_kb, scale_percent,
                do_preprocess, brightness, contrast, denoise, sharpen,
                do_rectify, do_crop, do_compress, do_scale, do_portrait,
                top_r, bottom_r, side_r, log=print):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"处理: {input_path.name}")
    img_bytes = np.fromfile(str(input_path), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if img is None:
        log("  无法读取，跳过。")
        return False

    # 1. 预处理
    if do_preprocess:
        img = preprocess_image(img, brightness, contrast, denoise, sharpen, log=log)
    else:
        log("  [跳过] 功能1：图像预处理")

    # 2. 几何校正（合并检测）
    if do_rectify or do_crop:
        quad = find_card_quad(img)
        img = rectify_and_crop(img, quad, do_rectify, do_crop, log=log)
    else:
        log("  [跳过] 功能2/3：透视与裁切")

    # 3. 人像截取
    portrait_pil = None
    if do_portrait:
        log("  [执行] 功能6：人像截取")
        face_bbox = detect_face_bbox(img, log=log)
        if face_bbox is None:
            log("  未检测到人脸，跳过人像截取。")
        else:
            portrait_bgr = extract_portrait(img, face_bbox, top_r, bottom_r, side_r, log=log)
            if portrait_bgr is not None:
                portrait_pil = to_pil_rgb(portrait_bgr)
    else:
        log("  [跳过] 功能6：人像截取")

    pil_img = to_pil_rgb(img)

    # 4. 尺寸缩放
    if do_scale:
        pil_img = scale_pil_by_percent(pil_img, scale_percent, log=log)
        if portrait_pil is not None:
            portrait_pil = scale_pil_by_percent(portrait_pil, scale_percent,
                                                log=lambda m: log("  [人像] " + m))
    else:
        log("  [跳过] 功能5：缩放尺寸")

    out_name = input_path.stem + ".jpg"
    out_path = output_dir / out_name

    # 5. 压缩文件大小
    if do_compress:
        data, q = compress_pil_to_target(pil_img, target_kb, log=log)
        with open(out_path, 'wb') as f:
            f.write(data)
        log(f"  已保存 {out_name} (质量={q}, {len(data) / 1024:.1f}KB)")
    else:
        log("  [跳过] 功能4：压缩大小")
        save_plain_jpg(pil_img, out_path, log=log)

    # 保存人像
    if portrait_pil is not None:
        portrait_name = input_path.stem + "_portrait.jpg"
        portrait_path = output_dir / portrait_name
        if do_compress:
            pdata, pq = compress_pil_to_target(portrait_pil, target_kb,
                                               log=lambda m: log("  [人像] " + m))
            with open(portrait_path, 'wb') as f:
                f.write(pdata)
            log(f"  已保存人像 {portrait_name} (质量={pq}, {len(pdata) / 1024:.1f}KB)")
        else:
            save_plain_jpg(portrait_pil, portrait_path,
                           log=lambda m: log("  [人像] " + m))

    return True


# ================= GUI =================
class CardProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("证件照片处理工具--符秀梅又高又帅的老公全力支持！")
        self.root.geometry("900x950")
        self.root.minsize(820, 850)

        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.mode = tk.StringVar(value="file")

        self.target_kb = tk.StringVar(value=str(DEFAULT_TARGET_KB))
        self.scale_percent = tk.StringVar(value=str(DEFAULT_SCALE_PERCENT))

        self.do_preprocess = tk.BooleanVar(value=True)
        self.do_rectify = tk.BooleanVar(value=True)
        self.do_crop = tk.BooleanVar(value=True)
        self.do_compress = tk.BooleanVar(value=True)
        self.do_scale = tk.BooleanVar(value=False)
        self.do_portrait = tk.BooleanVar(value=False)

        self.top_ratio = tk.StringVar(value=str(DEFAULT_TOP_RATIO))
        self.bottom_ratio = tk.StringVar(value=str(DEFAULT_BOTTOM_RATIO))
        self.side_ratio = tk.StringVar(value=str(DEFAULT_SIDE_RATIO))

        self.brightness = tk.StringVar(value=str(DEFAULT_BRIGHTNESS))
        self.contrast = tk.StringVar(value=str(DEFAULT_CONTRAST))
        self.denoise = tk.StringVar(value=str(DEFAULT_DENOISE))
        self.sharpen = tk.StringVar(value=str(DEFAULT_SHARPEN))

        self.msg_queue = queue.Queue()
        self.worker = None

        self._build_ui()
        self._poll_queue()

    def _build_ui(self):
        pad = {'padx': 10, 'pady': 4}

        mode_frame = ttk.LabelFrame(self.root, text="处理模式")
        mode_frame.pack(fill="x", **pad)
        ttk.Radiobutton(mode_frame, text="处理单张图片", variable=self.mode,
                        value="file", command=self._on_mode_change).pack(side="left", padx=12, pady=6)
        ttk.Radiobutton(mode_frame, text="处理整个文件夹", variable=self.mode,
                        value="folder", command=self._on_mode_change).pack(side="left", padx=12, pady=6)

        in_frame = ttk.LabelFrame(self.root, text="选择输入")
        in_frame.pack(fill="x", **pad)
        ttk.Entry(in_frame, textvariable=self.input_path).pack(
            side="left", fill="x", expand=True, padx=(12, 6), pady=6)
        ttk.Button(in_frame, text="浏览...", command=self._browse_input).pack(
            side="right", padx=(0, 12), pady=6)

        out_frame = ttk.LabelFrame(self.root, text="选择输出文件夹（留空则自动创建）")
        out_frame.pack(fill="x", **pad)
        ttk.Entry(out_frame, textvariable=self.output_path).pack(
            side="left", fill="x", expand=True, padx=(12, 6), pady=6)
        ttk.Button(out_frame, text="浏览...", command=self._browse_output).pack(
            side="right", padx=(0, 12), pady=6)

        func_frame = ttk.LabelFrame(self.root, text="选择要执行的功能（可任意组合）")
        func_frame.pack(fill="x", **pad)

        # 功能1：预处理
        row1 = ttk.Frame(func_frame)
        row1.pack(fill="x", padx=12, pady=(6, 2))
        ttk.Checkbutton(row1, text="功能1：图像预处理", variable=self.do_preprocess).pack(side="left")

        row1_sub = ttk.Frame(func_frame)
        row1_sub.pack(fill="x", padx=30, pady=(0, 4))
        ttk.Label(row1_sub, text="└─ 亮度").pack(side="left")
        ttk.Spinbox(row1_sub, from_=-100, to=100, increment=5, width=5, textvariable=self.brightness).pack(side="left",
                                                                                                           padx=(2, 8))
        ttk.Label(row1_sub, text="对比度").pack(side="left")
        ttk.Spinbox(row1_sub, from_=0.5, to=3.0, increment=0.1, width=5, textvariable=self.contrast).pack(side="left",
                                                                                                          padx=(2, 8))
        ttk.Label(row1_sub, text="降噪").pack(side="left")
        ttk.Spinbox(row1_sub, from_=0, to=50, increment=1, width=5, textvariable=self.denoise).pack(side="left",
                                                                                                    padx=(2, 8))
        ttk.Label(row1_sub, text="锐化").pack(side="left")
        ttk.Spinbox(row1_sub, from_=0.0, to=2.0, increment=0.1, width=5, textvariable=self.sharpen).pack(side="left",
                                                                                                         padx=(2, 2))

        # 功能2+3：透视校正与裁切
        row2 = ttk.Frame(func_frame)
        row2.pack(fill="x", padx=12, pady=2)
        ttk.Checkbutton(row2, text="功能2：自动校正倾斜 (含透视摆正)", variable=self.do_rectify).pack(side="left")

        row3 = ttk.Frame(func_frame)
        row3.pack(fill="x", padx=12, pady=2)
        ttk.Checkbutton(row3, text="功能3：沿证件边缘裁切", variable=self.do_crop).pack(side="left")
        ttk.Label(row3, text="(建议将功能2与功能3同时勾选，一步到位)", foreground="#777").pack(side="left", padx=6)

        # 功能4：压缩文件大小
        row4 = ttk.Frame(func_frame)
        row4.pack(fill="x", padx=12, pady=2)
        ttk.Checkbutton(row4, text="功能4：保持比例缩小文件大小至", variable=self.do_compress).pack(side="left")
        ttk.Spinbox(row4, from_=10, to=5000, increment=10, width=7, textvariable=self.target_kb).pack(side="left",
                                                                                                      padx=(4, 2))
        ttk.Label(row4, text="KB").pack(side="left")

        # 功能5：缩放尺寸
        row5 = ttk.Frame(func_frame)
        row5.pack(fill="x", padx=12, pady=2)
        ttk.Checkbutton(row5, text="功能5：保持比例缩放尺寸至", variable=self.do_scale).pack(side="left")
        ttk.Spinbox(row5, from_=1, to=500, increment=5, width=7, textvariable=self.scale_percent).pack(side="left",
                                                                                                       padx=(4, 2))
        ttk.Label(row5, text="%").pack(side="left")

        # 功能6：人像截取
        row6 = ttk.Frame(func_frame)
        row6.pack(fill="x", padx=12, pady=(2, 2))
        ttk.Checkbutton(row6, text="功能6：识别人像另存", variable=self.do_portrait).pack(side="left")
        model_ok = "模型已就绪" if Path(get_resource_path(YUNET_MODEL_NAME)).exists() else "模型未找到，请下载 ONNX 文件"
        ttk.Label(row6, text=f"（YuNet：{model_ok}）", foreground="#777").pack(side="left", padx=6)

        row6_sub = ttk.Frame(func_frame)
        row6_sub.pack(fill="x", padx=30, pady=(0, 6))
        ttk.Label(row6_sub, text="└─ 截取微调：头顶留白").pack(side="left")
        ttk.Spinbox(row6_sub, from_=0.1, to=2.0, increment=0.1, width=5, textvariable=self.top_ratio).pack(side="left",
                                                                                                           padx=(2, 6))
        ttk.Label(row6_sub, text="倍脸高，下巴以下").pack(side="left")
        ttk.Spinbox(row6_sub, from_=0.5, to=3.0, increment=0.1, width=5, textvariable=self.bottom_ratio).pack(
            side="left", padx=(2, 6))
        ttk.Label(row6_sub, text="倍脸高，左右").pack(side="left")
        ttk.Spinbox(row6_sub, from_=0.3, to=2.0, increment=0.05, width=5, textvariable=self.side_ratio).pack(
            side="left", padx=(2, 2))
        ttk.Label(row6_sub, text="倍脸宽").pack(side="left")

        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill="x", **pad)
        self.start_btn = ttk.Button(btn_frame, text="开始处理", command=self._start)
        self.start_btn.pack(side="left", padx=12, pady=4)
        ttk.Button(btn_frame, text="清空日志", command=self._clear_log).pack(side="left", padx=6, pady=4)
        ttk.Button(btn_frame, text="打开输出文件夹", command=self._open_output).pack(side="left", padx=6, pady=4)

        self.progress = ttk.Progressbar(self.root, mode="determinate")
        self.progress.pack(fill="x", padx=12, pady=(2, 4))

        log_frame = ttk.LabelFrame(self.root, text="处理日志")
        log_frame.pack(fill="both", expand=True, **pad)
        self.log_text = ScrolledText(log_frame, height=10, state="disabled", wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=8, pady=8)

    def _on_mode_change(self):
        self.input_path.set("")

    def _browse_input(self):
        if self.mode.get() == "file":
            path = filedialog.askopenfilename(
                title="选择证件照片",
                filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"), ("所有文件", "*.*")])
        else:
            path = filedialog.askdirectory(title="选择包含证件照片的文件夹")
        if path:
            self.input_path.set(path)

    def _browse_output(self):
        path = filedialog.askdirectory(title="选择输出文件夹")
        if path:
            self.output_path.set(path)

    def _open_output(self):
        out = self.output_path.get().strip()
        if not out:
            inp = self.input_path.get().strip()
            if not inp:
                messagebox.showinfo("提示", "请先选择输入路径。")
                return
            p = Path(inp)
            out = str(p.parent / "processed_output" if p.is_file() else p / "processed_output")
        p = Path(out)
        if not p.exists():
            messagebox.showinfo("提示", "输出文件夹尚不存在，处理完成后会自动创建。")
            return
        try:
            import subprocess, sys as _sys, os
            if _sys.platform.startswith("win"):
                os.startfile(str(p))
            elif _sys.platform == "darwin":
                subprocess.Popen(["open", str(p)])
            else:
                subprocess.Popen(["xdg-open", str(p)])
        except Exception as e:
            messagebox.showerror("无法打开", f"打开文件夹失败：{e}")

    def _clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

    def _log(self, msg):
        self.msg_queue.put(("log", msg))

    def _poll_queue(self):
        try:
            while True:
                kind, data = self.msg_queue.get_nowait()
                if kind == "log":
                    self.log_text.config(state="normal")
                    self.log_text.insert("end", str(data) + "\n")
                    self.log_text.see("end")
                    self.log_text.config(state="disabled")
                elif kind == "progress":
                    self.progress["maximum"] = data[1]
                    self.progress["value"] = data[0]
                elif kind == "done":
                    self.start_btn.config(state="normal")
                    self.progress["value"] = self.progress["maximum"]
                elif kind == "error":
                    messagebox.showerror("错误", data)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _start(self):
        inp = self.input_path.get().strip()
        if not inp:
            messagebox.showwarning("提示", "请先选择输入图片或文件夹。")
            return
        p = Path(inp)
        if not p.exists():
            messagebox.showerror("错误", f"路径不存在：\n{p}")
            return

        do_preprocess = self.do_preprocess.get()
        do_rectify = self.do_rectify.get()
        do_crop = self.do_crop.get()
        do_compress = self.do_compress.get()
        do_scale = self.do_scale.get()
        do_portrait = self.do_portrait.get()

        brightness, contrast, denoise, sharpen = 0, 1.0, 0, 0.0
        if do_preprocess:
            try:
                brightness = float(self.brightness.get())
                contrast = float(self.contrast.get())
                denoise = float(self.denoise.get())
                sharpen = float(self.sharpen.get())
            except ValueError:
                messagebox.showerror("错误", "功能1的参数必须是数字。")
                return

        target_kb = DEFAULT_TARGET_KB
        if do_compress:
            try:
                target_kb = int(float(self.target_kb.get()))
                if target_kb <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("错误", "功能4的目标大小必须是正整数（KB）。")
                return

        scale_percent = DEFAULT_SCALE_PERCENT
        if do_scale:
            try:
                scale_percent = float(self.scale_percent.get())
                if scale_percent <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("错误", "功能5的缩放百分比必须是正数。")
                return

        top_r = DEFAULT_TOP_RATIO
        bottom_r = DEFAULT_BOTTOM_RATIO
        side_r = DEFAULT_SIDE_RATIO
        if do_portrait:
            try:
                top_r = float(self.top_ratio.get())
                bottom_r = float(self.bottom_ratio.get())
                side_r = float(self.side_ratio.get())
            except ValueError:
                messagebox.showerror("错误", "功能6的微调参数必须是数字。")
                return

        out = self.output_path.get().strip()
        if not out:
            out = str(p.parent / "processed_output" if p.is_file() else p / "processed_output")

        self.start_btn.config(state="disabled")
        self.progress["value"] = 0
        self.progress["maximum"] = 1
        self._log("=" * 60)

        feats = [
            "功能1(预处理)" if do_preprocess else "功能1(跳过)",
            "功能2/3(透视与裁切)" if (do_rectify or do_crop) else "功能2/3(跳过)",
            f"功能4({target_kb:g}KB)" if do_compress else "功能4(跳过)",
            f"功能5({scale_percent:g}%)" if do_scale else "功能5(跳过)",
            "功能6(人像)" if do_portrait else "功能6(跳过)",
        ]
        self._log("本次启用: " + " + ".join(feats))

        self.worker = threading.Thread(
            target=self._run_job,
            args=(p, Path(out), target_kb, scale_percent,
                  do_preprocess, brightness, contrast, denoise, sharpen,
                  do_rectify, do_crop, do_compress, do_scale, do_portrait,
                  top_r, bottom_r, side_r),
            daemon=True)
        self.worker.start()

    def _run_job(self, target_path, out_dir, target_kb, scale_percent,
                 do_preprocess, brightness, contrast, denoise, sharpen,
                 do_rectify, do_crop, do_compress, do_scale, do_portrait,
                 top_r, bottom_r, side_r):
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            if target_path.is_file():
                if target_path.suffix.lower() not in SUPPORTED_EXT:
                    self._log(f"不支持的文件类型: {target_path.suffix}")
                    self.msg_queue.put(("error", f"不支持的文件类型：{target_path.suffix}"))
                    self.msg_queue.put(("done", None))
                    return
                self.msg_queue.put(("progress", (0, 1)))
                process_one(target_path, out_dir, target_kb, scale_percent,
                            do_preprocess, brightness, contrast, denoise, sharpen,
                            do_rectify, do_crop, do_compress, do_scale, do_portrait,
                            top_r, bottom_r, side_r, log=self._log)
                self.msg_queue.put(("progress", (1, 1)))
                self._log(f"\n处理完成！输出文件夹：{out_dir}")
            else:
                files = [f for f in target_path.iterdir()
                         if f.is_file() and f.suffix.lower() in SUPPORTED_EXT]
                if not files:
                    self._log("文件夹内没有支持的图片文件。")
                    self.msg_queue.put(("done", None))
                    return
                total = len(files)
                self._log(f"共发现 {total} 个图片文件。\n")
                success = 0
                for i, f in enumerate(files, 1):
                    self.msg_queue.put(("progress", (i - 1, total)))
                    try:
                        if process_one(f, out_dir, target_kb, scale_percent,
                                       do_preprocess, brightness, contrast, denoise, sharpen,
                                       do_rectify, do_crop, do_compress, do_scale,
                                       do_portrait, top_r, bottom_r, side_r,
                                       log=self._log):
                            success += 1
                    except Exception as e:
                        self._log(f"  处理失败：{e}")
                    self._log("")
                self.msg_queue.put(("progress", (total, total)))
                self._log(f"完成：成功处理 {success}/{total} 个文件。")
                self._log(f"输出文件夹：{out_dir}")
            self.msg_queue.put(("done", None))
        except Exception as e:
            self._log(f"发生错误：{e}")
            self.msg_queue.put(("error", str(e)))
            self.msg_queue.put(("done", None))


def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    CardProcessorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()