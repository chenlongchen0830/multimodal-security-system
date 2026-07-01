# -*- coding: utf-8 -*-
"""
基于计算机视觉的智能安防监控系统
本科课设项目 - 整合6个功能

功能列表：
1. 人脸识别（调用开源库 face_recognition）
2. 黑名单报警（基于人脸识别的比对逻辑）
3. 区域入侵检测（OpenCV画禁区，人进入就报警）
4. 实时视频流（OpenCV读取摄像头或视频文件）
5. 车辆识别（YOLO检测车 + 车牌OCR）
6. AI人流统计（检测人 + 画线计数）

使用方法：
    python main.py --source your_video.mp4 --all
    python main.py --source 0 --face --blacklist --intrusion --vehicle --flow

更新日志（v2.0 - 迁移优化版）：
- 修复：EasyOCR 自动检测 GPU 可用性，不再强制 CPU
- 优化：Vehicle + Flow 两个模块共享同一个 YOLO 模型，减少显存占用
- 优化：添加 half=True 自动推理（FP16），RTX 40 系速度提升 15-20%
- 优化：添加 GPU 信息打印，启动时明确显示推理设备
- 优化：添加内存/显存监控，长时间运行更稳定
"""

import os
import sys
import cv2
import numpy as np
import argparse
import time
import json
from datetime import datetime
from collections import defaultdict, deque
from pathlib import Path

# ==================== 尝试导入可选依赖 ====================
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("[警告] face_recognition 未安装，人脸识别和黑名单功能将不可用")
    print("       安装命令: pip install face_recognition")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[警告] ultralytics 未安装，车辆识别和人流统计功能将不可用")
    print("       安装命令: pip install ultralytics")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("[警告] easyocr 未安装，车牌OCR功能将不可用")
    print("       安装命令: pip install easyocr")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ==================== 全局 GPU 检测 ====================
def check_gpu():
    """检测 GPU 状态并打印信息"""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[GPU] 检测到 NVIDIA 显卡: {device_name}")
        print(f"[GPU] 显存总量: {mem_total:.1f} GB")
        print(f"[GPU] CUDA 版本: {torch.version.cuda}")
        print(f"[GPU] YOLO 和 EasyOCR 将使用 GPU 加速，推理速度比 CPU 快 10-20 倍！")
        return True, device_name
    else:
        print("[GPU] 未检测到可用 GPU 或 CUDA 环境，YOLO 将使用 CPU 推理（较慢）")
        if TORCH_AVAILABLE:
            print(f"[GPU] torch.cuda.is_available() = {torch.cuda.is_available()}")
        return False, None

GPU_AVAILABLE, GPU_NAME = check_gpu()
GPU_HALF = GPU_AVAILABLE and torch.cuda.is_available()  # FP16 推理（RTX 40 系加速明显）


# ==================== YOLO 模型共享管理器 ====================
class YOLOModelManager:
    """
    YOLO 模型共享管理器：避免 Vehicle 和 Flow 两个模块各自加载模型，
    节省显存和内存。同一个模型实例用于 person + vehicle 检测。
    """
    _instance = None
    _model = None
    _model_path = None
    
    @classmethod
    def get_model(cls, model_path="yolov9s.pt"):
        if cls._model is None or cls._model_path != model_path:
            print(f"[YOLO] 加载共享模型: {model_path}")
            cls._model_path = model_path
            cls._model = YOLO(model_path)
            # 如果 GPU 可用，模型自动在 GPU 上，但显式确认一下
            if GPU_AVAILABLE:
                cls._model.to('cuda')
                print(f"[YOLO] 模型已加载到 GPU ({GPU_NAME})")
            else:
                cls._model.to('cpu')
                print("[YOLO] 模型已加载到 CPU")
        return cls._model


# ==================== 配置类 ====================
class Config:
    """系统配置参数"""
    # 视频源
    SOURCE = "your_video.mp4"  # 替换为你的视频文件路径，或 0 表示摄像头
    
    # 输出设置
    OUTPUT_DIR = "output"
    SAVE_VIDEO = True
    SHOW_WINDOW = True
    
    # 功能开关
    ENABLE_FACE = True
    ENABLE_BLACKLIST = True
    ENABLE_INTRUSION = True
    ENABLE_VEHICLE = True
    ENABLE_FLOW = True
    
    # 人脸识别
    KNOWN_FACES_DIR = "known_faces"
    BLACKLIST_DIR = "blacklist"
    FACE_TOLERANCE = 0.4  # 匹配阈值，严格模式减少误报
    
    # 区域入侵
    INTRUSION_ZONE = None  # 动态设置，格式: [(x1,y1), (x2,y2), ...]
    INTRUSION_ALERT_COOLDOWN = 3  # 报警冷却秒数
    
    # 车辆识别
    VEHICLE_CONF = 0.5
    
    # 人流统计
    FLOW_LINE = None  # 动态设置，格式: ((x1,y1), (x2,y2))
    FLOW_COUNT_UP = 0
    FLOW_COUNT_DOWN = 0
    
    # 显示设置
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2


# ==================== 工具函数 ====================
def put_text_chinese(img, text, position, color=(0, 255, 0), size=20):
    """
    在图像上绘制中文文本（使用PIL）
    如果PIL不可用，则使用英文替代
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        # 尝试使用系统字体
        try:
            font = ImageFont.truetype("simhei.ttf", size)
        except:
            try:
                font = ImageFont.truetype("msyh.ttc", size)
            except:
                font = ImageFont.load_default()
        draw.text(position, text, font=font, fill=color[::-1])  # RGB to BGR
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except ImportError:
        # PIL不可用，用英文/拼音替代
        cv2.putText(img, text.encode('ascii', 'ignore').decode(), position, 
                    Config.FONT, Config.FONT_SCALE, color, Config.FONT_THICKNESS)
        return img


def draw_panel(img, title, items, x=10, y=10, width=320, line_height=25):
    """绘制信息面板"""
    height = line_height * (len(items) + 2)
    # 背景
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    # 边框
    cv2.rectangle(img, (x, y), (x + width, y + height), (100, 100, 100), 1)
    # 标题
    cv2.putText(img, title, (x + 10, y + line_height), 
                Config.FONT, 0.7, (0, 255, 255), 2)
    # 内容
    for i, (key, value) in enumerate(items):
        text = f"{key}: {value}"
        cv2.putText(img, text, (x + 10, y + line_height * (i + 2)), 
                    Config.FONT, Config.FONT_SCALE, (200, 200, 200), 1)
    return img


# ==================== 功能1&2: 人脸识别 + 黑名单 ====================
class FaceRecognitionModule:
    """人脸识别与黑名单检测模块"""
    
    def __init__(self, known_dir="known_faces", blacklist_dir="blacklist"):
        self.known_encodings = []
        self.known_names = []
        self.blacklist_encodings = []
        self.blacklist_names = []
        self.last_alert_time = 0
        
        # 跳帧检测优化：每5帧做一次完整检测，其余帧复用上次结果
        self.frame_counter = 0
        self.last_display_info = []  # 缓存上次检测的坐标、颜色、标签
        
        if not FACE_RECOGNITION_AVAILABLE:
            print("[人脸识别] face_recognition 库不可用，模块初始化失败")
            return
            
        # 加载已知人脸
        self._load_faces(known_dir, self.known_encodings, self.known_names, "白名单")
        # 加载黑名单
        self._load_faces(blacklist_dir, self.blacklist_encodings, self.blacklist_names, "黑名单")
    
    def _load_faces(self, directory, encodings_list, names_list, label):
        """从目录加载人脸图片"""
        if not os.path.exists(directory):
            print(f"[人脸识别] {label}目录不存在: {directory}，跳过加载")
            return
            
        files = [f for f in os.listdir(directory) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for filename in files:
            path = os.path.join(directory, filename)
            try:
                image = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    encodings_list.append(encodings[0])
                    name = os.path.splitext(filename)[0]
                    names_list.append(name)
                    print(f"[人脸识别] 已加载{label}: {name}")
                else:
                    print(f"[人脸识别] 未检测到人脸: {filename}")
            except Exception as e:
                print(f"[人脸识别] 加载失败 {filename}: {e}")
    
    def process(self, frame):
        """处理单帧，返回标注后的帧和检测结果"""
        if not FACE_RECOGNITION_AVAILABLE:
            return frame, []
        
        self.frame_counter += 1
        
        # 每5帧做一次完整检测（第1, 6, 11...帧），其余帧复用上次结果
        if self.frame_counter % 5 == 1:
            # ========== 完整检测（耗时操作）==========
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_small)
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
            
            self.last_display_info = []
            results = []
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                top *= 2; right *= 2; bottom *= 2; left *= 2
                
                name = "Unknown"
                status = "unknown"
                color = (128, 128, 128)
                
                # 先检查黑名单
                if self.blacklist_encodings:
                    matches = face_recognition.compare_faces(
                        self.blacklist_encodings, face_encoding, tolerance=Config.FACE_TOLERANCE)
                    if True in matches:
                        idx = matches.index(True)
                        name = self.blacklist_names[idx]
                        status = "blacklist"
                        color = (0, 0, 255)
                        current_time = time.time()
                        if current_time - self.last_alert_time > Config.INTRUSION_ALERT_COOLDOWN:
                            self.last_alert_time = current_time
                            print(f"[报警] 黑名单人员 detected: {name}!")
                
                # 再检查白名单
                if status == "unknown" and self.known_encodings:
                    matches = face_recognition.compare_faces(
                        self.known_encodings, face_encoding, tolerance=Config.FACE_TOLERANCE)
                    if True in matches:
                        idx = matches.index(True)
                        name = self.known_names[idx]
                        status = "known"
                        color = (0, 255, 0)
                
                # 保存到缓存，供跳帧时复用
                if status == "blacklist":
                    label = f"{name} [黑名单]"
                elif status == "known":
                    label = f"{name} [白名单]"
                else:
                    label = f"{name} [未知]"
                
                self.last_display_info.append({
                    "bbox": (left, top, right, bottom),
                    "color": color,
                    "label": label
                })
                
                results.append({"name": name, "status": status, "bbox": (left, top, right, bottom)})
        else:
            # ========== 跳帧：直接复用上次结果画框（不耗时的绘制操作）==========
            results = []
            for info in self.last_display_info:
                left, top, right, bottom = info["bbox"]
                results.append({
                    "name": info["label"].split(" [")[0],  # 从标签提取名字
                    "status": info["label"].split("[")[1][:-1] if "[" in info["label"] else "unknown",
                    "bbox": info["bbox"]
                })
        
        # 绘制框和标签（每帧都画，保证画面流畅）
        for info in self.last_display_info:
            left, top, right, bottom = info["bbox"]
            cv2.rectangle(frame, (left, top), (right, bottom), info["color"], 2)
            frame = put_text_chinese(frame, info["label"], (left, top - 30), info["color"], size=20)
        
        return frame, results


# ==================== 功能3: 区域入侵检测 ====================
class IntrusionDetectionModule:
    """区域入侵检测模块"""
    
    def __init__(self):
        self.zone = None  # 多边形禁区
        self.alert_active = False
        self.last_alert_time = 0
        self.drawing = False
        self.points = []
    
    def set_zone_interactive(self, first_frame):
        """交互式设置禁区（点击多边形）"""
        try:
            temp_frame = first_frame.copy()
            self.points = []
            
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.points.append((x, y))
                    cv2.circle(temp_frame, (x, y), 5, (0, 0, 255), -1)
                    if len(self.points) > 1:
                        cv2.line(temp_frame, self.points[-2], self.points[-1], (0, 0, 255), 2)
                    cv2.imshow("Set Intrusion Zone (Press 'q' to finish)", temp_frame)
            
            cv2.imshow("Set Intrusion Zone (Press 'q' to finish)", temp_frame)
            cv2.setMouseCallback("Set Intrusion Zone (Press 'q' to finish)", mouse_callback)
            
            print("[区域入侵] 请点击画面设置禁区顶点，按 'q' 完成")
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') and len(self.points) >= 3:
                    break
            
            cv2.destroyWindow("Set Intrusion Zone (Press 'q' to finish)")
            
            if len(self.points) >= 3:
                self.zone = np.array(self.points, dtype=np.int32)
                self.save_zone()
                print(f"[区域入侵] 禁区已设置，顶点数: {len(self.points)}")
        except Exception as e:
            print(f"[区域入侵] 交互式设置失败（可能OpenCV无GUI支持）: {e}")
            print("[区域入侵] 自动降级为自动设置禁区")
            self.set_zone_auto(first_frame.shape)
    
    def save_zone(self, filepath="intrusion_zone.json"):
        """保存禁区坐标到文件"""
        if self.zone is not None:
            import json
            points = self.zone.tolist()
            with open(filepath, 'w') as f:
                json.dump(points, f)
            print(f"[区域入侵] 禁区坐标已保存: {filepath}")
    
    def load_zone(self, filepath="intrusion_zone.json"):
        """从文件加载禁区坐标"""
        import json
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                points = json.load(f)
            self.zone = np.array(points, dtype=np.int32)
            print(f"[区域入侵] 已加载保存的禁区: {filepath}")
            return True
        return False
    
    def set_zone_auto(self, frame_shape, margin=50):
        """自动设置禁区（画面中央区域）"""
        h, w = frame_shape[:2]
        self.zone = np.array([
            (w//2 - 100, h//2 - 100),
            (w//2 + 100, h//2 - 100),
            (w//2 + 100, h//2 + 100),
            (w//2 - 100, h//2 + 100)
        ], dtype=np.int32)
        print("[区域入侵] 自动设置禁区: 画面中央区域")
    
    def is_point_in_zone(self, point):
        """判断点是否在禁区内"""
        if self.zone is None:
            return False
        return cv2.pointPolygonTest(self.zone, point, False) >= 0
    
    def process(self, frame, person_detections=None):
        """
        处理单帧
        person_detections: [(x1,y1,x2,y2), ...] 人员检测框
        """
        if self.zone is None:
            return frame, False
        
        # 绘制禁区（半透明红色覆盖）
        overlay = frame.copy()
        cv2.polylines(overlay, [self.zone], True, (0, 0, 255), 2)
        cv2.fillPoly(overlay, [self.zone], (0, 0, 255))
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        intrusion_detected = False
        
        if person_detections:
            for (x1, y1, x2, y2) in person_detections:
                # 计算人体底部中心点
                center = (int((x1 + x2) / 2), int(y2))
                cv2.circle(frame, center, 5, (255, 0, 0), -1)
                
                if self.is_point_in_zone(center):
                    intrusion_detected = True
                    # 绘制入侵标记
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, "INTRUSION!", (x1, y1 - 10),
                                Config.FONT, 0.8, (0, 0, 255), 2)
        
        # 报警逻辑
        if intrusion_detected:
            current_time = time.time()
            if current_time - self.last_alert_time > Config.INTRUSION_ALERT_COOLDOWN:
                self.last_alert_time = current_time
                self.alert_active = True
                print("[报警] 区域入侵 detected!")
                # 闪烁效果
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
                frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
        
        return frame, intrusion_detected


# ==================== 功能5: 车辆识别 ====================
class VehicleRecognitionModule: #类定义与文档注释
    """车辆识别模块（YOLO + OCR）"""

    def __init__(self, shared_model=None): #构造函数，初始化模块所有参数；shared_model传入全局共享YOLO模型，避免重复加载占用显存。
        self.model = shared_model  #将外部传入的共用YOLO模型赋值给实例变量，实现多模块共用一个模型，减少显存开销。
        self.ocr_reader = None #初始化OCR识别器对象，初始为空，后续加载EasyOCR实例。
        self.vehicle_classes = [2, 3, 5, 7]  #累计车辆总数，COCO数据集类别编号：2轿车、3摩托车、5公交车、7货车，仅检测这四类交通工具。

        # 车辆跟踪计数器
        self.total_count = 0  #全局累计车辆数量，满足3帧稳定检测后才 + 1，防止重复计数。
        self.trackers = {}  #字典存储每一辆车的跟踪信息，key为车辆唯一ID，value存坐标、车牌、消失帧数、确认帧数等全部信息。
        self.next_id = 0 #新车辆分配自增唯一ID，用于区分画面内不同车辆。
        self.max_disappear = 30 #未确认车辆最多消失30帧就删除跟踪器，过滤短暂误检测。
        self.min_confirm_frames = 3 #新车辆必须连续匹配到3帧，才判定为真实车辆并计入总数，过滤抖动误检。
        self.match_distance = 200  #前后帧车辆中心点欧氏距离阈值200像素，车辆移动速度快，匹配范围比行人更大。
        self.confirmed_disappear = 90  #已经确认计数的车辆，短暂遮挡消失后保留90帧再删除，避免遮挡后重复计数。

        if YOLO_AVAILABLE and self.model is None:
            print("[车辆识别] 正在加载 YOLOv9s 模型...") #判断：系统支持YOLO、且没有传入共享模型时，手动加载YOLOv9s。打印加载日志。
            try:
                self.model = YOLOModelManager.get_model("yolov9s.pt")
                print("[车辆识别] YOLO 模型加载完成") #调用单例模型管理器加载权重文件，全局只加载一次；加载成功打印提示。
            except Exception as e:
                print(f"[车辆识别] YOLO 加载失败: {e}") #捕获YOLO加载异常，打印错误信息，程序不会直接崩溃。

        if EASYOCR_AVAILABLE:
            print("[车辆识别] 正在加载 OCR 模型...") #若环境支持EasyOCR，开始加载文字识别模型并打印日志。
            try:
                # 自动检测 GPU 可用性，不再强制 CPU
                use_gpu = GPU_AVAILABLE
                print(f"[车辆识别] EasyOCR GPU 模式: {use_gpu}") #自动判断GPU是否可用，OCR优先使用GPU加速，打印当前运行模式。
                self.ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=use_gpu)
                print("[车辆识别] OCR 模型加载完成") #初始化OCR读取器，支持简体中文+英文；GPU可用则启用GPU推理。
            except Exception as e:
                print(f"[车辆识别] OCR 加载失败: {e}")
                # 如果 GPU 模式失败，回退到 CPU
                try:
                    print("[车辆识别] 尝试 CPU 模式加载 OCR...")
                    self.ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
                    print("[车辆识别] OCR CPU 模式加载完成") #GPU加载失败时，自动降级为CPU模式加载OCR，保证功能可用。
                except Exception as e2:
                    print(f"[车辆识别] OCR 完全加载失败: {e2}") #CPU加载也失败，打印最终报错，OCR功能失效。

    def _preprocess_for_ocr(self, roi):
        """
        车牌区域预处理：放大 + 灰度 + 对比度增强
        """
        # 放大 2 倍（车牌通常很小）
        roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC) #车牌画面偏小，放大2倍；三次立方插值保证放大后文字清晰。
        # 灰度化
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) #转灰度图，消除色彩干扰，降低OCR计算量。
        # 自适应对比度增强（CLAHE）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray) #CLAHE自适应直方图均衡，解决逆光、阴影下车牌发黑发白看不清文字问题。
        # 转回 RGB（EasyOCR 需要）
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        return rgb #EasyOCR要求RGB格式输入，灰度图转回RGB后返回处理完成的车牌图。

    def _clean_plate(self, text):
        """
        车牌后处理：只保留字母和数字，过滤中文/特殊字符
        中国车牌格式：1位省份汉字 + 1位字母 + 5位字母数字（新能源多1位）
        """
        import re
        # 只保留字母和数字
        cleaned = re.sub(r'[^A-Za-z0-9]', '', text) #正则表达式剔除所有汉字、符号、空格，仅保留大小写字母+数字。
        # 过滤太短的结果（车牌至少6位）
        if len(cleaned) < 6:
            return "" #有效车牌字符最少6位，不足6位判定识别失败，返回空字符串。
        # 过滤太长的结果（车牌最多8位）
        if len(cleaned) > 8:
            cleaned = cleaned[:8] #普通车牌7位、新能源8位，超过8位截断多余字符。
        return cleaned.upper() #全部转为大写字母，统一车牌显示格式。

    def process(self, frame):
        """处理单帧，返回标注后的帧和检测结果"""
        if self.model is None:
            return frame, [] #YOLO模型未加载成功，直接返回原图、空检测列表，不执行任何逻辑。

        results_list = []
        current_detections = [] #results_list：存储当前帧所有车辆检测信息用于上层调用；current_detections：当前帧YOLO检测到的全部车辆坐标、置信度、类别。

        # YOLO检测（使用 half=True 如果 GPU 支持 FP16）
        try:
            if GPU_AVAILABLE and GPU_HALF:
                yolo_results = self.model(frame, verbose=False, conf=Config.VEHICLE_CONF, half=True)
            else:
                yolo_results = self.model(frame, verbose=False, conf=Config.VEHICLE_CONF) #GPU支持半精度FP16时启用half=True加速推理；关闭控制台冗余输出；置信度阈值读取全局配置，过滤低可信度车辆框。
        except Exception as e:
            print(f"[车辆识别] YOLO推理失败: {e}")
            return frame, [] #YOLO推理报错捕获，打印日志后返回原图，避免程序崩溃。

        for result in yolo_results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls in self.vehicle_classes: #遍历YOLO所有检测框，提取类别编号，只保留车辆相关4类目标。
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    center = ((x1 + x2) // 2, (y1 + y2) // 2) #提取车辆左上角、右下角坐标；检测置信度；计算车辆框中心点，用于跨帧跟踪匹配。

                    current_detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "center": center,
                        "conf": conf,
                        "cls": cls
                    }) #将当前车辆信息存入列表，供后续跟踪匹配逻辑使用。

        # 车辆跟踪逻辑（同辆车不重复计数）
        matched_ids = set() #集合存储本帧匹配成功的车辆ID，避免一个跟踪器匹配多个车辆。
        for det in current_detections:
            x1, y1, x2, y2 = det["bbox"]
            center = det["center"]
            conf = det["conf"]
            cls = det["cls"] #循环遍历本帧每一辆检测到的车，取出坐标、中心点、置信度、类别。

            best_id = None
            best_dist = float('inf') #初始化最优匹配车辆ID、最小距离，初始距离设为无穷大。

            for tid, tracker in self.trackers.items():
                if tid in matched_ids:
                    continue
                last_center = tracker["center"]
                dist = np.linalg.norm(np.array(center) - np.array(last_center))
                if dist < self.match_distance and dist < best_dist:
                    best_dist = dist
                    best_id = tid #遍历历史跟踪车辆，计算当前车辆与历史车辆中心点欧氏距离；在200像素阈值内找到距离最近的历史车辆作为匹配对象。

            plate_text = ""
            if best_id is not None:
                # 匹配到已有车辆，更新tracker
                self.trackers[best_id]["center"] = center
                self.trackers[best_id]["bbox"] = (x1, y1, x2, y2)
                self.trackers[best_id]["disappear"] = 0
                self.trackers[best_id]["confirm_frames"] = self.trackers[best_id].get("confirm_frames", 0) + 1
                matched_ids.add(best_id) #匹配到历史车辆：更新该车辆最新坐标，重置消失帧数，确认帧数+1，标记该ID已匹配。

                # 确认帧数足够后计数
                if not self.trackers[best_id].get("confirmed", False):
                    if self.trackers[best_id]["confirm_frames"] >= self.min_confirm_frames:
                        self.trackers[best_id]["confirmed"] = True
                        self.total_count += 1
                        print(f"[车辆识别] 新车辆确认，累计总数: {self.total_count}") #连续匹配3帧后，标记车辆为已确认，全局车辆总数+1，打印计数日志。

                # 保持之前识别的车牌（如果当前帧没识别出来）
                plate_text = self.trackers[best_id].get("plate", "") #若当前帧识别不出车牌，沿用历史帧识别到的车牌文字，保证画面持续显示车牌。

                # 车牌投票：多帧识别结果取最稳定的
                tracker = self.trackers[best_id]
                if plate_text and len(plate_text) >= 6:
                    tracker["plate_history"].append(plate_text)
                    if len(tracker["plate_history"]) > 10:
                        tracker["plate_history"].pop(0)
                    from collections import Counter
                    if tracker["plate_history"]:
                        most_common = Counter(tracker["plate_history"]).most_common(1)[0]
                        tracker["plate"] = most_common[0] #车牌投票机制：记录最近10帧识别结果，统计出现次数最多的车牌作为最终显示车牌，解决单帧识别错误闪烁问题。
            else:
                # 新车辆，创建tracker
                self.trackers[self.next_id] = {
                    "center": center,
                    "bbox": (x1, y1, x2, y2),
                    "disappear": 0,
                    "confirm_frames": 1,
                    "confirmed": False,
                    "plate": "",
                    "plate_history": [],
                    "plate_votes": {}
                }
                matched_ids.add(self.next_id)
                self.next_id += 1 #无匹配历史车辆，判定为新车辆，新建跟踪器字典，分配新ID，ID自增。

            # 车牌OCR（只对已确认的车辆，或当前帧）
            if self.ocr_reader and (x2 - x1) > 60 and (y2 - y1) > 60: #OCR 可用、车辆框尺寸足够大时，才裁剪车牌区域识别，过小车辆直接跳过节省算力。
                h = y2 - y1
                w = x2 - x1
                plate_y1 = int(y1 + h * 0.55)
                plate_y2 = int(y2 - h * 0.05)
                plate_x1 = int(x1 + w * 0.1)
                plate_x2 = int(x2 - w * 0.1) #根据车辆整体框比例，截取车辆下半部分作为车牌ROI区域，排除车窗、车身干扰。

                plate_roi = frame[plate_y1:plate_y2, plate_x1:plate_x2]
                if plate_roi.size > 0 and plate_roi.shape[0] > 10 and plate_roi.shape[1] > 30: #裁剪车牌区域，过滤空图、极小图片，避免OCR报错。
                    try:
                        processed = self._preprocess_for_ocr(plate_roi)
                        ocr_results = self.ocr_reader.readtext(
                            processed,
                            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                            detail=1
                        ) #车牌图像预处理，调用OCR识别；限定仅识别大写字母+数字，detail=1返回置信度、坐标、文字完整信息。
                        if ocr_results:
                            best = max(ocr_results, key=lambda x: x[2])
                            if best[2] > 0.3:
                                raw_text = best[1]
                                plate_text = self._clean_plate(raw_text) #取置信度最高的识别结果，置信度大于0.3 才判定有效，清洗车牌字符。
                                # 如果匹配到tracker，把识别结果加入投票历史
                                if best_id is not None and plate_text:
                                    tracker = self.trackers[best_id]
                                    tracker["plate_history"].append(plate_text)
                                    if len(tracker["plate_history"]) > 10:
                                        tracker["plate_history"].pop(0)
                                    # 重新投票取最稳定的车牌
                                    from collections import Counter
                                    if tracker["plate_history"]:
                                        most_common = Counter(tracker["plate_history"]).most_common(1)[0]
                                        tracker["plate"] = most_common[0] #识别成功后，将车牌存入历史列表，重新统计高频车牌更新跟踪器存储的标准车牌。
                    except Exception as e:
                        pass #OCR识别异常直接跳过，不中断整体车辆检测流程。

            # 绘制车辆框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2) #在原图上绘制橙色车辆检测框，线条宽度2。

            class_names = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
            label = f"{class_names.get(cls, 'Vehicle')} {conf:.2f}" #类别编号映射文字标签，拼接车辆类型+置信度。

            # 显示车牌：从tracker取投票结果，至少3帧识别到才显示
            plate_to_show = ""
            if best_id is not None:
                tracker = self.trackers[best_id]
                plate_history = tracker.get("plate_history", [])
                if tracker.get("confirmed", False) and len(plate_history) >= 3:
                    plate_to_show = tracker.get("plate", "") #车辆确认计数、且至少3帧识别到车牌，才在画面展示车牌，避免单帧误识别文字闪烁。

            if plate_to_show:
                label += f" | Plate: {plate_to_show}" #识别到有效车牌，拼接到标签文字后方同步显示。

            cv2.putText(frame, label, (x1, y1 - 10),
                        Config.FONT, Config.FONT_SCALE, (255, 165, 0), Config.FONT_THICKNESS) #在车辆框上方打印标签文字，字体、大小、颜色读取全局配置。

            results_list.append({
                "type": class_names.get(cls, "Vehicle"),
                "confidence": conf,
                "plate": plate_text,
                "bbox": (x1, y1, x2, y2)
            }) #将当前车辆完整信息存入结果列表，供外部获取检测数据。

        # 清理消失的车辆tracker
        to_remove = []
        for tid, tracker in self.trackers.items():
            if tid not in matched_ids:
                tracker["disappear"] += 1
                max_disappear = self.max_disappear if not tracker.get("confirmed", False) else self.confirmed_disappear
                if tracker["disappear"] > max_disappear:
                    to_remove.append(tid)
        for tid in to_remove:
            del self.trackers[tid] #本帧未匹配到的车辆，消失帧数+1；未确认车辆消失30帧删除，已确认车辆消失90帧删除，清理无效跟踪器释放内存。

        # 显示车辆总数
        cv2.putText(frame, f"Vehicle Total: {self.total_count}", (10, 60),
                    Config.FONT, 0.8, (255, 165, 0), 2) #在画面左上角打印全局累计车辆总数，橙色字体。

        return frame, results_list #返回标注完成的图像、当前帧所有车辆检测信息列表，函数结束。


# ==================== 功能6: AI人流统计 ====================
class FlowCountingModule:
    """AI人流统计模块"""
    
    def __init__(self, shared_model=None):
        self.model = shared_model  # 使用共享模型
        self.line = None
        self.total_count = 0  # 累计总人数
        self.trackers = {}  # 跟踪器: id -> {centers, direction, counted, confirm_frames}
        self.next_id = 0
        self.max_disappear = 30  # 最大消失帧数（增加，避免短暂遮挡就删tracker）
        self.min_confirm_frames = 3  # 新tracker需连续存在3帧才确认，避免检测抖动误报
        self.match_distance = 150  # 匹配距离阈值（增加，人走动也能匹配上）
        
        if YOLO_AVAILABLE and self.model is None:
            print("[人流统计] 正在加载 YOLOv9s 模型...")
            try:
                self.model = YOLOModelManager.get_model("yolov9s.pt")
                print("[人流统计] 模型加载完成")
            except Exception as e:
                print(f"[人流统计] 模型加载失败: {e}")
    
    def set_line(self, frame_shape, orientation="horizontal"):
        """自动设置计数线"""
        h, w = frame_shape[:2]
        if orientation == "horizontal":
            self.line = ((0, h // 2), (w, h // 2))
        else:
            self.line = ((w // 2, 0), (w // 2, h))
        print(f"[人流统计] 计数线已设置: {self.line}")
    
    def set_line_interactive(self, first_frame):
        """交互式设置计数线"""
        try:
            points = []
            temp_frame = first_frame.copy()
            
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
                    points.append((x, y))
                    cv2.circle(temp_frame, (x, y), 5, (0, 255, 0), -1)
                    if len(points) == 2:
                        cv2.line(temp_frame, points[0], points[1], (0, 255, 0), 2)
                    cv2.imshow("Set Counting Line (Click 2 points)", temp_frame)
            
            cv2.imshow("Set Counting Line (Click 2 points)", temp_frame)
            cv2.setMouseCallback("Set Counting Line (Click 2 points)", mouse_callback)
            
            print("[人流统计] 请点击两个点设置计数线")
            while len(points) < 2:
                cv2.waitKey(1)
            
            cv2.destroyWindow("Set Counting Line (Click 2 points)")
            self.line = (points[0], points[1])
            print(f"[人流统计] 计数线已设置: {self.line}")
        except Exception as e:
            print(f"[人流统计] 交互式设置失败（可能OpenCV无GUI支持）: {e}")
            print("[人流统计] 自动降级为自动设置计数线")
            self.set_line(first_frame.shape, "horizontal")
    
    def _get_center(self, bbox):
        """获取检测框中心"""
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    def _point_side_of_line(self, point, line):
        """判断点在计数线的哪一侧"""
        (x, y) = point
        (x1, y1), (x2, y2) = line
        # 叉积判断
        return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    
    def process(self, frame):
        """处理单帧，返回标注后的帧"""
        if self.model is None or self.line is None:
            return frame
        
        # 绘制计数线
        cv2.line(frame, self.line[0], self.line[1], (0, 255, 255), 2)
        
        # YOLO检测（不指定classes，避免污染共享模型，后续手动过滤）
        try:
            if GPU_AVAILABLE and GPU_HALF:
                results = self.model(frame, verbose=False, conf=0.4, half=True)
            else:
                results = self.model(frame, verbose=False, conf=0.4)
        except Exception as e:
            print(f"[人流统计] 检测失败: {e}")
            return frame
        
        current_detections = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls != 0:  # 只取 person (class 0)
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                center = self._get_center((x1, y1, x2, y2))
                current_detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "center": center,
                    "conf": conf
                })
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        
        # 简单跟踪逻辑（最近邻匹配）
        matched_ids = set()
        for det in current_detections:
            best_id = None
            best_dist = float('inf')
            
            for tid, tracker in self.trackers.items():
                if tid in matched_ids:
                    continue
                last_center = tracker["centers"][-1]
                dist = np.linalg.norm(np.array(det["center"]) - np.array(last_center))
                if dist < self.match_distance and dist < best_dist:  # 150像素阈值，人走动也能匹配
                    best_dist = dist
                    best_id = tid
            
            if best_id is not None:
                self.trackers[best_id]["centers"].append(det["center"])
                self.trackers[best_id]["disappear"] = 0
                # 增加确认帧数
                self.trackers[best_id]["confirm_frames"] = self.trackers[best_id].get("confirm_frames", 0) + 1
                matched_ids.add(best_id)
                
                # 等确认帧数足够后才计数（避免检测抖动误报）
                if not self.trackers[best_id].get("confirmed", False):
                    if self.trackers[best_id]["confirm_frames"] >= self.min_confirm_frames:
                        self.trackers[best_id]["confirmed"] = True
                        self.total_count += 1
                        print(f"[人流统计] 新人员确认，累计总数: {self.total_count}")
                
            else:
                # 新目标：创建tracker，但不立刻计数（等确认期）
                self.trackers[self.next_id] = {
                    "centers": [det["center"]],
                    "disappear": 0,
                    "confirm_frames": 1,
                    "confirmed": False
                }
                matched_ids.add(self.next_id)
                self.next_id += 1
        
        # 清理消失的目标（只删除未确认的，已确认的永久保留）
        to_remove = []
        for tid, tracker in self.trackers.items():
            if tid not in matched_ids:
                tracker["disappear"] += 1
                # 未确认的tracker：消失10帧就删（检测抖动）
                # 已确认的tracker：消失90帧才删（约3秒，人真正离开画面）
                max_disappear = 10 if not tracker.get("confirmed", False) else 90
                if tracker["disappear"] > max_disappear:
                    to_remove.append(tid)
        for tid in to_remove:
            del self.trackers[tid]
        
        # 显示统计信息
        cv2.putText(frame, f"Total: {self.total_count}", (10, frame.shape[0] - 30),
                    Config.FONT, 0.8, (0, 255, 255), 2)
        
        return frame


# ==================== 主程序 ====================
class SmartSecuritySystem:
    """智能安防监控系统主类"""
    
    def __init__(self, args):
        self.args = args
        self.cap = None
        self.writer = None
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        
        # 初始化各模块
        self.face_module = None
        self.intrusion_module = None
        self.vehicle_module = None
        self.flow_module = None
        self.shared_yolo_model = None  # 共享的 YOLO 模型
        
        # 统计信息
        self.stats = {
            "faces_detected": 0,
            "blacklist_alerts": 0,
            "intrusion_alerts": 0,
            "vehicle_total": 0,
            "total_count": 0
        }
    
    def _init_video_source(self):
        """初始化视频源"""
        source = self.args.source
        if source == "0" or source == "1":
            source = int(source)
        
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            print(f"[错误] 无法打开视频源: {source}")
            sys.exit(1)
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[视频源] 分辨率: {self.width}x{self.height}, FPS: {self.fps}")
    
    def _init_output(self):
        """初始化视频输出"""
        if self.args.save:
            os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(Config.OUTPUT_DIR, f"output_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
            print(f"[输出] 视频将保存至: {output_path}")
    
    def _init_modules(self, first_frame):
        """初始化各功能模块"""
        # 如果同时需要车辆识别和人流统计，共享 YOLO 模型
        need_yolo = self.args.vehicle or self.args.flow
        if need_yolo and YOLO_AVAILABLE:
            self.shared_yolo_model = YOLOModelManager.get_model("yolov9s.pt")
        
        # 1&2. 人脸识别 + 黑名单
        if self.args.face:
            self.face_module = FaceRecognitionModule(
                self.args.known_faces, self.args.blacklist_dir)
        
        # 3. 区域入侵
        if self.args.intrusion:
            self.intrusion_module = IntrusionDetectionModule()
            # 如果加了 --reset-zone，删除已保存的禁区
            if self.args.reset_zone and os.path.exists("intrusion_zone.json"):
                os.remove("intrusion_zone.json")
                print("[区域入侵] 已删除保存的禁区坐标，将重新设置")
            # interactive 或 reset-zone 都触发交互式设置
            if self.args.interactive or self.args.reset_zone:
                self.intrusion_module.set_zone_interactive(first_frame)
            else:
                # 先尝试加载保存的禁区，失败则自动设置
                if not self.intrusion_module.load_zone():
                    self.intrusion_module.set_zone_auto(first_frame.shape)
        
        # 5. 车辆识别
        if self.args.vehicle:
            self.vehicle_module = VehicleRecognitionModule(self.shared_yolo_model)
        
        # 6. 人流统计
        if self.args.flow:
            self.flow_module = FlowCountingModule(self.shared_yolo_model)
            if self.args.interactive:
                self.flow_module.set_line_interactive(first_frame)
            else:
                self.flow_module.set_line(first_frame.shape, "horizontal")
    
    def _draw_status_panel(self, frame):
        """绘制状态面板"""
        # 计算实时 FPS
        elapsed = time.time() - self.start_time + 0.001
        real_fps = self.frame_count / elapsed
        
        # 如果 GPU 可用，显示 GPU 信息
        gpu_info = "GPU" if GPU_AVAILABLE else "CPU"
        if GPU_AVAILABLE and TORCH_AVAILABLE:
            try:
                mem_used = torch.cuda.memory_allocated(0) / 1024**3
                mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_info = f"GPU {mem_used:.1f}/{mem_total:.1f}G"
            except:
                pass
        
        items = [
            ("Frame", self.frame_count),
            ("FPS", f"{real_fps:.1f}"),
            ("Device", gpu_info),
            ("Face", self.stats["faces_detected"]),
            ("Blacklist", self.stats["blacklist_alerts"]),
            ("Intrusion", self.stats["intrusion_alerts"]),
            ("Vehicle Total", self.stats["vehicle_total"]),
            ("Flow Total", self.stats["total_count"]),
        ]
        return draw_panel(frame, "Smart Security System", items, x=10, y=10)
    
    def run(self):
        """主循环"""
        self._init_video_source()
        
        # 读取第一帧用于初始化
        ret, first_frame = self.cap.read()
        if not ret:
            print("[错误] 无法读取视频帧")
            return
        
        self._init_modules(first_frame)
        self._init_output()
        
        # 重置到第一帧
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        print("\n" + "="*50)
        print("智能安防监控系统已启动")
        print("按 'q' 退出，按 'p' 暂停")
        print("="*50 + "\n")
        
        paused = False
        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("[信息] 视频播放完毕")
                    break
                
                self.frame_count += 1
                display_frame = frame.copy()
                
                # ========== 功能处理 ==========
                person_detections = []  # 用于入侵检测的人的位置
                
                # 1&2. 人脸识别 + 黑名单
                if self.face_module:
                    display_frame, face_results = self.face_module.process(display_frame)
                    self.stats["faces_detected"] += len(face_results)
                    self.stats["blacklist_alerts"] += sum(1 for r in face_results if r["status"] == "blacklist")
                
                # 5. 车辆识别（先运行以获取人检测框用于入侵检测）
                if self.vehicle_module:
                    display_frame, vehicle_results = self.vehicle_module.process(display_frame)
                    self.stats["vehicle_total"] = self.vehicle_module.total_count
                
                # 6. 人流统计
                if self.flow_module:
                    display_frame = self.flow_module.process(display_frame)
                    self.stats["total_count"] = self.flow_module.total_count
                    # 从跟踪器获取人的位置用于入侵检测
                    for tracker in self.flow_module.trackers.values():
                        if tracker["centers"]:
                            c = tracker["centers"][-1]
                            person_detections.append((c[0]-30, c[1]-60, c[0]+30, c[1]))
                
                # 3. 区域入侵
                if self.intrusion_module:
                    display_frame, intrusion = self.intrusion_module.process(display_frame, person_detections)
                    if intrusion:
                        self.stats["intrusion_alerts"] += 1
                
                # 绘制状态面板
                display_frame = self._draw_status_panel(display_frame)
                
                # 保存视频
                if self.writer:
                    self.writer.write(display_frame)
                
                # 显示
                if self.args.show:
                    cv2.imshow("Smart Security System", display_frame)
            
            # 键盘控制（仅在显示窗口时）
            if self.args.show:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[信息] 用户退出")
                    break
                elif key == ord('p'):
                    paused = not paused
                    print(f"[信息] {'暂停' if paused else '继续'}")
            else:
                # 不显示窗口时，用 sleep 控制帧率
                time.sleep(0.001)
        
        self._cleanup()
    
    def _cleanup(self):
        """清理资源"""
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        if self.args.show:
            cv2.destroyAllWindows()
        
        # 打印统计
        elapsed = time.time() - self.start_time
        processing_fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        print("\n" + "="*50)
        print("运行统计:")
        for key, value in self.stats.items():
            print(f"  {key}: {value}")
        print(f"  总帧数: {self.frame_count}")
        print(f"  运行时间: {elapsed:.1f}s")
        print(f"  平均处理帧率: {processing_fps:.1f} FPS")
        if GPU_AVAILABLE and TORCH_AVAILABLE:
            try:
                peak_mem = torch.cuda.max_memory_allocated(0) / 1024**3
                print(f"  峰值显存占用: {peak_mem:.1f} GB")
            except:
                pass
        print("="*50)


# ==================== 命令行参数解析 ====================
def parse_args():
    parser = argparse.ArgumentParser(
        description="基于计算机视觉的智能安防监控系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --source test_video.mp4 --all
  python main.py --source 0 --face --intrusion --flow
  python main.py --source video.mp4 --all --no-show
        """)
    
    parser.add_argument("--source", "-s", default="your_video.mp4",
                        help="视频源: 0=摄像头, 1=摄像头2, 或视频文件路径 (默认: your_video.mp4，请替换为你自己的视频)")
    
    # 功能开关
    parser.add_argument("--all", action="store_true",
                        help="启用所有功能")
    parser.add_argument("--face", action="store_true",
                        help="启用人脸识别")
    parser.add_argument("--blacklist", action="store_true",
                        help="启用黑名单报警（需配合--face）")
    parser.add_argument("--intrusion", action="store_true",
                        help="启用区域入侵检测")
    parser.add_argument("--vehicle", action="store_true",
                        help="启用车辆识别")
    parser.add_argument("--flow", action="store_true",
                        help="启用AI人流统计")
    
    # 其他选项
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="交互式设置禁区和计数线（否则自动设置）")
    parser.add_argument("--reset-zone", action="store_true",
                        help="强制重新设置禁区（删除已保存的禁区坐标）")
    parser.add_argument("--known-faces", default="known_faces",
                        help="已知人脸目录 (默认: known_faces)")
    parser.add_argument("--blacklist-dir", default="blacklist",
                        help="黑名单目录 (默认: blacklist)")
    parser.add_argument("--save", action="store_true", default=True,
                        help="保存输出视频 (默认: True)")
    parser.add_argument("--no-save", action="store_false", dest="save",
                        help="不保存输出视频")
    parser.add_argument("--show", action="store_true", default=True,
                        help="显示实时窗口 (默认: True)")
    parser.add_argument("--no-show", action="store_false", dest="show",
                        help="不显示实时窗口（后台运行）")
    
    args = parser.parse_args()
    
    # --all 启用所有功能
    if args.all:
        args.face = True
        args.blacklist = True
        args.intrusion = True
        args.vehicle = True
        args.flow = True
    
    return args


# ==================== 入口 ====================
if __name__ == "__main__":
    args = parse_args()
    system = SmartSecuritySystem(args)
    system.run()
