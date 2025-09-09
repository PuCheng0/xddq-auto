import sys
import os
# Set Qt DPI behavior early on Windows to avoid access denied warning and ensure consistent scaling
if os.name == "nt":
    os.environ.setdefault("QT_QPA_PLATFORM", "windows:dpiawareness=1")
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, QRect, QPoint, Signal, QObject, QSettings
from PySide6.QtGui import QPainter, QColor, QPen, QGuiApplication
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel,
    QTextEdit, QCheckBox, QGroupBox, QGridLayout, QDoubleSpinBox, QLineEdit, QFileDialog, QScrollArea, QTabWidget, QProgressBar, QFrame
)

# Image/automation libs
import numpy as np
import cv2
import pyautogui
import pytesseract

# ---------- Utilities ----------

def load_image(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    try:
        # Read as binary to support Unicode paths and avoid Windows-specific issues
        with open(path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            # Fallback to cv2.imread if imdecode failed
            img = cv2.imread(path, cv2.IMREAD_COLOR)
        return img
    except Exception:
        try:
            # Final fallback attempt
            return cv2.imread(path, cv2.IMREAD_COLOR)
        except Exception:
            return None


def screenshot_region(region: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    if region is None:
        shot = pyautogui.screenshot()
    else:
        x, y, w, h = region
        shot = pyautogui.screenshot(region=(x, y, w, h))
    return cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)


def match_template_with_bbox(screen_bgr: np.ndarray, template_bgr: np.ndarray, confidence: float = 0.8) -> Optional[Tuple[int, int, int, int]]:
    try:
        screen_gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)
        templ_gray_orig = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

        # Try multiple scales to handle DPI/zoom differences
        scales = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
        best = None  # (max_val, x, y, w, h)
        for s in scales:
            if s == 1.0:
                templ_gray = templ_gray_orig
            else:
                h0, w0 = templ_gray_orig.shape[:2]
                w = max(1, int(w0 * s))
                h = max(1, int(h0 * s))
                templ_gray = cv2.resize(templ_gray_orig, (w, h), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC)
            if templ_gray.shape[0] > screen_gray.shape[0] or templ_gray.shape[1] > screen_gray.shape[1]:
                continue
            res = cv2.matchTemplate(screen_gray, templ_gray, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if best is None or max_val > best[0]:
                h, w = templ_gray.shape[:2]
                x1, y1 = max_loc
                best = (max_val, x1, y1, w, h)
        if best and best[0] >= confidence:
            return (best[1], best[2], best[3], best[4])
        return None
    except Exception:
        return None


def match_template(screen_bgr: np.ndarray, template_bgr: np.ndarray, confidence: float = 0.8) -> Optional[Tuple[int, int]]:
    bbox = match_template_with_bbox(screen_bgr, template_bgr, confidence)
    if bbox is None:
        return None
    x1, y1, w, h = bbox
    return (x1 + w // 2, y1 + h // 2)


def click_point(x: int, y: int, move_duration: float = 0.2, clicks: int = 1, interval: float = 0.2):
    pyautogui.moveTo(x, y, duration=move_duration)
    for i in range(clicks):
        pyautogui.click()
        if i < clicks - 1:
            time.sleep(interval)


# ---------- Region Selector Overlay ----------

class RegionOverlay(QWidget):
    regionSelected = Signal(QRect)

    def __init__(self):
        super().__init__(None, Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFocusPolicy(Qt.StrongFocus)
        self.start_pos: Optional[QPoint] = None
        self.end_pos: Optional[QPoint] = None
        self.setMouseTracking(True)
        geometry = QGuiApplication.primaryScreen().geometry()
        for scr in QGuiApplication.screens():
            geometry = geometry.united(scr.geometry())
        self.setGeometry(geometry)

    def paintEvent(self, event):
        if not (self.start_pos and self.end_pos):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.fillRect(self.rect(), QColor(0, 0, 0, 80))
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 80))
        rect = QRect(self.start_pos, self.end_pos).normalized()
        painter.setPen(QPen(QColor(46, 204, 113, 255), 2, Qt.SolidLine))
        painter.fillRect(rect, QColor(46, 204, 113, 60))
        painter.drawRect(rect)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = event.globalPosition().toPoint()
            self.end_pos = self.start_pos
            self.update()

    def mouseMoveEvent(self, event):
        if self.start_pos is not None:
            self.end_pos = event.globalPosition().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.start_pos and self.end_pos:
            rect = QRect(self.start_pos, self.end_pos).normalized()
            self.regionSelected.emit(rect)
            self.close()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Escape, Qt.Key_Q):
            self.close()


# ---------- Worker Thread ----------

@dataclass
class ResourceConfig:
    tiers: Dict[str, List[int]]


class AutomationWorker(QObject):
    logSignal = Signal(str)
    finished = Signal()
    progressSignal = Signal(int, str)

    def __init__(self, assets_dir: str, region: Optional[Tuple[int, int, int, int]], resource_cfg: ResourceConfig, confidence: float = 0.82, relax_threshold: bool = True):
        super().__init__()
        self.assets_dir = assets_dir
        self.region = region
        self.resource_cfg = resource_cfg
        self._stop = threading.Event()
        self.confidence = float(max(0.5, min(0.99, confidence)))
        self.relax_threshold = bool(relax_threshold)

        self.templates: Dict[str, Optional[np.ndarray]] = {}
        names = [
            "target.png", "洞府.png", "福地.png", "探寻.png", "刷新.png", "前往.png",
            "关闭.png", "加号.png", "采集.png", "采集完成.png", "采集时间.png", "无鼠.png"
        ]
        resource_names = [
            "三级桃.png", "四级桃.png", "五级桃.png",
            "三级玻璃珠.png", "四级玻璃珠.png", "五级玻璃珠.png",
            "三级仙玉.png", "四级仙玉.png", "五级仙玉.png",
            "三级净瓶水.png", "四级净瓶水.png", "五级净瓶水.png",
            "三级天衍令.png", "四级天衍令.png", "五级天衍令.png",
        ]
        for n in names + resource_names:
            self.templates[n] = load_image(os.path.join(self.assets_dir, n))

    def stop(self):
        self._stop.set()

    def log(self, text: str):
        try:
            self.logSignal.emit(text)
        except RuntimeError:
            # UI may have been closed or signal source deleted; ignore safely
            pass

    def find_and_click(self, name: str, attempts: int = 3) -> bool:
        templ = self.templates.get(name)
        if templ is None:
            self.log(f"模板缺失: {name}")
            return False
        # First try within selected region (if any)
        for i in range(attempts):
            if self._stop.is_set():
                return False
            scr = screenshot_region(self.region)
            pt = match_template(scr, templ, self.confidence)
            if pt is not None:
                x, y = pt
                if self.region is not None:
                    x += self.region[0]
                    y += self.region[1]
                click_point(x, y, move_duration=0.12)
                self.log(f"点击: {name}")
                return True
            time.sleep(0.5)
        # Region failed; try full-screen fallback if a region was set
        if self.region is not None:
            scr = screenshot_region(None)
            pt = match_template(scr, templ, self.confidence)
            if pt is not None:
                x, y = pt  # full-screen coords, no offset
                click_point(x, y, move_duration=0.12)
                self.log(f"点击: {name} (全屏回退)")
                return True
            # Lower-confidence fallback for full-screen to improve recall
            pt_low = match_template(scr, templ, max(0.72, self.confidence - 0.06))
            if pt_low is not None:
                x, y = pt_low
                click_point(x, y, move_duration=0.12)
                self.log(f"点击: {name} (全屏低阈值)")
                return True
        self.log(f"未找到: {name}")
        # Debug: log best scores to help diagnose
        try:
            self._log_best_match_scores(name)
        except Exception:
            pass
        return False

    def _log_best_match_scores(self, name: str):
        templ = self.templates.get(name)
        if templ is None:
            return
        def best_score(screen_bgr: np.ndarray, template_bgr: np.ndarray) -> float:
            try:
                screen_gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)
                templ_gray_orig = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
                scales = [0.8, 0.9, 1.0, 1.1, 1.2]
                best = 0.0
                for s in scales:
                    if s == 1.0:
                        templ_gray = templ_gray_orig
                    else:
                        h0, w0 = templ_gray_orig.shape[:2]
                        w = max(1, int(w0 * s))
                        h = max(1, int(h0 * s))
                        templ_gray = cv2.resize(templ_gray_orig, (w, h), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC)
                    if templ_gray.shape[0] > screen_gray.shape[0] or templ_gray.shape[1] > screen_gray.shape[1]:
                        continue
                    res = cv2.matchTemplate(screen_gray, templ_gray, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    if max_val > best:
                        best = max_val
                return float(best)
            except Exception:
                return 0.0
        scr_region = screenshot_region(self.region)
        score_region = best_score(scr_region, templ)
        scr_full = screenshot_region(None)
        score_full = best_score(scr_full, templ)
        reg_desc = "已选区域" if self.region is not None else "全屏"
        self.log(f"[{name}] 最佳相似度 - {reg_desc}: {score_region:.2f}，全屏: {score_full:.2f}，当前阈值: {self.confidence}")

    def find_all_bboxes_sorted(self, name: str, max_count: int = 6) -> List[Tuple[int, int, int, int]]:
        templ = self.templates.get(name)
        if templ is None:
            return []
        # Determine effective threshold based on relaxation toggle
        effective_threshold = max(0.72, self.confidence - 0.06) if self.relax_threshold else self.confidence
        # First pass: work on a copy of the selected region (if any)
        screen_local = screenshot_region(self.region)
        results: List[Tuple[int, int, int, int]] = []
        def scan_screen(screen_img: np.ndarray, is_fullscreen: bool) -> List[Tuple[int, int, int, int]]:
            found: List[Tuple[int, int, int, int]] = []
            work = screen_img.copy()
            for _ in range(max_count * 2):  # safety upper bound to avoid infinite loops
                bbox = match_template_with_bbox(work, templ, effective_threshold)
                if bbox is None:
                    break
                x, y, w, h = bbox
                if self.region is not None and not is_fullscreen:
                    gx, gy = x + self.region[0], y + self.region[1]
                else:
                    gx, gy = x, y
                found.append((gx, gy, w, h))
                # Mask the matched area to find the next one
                x2, y2 = x + w, y + h
                x, y = max(0, x), max(0, y)
                x2 = min(work.shape[1], x2)
                y2 = min(work.shape[0], y2)
                work[y:y2, x:x2] = 0
                if len(found) >= max_count:
                    break
            return found
        # Scan within region (or full screen if region is None)
        results = scan_screen(screen_local, is_fullscreen=(self.region is None))
        # Fallback: if nothing found in region but we have a region, try full-screen with relaxed threshold
        if not results and self.region is not None:
            screen_full = screenshot_region(None)
            results = scan_screen(screen_full, is_fullscreen=True)
        # Sort by Y (top to bottom)
        results.sort(key=lambda b: b[1])
        return results

    def perform_collect_actions(self):
        # If previous collection still ongoing, wait until it completes
        ongoing = False
        # Consider ongoing only if time indicator present and collect button not present
        if self.find("采集时间.png", attempts=2) and not self.find("采集.png", attempts=1):
            ongoing = True
        if ongoing:
            self.log("检测到上一轮未完成，等待完成...")
            self.wait_until_collection_finished()
        # Click plus twice
        self.find_and_click("加号.png", 3)
        self.find_and_click("加号.png", 3)
        # OCR collection time BEFORE clicking start collect
        pre_seconds = self.ocr_collect_seconds()
        if pre_seconds is not None:
            self.log(f"识别到采集时间：{pre_seconds} 秒（采集前）")
        else:
            self.log("未识别到采集时间（采集前），稍后将轮询采集完成按钮")
        # Start collecting
        self.find_and_click("采集.png", 5)
        # Prefer pre-ocr seconds if found, otherwise fallback to existing logic
        if pre_seconds is not None:
            self.wait_collect_seconds_then_complete(pre_seconds)
        else:
            self.wait_collect_time_and_complete()

    def try_frontiers_collect(self) -> bool:
        # Find up to six "前往" buttons, top-to-bottom
        frontiers = self.find_all_bboxes_sorted("前往.png", max_count=6)
        if not frontiers:
            return False
        for (x, y, w, h) in frontiers:
            if self._stop.is_set():
                return False
            cx, cy = x + w // 2, y + h // 2
            click_point(cx, cy, move_duration=0.12)
            time.sleep(0.4)
            if self.click_resource_by_priority():
                self.perform_collect_actions()
                return True
            else:
                self.find_and_click("关闭.png", 3)
                time.sleep(0.5)
        return False

    def find_bbox(self, name: str, attempts: int = 2) -> Optional[Tuple[int, int, int, int]]:
        templ = self.templates.get(name)
        if templ is None:
            return None
        for i in range(attempts):
            scr = screenshot_region(self.region)
            bbox = match_template_with_bbox(scr, templ, self.confidence)
            if bbox is not None:
                if self.region is not None:
                    return (bbox[0] + self.region[0], bbox[1] + self.region[1], bbox[2], bbox[3])
                return bbox
            time.sleep(0.3)
        return None

    def ocr_collect_seconds(self) -> Optional[int]:
        bbox = self.find_bbox("采集时间.png", attempts=3)
        if bbox is None:
            # Fallback: try full-screen with a relaxed threshold to locate the time indicator
            templ = self.templates.get("采集时间.png")
            if templ is not None:
                scr_full = screenshot_region(None)
                relaxed = max(0.65, self.confidence - 0.10)
                bbox_local = match_template_with_bbox(scr_full, templ, relaxed)
                if bbox_local is not None:
                    x, y, w, h = bbox_local
                    bbox = (x, y, w, h)
            if bbox is None:
                return None
        x, y, w, h = bbox
        pad_h = int(h * 0.2)
        crop_x1 = x + w + 4
        # Expand ROI width to cover longer time strings
        crop_x2 = crop_x1 + int(w * 4.5)
        crop_y2 = y + h + pad_h
        screen_full = screenshot_region(None)
        H, W = screen_full.shape[:2]
        crop_y1 = max(0, min(H - 1, max(0, y - pad_h)))
        crop_x1 = max(0, min(W - 1, crop_x1))
        crop_x2 = max(1, min(W, crop_x2))
        crop_y2 = max(1, min(H, crop_y2))
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return None
        roi = screen_full[crop_y1:crop_y2, crop_x1:crop_x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        variants = []
        # OTSU binarization (normal and inverted)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, th_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        variants += [th, th_inv]
        # Adaptive threshold
        adp = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 10)
        adp_inv = cv2.bitwise_not(adp)
        variants += [adp, adp_inv]
        # Morphological cleanups
        kernel = np.ones((2, 2), np.uint8)
        variants += [cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel), cv2.morphologyEx(th_inv, cv2.MORPH_OPEN, kernel)]

        def normalize_text(t: str) -> str:
            t = (t or "").strip().replace(" ", "")
            t = t.replace("：", ":")
            # Common OCR confusions
            t = t.replace("O", "0").replace("o", "0")
            t = t.replace("I", "1").replace("l", "1").replace("|", "1")
            t = t.replace("S", "5")
            t = t.replace("s", "秒")
            t = t.replace("m", "分").replace("M", "分")
            t = t.replace("（", "(").replace(")", ")")
            return t

        texts: List[str] = []
        for img in variants:
            for psm in (7, 6):
                config = f"--psm {psm} -l chi_sim+eng -c tessedit_char_whitelist=0123456789:分秒"
                t = pytesseract.image_to_string(img, config=config)
                if t:
                    texts.append(normalize_text(t))
        # Deduplicate while preserving order
        seen = set()
        texts = [x for x in texts if not (x in seen or seen.add(x))]

        text = next(iter(texts), "")
        if not text:
            return None
        try:
            if "分" in text or "秒" in text:
                minutes = 0
                seconds = 0
                if "分" in text:
                    parts = text.split("分")
                    minutes = int(''.join(ch for ch in parts[0] if ch.isdigit()) or 0)
                    tail = parts[1] if len(parts) > 1 else ""
                    seconds = int(''.join(ch for ch in tail if ch.isdigit()) or 0)
                else:
                    seconds = int(''.join(ch for ch in text if ch.isdigit()) or 0)
                total = minutes * 60 + seconds
                return total if total > 0 else None
            if ":" in text:
                mm, ss = text.split(":", 1)
                minutes = int(''.join(ch for ch in mm if ch.isdigit()) or 0)
                seconds = int(''.join(ch for ch in ss if ch.isdigit()) or 0)
                total = minutes * 60 + seconds
                return total if total > 0 else None
            seconds = int(''.join(ch for ch in text if ch.isdigit()) or 0)
            return seconds if seconds > 0 else None
        except Exception:
            return None

    def find(self, name: str, attempts: int = 1) -> bool:
        templ = self.templates.get(name)
        if templ is None:
            return False
        for i in range(attempts):
            scr = screenshot_region(self.region)
            if match_template(scr, templ, self.confidence) is not None:
                return True
            time.sleep(0.3)
        return False

    def click_resource_by_priority(self) -> bool:
        order: List[str] = []
        mapping = {
            "桃": ["三级桃.png", "四级桃.png", "五级桃.png"],
            "玻璃珠": ["三级玻璃珠.png", "四级玻璃珠.png", "五级玻璃珠.png"],
            "仙玉": ["三级仙玉.png", "四级仙玉.png", "五级仙玉.png"],
            "净瓶水": ["三级净瓶水.png", "四级净瓶水.png", "五级净瓶水.png"],
            "天衍令": ["三级天衍令.png", "四级天衍令.png", "五级天衍令.png"],
        }
        for key, tiers in self.resource_cfg.tiers.items():
            names = mapping.get(key, [])
            for tier in tiers:
                idx = max(0, min(2, tier - 3))
                if idx < len(names):
                    order.append(names[idx])
        if not order:
            return False
        # Take one screenshot for speed
        screen = screenshot_region(self.region)
        # Use tighter scale set for speed; include 0.9~1.2 common DPI range
        fast_scales = [0.9, 1.0, 1.1, 1.2]
        best_hit: Optional[Tuple[str, Tuple[int, int, int, int], float]] = None
        # Determine effective threshold based on relaxation toggle
        effective_threshold = max(0.72, self.confidence - 0.06) if self.relax_threshold else self.confidence
        def match_one(name: str):
            templ = self.templates.get(name)
            if templ is None:
                return (name, None, 0.0)
            bbox_local, score = self._fast_match_with_score(screen, templ, fast_scales)
            return (name, bbox_local, score)

        results: List[Tuple[str, Optional[Tuple[int, int, int, int]], float]] = []
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(order)))) as ex:
            future_to_name = {ex.submit(match_one, name): name for name in order}
            for fut in as_completed(future_to_name):
                try:
                    name, bbox_local, score = fut.result()
                except Exception:
                    continue
                # Log score for visibility
                self.log(f"资源匹配 [{name}] 相似度：{score:.2f} (阈值 {effective_threshold:.2f}{'，放宽' if self.relax_threshold else ''})")
                if bbox_local is None:
                    continue
                x, y, w, h = bbox_local
                if self.region is not None:
                    x += self.region[0]
                    y += self.region[1]
                # Respect original order priority: only override if better score and above threshold policy
                if score >= self.confidence:
                    # Immediate accept if this name is earlier in order than any current best
                    if best_hit is None or (order.index(name) < order.index(best_hit[0])):
                        best_hit = (name, (x, y, w, h), score)
                else:
                    if best_hit is None or score > best_hit[2]:
                        best_hit = (name, (x, y, w, h), score)

        if best_hit is None:
            return False
        name, bbox, score = best_hit
        if score < effective_threshold:
            # Too low to trust
            return False
        # Log success score for the recognized match
        self.log(f"匹配成功 [{name}] 相似度：{score:.2f}")
        # OCR level for logging
        level = self.ocr_level_near_resource(bbox)
        if level is not None:
            self.log(f"识别到 {name} 等级：{level}")
        else:
            self.log(f"识别到 {name}，等级未识别")
        x, y, w, h = bbox
        click_point(x + w // 2, y + h // 2, move_duration=0.1)
        self.log(f"点击: {name}")
        return True
        return False

    def wait_until_collection_finished(self, timeout_seconds: int = 15 * 60) -> bool:
        start = time.time()
        no_indicator_secs = 0
        self.progressSignal.emit(0, "等待采集完成…")
        # Fast path: if completion button appears, click and finish
        while not self._stop.is_set() and (time.time() - start) < timeout_seconds:
            # If explicit completion is visible, finish
            if self.find_and_click("采集完成.png", attempts=1):
                self.progressSignal.emit(100, "采集完成")
                return True
            # If time indicator still present, keep waiting briefly
            if self.find("采集时间.png", attempts=1):
                time.sleep(1)
                no_indicator_secs = 0
                elapsed = int(time.time() - start)
                pct = max(1, min(99, int((elapsed / max(1, timeout_seconds)) * 100)))
                self.progressSignal.emit(pct, "采集中…")
                continue
            # Otherwise short sleep and retry
            time.sleep(1)
            no_indicator_secs += 1
            # If indicator is gone for a while, assume no ongoing collection
            if no_indicator_secs >= 5:
                break
        self.progressSignal.emit(0, "等待结束")
        return False

    def wait_collect_time_and_complete(self):
        seconds = self.ocr_collect_seconds()
        if seconds is None:
            self.log("未识别到采集时间，切换为轮询采集完成按钮")
            if not self.wait_until_collection_finished():
                # As a fallback, wait a conservative default and try again
                fallback = 60
                self.log(f"轮询未命中，回退等待 {fallback} 秒后再试")
                for i in range(fallback):
                    if self._stop.is_set():
                        return
                    time.sleep(1)
                    pct = int(((i + 1) / fallback) * 100)
                    self.progressSignal.emit(pct, "回退等待…")
                self.find_and_click("采集完成.png", attempts=10)
            return
        self.log(f"识别到采集时间：{seconds} 秒")
        for i in range(seconds):
            if self._stop.is_set():
                return
            time.sleep(1)
            # If completion appears earlier than estimated, finish early
            if self.find_and_click("采集完成.png", attempts=1):
                self.progressSignal.emit(100, "采集完成")
                return
        self.find_and_click("采集完成.png", attempts=10)
        self.progressSignal.emit(0, "等待结束")

    def wait_collect_seconds_then_complete(self, seconds: int):
        seconds = int(max(1, seconds))
        for i in range(seconds):
            if self._stop.is_set():
                return
            time.sleep(0.9)
            # Early finish if completion appears
            if self.find_and_click("采集完成.png", attempts=1):
                self.progressSignal.emit(100, "采集完成")
                return
        self.find_and_click("采集完成.png", attempts=10)
        self.progressSignal.emit(0, "等待结束")

    def run_collect_loop(self):
        pyautogui.PAUSE = 0.15
        self.log("开始福地采集")
        while not self._stop.is_set():
            if not self.find_and_click("target.png", 4):
                self.log("目标入口未找到，重试...")
                time.sleep(0.6)
                continue
            self.find_and_click("洞府.png", 4)
            self.find_and_click("福地.png", 4)

            if self.click_resource_by_priority():
                # Use unified flow that OCRs time before starting collection
                self.perform_collect_actions()
                continue

            self.find_and_click("探寻.png", 3)
            # Try traversing the six "前往" from top to bottom
            if self.try_frontiers_collect():
                continue

            # If none of the six had desired resources, perform refresh cooldown then retry traversal
            if self.find_and_click("刷新.png", 3):
                # Immediately retry traversal after the first refresh
                if self.try_frontiers_collect():
                    continue
                # If still none, wait for cooldown then retry once more
                cooldown = 5 * 60
                self.log(f"刷新冷却中，等待 {cooldown} 秒后重试")
                for s in range(cooldown):
                    if self._stop.is_set():
                        break
                    time.sleep(0.9)
                if self.try_frontiers_collect():
                    continue
            else:
                self.log("未找到刷新按钮，重试循环")
                time.sleep(1.2)
                continue

        self.log("采集流程结束")
        try:
            self.finished.emit()
        except RuntimeError:
            pass

    def _ocr_digits_from_image(self, img_bgr: np.ndarray) -> Optional[int]:
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
            candidates = []
            # Global OTSU (normal and inverted)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, th_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            candidates.append(th)
            candidates.append(th_inv)
            # Adaptive threshold
            adp = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 10)
            adp_inv = cv2.bitwise_not(adp)
            candidates.append(adp)
            candidates.append(adp_inv)
            # Morph tweaks
            kernel = np.ones((2, 2), np.uint8)
            candidates.append(cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel))
            candidates.append(cv2.morphologyEx(th_inv, cv2.MORPH_OPEN, kernel))

            for img in candidates:
                for psm in (7, 6):
                    # Only allow digits 3,4,5 since resource level is within {3,4,5}
                    config = f"--oem 3 --psm {psm} -c tessedit_char_whitelist=345"
                    text = pytesseract.image_to_string(img, config=config)
                    if not text:
                        continue
                    digits = ''.join(ch for ch in text if ch.isdigit())
                    if digits:
                        try:
                            val = int(digits)
                        except Exception:
                            continue
                        if val in (3, 4, 5):
                            return val
            return None
        except Exception:
            return None

    def ocr_number_right_of_bbox(self, bbox: Tuple[int, int, int, int]) -> Optional[int]:
        x, y, w, h = bbox
        pad_y = int(h * 0.25)
        rx1 = x + w + 4
        ry1 = max(0, y - pad_y)
        rx2 = rx1 + int(w * 1.6)
        ry2 = y + h + pad_y
        screen_full = screenshot_region(None)
        H, W = screen_full.shape[:2]
        rx1 = max(0, min(W - 1, rx1))
        ry1 = max(0, min(H - 1, ry1))
        rx2 = max(1, min(W, rx2))
        ry2 = max(1, min(H, ry2))
        if rx2 <= rx1 or ry2 <= ry1:
            return None
        roi = screen_full[ry1:ry2, rx1:rx2]
        return self._ocr_digits_from_image(roi)

    def ocr_level_near_resource(self, bbox: Tuple[int, int, int, int]) -> Optional[int]:
        x, y, w, h = bbox
        # Primary: area above the icon where level digit typically sits
        pad_x = int(w * 0.2)
        ax1 = max(0, x - pad_x)
        ax2 = min(x + w + pad_x, (screenshot_region(None).shape[1] if True else x + w + pad_x))
        ah = int(h * 0.5)
        ay2 = y - 4
        ay1 = max(0, ay2 - ah)
        if ay2 > ay1 and ax2 > ax1:
            roi_above = screenshot_region(None)[ay1:ay2, ax1:ax2]
            val = self._ocr_digits_from_image(roi_above)
            if val is not None and val in (3, 4, 5):
                return val
        # Fallback: number to the right of the icon
        right_val = self.ocr_number_right_of_bbox(bbox)
        return right_val if right_val in (3, 4, 5) else None

    def _fast_match_with_score(self, screen_bgr: np.ndarray, template_bgr: np.ndarray, scales: List[float]) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        try:
            screen_gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)
            templ_gray_orig = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
            best = (None, 0.0)
            for s in scales:
                if s == 1.0:
                    templ_gray = templ_gray_orig
                else:
                    h0, w0 = templ_gray_orig.shape[:2]
                    w = max(1, int(w0 * s))
                    h = max(1, int(h0 * s))
                    templ_gray = cv2.resize(templ_gray_orig, (w, h), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC)
                if templ_gray.shape[0] > screen_gray.shape[0] or templ_gray.shape[1] > screen_gray.shape[1]:
                    continue
                res = cv2.matchTemplate(screen_gray, templ_gray, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > best[1]:
                    h, w = templ_gray.shape[:2]
                    x1, y1 = max_loc
                    best = (((x1, y1, w, h)), float(max_val))
            return best
        except Exception:
            return (None, 0.0)

    # ---------- Main Window ----------

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XDDQ Auto")
        self.resize(840, 620)
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        self.assets_dir = os.path.join(base_path, "assets")
        self.region: Optional[Tuple[int, int, int, int]] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.worker: Optional[AutomationWorker] = None
        self.overlay: Optional[RegionOverlay] = None
        self.settings = QSettings("xddq", "xddq-auto")

        self.setStyleSheet("""
            QWidget { background: #e8eef5; color: #2b2f38; font-family: 'Segoe UI', 'Microsoft YaHei'; }
            QPushButton { background: #e8eef5; border: 1px solid #d1d9e6; padding: 10px 14px; border-radius: 12px; }
            QPushButton:hover { background: #f2f6fb; }
            QPushButton:pressed { background: #dde6f1; }
            QGroupBox { border: 1px solid #d1d9e6; border-radius: 12px; margin-top: 10px; padding-top: 10px; }
            QTextEdit { background: #f7f9fc; border: 1px solid #d1d9e6; border-radius: 12px; }
            QLabel { color: #3c4452; }
            QCheckBox::indicator { width: 18px; height: 18px; border-radius: 4px; border: 1px solid #b8c4d6; background: #ffffff; }
            QCheckBox::indicator:checked { background: #12b66a; border: 1px solid #0f9a59; }
            QCheckBox:checked { color: #0f9a59; font-weight: 600; }
            #StatusBadge { background: #12b66a; color: #ffffff; border-radius: 10px; padding: 2px 8px; }
            QProgressBar { border: 1px solid #d1d9e6; border-radius: 8px; background: #f7f9fc; height: 14px; }
            QProgressBar::chunk { background-color: #12b66a; border-radius: 8px; }
        """)

        self.build_ui()

    def build_ui(self):
        layout = QVBoxLayout(self)

        # Tabs
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        # --- Tab: 采集 ---
        tab_collect = QWidget()
        collect_layout = QVBoxLayout(tab_collect)

        # Top controls row
        row_top = QHBoxLayout()
        self.btn_select_region = QPushButton("选择区域")
        self.lbl_conf = QLabel("阈值：")
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.50, 0.99)
        self.spin_conf.setDecimals(2)
        self.spin_conf.setSingleStep(0.01)
        self.spin_conf.setValue(0.94)
        self.spin_conf.setToolTip("模板匹配相似度阈值，越高越严格")
        self.spin_conf.setMinimumWidth(160)
        self.spin_conf.setStyleSheet("font-size: 14px; padding: 4px 8px;")
        self.chk_relax = QCheckBox("近阈值放宽")
        self.chk_relax.setChecked(True)
        self.chk_relax.setToolTip("允许在阈值基础上放宽 0.06（最低 0.72）")
        row_top.addWidget(self.btn_select_region)
        row_top.addStretch(1)
        row_top.addWidget(self.lbl_conf)
        row_top.addWidget(self.spin_conf)
        row_top.addWidget(self.chk_relax)
        collect_layout.addLayout(row_top)

        # Middle: left (resources) + right (region preview)
        row_mid = QHBoxLayout()
        # Resources group with scroll
        grp_res = QGroupBox("资源选择（可多选）")
        grid = QGridLayout()
        self.chk: Dict[str, Dict[int, QCheckBox]] = {}
        categories = ["桃", "玻璃珠", "仙玉", "净瓶水", "天衍令"]
        for ci, cat in enumerate(categories):
            self.chk[cat] = {}
            grid.addWidget(QLabel(cat), ci, 0)
            for ti, tier in enumerate([3, 4, 5]):
                cb = QCheckBox(f"{tier}级")
                grid.addWidget(cb, ci, ti + 1)
                self.chk[cat][tier] = cb
        grp_res.setLayout(grid)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(grp_res)
        row_mid.addWidget(scroll, 1)

        # Region preview panel
        preview_panel = QGroupBox("区域预览")
        pv_layout = QVBoxLayout()
        self.preview_label = QLabel("未选择区域")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(180)
        self.preview_label.setStyleSheet("background:#f7f9fc;border:1px solid #d1d9e6;border-radius:12px;")
        pv_layout.addWidget(self.preview_label)
        preview_panel.setLayout(pv_layout)
        row_mid.addWidget(preview_panel, 1)
        collect_layout.addLayout(row_mid, 1)

        # Bottom: actions and status bar
        row_actions = QHBoxLayout()
        self.btn_start = QPushButton("开始")
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)
        row_actions.addWidget(self.btn_start)
        row_actions.addWidget(self.btn_stop)
        row_actions.addStretch(1)
        self.status_badge = QLabel("就绪")
        self.status_badge.setObjectName("StatusBadge")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedWidth(200)
        row_actions.addWidget(self.progress)
        row_actions.addWidget(self.status_badge)
        collect_layout.addLayout(row_actions)

        self.tabs.addTab(tab_collect, "采集")

        # --- Tab: 设置 ---
        tab_settings = QWidget()
        st_layout = QVBoxLayout(tab_settings)
        row_tess = QHBoxLayout()
        self.lbl_tess = QLabel("Tesseract 路径：")
        self.le_tess = QLineEdit()
        saved_path = str(self.settings.value("tesseract_path", "") or "")
        if not saved_path:
            default_win = r"D:\\Program Files\\Tesseract-OCR\\tesseract.exe"
            saved_path = default_win if os.path.exists(default_win) else ""
        self.le_tess.setText(saved_path)
        self.btn_browse_tess = QPushButton("浏览…")
        row_tess.addWidget(self.lbl_tess)
        row_tess.addWidget(self.le_tess, 1)
        row_tess.addWidget(self.btn_browse_tess)
        st_layout.addLayout(row_tess)

        # Screen params row
        row_screen = QHBoxLayout()
        self.chk_override_scale = QCheckBox("手动缩放覆盖")
        self.lbl_screen = QLabel("屏幕分辨率：")
        from PySide6.QtWidgets import QSpinBox
        self.spin_screen_w = QSpinBox()
        self.spin_screen_w.setRange(640, 10000)
        self.spin_screen_w.setSuffix(" px")
        self.spin_screen_w.setMinimumWidth(160)
        self.spin_screen_w.setStyleSheet("font-size: 14px; padding: 4px 8px;")
        self.spin_screen_h = QSpinBox()
        self.spin_screen_h.setRange(480, 10000)
        self.spin_screen_h.setSuffix(" px")
        self.spin_screen_h.setMinimumWidth(160)
        self.spin_screen_h.setStyleSheet("font-size: 14px; padding: 4px 8px;")
        self.lbl_scale = QLabel("缩放：")
        self.lbl_scale.setStyleSheet("font-size: 14px;")
        self.spin_scale = QDoubleSpinBox()
        self.spin_scale.setRange(50.0, 300.0)
        self.spin_scale.setDecimals(0)
        self.spin_scale.setSingleStep(25.0)
        self.spin_scale.setSuffix(" %")
        self.spin_scale.setMinimumWidth(160)
        self.spin_scale.setStyleSheet("font-size: 14px; padding: 4px 8px;")
        self.lbl_screen.setStyleSheet("font-size: 14px;")
        self.chk_override_scale.setStyleSheet("font-size: 14px;")
        self.btn_apply_screen = QPushButton("应用")
        row_screen.addWidget(self.chk_override_scale)
        row_screen.addStretch(1)
        row_screen.addWidget(self.lbl_screen)
        row_screen.addWidget(self.spin_screen_w)
        row_screen.addWidget(QLabel("×"))
        row_screen.addWidget(self.spin_screen_h)
        row_screen.addStretch(1)
        row_screen.addWidget(self.lbl_scale)
        row_screen.addWidget(self.spin_scale)
        row_screen.addWidget(self.btn_apply_screen)
        st_layout.addLayout(row_screen)

        # Initialize screen params from system and settings
        try:
            pr = QGuiApplication.primaryScreen()
            geo = pr.geometry()
            # Estimate device pixel size from logical size and scaling
            try:
                auto_scale = max(1.0, float(pr.logicalDotsPerInch()) / 96.0)
            except Exception:
                try:
                    auto_scale = max(1.0, float(pr.devicePixelRatio()))
                except Exception:
                    auto_scale = 1.0
            dev_w = int(geo.width() * auto_scale)
            dev_h = int(geo.height() * auto_scale)
        except Exception:
            dev_w, dev_h, auto_scale = 1920, 1080, 1.0
        saved_w = int(self.settings.value("screen_width", dev_w) or dev_w)
        saved_h = int(self.settings.value("screen_height", dev_h) or dev_h)
        saved_scale = float(self.settings.value("screen_scale_percent", int(auto_scale * 100)) or int(auto_scale * 100))
        saved_override = bool(self.settings.value("override_scale", False) in (True, "true", "1", 1))
        self.spin_screen_w.setValue(saved_w)
        self.spin_screen_h.setValue(saved_h)
        self.spin_scale.setValue(saved_scale)
        self.chk_override_scale.setChecked(saved_override)
        # Stash into instance for runtime use
        self.screen_width = saved_w
        self.screen_height = saved_h
        self.user_scale = float(saved_scale) / 100.0
        self.override_scale = saved_override

        st_layout.addStretch(1)
        self.tabs.addTab(tab_settings, "设置")

        # --- Tab: 日志 ---
        tab_logs = QWidget()
        log_layout = QVBoxLayout(tab_logs)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        log_layout.addWidget(self.log_box)
        self.tabs.addTab(tab_logs, "日志")

        # Wire signals
        self.btn_select_region.clicked.connect(self.on_select_region)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_browse_tess.clicked.connect(self.on_browse_tesseract)
        self.le_tess.editingFinished.connect(self.on_tesseract_path_entered)
        self.btn_apply_screen.clicked.connect(self.on_apply_screen_params)

        # Apply Tesseract path at startup
        self.apply_tesseract_path(self.le_tess.text())

    def append_log(self, text: str):
        self.log_box.append(text)

    def update_status(self, percent: int, text: str):
        self.progress.setValue(max(0, min(100, int(percent))))
        self.status_badge.setText(text or "")

    def normalize_tesseract_path(self, raw_path: str) -> str:
        path = (raw_path or "").strip().strip('"')
        if not path:
            return ""
        # If a directory is provided, append tesseract.exe on Windows
        if os.path.isdir(path):
            exe = os.path.join(path, "tesseract.exe" if os.name == "nt" else "tesseract")
            return exe
        return path

    def apply_tesseract_path(self, raw_path: str):
        exe_path = self.normalize_tesseract_path(raw_path)
        if exe_path and os.path.exists(exe_path):
            pytesseract.pytesseract.tesseract_cmd = exe_path
            self.settings.setValue("tesseract_path", exe_path)
            self.append_log(f"已设置 Tesseract：{exe_path}")
        else:
            self.append_log("未找到有效的 Tesseract 可执行文件路径，请检查设置")

    def on_browse_tesseract(self):
        if os.name == "nt":
            start_dir = os.path.dirname(self.le_tess.text()) or r"C:\\Program Files"
            file, _ = QFileDialog.getOpenFileName(self, "选择 tesseract.exe", start_dir, "可执行文件 (*.exe)")
        else:
            start_dir = os.path.dirname(self.le_tess.text()) or "/usr/bin"
            file, _ = QFileDialog.getOpenFileName(self, "选择 tesseract", start_dir)
        if file:
            self.le_tess.setText(file)
            self.apply_tesseract_path(file)

    def on_tesseract_path_entered(self):
        self.apply_tesseract_path(self.le_tess.text())

    def on_select_region(self):
        if self.overlay is not None:
            try:
                self.overlay.close()
            except Exception:
                pass
        overlay = RegionOverlay()
        if overlay is None:
            self.append_log("创建选择区域覆盖层失败")
            return
        self.overlay = overlay
        self.overlay.regionSelected.connect(self.on_region_selected)
        self.overlay.destroyed.connect(lambda: setattr(self, 'overlay', None))
        self.overlay.setWindowState(self.overlay.windowState() | Qt.WindowFullScreen)
        self.overlay.show()
        self.overlay.raise_()
        self.overlay.activateWindow()

    def on_region_selected(self, rect: QRect):
        # Convert Qt logical coordinates to device pixels according to the screen's scaling
        try:
            center_point = rect.center()
            screen = QGuiApplication.screenAt(center_point)
            if screen is None:
                screen = QGuiApplication.primaryScreen()
            # Prefer manual override if enabled
            if getattr(self, 'override_scale', False):
                scale = max(1.0, float(getattr(self, 'user_scale', 1.0)))
            else:
                # Prefer logical DPI ratio to 96 dpi baseline; fallback to devicePixelRatio
                try:
                    scale = max(1.0, float(screen.logicalDotsPerInch()) / 96.0)
                except Exception:
                    try:
                        scale = max(1.0, float(screen.devicePixelRatio()))
                    except Exception:
                        scale = 1.0
            x = int(rect.x() * scale)
            y = int(rect.y() * scale)
            w = int(rect.width() * scale)
            h = int(rect.height() * scale)
            self.region = (x, y, w, h)
            self.append_log(f"已设置识别区域: x={x}, y={y}, w={w}, h={h} (缩放: {scale:.2f}{' 手动' if getattr(self, 'override_scale', False) else ''})")
        except Exception:
            # Fallback to original logical coordinates if scaling fails
            self.region = (rect.x(), rect.y(), rect.width(), rect.height())
            self.append_log(f"已设置识别区域: x={rect.x()}, y={rect.y()}, w={rect.width()}, h={rect.height()} (未应用缩放)")
        try:
            if self.overlay is not None:
                self.overlay.close()
        except Exception:
            pass
        self.overlay = None
        # Update preview snapshot
        try:
            shot = screenshot_region(self.region)
            # Convert to QImage via temporary fileless approach
            rgb = cv2.cvtColor(shot, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            from PySide6.QtGui import QImage, QPixmap
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pm = QPixmap.fromImage(qimg).scaled(self.preview_label.width(), self.preview_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview_label.setPixmap(pm)
        except Exception:
            self.preview_label.setText("预览失败")

    def build_resource_cfg(self) -> ResourceConfig:
        tiers: Dict[str, List[int]] = {}
        for cat, tier_map in self.chk.items():
            selected = [tier for tier, cb in tier_map.items() if cb.isChecked()]
            if selected:
                tiers[cat] = sorted(selected)
        return ResourceConfig(tiers=tiers)

    def on_start(self):
        if self.worker_thread and self.worker_thread.is_alive():
            return
        cfg = self.build_resource_cfg()
        if not cfg.tiers:
            self.append_log("请至少选择一个资源等级")
            return
        self.append_log("启动中...")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.update_status(0, "启动中…")

        self.worker = AutomationWorker(self.assets_dir, self.region, cfg, self.spin_conf.value(), self.chk_relax.isChecked())
        self.worker.logSignal.connect(self.append_log)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.progressSignal.connect(self.update_status)

        # Diagnostics: report missing templates and assets path
        try:
            missing_templates = [name for name, img in self.worker.templates.items() if img is None]
            if missing_templates:
                self.append_log(f"资源目录: {self.assets_dir}")
                self.append_log("以下模板加载失败，请检查文件是否存在或命名是否正确：")
                for name in missing_templates:
                    self.append_log(f" - {name}")
        except Exception:
            pass

        def target():
            try:
                self.worker.run_collect_loop()
            finally:
                pass
        self.worker_thread = threading.Thread(target=target, daemon=True)
        self.worker_thread.start()

    def on_stop(self):
        if self.worker:
            self.worker.stop()
        self.btn_stop.setEnabled(False)
        self.update_status(0, "停止…")

    def on_worker_finished(self):
        self.append_log("已停止")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.update_status(0, "就绪")

    def on_apply_screen_params(self):
        self.screen_width = self.spin_screen_w.value()
        self.screen_height = self.spin_screen_h.value()
        self.user_scale = float(self.spin_scale.value()) / 100.0
        self.override_scale = self.chk_override_scale.isChecked()
        # Persist settings
        try:
            self.settings.setValue("screen_width", self.screen_width)
            self.settings.setValue("screen_height", self.screen_height)
            self.settings.setValue("screen_scale_percent", int(self.user_scale * 100))
            self.settings.setValue("override_scale", self.override_scale)
        except Exception:
            pass
        self.append_log(f"屏幕参数已应用: 宽度={self.screen_width}, 高度={self.screen_height}, 缩放={self.user_scale * 100:.0f}%")


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 