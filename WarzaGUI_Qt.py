"""WarzaVision Pro 9.6 - FIRE GUI
Real-time sync | All inputs | Network stats | Shot history
"""

import sys
import os
import json
import socket
import threading
import time
import base64

DEFAULT_PORT = 59420
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from PyQt6.QtWidgets import *
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QByteArray
    from PyQt6.QtGui import QColor, QPainter, QPen, QBrush, QFont, QPixmap, QImage, QCursor, QLinearGradient
except:
    print("PyQt6 required: pip install PyQt6")
    sys.exit(1)

# Fire color scheme
C = {
    "bg": "#0d0d12",
    "card": "#1a1a24",
    "card2": "#252535",
    "accent": "#00ff88",
    "accent2": "#00ccff",
    "hot": "#ff4d6d",
    "warning": "#ffd000",
    "text": "#ffffff",
    "dim": "#666680",
    "border": "#333355",
}

STYLE = f"""
QWidget {{ background: {C['bg']}; color: {C['text']}; font-family: 'Segoe UI', Arial; font-size: 11px; }}
QFrame#card {{ background: {C['card']}; border-radius: 8px; border: 1px solid {C['border']}; }}
QFrame#card2 {{ background: {C['card2']}; border-radius: 6px; }}
QLabel#title {{ font-size: 16px; font-weight: bold; color: {C['accent']}; }}
QLabel#section {{ font-size: 9px; font-weight: bold; color: {C['dim']}; text-transform: uppercase; letter-spacing: 1px; }}
QLabel#value {{ font-size: 18px; font-weight: bold; }}
QLabel#stat {{ font-size: 10px; color: {C['dim']}; }}
QPushButton {{ background: {C['card2']}; border: 1px solid {C['border']}; border-radius: 4px; padding: 6px 14px; font-size: 10px; font-weight: bold; }}
QPushButton:hover {{ background: {C['accent']}; color: #000; border-color: {C['accent']}; }}
QPushButton:pressed {{ background: {C['accent2']}; }}
QLineEdit, QSpinBox, QDoubleSpinBox {{ background: {C['bg']}; border: 1px solid {C['border']}; border-radius: 4px; padding: 5px 8px; }}
QComboBox {{ background: {C['bg']}; border: 1px solid {C['border']}; border-radius: 4px; padding: 5px 8px; }}
QComboBox::drop-down {{ border: none; }}
QComboBox::down-arrow {{ image: none; }}
QSlider::groove:horizontal {{ height: 4px; background: {C['border']}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {C['accent']}; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}
QSlider::sub-page:horizontal {{ background: qlineargradient(x1:0, x2:1, stop:0 {C['accent2']}, stop:1 {C['accent']}); border-radius: 2px; }}
QCheckBox {{ spacing: 6px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid {C['border']}; background: {C['bg']}; }}
QCheckBox::indicator:checked {{ background: {C['accent']}; border-color: {C['accent']}; }}
QProgressBar {{ background: {C['border']}; border-radius: 3px; height: 6px; text-align: center; }}
QProgressBar::chunk {{ background: qlineargradient(x1:0, x2:1, stop:0 {C['accent2']}, stop:1 {C['accent']}); border-radius: 3px; }}
QScrollArea {{ border: none; }}
QScrollBar:vertical {{ width: 8px; background: {C['bg']}; }}
QScrollBar::handle:vertical {{ background: {C['border']}; border-radius: 4px; min-height: 20px; }}
"""


class SocketClient(QThread):
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    message = pyqtSignal(dict)
    
    def __init__(self, port):
        super().__init__()
        self.port = port
        self.sock = None
        self.running = True
        self._connected = False
        
    def run(self):
        while self.running:
            try:
                self.sock = socket.create_connection(("127.0.0.1", self.port), timeout=2)
                self.sock.settimeout(0.05)
                self._connected = True
                self.connected.emit()
                buf = b""
                while self.running and self._connected:
                    try:
                        data = self.sock.recv(8192)
                        if not data:
                            break
                        buf += data
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            if line.strip():
                                try:
                                    self.message.emit(json.loads(line.decode()))
                                except:
                                    pass
                    except socket.timeout:
                        pass
                    except:
                        break
            except:
                pass
            self._connected = False
            self.disconnected.emit()
            if self.sock:
                try: self.sock.close()
                except: pass
            self.sock = None
            if self.running:
                time.sleep(0.5)
                
    def send(self, data):
        if self.sock and self._connected:
            try:
                self.sock.sendall((json.dumps(data) + "\n").encode())
            except:
                pass
                
    def stop(self):
        self.running = False


class ColorPickerDialog(QDialog):
    color_picked = pyqtSignal(int, int, int)
    
    def __init__(self, image_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pick Meter Color")
        self.setFixedSize(700, 450)
        self.setStyleSheet(STYLE)
        
        img_bytes = base64.b64decode(image_data)
        qimg = QImage.fromData(QByteArray(img_bytes))
        self.pixmap = QPixmap.fromImage(qimg)
        self.tolerance = 30
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        lbl = QLabel("Click on the shot meter color")
        lbl.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {C['accent']};")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl)
        
        row = QHBoxLayout()
        row.addWidget(QLabel("Tolerance:"))
        self.tol_slider = QSlider(Qt.Orientation.Horizontal)
        self.tol_slider.setRange(10, 60)
        self.tol_slider.setValue(30)
        self.tol_slider.valueChanged.connect(lambda v: setattr(self, 'tolerance', v))
        row.addWidget(self.tol_slider)
        self.tol_lbl = QLabel("30")
        self.tol_slider.valueChanged.connect(lambda v: self.tol_lbl.setText(str(v)))
        row.addWidget(self.tol_lbl)
        layout.addLayout(row)
        
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        self.img_label.mousePressEvent = self._on_click
        layout.addWidget(self.img_label, 1)
        
        scaled = self.pixmap.scaled(670, 350, Qt.AspectRatioMode.KeepAspectRatio)
        self.scaled_pixmap = scaled
        self.img_label.setPixmap(scaled)
        
    def _on_click(self, event):
        if not self.scaled_pixmap:
            return
        lx, ly = event.position().x(), event.position().y()
        img_rect = self.scaled_pixmap.rect()
        lbl_rect = self.img_label.rect()
        ox = (lbl_rect.width() - img_rect.width()) / 2
        oy = (lbl_rect.height() - img_rect.height()) / 2
        cx, cy = lx - ox, ly - oy
        sx = self.pixmap.width() / self.scaled_pixmap.width()
        sy = self.pixmap.height() / self.scaled_pixmap.height()
        self.color_picked.emit(int(cx * sx), int(cy * sy), self.tolerance)
        self.accept()


class InputIndicator(QWidget):
    """Visual input indicator"""
    def __init__(self, label, parent=None):
        super().__init__(parent)
        self.label = label
        self.value = 0
        self.setFixedSize(50, 30)
        
    def set_value(self, v):
        self.value = v
        self.update()
        
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        if self.value > 50:
            p.setBrush(QBrush(QColor(C['accent'])))
        elif self.value > 0:
            p.setBrush(QBrush(QColor(C['warning'])))
        else:
            p.setBrush(QBrush(QColor(C['card2'])))
        p.setPen(QPen(QColor(C['border']), 1))
        p.drawRoundedRect(0, 0, self.width(), self.height(), 4, 4)
        
        # Label
        p.setPen(QColor("#000" if self.value > 50 else C['text']))
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.label)


class MainWindow(QMainWindow):
    def __init__(self, port):
        super().__init__()
        self.setWindowTitle("WarzaVision 9.6")
        self.setFixedSize(520, 680)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        self._updating = False
        self.client = SocketClient(port)
        self.client.connected.connect(self._on_connected)
        self.client.disconnected.connect(self._on_disconnected)
        self.client.message.connect(self._on_message)
        self.client.start()
        
        self._build_ui()
        self.drag_pos = None
        
        self.timer = QTimer()
        self.timer.timeout.connect(lambda: self.client.send({"type": "get_status"}))
        self.timer.start(100)  # Fast updates
        
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QVBoxLayout(central)
        main.setContentsMargins(12, 12, 12, 12)
        main.setSpacing(8)
        
        # === HEADER ===
        header = QHBoxLayout()
        title = QLabel("WarzaVision 9.6")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch()
        
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet(f"color: {C['hot']}; font-size: 14px;")
        header.addWidget(self.status_dot)
        
        self.gtuner_lbl = QLabel("GTuner: --")
        self.gtuner_lbl.setStyleSheet(f"color: {C['dim']}; font-size: 10px;")
        header.addWidget(self.gtuner_lbl)
        
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(28, 28)
        close_btn.clicked.connect(self.close)
        header.addWidget(close_btn)
        main.addLayout(header)
        
        # === STATUS CARD ===
        status_card = QFrame()
        status_card.setObjectName("card")
        sc = QHBoxLayout(status_card)
        sc.setContentsMargins(12, 10, 12, 10)
        
        # Shot status
        left = QVBoxLayout()
        self.shot_label = QLabel("READY")
        self.shot_label.setStyleSheet(f"font-size: 22px; font-weight: bold; color: {C['dim']};")
        left.addWidget(self.shot_label)
        self.meter_label = QLabel("Meter: --")
        self.meter_label.setStyleSheet(f"color: {C['dim']}; font-size: 11px;")
        left.addWidget(self.meter_label)
        sc.addLayout(left)
        
        sc.addStretch()
        
        # Stats
        right = QVBoxLayout()
        right.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.shots_lbl = QLabel("Shots: 0")
        self.shots_lbl.setStyleSheet(f"color: {C['text']}; font-size: 11px;")
        right.addWidget(self.shots_lbl)
        self.greens_lbl = QLabel("Greens: 0 (0%)")
        self.greens_lbl.setStyleSheet(f"color: {C['accent']}; font-size: 11px;")
        right.addWidget(self.greens_lbl)
        sc.addLayout(right)
        
        main.addWidget(status_card)
        
        # === TWO COLUMNS ===
        cols = QHBoxLayout()
        cols.setSpacing(10)
        
        # LEFT COLUMN: Timing & Detection
        left_col = QVBoxLayout()
        left_col.setSpacing(6)
        
        left_col.addWidget(self._section("TIMING (ms)"))
        
        # Timing controls
        for name, key, default, max_val in [
            ("Regular:", "regular_timing_ms", 165, 500),
            ("Rhythm:", "rhythm_timing_ms", 28, 200),
            ("Tempo:", "rhythm_tempo", 62, 200),
            ("Post:", "post_timing_ms", 64, 200),
            ("Dunk:", "dunk_timing_ms", 945, 2000),
            ("Green Offset:", "green_offset", 0, 50),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(name))
            spin = QSpinBox()
            spin.setRange(-50 if "offset" in key else 0, max_val)
            spin.setValue(default)
            spin.setObjectName(key)
            spin.valueChanged.connect(lambda v, k=key: self._send(k, v))
            row.addWidget(spin)
            left_col.addLayout(row)
            setattr(self, f"{key}_spin", spin)
            
        left_col.addWidget(self._section("DETECTION"))
        
        # Color
        row = QHBoxLayout()
        row.addWidget(QLabel("Color:"))
        self.color_combo = QComboBox()
        self.color_combo.addItems(["purple", "green", "red", "yellow", "blue", "white", "custom"])
        self.color_combo.currentTextChanged.connect(lambda v: self._send("meter_color", v))
        row.addWidget(self.color_combo)
        left_col.addLayout(row)
        
        # Space
        row = QHBoxLayout()
        row.addWidget(QLabel("Space:"))
        self.space_combo = QComboBox()
        self.space_combo.addItems(["LAB", "HSV"])
        self.space_combo.currentTextChanged.connect(lambda v: self._send("color_space", v))
        row.addWidget(self.space_combo)
        left_col.addLayout(row)
        
        # Min/Max area
        for name, key, default, min_v, max_v in [
            ("Min Area:", "min_area", 40, 10, 500),
            ("Max Area:", "max_area", 30000, 1000, 50000),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(name))
            spin = QSpinBox()
            spin.setRange(min_v, max_v)
            spin.setValue(default)
            spin.setSingleStep(100 if "max" in key else 10)
            spin.valueChanged.connect(lambda v, k=key: self._send(k, v))
            row.addWidget(spin)
            left_col.addLayout(row)
            setattr(self, f"{key}_spin", spin)
            
        # Pick color button
        self.pick_btn = QPushButton("🎨 Pick from Screen")
        self.pick_btn.clicked.connect(self._pick_color)
        left_col.addWidget(self.pick_btn)
        
        left_col.addStretch()
        cols.addLayout(left_col, 1)
        
        # RIGHT COLUMN: Network
        right_col = QVBoxLayout()
        right_col.setSpacing(6)
        
        right_col.addWidget(self._section("NETWORK"))
        
        # Quality indicator
        self.quality_lbl = QLabel("UNKNOWN")
        self.quality_lbl.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {C['dim']};")
        right_col.addWidget(self.quality_lbl)
        
        # Server
        self.server_lbl = QLabel("Server: --")
        self.server_lbl.setObjectName("stat")
        right_col.addWidget(self.server_lbl)
        
        # Ping stats
        self.ping_lbl = QLabel("Ping: -- ms")
        right_col.addWidget(self.ping_lbl)
        
        self.baseline_lbl = QLabel("Baseline: -- ms")
        self.baseline_lbl.setObjectName("stat")
        right_col.addWidget(self.baseline_lbl)
        
        self.jitter_lbl = QLabel("Jitter: -- ms")
        self.jitter_lbl.setObjectName("stat")
        right_col.addWidget(self.jitter_lbl)
        
        self.stability_lbl = QLabel("Stability: --%")
        right_col.addWidget(self.stability_lbl)
        
        # Compensation
        comp_frame = QFrame()
        comp_frame.setObjectName("card2")
        comp_layout = QVBoxLayout(comp_frame)
        comp_layout.setContentsMargins(8, 6, 8, 6)
        self.comp_lbl = QLabel("Compensation: -0ms")
        self.comp_lbl.setStyleSheet(f"font-size: 13px; font-weight: bold; color: {C['accent']};")
        comp_layout.addWidget(self.comp_lbl)
        self.comp_detail = QLabel("Base:0 Jitter:0 Spike:0")
        self.comp_detail.setObjectName("stat")
        comp_layout.addWidget(self.comp_detail)
        right_col.addWidget(comp_frame)
        
        # Calibration bar
        self.cal_bar = QProgressBar()
        self.cal_bar.setRange(0, 100)
        self.cal_bar.setTextVisible(False)
        self.cal_bar.setFixedHeight(6)
        right_col.addWidget(self.cal_bar)
        
        right_col.addWidget(self._section("INPUTS"))
        
        # Input indicators
        inputs_grid = QGridLayout()
        inputs_grid.setSpacing(4)
        
        self.input_indicators = {}
        inputs = [("RS", 0, 0), ("SQ/X", 0, 1), ("L2", 1, 0), ("R2", 1, 1), ("R3", 2, 0), ("L3", 2, 1)]
        for label, row, col in inputs:
            ind = InputIndicator(label)
            inputs_grid.addWidget(ind, row, col)
            self.input_indicators[label] = ind
            
        right_col.addLayout(inputs_grid)
        
        right_col.addStretch()
        cols.addLayout(right_col, 1)
        
        main.addLayout(cols)
        
        # === TOGGLES ===
        toggles = QHBoxLayout()
        
        self.chk_comp = QCheckBox("Auto Comp")
        self.chk_comp.setChecked(True)
        self.chk_comp.stateChanged.connect(lambda s: self._send("auto_comp", s == 2))
        toggles.addWidget(self.chk_comp)
        
        self.chk_overlay = QCheckBox("Overlay")
        self.chk_overlay.setChecked(True)
        self.chk_overlay.stateChanged.connect(lambda s: self._send("show_overlay", s == 2))
        toggles.addWidget(self.chk_overlay)
        
        self.chk_aggressive = QCheckBox("Aggressive")
        self.chk_aggressive.stateChanged.connect(lambda s: self._send("aggressive_mode", s == 2))
        toggles.addWidget(self.chk_aggressive)
        
        toggles.addStretch()
        
        self.fps_lbl = QLabel("FPS: --")
        self.fps_lbl.setStyleSheet(f"color: {C['dim']};")
        toggles.addWidget(self.fps_lbl)
        
        main.addLayout(toggles)
        
    def _section(self, text):
        lbl = QLabel(text)
        lbl.setObjectName("section")
        return lbl
        
    def _send(self, key, value):
        if not self._updating:
            self.client.send({"type": "update", key: value})
            
    def _pick_color(self):
        self.client.send({"type": "get_screenshot"})
        
    def _on_screenshot(self, data):
        dlg = ColorPickerDialog(data, self)
        dlg.color_picked.connect(self._on_color_picked)
        dlg.exec()
        
    def _on_color_picked(self, x, y, tol):
        self.client.send({"type": "pick_color", "x": x, "y": y, "tolerance": tol})
        
    def _on_connected(self):
        self.status_dot.setStyleSheet(f"color: {C['accent']}; font-size: 14px;")
        
    def _on_disconnected(self):
        self.status_dot.setStyleSheet(f"color: {C['hot']}; font-size: 14px;")
        
    def _on_message(self, msg):
        if msg.get("type") == "screenshot":
            self._on_screenshot(msg.get("image", ""))
            return
            
        if msg.get("type") != "status":
            return
            
        self._updating = True
        try:
            cfg = msg.get("config", {})
            net = msg.get("network", {})
            meter = msg.get("meter", {})
            shot = msg.get("shot", {})
            inputs = msg.get("inputs", {})
            
            # GTuner status
            gtuner = msg.get("gtuner", False)
            self.gtuner_lbl.setText(f"GTuner: {'ON' if gtuner else 'OFF'}")
            self.gtuner_lbl.setStyleSheet(f"color: {C['accent'] if gtuner else C['dim']}; font-size: 10px;")
            
            # Update spins
            for key in ["regular_timing_ms", "rhythm_timing_ms", "rhythm_tempo", "post_timing_ms", "dunk_timing_ms", "green_offset", "min_area", "max_area"]:
                spin = getattr(self, f"{key}_spin", None)
                if spin:
                    spin.blockSignals(True)
                    spin.setValue(int(cfg.get(key, spin.value())))
                    spin.blockSignals(False)
                    
            # Color
            color = cfg.get("meter_color", "purple")
            idx = self.color_combo.findText(color)
            if idx >= 0:
                self.color_combo.blockSignals(True)
                self.color_combo.setCurrentIndex(idx)
                self.color_combo.blockSignals(False)
                
            space = cfg.get("color_space", "LAB")
            idx = self.space_combo.findText(space)
            if idx >= 0:
                self.space_combo.blockSignals(True)
                self.space_combo.setCurrentIndex(idx)
                self.space_combo.blockSignals(False)
                
            # Toggles
            self.chk_comp.blockSignals(True)
            self.chk_comp.setChecked(cfg.get("auto_comp", True))
            self.chk_comp.blockSignals(False)
            
            self.chk_overlay.blockSignals(True)
            self.chk_overlay.setChecked(cfg.get("show_overlay", True))
            self.chk_overlay.blockSignals(False)
            
            self.chk_aggressive.blockSignals(True)
            self.chk_aggressive.setChecked(cfg.get("aggressive_mode", False))
            self.chk_aggressive.blockSignals(False)
            
            # Network
            quality = net.get('quality', 'UNKNOWN')
            self.quality_lbl.setText(quality)
            qc = net.get('quality_color', [128, 128, 128])
            self.quality_lbl.setStyleSheet(f"font-size: 14px; font-weight: bold; color: rgb({qc[2]},{qc[1]},{qc[0]});")
            
            self.server_lbl.setText(f"Server: {net.get('server', '--')} ({net.get('server_region', '')})")
            self.ping_lbl.setText(f"Ping: {net.get('ping', 0)} ms")
            self.baseline_lbl.setText(f"Baseline: {net.get('baseline', 0)} ms")
            self.jitter_lbl.setText(f"Jitter: {net.get('jitter', 0)} ms")
            
            stab = net.get('stability', 0)
            stab_color = C['accent'] if stab >= 80 else C['warning'] if stab >= 50 else C['hot']
            self.stability_lbl.setText(f"Stability: {stab}%")
            self.stability_lbl.setStyleSheet(f"color: {stab_color};")
            
            comp = net.get('compensation', 0)
            self.comp_lbl.setText(f"Compensation: -{comp}ms")
            self.comp_detail.setText(f"Base:{net.get('comp_base',0)} Jitter:{net.get('comp_jitter',0)} Spike:{net.get('comp_spike',0)}")
            
            if net.get('calibrated', False):
                self.cal_bar.setVisible(False)
            else:
                self.cal_bar.setVisible(True)
                self.cal_bar.setValue(net.get('calibration_pct', 0))
                
            # Meter
            fill = meter.get('fill', 0)
            detected = meter.get('detected', False)
            in_green = meter.get('in_green', False)
            shooting = meter.get('shooting', False)
            
            if detected:
                self.meter_label.setText(f"Meter: {fill}%")
                self.meter_label.setStyleSheet(f"color: {C['accent'] if in_green else C['text']}; font-size: 11px;")
            elif shooting:
                self.meter_label.setText("Searching...")
                self.meter_label.setStyleSheet(f"color: {C['warning']}; font-size: 11px;")
            else:
                self.meter_label.setText("Meter: --")
                self.meter_label.setStyleSheet(f"color: {C['dim']}; font-size: 11px;")
                
            # Shot status
            if shot.get('active', False):
                stype = shot.get('type', 'NONE')
                self.shot_label.setText(stype)
                self.shot_label.setStyleSheet(f"font-size: 22px; font-weight: bold; color: {C['accent']};")
            elif shooting:
                self.shot_label.setText("SHOOTING")
                self.shot_label.setStyleSheet(f"font-size: 22px; font-weight: bold; color: {C['warning']};")
            else:
                self.shot_label.setText("READY")
                self.shot_label.setStyleSheet(f"font-size: 22px; font-weight: bold; color: {C['dim']};")
                
            # Shot stats
            total = shot.get('total', 0)
            greens = shot.get('greens', 0)
            pct = int(greens / total * 100) if total > 0 else 0
            self.shots_lbl.setText(f"Shots: {total}")
            self.greens_lbl.setText(f"Greens: {greens} ({pct}%)")
            
            # Inputs
            self.input_indicators.get("RS", InputIndicator("")).set_value(max(abs(inputs.get('rx', 0)), abs(inputs.get('ry', 0))))
            self.input_indicators.get("SQ/X", InputIndicator("")).set_value(inputs.get('square', 0))
            self.input_indicators.get("L2", InputIndicator("")).set_value(inputs.get('l2', 0))
            self.input_indicators.get("R2", InputIndicator("")).set_value(inputs.get('r2', 0))
            self.input_indicators.get("R3", InputIndicator("")).set_value(inputs.get('r3', 0))
            self.input_indicators.get("L3", InputIndicator("")).set_value(inputs.get('l3', 0))
            
            # FPS
            self.fps_lbl.setText(f"FPS: {msg.get('fps', 0)}")
            
        finally:
            self._updating = False
            
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.drag_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
            
    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.MouseButton.LeftButton and self.drag_pos:
            self.move(e.globalPosition().toPoint() - self.drag_pos)
            
    def mouseReleaseEvent(self, e):
        self.drag_pos = None
        
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor(C['border']), 1))
        p.setBrush(QBrush(QColor(C['bg'])))
        p.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 12, 12)
        
    def closeEvent(self, e):
        self.timer.stop()
        self.client.stop()
        self.client.wait(500)
        e.accept()


if __name__ == "__main__":
    port = DEFAULT_PORT
    if len(sys.argv) > 1:
        try: port = int(sys.argv[1])
        except: pass
    elif 'WARZAVISION_PORT' in os.environ:
        try: port = int(os.environ['WARZAVISION_PORT'])
        except: pass
    else:
        pf = os.path.join(SCRIPT_DIR, '.warzavision_port')
        if os.path.exists(pf):
            try:
                with open(pf) as f: port = int(f.read().strip())
            except: pass
    
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    window = MainWindow(port)
    window.show()
    sys.exit(app.exec())
