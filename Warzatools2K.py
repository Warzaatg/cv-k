"""WarzaVision Pro 9.6 - MAXIMUM PRECISION EDITION
NBA 2K | EVERY INPUT | PERFECT GREEN | AUTO NETWORK

FEATURES:
- Catches EVERY controller input
- Auto-detects network adapter & shared connections
- Real-time ping compensation with Kalman filtering
- Sub-millisecond shot timing precision
- Full Xbox & PlayStation support
- Live GUI sync
"""

import cv2
import numpy as np
import time
import logging
import json
import os
import threading
import socket
import subprocess
import platform
import re
import struct
from collections import deque
from datetime import datetime

VERSION = "9.6"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, 'warzavision.log')
CONFIG_FILE = os.path.join(SCRIPT_DIR, 'config.json')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, mode='a'), logging.StreamHandler()]
)

WARZAVISION_PORT = 59420
FONT = cv2.FONT_HERSHEY_SIMPLEX

# =============================================================================
# GTUNER INTERFACE - UNIVERSAL XBOX/PLAYSTATION SUPPORT
# =============================================================================

GTUNER_AVAILABLE = False

# Universal GTuner button indices - SAME for Xbox AND PlayStation!
# GTuner normalizes all controller inputs to these indices
class BTN:
    """Universal button mapping for GTuner/Titan Two"""
    # System
    PS_XBOX = 0
    SHARE_VIEW = 1
    OPTIONS_MENU = 2
    
    # Shoulders & Triggers
    R1_RB = 3
    R3_RS = 4        # Right stick click - DUNK TRIGGER
    R2_RT = 5        # Right trigger - SPRINT
    L1_LB = 6
    L3_LS = 7        # Left stick click
    L2_LT = 8        # Left trigger - POST MODIFIER
    
    # Sticks
    RX = 9           # Right stick X - RHYTHM
    RY = 10          # Right stick Y - RHYTHM  
    LX = 11          # Left stick X
    LY = 12          # Left stick Y
    
    # D-pad
    UP = 13
    DOWN = 14
    LEFT = 15
    RIGHT = 16
    
    # Face buttons
    SQUARE_X = 17    # Square/X - SHOOT
    CROSS_A = 18     # Cross/A - PASS
    CIRCLE_B = 19    # Circle/B
    TRIANGLE_Y = 20  # Triangle/Y
    
    # Touch/Special
    TOUCH = 27
    
# Thresholds
STICK_DEAD_ZONE = 15
STICK_SHOOT_THRESHOLD = 79
TRIGGER_THRESHOLD = 50
BUTTON_THRESHOLD = 50

class MockGTuner:
    """Mock GTuner for standalone testing - simulates all inputs"""
    _states = {}
    
    # Button constants matching real gtuner module
    BUTTON_0 = 0
    BUTTON_1 = 1
    BUTTON_2 = 2
    BUTTON_3 = 3
    BUTTON_4 = 4
    BUTTON_5 = 5
    BUTTON_6 = 6
    BUTTON_7 = 7
    BUTTON_8 = 8
    BUTTON_9 = 9
    BUTTON_10 = 10
    BUTTON_11 = 11
    BUTTON_12 = 12
    BUTTON_13 = 13
    BUTTON_14 = 14
    BUTTON_15 = 15
    BUTTON_16 = 16
    BUTTON_17 = 17
    BUTTON_18 = 18
    BUTTON_19 = 19
    BUTTON_20 = 20
    STICK_1_X = 9
    STICK_1_Y = 10
    STICK_2_X = 11
    STICK_2_Y = 12
    
    @staticmethod
    def get_actual(btn): return MockGTuner._states.get(btn, 0)
    @staticmethod
    def get_val(btn): return MockGTuner._states.get(btn, 0)
    @staticmethod
    def set_val(btn, val): MockGTuner._states[btn] = val
    @staticmethod
    def gcv_ready(): return True
    @staticmethod
    def gcv_write(i, v): pass
    @staticmethod
    def gcv_read(i): return 0

gtuner = MockGTuner()

try:
    import gtuner as _gtuner
    gtuner = _gtuner
    GTUNER_AVAILABLE = True
    logging.info(f"GTuner CONNECTED - Xbox & PlayStation ready")
except ImportError:
    logging.info("Standalone mode - GTuner not available")
except Exception as e:
    logging.warning(f"GTuner error: {e}")

# =============================================================================
# GCV OUTPUT INDICES - MATCH GPC SCRIPT!
# =============================================================================
GCV_RHYTHM_SHOT = 1
GCV_REGULAR_SHOT = 2  
GCV_RHYTHM_TIMING = 3
GCV_RHYTHM_TEMPO = 4
GCV_REGULAR_TIMING = 5
GCV_POST_SHOT = 6
GCV_POST_TIMING = 7
GCV_DUNK_READY = 8
GCV_DUNK_TIMING_LO = 9
GCV_DUNK_TIMING_HI = 10
GCV_PING_COMP = 11
GCV_METER_FILL = 12
GCV_IN_GREEN = 13
GCV_SHOT_QUALITY = 14
GCV_CONNECTION_STATUS = 15

# =============================================================================
# NETWORK AUTO-DETECTION & STABILIZER
# =============================================================================

class NetworkDetector:
    """Auto-detect network adapters, IPv4, and shared connections"""
    
    def __init__(self):
        self.adapters = []
        self.primary_ip = None
        self.gateway = None
        self.shared_connection = None
        self.connection_type = "Unknown"
        
    def detect_all(self):
        """Detect all network info"""
        self._detect_adapters()
        self._detect_gateway()
        self._detect_shared_connection()
        return self.get_info()
        
    def _detect_adapters(self):
        """Find all network adapters and their IPs"""
        self.adapters = []
        try:
            if platform.system() == 'Windows':
                result = subprocess.run(
                    ['ipconfig'], capture_output=True, text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                current_adapter = None
                for line in result.stdout.split('\n'):
                    if 'adapter' in line.lower() and ':' in line:
                        current_adapter = line.split(':')[0].strip()
                    elif 'IPv4' in line and current_adapter:
                        ip = line.split(':')[-1].strip()
                        if ip and not ip.startswith('127.'):
                            self.adapters.append({
                                'name': current_adapter,
                                'ip': ip,
                                'type': self._guess_adapter_type(current_adapter)
                            })
                            if not self.primary_ip:
                                self.primary_ip = ip
            else:
                # Linux/Mac
                result = subprocess.run(
                    ['ip', 'addr'], capture_output=True, text=True
                )
                for line in result.stdout.split('\n'):
                    if 'inet ' in line and '127.0.0.1' not in line:
                        parts = line.strip().split()
                        ip = parts[1].split('/')[0]
                        self.adapters.append({'name': 'eth', 'ip': ip, 'type': 'Ethernet'})
                        if not self.primary_ip:
                            self.primary_ip = ip
        except Exception as e:
            logging.error(f"Adapter detection error: {e}")
            
    def _detect_gateway(self):
        """Find default gateway"""
        try:
            if platform.system() == 'Windows':
                result = subprocess.run(
                    ['ipconfig'], capture_output=True, text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                for line in result.stdout.split('\n'):
                    if 'Default Gateway' in line:
                        gw = line.split(':')[-1].strip()
                        if gw and gw[0].isdigit():
                            self.gateway = gw
                            break
            else:
                result = subprocess.run(
                    ['ip', 'route'], capture_output=True, text=True
                )
                for line in result.stdout.split('\n'):
                    if 'default' in line:
                        parts = line.split()
                        if 'via' in parts:
                            idx = parts.index('via')
                            self.gateway = parts[idx + 1]
                            break
        except Exception as e:
            logging.error(f"Gateway detection error: {e}")
            
    def _detect_shared_connection(self):
        """Detect if using Internet Connection Sharing"""
        try:
            if platform.system() == 'Windows':
                # Check for ICS indicators
                for adapter in self.adapters:
                    if '192.168.137.' in adapter['ip']:
                        self.shared_connection = adapter['ip']
                        self.connection_type = "ICS (Shared)"
                        return
                # Check for mobile hotspot
                result = subprocess.run(
                    ['netsh', 'wlan', 'show', 'hostednetwork'],
                    capture_output=True, text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                if 'Started' in result.stdout:
                    self.connection_type = "Mobile Hotspot"
                    return
            # Determine connection type
            if self.primary_ip:
                if self.primary_ip.startswith('192.168.'):
                    self.connection_type = "Private Network"
                elif self.primary_ip.startswith('10.'):
                    self.connection_type = "Corporate/VPN"
                elif self.primary_ip.startswith('172.'):
                    self.connection_type = "Private (Class B)"
                else:
                    self.connection_type = "Direct/Public"
        except Exception as e:
            logging.error(f"Shared connection detection error: {e}")
            
    def _guess_adapter_type(self, name):
        """Guess adapter type from name"""
        name_lower = name.lower()
        if 'wi-fi' in name_lower or 'wireless' in name_lower or 'wlan' in name_lower:
            return 'WiFi'
        elif 'ethernet' in name_lower or 'eth' in name_lower or 'lan' in name_lower:
            return 'Ethernet'
        elif 'mobile' in name_lower or 'cellular' in name_lower:
            return 'Mobile'
        elif 'vpn' in name_lower or 'virtual' in name_lower:
            return 'VPN'
        return 'Other'
        
    def get_info(self):
        return {
            'adapters': self.adapters,
            'primary_ip': self.primary_ip,
            'gateway': self.gateway,
            'shared_connection': self.shared_connection,
            'connection_type': self.connection_type
        }


class SuperConnectionStabilizer:
    """Advanced connection stabilization with Kalman filtering and predictive compensation"""
    
    # 2K Server IPs
    SERVERS = [
        {"name": "US-East", "ip": "104.255.107.140", "region": "NA"},
        {"name": "US-West", "ip": "104.255.104.100", "region": "NA"},
        {"name": "EU-West", "ip": "185.56.65.100", "region": "EU"},
        {"name": "EU-Central", "ip": "185.56.67.100", "region": "EU"},
        {"name": "Asia", "ip": "203.107.43.100", "region": "ASIA"},
    ]
    
    def __init__(self):
        self._lock = threading.Lock()
        
        # Network detection
        self.net_detector = NetworkDetector()
        self.network_info = {}
        
        # Server selection
        self.active_server = self.SERVERS[0]
        self.server_pings = {}
        
        # Ping tracking
        self.current_ping = 0.0
        self.ping_history = deque(maxlen=100)
        self.recent_pings = deque(maxlen=20)
        
        # Baseline establishment
        self.baseline_ping = 0.0
        self.baseline_samples = deque(maxlen=30)
        self.baseline_established = False
        self.calibration_progress = 0
        
        # Statistics
        self.min_ping = 999.0
        self.max_ping = 0.0
        self.avg_ping = 0.0
        self.jitter = 0.0
        self.jitter_trend = 0.0
        self.packet_loss = 0.0
        self.stability_score = 100.0
        
        # Spike detection
        self.spike_threshold = 1.5
        self.spike_detected = False
        self.spike_count = 0
        self.consecutive_spikes = 0
        
        # Kalman filter state
        self.kalman_estimate = 0.0
        self.kalman_error = 1.0
        self.kalman_q = 0.1  # Process noise
        self.kalman_r = 0.5  # Measurement noise
        
        # Compensation calculation
        self.base_compensation = 0.0
        self.jitter_compensation = 0.0
        self.spike_compensation = 0.0
        self.predictive_compensation = 0.0
        self.total_compensation = 0.0
        
        # Connection quality
        self.quality = "UNKNOWN"
        self.quality_color = (128, 128, 128)
        
        # Control
        self.running = False
        self.ping_thread = None
        self.failed_pings = 0
        
    def start(self):
        """Start the stabilizer"""
        if self.running:
            return
        self.running = True
        
        # Detect network first
        threading.Thread(target=self._init_network, daemon=True).start()
        
    def _init_network(self):
        """Initialize network detection and server selection"""
        logging.info("Detecting network configuration...")
        self.network_info = self.net_detector.detect_all()
        logging.info(f"Network: {self.network_info['connection_type']} | IP: {self.network_info['primary_ip']}")
        
        # Find best server
        self._find_best_server()
        
        # Start ping loop
        self.ping_thread = threading.Thread(target=self._ping_loop, daemon=True)
        self.ping_thread.start()
        
    def _find_best_server(self):
        """Test all servers and find the best one"""
        logging.info("Finding best 2K server...")
        best_server = None
        best_ping = 999
        
        for server in self.SERVERS:
            ping = self._ping(server['ip'])
            if ping:
                self.server_pings[server['name']] = ping
                logging.info(f"  {server['name']}: {ping}ms")
                if ping < best_ping:
                    best_ping = ping
                    best_server = server
                    
        if best_server:
            self.active_server = best_server
            logging.info(f"Selected server: {best_server['name']} ({best_ping}ms)")
        else:
            logging.warning("Could not reach any 2K servers, using default")
            
    def _ping(self, ip):
        """Ping an IP and return latency in ms"""
        try:
            if platform.system() == 'Windows':
                result = subprocess.run(
                    ['ping', '-n', '1', '-w', '1000', ip],
                    capture_output=True, text=True, timeout=3,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                match = re.search(r'time[=<](\d+)ms', result.stdout)
            else:
                result = subprocess.run(
                    ['ping', '-c', '1', '-W', '1', ip],
                    capture_output=True, text=True, timeout=3
                )
                match = re.search(r'time=(\d+\.?\d*)', result.stdout)
            if match:
                return float(match.group(1))
        except:
            pass
        return None
        
    def _ping_loop(self):
        """Main ping loop"""
        while self.running:
            ping = self._ping(self.active_server['ip'])
            
            if ping is not None:
                self._process_ping(ping)
                self.failed_pings = 0
            else:
                self.failed_pings += 1
                if self.failed_pings >= 3:
                    self.packet_loss = min(100, self.packet_loss + 5)
                    
            # Adaptive sleep - faster when calibrating
            sleep_time = 0.3 if not self.baseline_established else 0.5
            time.sleep(sleep_time)
            
    def _process_ping(self, ping):
        """Process a new ping measurement"""
        with self._lock:
            self.current_ping = ping
            self.ping_history.append(ping)
            self.recent_pings.append(ping)
            
            # Track min/max
            self.min_ping = min(self.min_ping, ping)
            self.max_ping = max(self.max_ping, ping)
            
            # Baseline establishment
            if not self.baseline_established:
                self.baseline_samples.append(ping)
                self.calibration_progress = min(100, int(len(self.baseline_samples) / 30 * 100))
                
                if len(self.baseline_samples) >= 30:
                    sorted_samples = sorted(self.baseline_samples)
                    # Use median of middle 60% to exclude outliers
                    start = int(len(sorted_samples) * 0.2)
                    end = int(len(sorted_samples) * 0.8)
                    self.baseline_ping = np.mean(sorted_samples[start:end])
                    self.baseline_established = True
                    self.kalman_estimate = self.baseline_ping
                    logging.info(f"Baseline established: {self.baseline_ping:.1f}ms")
                return
                
            # Kalman filter update
            self._kalman_update(ping)
            
            # Calculate jitter
            if len(self.recent_pings) >= 2:
                pings = list(self.recent_pings)
                diffs = [abs(pings[i] - pings[i-1]) for i in range(1, len(pings))]
                self.jitter = np.mean(diffs)
                
                # Jitter trend (increasing or decreasing)
                if len(diffs) >= 5:
                    recent_jitter = np.mean(diffs[-5:])
                    older_jitter = np.mean(diffs[:-5]) if len(diffs) > 5 else recent_jitter
                    self.jitter_trend = recent_jitter - older_jitter
                    
            # Average ping
            self.avg_ping = np.mean(list(self.recent_pings))
            
            # Spike detection
            self._detect_spike(ping)
            
            # Calculate stability score
            self._calculate_stability()
            
            # Calculate compensation
            self._calculate_compensation()
            
            # Update quality rating
            self._update_quality()
            
    def _kalman_update(self, measurement):
        """Kalman filter for smooth ping prediction"""
        # Prediction
        pred_estimate = self.kalman_estimate
        pred_error = self.kalman_error + self.kalman_q
        
        # Update
        kalman_gain = pred_error / (pred_error + self.kalman_r)
        self.kalman_estimate = pred_estimate + kalman_gain * (measurement - pred_estimate)
        self.kalman_error = (1 - kalman_gain) * pred_error
        
    def _detect_spike(self, ping):
        """Detect ping spikes"""
        if self.baseline_ping > 0:
            threshold = self.baseline_ping * self.spike_threshold
            if ping > threshold:
                self.spike_detected = True
                self.spike_count += 1
                self.consecutive_spikes += 1
            else:
                self.spike_detected = False
                self.consecutive_spikes = 0
                
    def _calculate_stability(self):
        """Calculate overall stability score (0-100)"""
        score = 100.0
        
        # Penalize for jitter (up to -30)
        jitter_penalty = min(30, self.jitter * 2)
        score -= jitter_penalty
        
        # Penalize for deviation from baseline (up to -25)
        if self.baseline_ping > 0:
            deviation = abs(self.current_ping - self.baseline_ping)
            deviation_penalty = min(25, deviation * 0.5)
            score -= deviation_penalty
            
        # Penalize for spikes (up to -20)
        if self.consecutive_spikes > 0:
            spike_penalty = min(20, self.consecutive_spikes * 5)
            score -= spike_penalty
            
        # Penalize for packet loss (up to -25)
        loss_penalty = min(25, self.packet_loss * 0.5)
        score -= loss_penalty
        
        self.stability_score = max(0, score)
        
    def _calculate_compensation(self):
        """Calculate total compensation for shot timing"""
        # Base compensation: half of predicted ping
        self.base_compensation = self.kalman_estimate * 0.5
        
        # Jitter compensation: buffer for variability
        self.jitter_compensation = self.jitter * 0.8
        
        # Spike compensation: extra buffer if spikes detected
        if self.spike_detected or self.consecutive_spikes > 0:
            self.spike_compensation = min(30, self.consecutive_spikes * 5)
        else:
            self.spike_compensation = max(0, self.spike_compensation - 1)
            
        # Predictive compensation: adjust for jitter trend
        if self.jitter_trend > 0:  # Jitter increasing
            self.predictive_compensation = self.jitter_trend * 2
        else:
            self.predictive_compensation = 0
            
        # Total compensation (capped at reasonable values)
        self.total_compensation = min(100, max(5,
            self.base_compensation + 
            self.jitter_compensation + 
            self.spike_compensation + 
            self.predictive_compensation
        ))
        
    def _update_quality(self):
        """Update connection quality rating"""
        if self.stability_score >= 85:
            self.quality = "EXCELLENT"
            self.quality_color = (0, 255, 0)  # Green
        elif self.stability_score >= 70:
            self.quality = "GOOD"
            self.quality_color = (0, 255, 128)
        elif self.stability_score >= 50:
            self.quality = "FAIR"
            self.quality_color = (0, 255, 255)  # Yellow
        elif self.stability_score >= 30:
            self.quality = "POOR"
            self.quality_color = (0, 165, 255)  # Orange
        else:
            self.quality = "BAD"
            self.quality_color = (0, 0, 255)  # Red
            
    def get_compensation_ms(self):
        """Get current compensation in milliseconds"""
        with self._lock:
            return int(round(self.total_compensation))
            
    def get_stats(self):
        """Get all stats for GUI"""
        with self._lock:
            return {
                'server': self.active_server['name'],
                'server_region': self.active_server['region'],
                'ping': int(round(self.current_ping)),
                'ping_min': int(round(self.min_ping)) if self.min_ping < 999 else 0,
                'ping_max': int(round(self.max_ping)),
                'ping_avg': int(round(self.avg_ping)),
                'ping_predicted': int(round(self.kalman_estimate)),
                'baseline': int(round(self.baseline_ping)),
                'jitter': round(self.jitter, 1),
                'jitter_trend': round(self.jitter_trend, 2),
                'stability': int(round(self.stability_score)),
                'quality': self.quality,
                'quality_color': self.quality_color,
                'spike_detected': self.spike_detected,
                'spike_count': self.spike_count,
                'packet_loss': round(self.packet_loss, 1),
                'compensation': int(round(self.total_compensation)),
                'comp_base': int(round(self.base_compensation)),
                'comp_jitter': int(round(self.jitter_compensation)),
                'comp_spike': int(round(self.spike_compensation)),
                'comp_predictive': int(round(self.predictive_compensation)),
                'calibrated': self.baseline_established,
                'calibration_pct': self.calibration_progress,
                'network': self.network_info,
            }
            
    def stop(self):
        self.running = False


# =============================================================================
# COLOR MANAGER
# =============================================================================

class ColorManager:
    PRESETS = {
        "purple": {"lab": ([20, 140, 100], [200, 200, 180]), "hsv": ([130, 50, 50], [170, 255, 255])},
        "green": {"lab": ([50, 80, 130], [200, 120, 200]), "hsv": ([35, 50, 50], [85, 255, 255])},
        "red": {"lab": ([20, 150, 128], [200, 200, 200]), "hsv": ([0, 50, 50], [10, 255, 255])},
        "yellow": {"lab": ([150, 100, 150], [255, 140, 220]), "hsv": ([20, 50, 50], [35, 255, 255])},
        "blue": {"lab": ([20, 128, 50], [200, 200, 120]), "hsv": ([100, 50, 50], [130, 255, 255])},
        "white": {"lab": ([200, 118, 118], [255, 138, 138]), "hsv": ([0, 0, 200], [180, 30, 255])},
        "custom": {"lab": ([0, 0, 0], [255, 255, 255]), "hsv": ([0, 0, 0], [180, 255, 255])},
    }
    
    def __init__(self):
        self.current = "purple"
        self.space = "LAB"
        self.tolerance = 30
        
    def set_color(self, name):
        if name in self.PRESETS:
            self.current = name
            
    def set_space(self, space):
        self.space = space.upper()
        
    def set_custom_bgr(self, bgr, tolerance=30):
        b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
        self.tolerance = tolerance
        
        # LAB
        lab = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2LAB)[0, 0]
        self.PRESETS["custom"]["lab"] = (
            [max(0, lab[0]-tolerance), max(0, lab[1]-tolerance), max(0, lab[2]-tolerance)],
            [min(255, lab[0]+tolerance), min(255, lab[1]+tolerance), min(255, lab[2]+tolerance)]
        )
        # HSV
        hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0]
        self.PRESETS["custom"]["hsv"] = (
            [max(0, hsv[0]-15), max(0, hsv[1]-50), max(0, hsv[2]-50)],
            [min(180, hsv[0]+15), min(255, hsv[1]+50), min(255, hsv[2]+50)]
        )
        self.current = "custom"
        
    def get_bounds(self):
        preset = self.PRESETS.get(self.current, self.PRESETS["purple"])
        key = "lab" if self.space == "LAB" else "hsv"
        return np.array(preset[key][0], np.uint8), np.array(preset[key][1], np.uint8)
        
    def get_name(self):
        return self.current


# =============================================================================
# CONFIG MANAGER  
# =============================================================================

class Config:
    TIMING_KEYS = ["rhythm_timing_ms", "rhythm_tempo", "regular_timing_ms",
                   "post_timing_ms", "dunk_timing_ms", "green_offset"]
    
    def __init__(self):
        self._lock = threading.Lock()
        self.data = self._load()
        
    def _defaults(self):
        return {
            "rhythm_timing_ms": 28,
            "rhythm_tempo": 62,
            "regular_timing_ms": 165,
            "post_timing_ms": 64,
            "dunk_timing_ms": 945,
            "green_offset": 0,
            "meter_color": "purple",
            "color_space": "LAB",
            "tolerance": 30,
            "min_area": 40,
            "max_area": 30000,
            "auto_comp": True,
            "show_overlay": True,
            "aggressive_mode": False,
        }
        
    def _load(self):
        defaults = self._defaults()
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as f:
                    loaded = json.load(f)
                for k, v in defaults.items():
                    if k not in loaded:
                        loaded[k] = v
                return loaded
        except Exception as e:
            logging.error(f"Config error: {e}")
        return defaults
        
    def _save(self):
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.data, f, indent=2)
        except:
            pass
            
    def get(self, key, default=None):
        with self._lock:
            val = self.data.get(key, default)
            return int(round(val)) if key in self.TIMING_KEYS and val else val
            
    def set(self, key, value):
        with self._lock:
            if key in self.TIMING_KEYS:
                value = int(round(value))
            self.data[key] = value
            self._save()
            
    def update(self, updates):
        with self._lock:
            for k, v in updates.items():
                if k in self.TIMING_KEYS:
                    v = int(round(v))
                self.data[k] = v
            self._save()
            
    def get_all(self):
        with self._lock:
            return self.data.copy()

CONFIG = Config()


# =============================================================================
# JSON SERVER FOR GUI - HIGH SPEED SYNC
# =============================================================================

class JSONServer(threading.Thread):
    def __init__(self, port=WARZAVISION_PORT):
        super().__init__(daemon=True)
        self.port = port
        self.sock = None
        self.client = None
        self.connected = False
        self.stop_flag = False
        self.on_update = None
        self._lock = threading.Lock()
        
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind(("127.0.0.1", port))
            self.sock.listen(1)
        except:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind(("127.0.0.1", 0))
            self.port = self.sock.getsockname()[1]
            self.sock.listen(1)
            
    def run(self):
        self.sock.settimeout(0.5)
        buf = b""
        while not self.stop_flag:
            try:
                if not self.client:
                    try:
                        c, _ = self.sock.accept()
                        c.settimeout(0.05)  # Fast timeout for responsiveness
                        with self._lock:
                            self.client = c
                            self.connected = True
                            logging.info("GUI connected")
                    except socket.timeout:
                        continue
                else:
                    try:
                        data = self.client.recv(4096)
                        if not data:
                            self._disconnect()
                            buf = b""
                            continue
                        buf += data
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            if line.strip() and self.on_update:
                                try:
                                    self.on_update(json.loads(line.decode()))
                                except:
                                    pass
                    except socket.timeout:
                        pass
                    except:
                        self._disconnect()
                        buf = b""
            except:
                time.sleep(0.05)
                
    def _disconnect(self):
        with self._lock:
            if self.client:
                try: self.client.close()
                except: pass
            self.client = None
            self.connected = False
            logging.info("GUI disconnected")
                
    def send(self, data):
        with self._lock:
            if self.client and self.connected:
                try:
                    self.client.sendall((json.dumps(data) + "\n").encode())
                except:
                    pass
                    
    def stop(self):
        self.stop_flag = True


# =============================================================================
# MAIN GCV WORKER - CATCHES EVERYTHING
# =============================================================================

class GCVWorker:
    def __init__(self, *args, **kwargs):
        self.width = 1920
        self.height = 1080
        
        self.gcvdata = bytearray(16)
        
        self.color_manager = ColorManager()
        self.color_manager.set_color(CONFIG.get("meter_color", "purple"))
        self.color_manager.set_space(CONFIG.get("color_space", "LAB"))
        
        # Advanced connection stabilizer
        self.network = SuperConnectionStabilizer()
        
        # ===========================================
        # SHOT STATE
        # ===========================================
        self.shot_in_progress = False
        self.rhythm_shot = False
        self.regular_shot = False
        self.post_shot = False
        self.auto_dunking = False
        self.shot_timer = None
        self.shot_start_time = None
        self.last_shot_type = None
        self.last_shot_timing = 0
        
        # Shot statistics
        self.shots_taken = 0
        self.greens_hit = 0
        self.shot_history = deque(maxlen=20)
        
        # ===========================================
        # INPUT TRACKING - CATCH EVERYTHING
        # ===========================================
        self.inputs = {
            'rx': 0, 'ry': 0, 'lx': 0, 'ly': 0,
            'l2': 0, 'r2': 0, 'l1': 0, 'r1': 0,
            'l3': 0, 'r3': 0,
            'square': 0, 'cross': 0, 'circle': 0, 'triangle': 0,
            'up': 0, 'down': 0, 'left': 0, 'right': 0,
        }
        self.l2_held = False
        self.r2_held = False
        self.shooting_detected = False
        
        # ===========================================
        # METER DETECTION
        # ===========================================
        self.meter_detected = False
        self.meter_fill_pct = 0
        self.in_green_zone = False
        self.meter_bbox = None
        self.color_detected = False
        
        kernel_size = (3, 3)
        self.morph_kernel = np.ones(kernel_size, np.uint8)
        self.min_area = CONFIG.get("min_area", 40)
        self.max_area = CONFIG.get("max_area", 30000)
        
        self.last_frame = None
        
        # GUI server
        self.server = JSONServer(WARZAVISION_PORT)
        self.server.on_update = self._on_gui_msg
        self.server.start()
        
        # Start network stabilizer
        self.network.start()
        
        # FPS tracking
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.last_status_time = 0
        
        self._lock = threading.Lock()
        self._write_port_file()
        self._print_startup()
        
    def _write_port_file(self):
        try:
            with open(os.path.join(SCRIPT_DIR, '.warzavision_port'), 'w') as f:
                f.write(str(self.server.port))
        except:
            pass
            
    def _print_startup(self):
        print("\n" + "="*65)
        print(f"  WARZAVISION PRO {VERSION} - MAXIMUM PRECISION")
        print("  Xbox & PlayStation | Every Input | Perfect Green")
        print("="*65)
        print("  SHOT CONTROLS:")
        print("    Right Stick      -> Rhythm Shot (Pro Stick)")
        print("    Square/X         -> Regular Jump Shot")
        print("    L2/LT + Square/X -> Post Shot")
        print("    R3/RS + R2/RT    -> Dunk (while driving)")
        print("="*65)
        print(f"  Port: {self.server.port}")
        print(f"  GTuner: {'CONNECTED' if GTUNER_AVAILABLE else 'STANDALONE'}")
        print("="*65 + "\n")
        
    def _on_gui_msg(self, msg):
        """Handle GUI messages"""
        msg_type = msg.get("type", "")
        
        if msg_type == "get_status":
            self._send_status()
        elif msg_type == "update":
            updates = {k: v for k, v in msg.items() if k != "type"}
            CONFIG.update(updates)
            if "meter_color" in updates:
                self.color_manager.set_color(updates["meter_color"])
            if "color_space" in updates:
                self.color_manager.set_space(updates["color_space"])
            if "min_area" in updates:
                self.min_area = int(updates["min_area"])
            if "max_area" in updates:
                self.max_area = int(updates["max_area"])
            self._send_status()
        elif msg_type == "get_screenshot":
            if self.last_frame is not None:
                _, encoded = cv2.imencode('.jpg', self.last_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                import base64
                self.server.send({"type": "screenshot", "image": base64.b64encode(encoded).decode()})
        elif msg_type == "pick_color":
            x, y = int(msg.get("x", 0)), int(msg.get("y", 0))
            if self.last_frame is not None:
                h, w = self.last_frame.shape[:2]
                x, y = min(max(x, 0), w-1), min(max(y, 0), h-1)
                self.color_manager.set_custom_bgr(self.last_frame[y, x], msg.get("tolerance", 30))
                CONFIG.set("meter_color", "custom")
                self._send_status()
                
    def _send_status(self):
        """Send full status to GUI"""
        net = self.network.get_stats()
        
        self.server.send({
            "type": "status",
            "version": VERSION,
            "config": CONFIG.get_all(),
            "network": net,
            "inputs": self.inputs,
            "meter": {
                "detected": self.meter_detected,
                "fill": self.meter_fill_pct,
                "in_green": self.in_green_zone,
                "color": self.color_manager.get_name(),
                "shooting": self.shooting_detected,
            },
            "shot": {
                "active": self.shot_in_progress,
                "type": self._get_shot_type(),
                "last_type": self.last_shot_type,
                "last_timing": self.last_shot_timing,
                "total": self.shots_taken,
                "greens": self.greens_hit,
                "history": list(self.shot_history),
            },
            "fps": self.fps,
            "gtuner": GTUNER_AVAILABLE,
        })
        
    def _get_shot_type(self):
        if self.rhythm_shot: return "RHYTHM"
        if self.regular_shot: return "REGULAR"
        if self.post_shot: return "POST"
        if self.auto_dunking: return "DUNK"
        return "NONE"
        
    # =========================================================================
    # INPUT READING - CATCHES EVERYTHING
    # =========================================================================
    
    def _read_all_inputs(self):
        """Read ALL controller inputs"""
        try:
            # Sticks
            self.inputs['rx'] = gtuner.get_actual(gtuner.STICK_1_X) if hasattr(gtuner, 'STICK_1_X') else gtuner.get_actual(BTN.RX)
            self.inputs['ry'] = gtuner.get_actual(gtuner.STICK_1_Y) if hasattr(gtuner, 'STICK_1_Y') else gtuner.get_actual(BTN.RY)
            self.inputs['lx'] = gtuner.get_actual(gtuner.STICK_2_X) if hasattr(gtuner, 'STICK_2_X') else gtuner.get_actual(BTN.LX)
            self.inputs['ly'] = gtuner.get_actual(gtuner.STICK_2_Y) if hasattr(gtuner, 'STICK_2_Y') else gtuner.get_actual(BTN.LY)
            
            # Triggers
            self.inputs['l2'] = gtuner.get_actual(gtuner.BUTTON_8) if hasattr(gtuner, 'BUTTON_8') else gtuner.get_actual(BTN.L2_LT)
            self.inputs['r2'] = gtuner.get_actual(gtuner.BUTTON_5) if hasattr(gtuner, 'BUTTON_5') else gtuner.get_actual(BTN.R2_RT)
            
            # Bumpers
            self.inputs['l1'] = gtuner.get_actual(gtuner.BUTTON_6) if hasattr(gtuner, 'BUTTON_6') else gtuner.get_actual(BTN.L1_LB)
            self.inputs['r1'] = gtuner.get_actual(gtuner.BUTTON_3) if hasattr(gtuner, 'BUTTON_3') else gtuner.get_actual(BTN.R1_RB)
            
            # Stick clicks
            self.inputs['l3'] = gtuner.get_actual(gtuner.BUTTON_7) if hasattr(gtuner, 'BUTTON_7') else gtuner.get_actual(BTN.L3_LS)
            self.inputs['r3'] = gtuner.get_actual(gtuner.BUTTON_4) if hasattr(gtuner, 'BUTTON_4') else gtuner.get_actual(BTN.R3_RS)
            
            # Face buttons
            self.inputs['square'] = gtuner.get_actual(gtuner.BUTTON_17) if hasattr(gtuner, 'BUTTON_17') else gtuner.get_actual(BTN.SQUARE_X)
            self.inputs['cross'] = gtuner.get_actual(gtuner.BUTTON_18) if hasattr(gtuner, 'BUTTON_18') else gtuner.get_actual(BTN.CROSS_A)
            self.inputs['circle'] = gtuner.get_actual(gtuner.BUTTON_19) if hasattr(gtuner, 'BUTTON_19') else gtuner.get_actual(BTN.CIRCLE_B)
            self.inputs['triangle'] = gtuner.get_actual(gtuner.BUTTON_20) if hasattr(gtuner, 'BUTTON_20') else gtuner.get_actual(BTN.TRIANGLE_Y)
            
            # D-pad
            self.inputs['up'] = gtuner.get_actual(gtuner.BUTTON_13) if hasattr(gtuner, 'BUTTON_13') else gtuner.get_actual(BTN.UP)
            self.inputs['down'] = gtuner.get_actual(gtuner.BUTTON_14) if hasattr(gtuner, 'BUTTON_14') else gtuner.get_actual(BTN.DOWN)
            self.inputs['left'] = gtuner.get_actual(gtuner.BUTTON_15) if hasattr(gtuner, 'BUTTON_15') else gtuner.get_actual(BTN.LEFT)
            self.inputs['right'] = gtuner.get_actual(gtuner.BUTTON_16) if hasattr(gtuner, 'BUTTON_16') else gtuner.get_actual(BTN.RIGHT)
            
            # Track held states
            self.l2_held = self.inputs['l2'] > TRIGGER_THRESHOLD
            self.r2_held = self.inputs['r2'] > TRIGGER_THRESHOLD
            
        except Exception as e:
            logging.debug(f"Input read error: {e}")
            
    def _detect_shooting(self):
        """
        Detect if player is shooting and what type.
        Returns: (is_shooting, shot_type)
        """
        rx, ry = self.inputs['rx'], self.inputs['ry']
        square = self.inputs['square']
        r3 = self.inputs['r3']
        
        # Priority order:
        
        # 1. DUNK: R3 + R2 (stick click while sprinting)
        if r3 > BUTTON_THRESHOLD and self.r2_held:
            return True, 'dunk'
            
        # 2. RHYTHM: Right stick pushed significantly
        if abs(rx) > STICK_SHOOT_THRESHOLD or abs(ry) > STICK_SHOOT_THRESHOLD:
            return True, 'rhythm'
            
        # 3. POST: L2 + Square
        if self.l2_held and square > BUTTON_THRESHOLD:
            return True, 'post'
            
        # 4. REGULAR: Square alone
        if square > BUTTON_THRESHOLD and not self.l2_held:
            return True, 'regular'
            
        # Check for any shooting indication (smaller stick movements)
        if abs(rx) > STICK_DEAD_ZONE or abs(ry) > STICK_DEAD_ZONE:
            return True, None  # Shooting but type unclear
            
        return False, None
        
    # =========================================================================
    # METER DETECTION - FULL FRAME SCAN
    # =========================================================================
    
    def _detect_meter(self, frame):
        """Detect meter anywhere on screen"""
        is_shooting, shot_type = self._detect_shooting()
        self.shooting_detected = is_shooting
        
        # Only actively scan when shooting
        if not is_shooting and not self.shot_in_progress:
            self.meter_detected = False
            self.meter_bbox = None
            self.meter_fill_pct = 0
            self.in_green_zone = False
            self.color_detected = False
            return
            
        # Convert color space
        if self.color_manager.space == "LAB":
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        else:
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
        lower, upper = self.color_manager.get_bounds()
        mask = cv2.inRange(converted, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.meter_detected = False
        self.meter_bbox = None
        best_area = 0
        
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area:
                continue
                
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            if solidity < 0.25:
                continue
                
            if area > best_area:
                x, y, w, h = cv2.boundingRect(c)
                self.meter_bbox = (x, y, w, h)
                self.meter_detected = True
                self.color_detected = True
                best_area = area
                
                fill = (area - self.min_area) / max(1, self.max_area - self.min_area)
                self.meter_fill_pct = int(min(100, max(0, fill * 100)))
                
        # Green zone: 70-95% fill during active shot
        if self.shot_in_progress and self.meter_detected:
            self.in_green_zone = 70 <= self.meter_fill_pct <= 95
        else:
            self.in_green_zone = False
            
    # =========================================================================
    # SHOT CONTROL - PRECISION TIMING
    # =========================================================================
    
    def _calc_timing(self, base):
        """Calculate precise timing with compensation"""
        comp = self.network.get_compensation_ms() if CONFIG.get("auto_comp", True) else 0
        offset = CONFIG.get("green_offset", 0)
        
        # Aggressive mode: slightly earlier release
        if CONFIG.get("aggressive_mode", False):
            comp += 3
            
        final = max(5, base - comp + offset)
        return int(final), comp
        
    def start_rhythm_shot(self):
        with self._lock:
            if self.shot_in_progress:
                return
            self.shot_in_progress = True
            self.rhythm_shot = True
            self.shot_start_time = time.perf_counter()
            
            base = CONFIG.get("rhythm_timing_ms", 28)
            timing, comp = self._calc_timing(base)
            tempo = int(CONFIG.get("rhythm_tempo", 62))
            
            self.gcvdata[GCV_RHYTHM_SHOT] = 0x01
            self.gcvdata[GCV_RHYTHM_TIMING] = int(timing) & 0xFF
            self.gcvdata[GCV_RHYTHM_TEMPO] = int(tempo) & 0xFF
            
            if self.shot_timer:
                self.shot_timer.cancel()
            self.shot_timer = threading.Timer(timing / 1000.0, self._release, args=("rhythm", timing))
            self.shot_timer.start()
            
            self.shots_taken += 1
            logging.info(f"RHYTHM: {timing}ms (comp:{comp}, tempo:{tempo})")
            
    def start_regular_shot(self):
        with self._lock:
            if self.shot_in_progress:
                return
            self.shot_in_progress = True
            self.regular_shot = True
            self.shot_start_time = time.perf_counter()
            
            base = CONFIG.get("regular_timing_ms", 165)
            timing, comp = self._calc_timing(base)
            
            self.gcvdata[GCV_REGULAR_SHOT] = 0x01
            self.gcvdata[GCV_REGULAR_TIMING] = int(timing) & 0xFF
            
            if self.shot_timer:
                self.shot_timer.cancel()
            self.shot_timer = threading.Timer(timing / 1000.0, self._release, args=("regular", timing))
            self.shot_timer.start()
            
            self.shots_taken += 1
            logging.info(f"REGULAR: {timing}ms (comp:{comp})")
            
    def start_post_shot(self):
        with self._lock:
            if self.shot_in_progress:
                return
            self.shot_in_progress = True
            self.post_shot = True
            self.shot_start_time = time.perf_counter()
            
            base = CONFIG.get("post_timing_ms", 64)
            timing, comp = self._calc_timing(base)
            
            self.gcvdata[GCV_POST_SHOT] = 0x01
            self.gcvdata[GCV_POST_TIMING] = int(timing) & 0xFF
            
            if self.shot_timer:
                self.shot_timer.cancel()
            self.shot_timer = threading.Timer(timing / 1000.0, self._release, args=("post", timing))
            self.shot_timer.start()
            
            self.shots_taken += 1
            logging.info(f"POST: {timing}ms (comp:{comp})")
            
    def start_dunk(self):
        with self._lock:
            if self.shot_in_progress:
                return
            self.shot_in_progress = True
            self.auto_dunking = True
            self.shot_start_time = time.perf_counter()
            
            timing = int(CONFIG.get("dunk_timing_ms", 945))
            
            self.gcvdata[GCV_DUNK_READY] = 0x01
            self.gcvdata[GCV_DUNK_TIMING_LO] = int(timing) & 0xFF
            self.gcvdata[GCV_DUNK_TIMING_HI] = int((timing >> 8)) & 0xFF
            
            if self.shot_timer:
                self.shot_timer.cancel()
            self.shot_timer = threading.Timer(timing / 1000.0, self._release, args=("dunk", timing))
            self.shot_timer.start()
            
            self.shots_taken += 1
            logging.info(f"DUNK: {timing}ms")
            
    def _release(self, shot_type, timing):
        """Release shot"""
        with self._lock:
            # Record shot
            was_green = self.in_green_zone
            if was_green:
                self.greens_hit += 1
            self.shot_history.append({
                'type': shot_type,
                'timing': timing,
                'green': was_green,
                'time': datetime.now().strftime('%H:%M:%S')
            })
            
            self.last_shot_type = shot_type
            self.last_shot_timing = timing
            self.shot_in_progress = False
            self.shot_start_time = None
            self.in_green_zone = False
            
            if shot_type == "rhythm":
                self.rhythm_shot = False
                self.gcvdata[GCV_RHYTHM_SHOT] = 0x00
            elif shot_type == "regular":
                self.regular_shot = False
                self.gcvdata[GCV_REGULAR_SHOT] = 0x00
            elif shot_type == "post":
                self.post_shot = False
                self.gcvdata[GCV_POST_SHOT] = 0x00
            elif shot_type == "dunk":
                self.auto_dunking = False
                self.gcvdata[GCV_DUNK_READY] = 0x00
                self.gcvdata[GCV_DUNK_TIMING_LO] = 0x00
                self.gcvdata[GCV_DUNK_TIMING_HI] = 0x00
                
    def _trigger_shots(self):
        """Trigger shots based on input detection"""
        if self.shot_in_progress:
            return
            
        is_shooting, shot_type = self._detect_shooting()
        
        if not is_shooting:
            return
            
        # Dunk doesn't need meter
        if shot_type == 'dunk':
            self.start_dunk()
            return
            
        # Other shots need meter OR color detected
        if self.meter_detected or self.color_detected:
            if shot_type == 'rhythm':
                self.start_rhythm_shot()
            elif shot_type == 'regular':
                self.start_regular_shot()
            elif shot_type == 'post':
                self.start_post_shot()
                
    def _update_gcv(self):
        """Update GCV buffer"""
        with self._lock:
            # Shot flags
            self.gcvdata[GCV_RHYTHM_SHOT] = 0x01 if self.rhythm_shot else 0x00
            self.gcvdata[GCV_REGULAR_SHOT] = 0x01 if self.regular_shot else 0x00
            self.gcvdata[GCV_POST_SHOT] = 0x01 if self.post_shot else 0x00
            self.gcvdata[GCV_DUNK_READY] = 0x01 if self.auto_dunking else 0x00
            
            # Timings
            self.gcvdata[GCV_RHYTHM_TIMING] = int(CONFIG.get("rhythm_timing_ms", 28)) & 0xFF
            self.gcvdata[GCV_RHYTHM_TEMPO] = int(CONFIG.get("rhythm_tempo", 62)) & 0xFF
            self.gcvdata[GCV_REGULAR_TIMING] = int(CONFIG.get("regular_timing_ms", 165)) & 0xFF
            self.gcvdata[GCV_POST_TIMING] = int(CONFIG.get("post_timing_ms", 64)) & 0xFF
            
            dunk = int(CONFIG.get("dunk_timing_ms", 945))
            self.gcvdata[GCV_DUNK_TIMING_LO] = int(dunk) & 0xFF
            self.gcvdata[GCV_DUNK_TIMING_HI] = int((dunk >> 8)) & 0xFF
            
            # Network
            comp = int(self.network.get_compensation_ms())
            self.gcvdata[GCV_PING_COMP] = int(min(255, max(0, comp)))
            self.gcvdata[GCV_METER_FILL] = int(min(255, max(0, self.meter_fill_pct)))
            self.gcvdata[GCV_IN_GREEN] = 0x01 if self.in_green_zone else 0x00
            
            # Quality indicator
            net = self.network.get_stats()
            self.gcvdata[GCV_SHOT_QUALITY] = int(min(255, net['stability']))
            self.gcvdata[GCV_CONNECTION_STATUS] = 0x01 if net['calibrated'] else 0x00
            
    # =========================================================================
    # OVERLAY - COMPREHENSIVE INFO
    # =========================================================================
    
    def _draw_overlay(self, frame):
        if not CONFIG.get("show_overlay", True):
            return
            
        h, w = frame.shape[:2]
        white = (255, 255, 255)
        green = (0, 255, 0)
        yellow = (0, 255, 255)
        red = (0, 0, 255)
        cyan = (255, 255, 0)
        gray = (100, 100, 100)
        
        net = self.network.get_stats()
        
        # === TOP LEFT: Version & FPS ===
        y = 25
        cv2.putText(frame, f"WarzaVision {VERSION}", (10, y), FONT, 0.5, cyan, 1)
        y += 18
        cv2.putText(frame, f"FPS: {self.fps}", (10, y), FONT, 0.4, white, 1)
        y += 16
        cv2.putText(frame, f"Color: {self.color_manager.get_name()}", (10, y), FONT, 0.4, white, 1)
        
        # === TOP RIGHT: Network ===
        rx = w - 180
        y = 25
        qc = tuple(net['quality_color'][::-1])  # BGR to RGB
        cv2.putText(frame, f"{net['quality']}", (rx, y), FONT, 0.5, qc, 1)
        y += 18
        cv2.putText(frame, f"Server: {net['server']}", (rx, y), FONT, 0.35, white, 1)
        y += 16
        cv2.putText(frame, f"Ping: {net['ping']}ms (baseline: {net['baseline']})", (rx, y), FONT, 0.35, white, 1)
        y += 16
        cv2.putText(frame, f"Jitter: {net['jitter']}ms | Stab: {net['stability']}%", (rx, y), FONT, 0.35, white, 1)
        y += 16
        comp_color = green if net['compensation'] > 0 else gray
        cv2.putText(frame, f"Compensation: -{net['compensation']}ms", (rx, y), FONT, 0.4, comp_color, 1)
        
        # === SHOT STATUS ===
        if self.shot_in_progress:
            shot_type = self._get_shot_type()
            text = f"SHOT: {shot_type}"
            sz = cv2.getTextSize(text, FONT, 0.8, 2)[0]
            cv2.putText(frame, text, ((w-sz[0])//2, h-50), FONT, 0.8, green, 2)
            
            if self.in_green_zone:
                cv2.putText(frame, "GREEN!", ((w-100)//2, h-90), FONT, 1.0, green, 2)
        elif self.shooting_detected:
            cv2.putText(frame, "SHOOTING...", ((w-150)//2, h-50), FONT, 0.6, yellow, 1)
            
        # === METER BOX ===
        if self.meter_bbox and self.meter_detected:
            x, my, mw, mh = self.meter_bbox
            color = green if self.in_green_zone else yellow if self.shot_in_progress else cyan
            cv2.rectangle(frame, (x, my), (x+mw, my+mh), color, 2)
            cv2.putText(frame, f"{self.meter_fill_pct}%", (x, my-8), FONT, 0.5, color, 1)
            
        # === INPUTS (bottom left) ===
        y = h - 100
        if abs(self.inputs['rx']) > STICK_DEAD_ZONE or abs(self.inputs['ry']) > STICK_DEAD_ZONE:
            cv2.putText(frame, f"RS: {self.inputs['rx']},{self.inputs['ry']}", (10, y), FONT, 0.35, cyan, 1)
        y += 14
        if self.inputs['square'] > BUTTON_THRESHOLD:
            cv2.putText(frame, "Square/X", (10, y), FONT, 0.35, green, 1)
        y += 14
        if self.l2_held:
            cv2.putText(frame, "L2/LT", (10, y), FONT, 0.35, yellow, 1)
        if self.r2_held:
            cv2.putText(frame, "R2/RT", (60, y), FONT, 0.35, yellow, 1)
            
        # === SHOT STATS (bottom right) ===
        rx = w - 120
        y = h - 60
        green_pct = int(self.greens_hit / self.shots_taken * 100) if self.shots_taken > 0 else 0
        cv2.putText(frame, f"Shots: {self.shots_taken}", (rx, y), FONT, 0.4, white, 1)
        y += 18
        cv2.putText(frame, f"Greens: {self.greens_hit} ({green_pct}%)", (rx, y), FONT, 0.4, green, 1)
        
        # === CALIBRATION ===
        if not net['calibrated']:
            cal_text = f"Calibrating: {net['calibration_pct']}%"
            sz = cv2.getTextSize(cal_text, FONT, 0.7, 1)[0]
            cv2.putText(frame, cal_text, ((w-sz[0])//2, h//2), FONT, 0.7, yellow, 1)
            
    # =========================================================================
    # MAIN PROCESS
    # =========================================================================
    
    def process(self, frame, *args, **kwargs):
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            
        self.last_frame = frame.copy()
        
        # FPS
        self.frame_count += 1
        now = time.time()
        if now - self.last_fps_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_fps_time = now
            
        # Read all inputs
        self._read_all_inputs()
        
        # Detect meter
        self._detect_meter(frame)
        
        # Trigger shots
        self._trigger_shots()
        
        # Update GCV
        self._update_gcv()
        
        # Send to GUI (high frequency)
        if self.server.connected and (now - self.last_status_time) > 0.1:
            self._send_status()
            self.last_status_time = now
            
        # Draw overlay
        self._draw_overlay(frame)
        
        return frame, self.gcvdata
        
    def __del__(self):
        try:
            if hasattr(self, 'shot_timer') and self.shot_timer:
                self.shot_timer.cancel()
            if hasattr(self, 'server'):
                self.server.stop()
            if hasattr(self, 'network'):
                self.network.stop()
        except:
            pass


# =============================================================================
# GTUNER INTERFACE
# =============================================================================

_WORKER = None

def get_worker():
    global _WORKER
    if _WORKER is None:
        _WORKER = GCVWorker()
    return _WORKER

def init(*args, **kwargs):
    global _WORKER
    try:
        _WORKER = GCVWorker()
        logging.info(f"WarzaVision {VERSION} initialized")
    except Exception as e:
        logging.error(f"Init error: {e}")

def process(frame, gcv_output=None, *args, **kwargs):
    try:
        worker = get_worker()
        frame, out = worker.process(frame)
        
        if gcv_output is not None:
            for i in range(min(len(out), len(gcv_output))):
                gcv_output[i] = out[i]
        elif GTUNER_AVAILABLE and gtuner.gcv_ready():
            for i, v in enumerate(out):
                gtuner.gcv_write(i, v)
                
        return frame
    except Exception as e:
        logging.error(f"Process error: {e}")
        return frame

def cleanup(*args, **kwargs):
    global _WORKER
    try:
        if _WORKER:
            del _WORKER
            _WORKER = None
        logging.info("WarzaVision cleanup complete")
    except Exception as e:
        logging.error(f"Cleanup error: {e}")
