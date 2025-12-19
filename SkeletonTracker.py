"""
WarzaVision Pro - Skeleton Tracker Module
MediaPipe Pose Estimation for Player Movement Detection

Features:
- Full body skeleton tracking (33 keypoints)
- Shooting form detection (arm angles, release point)
- Fadeaway/stepback motion detection
- No-meter mode support via animation tracking
"""

import cv2
import numpy as np
import time
import logging
from collections import deque
from enum import Enum

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipe not installed. Run: pip install mediapipe")

# ============================================================================
# POSE KEYPOINTS (MediaPipe 33 landmarks)
# ============================================================================

class PoseLandmark(Enum):
    """MediaPipe pose landmark indices"""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

# Shooting-relevant keypoints
SHOOTING_ARM_RIGHT = [PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST]
SHOOTING_ARM_LEFT = [PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST]


# ============================================================================
# TIMING CUES - The 4 selectable visual cues in 2K26
# ============================================================================

class TimingCue(Enum):
    """
    NBA 2K26 Visual Timing Cues
    These determine WHEN to release based on the shooting animation.
    """
    JUMP = "jump"           # Release at jump start (earliest)
    SET_POINT = "set_point" # Release at set point (ball at shooting pocket)
    PUSH = "push"           # Release at push (hand flicking forward)
    RELEASE = "release"     # Release at ball release (latest)

# Timing cue order (earliest to latest)
TIMING_CUE_ORDER = [TimingCue.JUMP, TimingCue.SET_POINT, TimingCue.PUSH, TimingCue.RELEASE]

# Approximate timing offsets in milliseconds relative to RELEASE
# Negative = earlier, these are approximate and vary by jumpshot
TIMING_CUE_OFFSETS_MS = {
    TimingCue.JUMP: -200,       # ~200ms before release
    TimingCue.SET_POINT: -120,  # ~120ms before release
    TimingCue.PUSH: -60,        # ~60ms before release
    TimingCue.RELEASE: 0,       # At release point
}


# ============================================================================
# SHOOTING PHASES
# ============================================================================

class ShootingPhase(Enum):
    """Phases of a jumpshot animation - maps to timing cues"""
    IDLE = "idle"              # Not shooting
    STANCE = "stance"          # Setting up
    CROUCH = "crouch"          # Loading/dipping (DIP phase)
    JUMP = "jump"              # Leaving ground - TIMING CUE 1
    SET_POINT = "set_point"    # Ball at set point
    PUSH = "push"              # Pushing ball up
    RELEASE = "release"        # Ball release
    FOLLOW_THROUGH = "follow"  # After release
    LANDING = "landing"        # Coming down


# ============================================================================
# SKELETON TRACKER
# ============================================================================

class SkeletonTracker:
    """
    Tracks player skeleton using MediaPipe Pose.
    Detects shooting form, motion type, and optimal release timing.
    Supports all 4 timing cues: Jump, Set Point, Push, Release
    """
    
    def __init__(self):
        self.enabled = MEDIAPIPE_AVAILABLE
        self.pose = None
        self.mp_pose = None
        self.mp_drawing = None
        
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,  # 0=fast, 1=balanced, 2=accurate
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        
        # Current pose data
        self.landmarks = None
        self.landmark_visibility = {}
        self.pose_detected = False
        
        # Shooting analysis
        self.shooting_phase = ShootingPhase.IDLE
        self.shooting_arm = "right"  # or "left"
        self.elbow_angle = 0
        self.wrist_height = 0
        self.shoulder_to_wrist_angle = 0
        
        # Timing Cue Configuration
        self.timing_cue = TimingCue.PUSH  # Default timing cue
        self.timing_cue_reached = {cue: False for cue in TimingCue}
        
        # No-dip detection
        self.is_no_dip = False  # True if catch-and-shoot without dip
        self.dip_detected = False
        self.min_wrist_y = 0  # Track lowest wrist position
        
        # Motion tracking
        self.position_history = deque(maxlen=30)  # Last 30 frames
        self.velocity = np.array([0.0, 0.0])
        self.vertical_velocity = 0.0  # For jump detection
        self.motion_type = "stationary"  # stationary, moving_left, moving_right, jumping, fading
        
        # Ground detection for jump timing
        self.ground_y = 0  # Ankle Y when on ground
        self.is_airborne = False
        self.jump_start_time = 0
        
        # No-meter timing
        self.shot_start_time = 0
        self.shot_phase_times = {}
        self.release_predicted = False
        self.optimal_release_time = 0
        
        # Ping compensation
        self.ping_compensation_ms = 0
        
        # Calibration
        self.calibrated = False
        self.player_height_px = 0
        self.baseline_positions = {}
        
        # Cache to prevent spam logging
        self._last_timing_cue = None
        self._last_ping_compensation = 0
        
    def set_timing_cue(self, cue):
        """Set which timing cue to use for release detection. Only logs on change."""
        if isinstance(cue, str):
            try:
                cue = TimingCue(cue)
            except ValueError:
                cue = TimingCue.PUSH  # Default fallback
        
        # Only update and log if actually changed
        if cue != self._last_timing_cue:
            self.timing_cue = cue
            self._last_timing_cue = cue
            logging.info(f"Timing cue changed to: {cue.value}")
        
    def set_ping_compensation(self, ping_ms):
        """
        Set ping compensation.
        Higher ping = fire EARLIER to compensate for network delay.
        Only logs on significant change.
        """
        new_comp = ping_ms / 2
        # Only log if changed by more than 5ms
        if abs(new_comp - self._last_ping_compensation) > 5:
            self.ping_compensation_ms = new_comp
            self._last_ping_compensation = new_comp
            logging.debug(f"Ping compensation adjusted: {self.ping_compensation_ms:.0f}ms")
        else:
            self.ping_compensation_ms = new_comp
        
    def process(self, frame, ping_ms=0):
        """
        Process a frame and extract pose data.
        Returns: dict with pose info
        """
        if not self.enabled or self.pose is None:
            return self._empty_result()
        
        # Update ping compensation
        if ping_ms > 0:
            self.set_ping_compensation(ping_ms)
            
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process pose
        results = self.pose.process(rgb)
        
        if not results.pose_landmarks:
            self.pose_detected = False
            return self._empty_result()
            
        self.pose_detected = True
        self.landmarks = results.pose_landmarks
        
        # Extract landmark positions
        h, w = frame.shape[:2]
        positions = {}
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            positions[idx] = {
                'x': lm.x * w,
                'y': lm.y * h,
                'z': lm.z,
                'visibility': lm.visibility
            }
        self.landmark_visibility = positions
        
        # Track center of mass position
        hip_center = self._get_hip_center(positions)
        self.position_history.append(hip_center)
        
        # Detect jump (for JUMP timing cue)
        self._detect_jump(positions)
        
        # Detect no-dip (catch and shoot)
        self._detect_no_dip(positions)
        
        # Calculate motion
        self._analyze_motion()
        
        # Analyze shooting form
        self._analyze_shooting_form(positions)
        
        # Detect shooting phase
        self._detect_shooting_phase(positions)
        
        # Check if selected timing cue is reached
        should_fire = self._check_timing_cue()
        
        return {
            'pose_detected': True,
            'landmarks': positions,
            'shooting_phase': self.shooting_phase.value,
            'timing_cue': self.timing_cue.value,
            'timing_cue_reached': self.timing_cue_reached,
            'should_fire': should_fire,
            'is_no_dip': self.is_no_dip,
            'is_airborne': self.is_airborne,
            'motion_type': self.motion_type,
            'elbow_angle': round(self.elbow_angle, 1),
            'wrist_height': round(self.wrist_height, 1),
            'velocity': self.velocity.tolist(),
            'vertical_velocity': round(self.vertical_velocity, 2),
            'release_predicted': self.release_predicted,
            'optimal_release_time': self.optimal_release_time,
            'ping_compensation_ms': self.ping_compensation_ms,
        }
        
    def _empty_result(self):
        return {
            'pose_detected': False,
            'landmarks': {},
            'shooting_phase': ShootingPhase.IDLE.value,
            'timing_cue': self.timing_cue.value,
            'timing_cue_reached': {cue.value: False for cue in TimingCue},
            'should_fire': False,
            'is_no_dip': False,
            'is_airborne': False,
            'motion_type': 'unknown',
            'elbow_angle': 0,
            'wrist_height': 0,
            'velocity': [0, 0],
            'vertical_velocity': 0,
            'release_predicted': False,
            'optimal_release_time': 0,
            'ping_compensation_ms': 0,
        }
    
    def _detect_jump(self, positions):
        """Detect when player leaves ground (for JUMP timing cue)"""
        ankle = positions.get(PoseLandmark.RIGHT_ANKLE.value, {})
        ankle_y = ankle.get('y', 0)
        
        # Track ground level when stationary
        if self.shooting_phase == ShootingPhase.IDLE and ankle_y > 0:
            self.ground_y = ankle_y
            
        # Detect airborne (ankle significantly above ground)
        if self.ground_y > 0:
            lift = self.ground_y - ankle_y
            was_airborne = self.is_airborne
            self.is_airborne = lift > 20  # 20 pixels above ground
            
            # Just jumped
            if self.is_airborne and not was_airborne:
                self.jump_start_time = time.time()
                self.timing_cue_reached[TimingCue.JUMP] = True
                logging.debug("JUMP detected!")
                
        # Track vertical velocity
        if len(self.position_history) >= 2:
            recent = list(self.position_history)[-2:]
            self.vertical_velocity = recent[0][1] - recent[-1][1]  # Positive = moving up
            
    def _detect_no_dip(self, positions):
        """
        Detect no-dip catch-and-shoot.
        In a normal shot, wrist dips down before coming up.
        In no-dip, ball goes straight up from catch.
        """
        wrist = positions.get(PoseLandmark.RIGHT_WRIST.value, {})
        shoulder = positions.get(PoseLandmark.RIGHT_SHOULDER.value, {})
        wrist_y = wrist.get('y', 0)
        shoulder_y = shoulder.get('y', 0)
        
        if self.shooting_phase == ShootingPhase.IDLE:
            # Reset for new shot
            self.dip_detected = False
            self.min_wrist_y = wrist_y
            self.is_no_dip = False
            
        elif self.shooting_phase in [ShootingPhase.CROUCH, ShootingPhase.STANCE]:
            # Track lowest wrist position
            if wrist_y > self.min_wrist_y:  # Lower on screen = higher Y
                self.min_wrist_y = wrist_y
                
            # Check if wrist dipped below shoulder
            if wrist_y > shoulder_y + 30:
                self.dip_detected = True
                
        elif self.shooting_phase == ShootingPhase.SET_POINT:
            # By set point, determine if it was no-dip
            if not self.dip_detected:
                self.is_no_dip = True
                logging.debug("No-dip shot detected!")
                
    def _check_timing_cue(self):
        """
        Check if selected timing cue has been reached.
        Returns True when it's time to fire.
        """
        # Map shooting phases to timing cues
        phase_to_cue = {
            ShootingPhase.JUMP: TimingCue.JUMP,
            ShootingPhase.SET_POINT: TimingCue.SET_POINT,
            ShootingPhase.PUSH: TimingCue.PUSH,
            ShootingPhase.RELEASE: TimingCue.RELEASE,
        }
        
        # Update which cues have been reached
        if self.shooting_phase in phase_to_cue:
            reached_cue = phase_to_cue[self.shooting_phase]
            self.timing_cue_reached[reached_cue] = True
            
            # Also mark earlier cues as reached
            for cue in TIMING_CUE_ORDER:
                if cue == reached_cue:
                    break
                self.timing_cue_reached[cue] = True
        
        # Check if our selected cue is reached
        should_fire = self.timing_cue_reached.get(self.timing_cue, False)
        
        # Apply ping compensation - fire earlier if high ping
        # This is handled by firing at an earlier cue
        if self.ping_compensation_ms > 30:
            # High ping - consider firing one cue earlier
            current_idx = TIMING_CUE_ORDER.index(self.timing_cue) if self.timing_cue in TIMING_CUE_ORDER else 2
            if current_idx > 0 and self.ping_compensation_ms > 50:
                # Fire at previous cue
                earlier_cue = TIMING_CUE_ORDER[current_idx - 1]
                should_fire = self.timing_cue_reached.get(earlier_cue, False)
                
        return should_fire
        
    def _get_hip_center(self, positions):
        """Get center of hips (body center)"""
        left_hip = positions.get(PoseLandmark.LEFT_HIP.value, {})
        right_hip = positions.get(PoseLandmark.RIGHT_HIP.value, {})
        
        if left_hip and right_hip:
            return np.array([
                (left_hip.get('x', 0) + right_hip.get('x', 0)) / 2,
                (left_hip.get('y', 0) + right_hip.get('y', 0)) / 2
            ])
        return np.array([0.0, 0.0])
        
    def _analyze_motion(self):
        """Analyze player motion from position history"""
        if len(self.position_history) < 5:
            self.motion_type = "stationary"
            self.velocity = np.array([0.0, 0.0])
            return
            
        # Calculate velocity from last few frames
        recent = list(self.position_history)[-5:]
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        self.velocity = np.array([dx, dy]) / 5  # Per frame
        
        # Determine motion type
        speed = np.linalg.norm(self.velocity)
        
        if speed < 2:
            self.motion_type = "stationary"
        elif dy < -5:  # Moving up significantly (remember Y is inverted)
            self.motion_type = "jumping"
        elif abs(dx) > 5:
            if dx > 0:
                self.motion_type = "moving_right"
            else:
                self.motion_type = "moving_left"
            # Check if also fading (moving away while shooting)
            if abs(dy) > 3:
                self.motion_type = "fading"
        else:
            self.motion_type = "stationary"
            
    def _analyze_shooting_form(self, positions):
        """Analyze shooting arm form"""
        # Get arm keypoints based on shooting hand
        if self.shooting_arm == "right":
            shoulder = positions.get(PoseLandmark.RIGHT_SHOULDER.value, {})
            elbow = positions.get(PoseLandmark.RIGHT_ELBOW.value, {})
            wrist = positions.get(PoseLandmark.RIGHT_WRIST.value, {})
        else:
            shoulder = positions.get(PoseLandmark.LEFT_SHOULDER.value, {})
            elbow = positions.get(PoseLandmark.LEFT_ELBOW.value, {})
            wrist = positions.get(PoseLandmark.LEFT_WRIST.value, {})
            
        if not all([shoulder, elbow, wrist]):
            return
            
        # Calculate elbow angle
        self.elbow_angle = self._calculate_angle(
            [shoulder.get('x', 0), shoulder.get('y', 0)],
            [elbow.get('x', 0), elbow.get('y', 0)],
            [wrist.get('x', 0), wrist.get('y', 0)]
        )
        
        # Track wrist height relative to shoulder
        self.wrist_height = shoulder.get('y', 0) - wrist.get('y', 0)  # Positive = above shoulder
        
        # Calculate shoulder-to-wrist angle (arm extension)
        self.shoulder_to_wrist_angle = self._calculate_vertical_angle(
            [shoulder.get('x', 0), shoulder.get('y', 0)],
            [wrist.get('x', 0), wrist.get('y', 0)]
        )
        
    def _calculate_angle(self, p1, p2, p3):
        """Calculate angle at p2 between p1-p2-p3"""
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        
        ba = a - b
        bc = c - b
        
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        
        return np.degrees(angle)
        
    def _calculate_vertical_angle(self, p1, p2):
        """Calculate angle from vertical"""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return np.degrees(np.arctan2(dx, -dy))  # 0 = straight up
        
    def _detect_shooting_phase(self, positions):
        """
        Detect current phase of shooting animation.
        Maps to timing cues: JUMP -> SET_POINT -> PUSH -> RELEASE
        Critical for no-meter timing!
        """
        # Get key positions
        shoulder = positions.get(PoseLandmark.RIGHT_SHOULDER.value, {})
        wrist = positions.get(PoseLandmark.RIGHT_WRIST.value, {})
        hip = positions.get(PoseLandmark.RIGHT_HIP.value, {})
        knee = positions.get(PoseLandmark.RIGHT_KNEE.value, {})
        
        if not all([shoulder, wrist, hip, knee]):
            return
            
        now = time.time()
        
        # Phase detection logic based on body position
        wrist_y = wrist.get('y', 0)
        shoulder_y = shoulder.get('y', 0)
        hip_y = hip.get('y', 0)
        
        # Wrist above shoulder = shooting motion
        wrist_above_shoulder = wrist_y < shoulder_y - 20
        
        # Wrist at or above head level
        nose = positions.get(PoseLandmark.NOSE.value, {})
        nose_y = nose.get('y', 0) if nose else shoulder_y - 50
        wrist_at_head = wrist_y < nose_y
        
        # Arm nearly extended (elbow angle > 150)
        arm_extended = self.elbow_angle > 150
        
        # Knee bent (crouch)
        knee_bent = self._is_knee_bent(positions)
        
        # State machine
        prev_phase = self.shooting_phase
        
        if not wrist_above_shoulder:
            # Arm low - either idle or crouch
            if knee_bent:
                self.shooting_phase = ShootingPhase.CROUCH
            else:
                self.shooting_phase = ShootingPhase.IDLE
                self.shot_start_time = 0
                # Reset timing cues for new shot
                self.timing_cue_reached = {cue: False for cue in TimingCue}
                self.is_airborne = False
                self.is_no_dip = False
                
        elif self.is_airborne and not wrist_above_shoulder:
            # Just left ground - JUMP timing cue
            self.shooting_phase = ShootingPhase.JUMP
            self.timing_cue_reached[TimingCue.JUMP] = True
            if prev_phase not in [ShootingPhase.JUMP, ShootingPhase.SET_POINT, ShootingPhase.PUSH, ShootingPhase.RELEASE]:
                self.shot_start_time = now
                
        elif wrist_above_shoulder and not wrist_at_head:
            # Arm raising - set point - TIMING CUE 2
            self.shooting_phase = ShootingPhase.SET_POINT
            self.timing_cue_reached[TimingCue.SET_POINT] = True
            self.shot_phase_times['set_point'] = now
            if prev_phase == ShootingPhase.CROUCH or prev_phase == ShootingPhase.JUMP:
                if self.shot_start_time == 0:
                    self.shot_start_time = now
                
        elif wrist_at_head and not arm_extended:
            # Ball at head, arm cocked - PUSH timing cue - TIMING CUE 3
            self.shooting_phase = ShootingPhase.PUSH
            self.timing_cue_reached[TimingCue.PUSH] = True
            self.shot_phase_times['push'] = now
            
        elif wrist_at_head and arm_extended:
            # Arm extended - RELEASE timing cue - TIMING CUE 4
            self.shooting_phase = ShootingPhase.RELEASE
            self.timing_cue_reached[TimingCue.RELEASE] = True
            self.release_predicted = True
            self.optimal_release_time = now
            self.shot_phase_times['release'] = now
            
        # Log phase transitions
        if prev_phase != self.shooting_phase:
            logging.debug(f"Shot phase: {prev_phase.value} -> {self.shooting_phase.value}")
            
    def _is_knee_bent(self, positions):
        """Check if knees are bent (crouching)"""
        hip = positions.get(PoseLandmark.RIGHT_HIP.value, {})
        knee = positions.get(PoseLandmark.RIGHT_KNEE.value, {})
        ankle = positions.get(PoseLandmark.RIGHT_ANKLE.value, {})
        
        if not all([hip, knee, ankle]):
            return False
            
        knee_angle = self._calculate_angle(
            [hip.get('x', 0), hip.get('y', 0)],
            [knee.get('x', 0), knee.get('y', 0)],
            [ankle.get('x', 0), ankle.get('y', 0)]
        )
        
        return knee_angle < 160  # Bent if less than 160 degrees
        
    def draw_skeleton(self, frame):
        """Draw skeleton overlay on frame"""
        if not self.enabled or not self.pose_detected or self.landmarks is None:
            return frame
            
        # Draw landmarks and connections
        self.mp_drawing.draw_landmarks(
            frame,
            self.landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2)
        )
        
        return frame
        
    def draw_shooting_analysis(self, frame):
        """Draw shooting form analysis overlay"""
        if not self.pose_detected:
            return frame
            
        h, w = frame.shape[:2]
        
        # Draw phase indicator
        phase_colors = {
            ShootingPhase.IDLE: (128, 128, 128),
            ShootingPhase.STANCE: (255, 255, 0),
            ShootingPhase.CROUCH: (255, 165, 0),
            ShootingPhase.JUMP: (0, 255, 255),
            ShootingPhase.SET_POINT: (255, 0, 255),
            ShootingPhase.PUSH: (0, 255, 0),
            ShootingPhase.RELEASE: (0, 255, 0),
            ShootingPhase.FOLLOW_THROUGH: (0, 200, 0),
            ShootingPhase.LANDING: (100, 100, 100),
        }
        
        color = phase_colors.get(self.shooting_phase, (255, 255, 255))
        
        # Phase text
        cv2.putText(frame, f"Phase: {self.shooting_phase.value.upper()}", 
                    (10, h - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Elbow angle
        cv2.putText(frame, f"Elbow: {self.elbow_angle:.0f} deg", 
                    (10, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Wrist height
        cv2.putText(frame, f"Wrist Height: {self.wrist_height:.0f}px", 
                    (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Motion type
        cv2.putText(frame, f"Motion: {self.motion_type}", 
                    (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # Release indicator
        if self.shooting_phase == ShootingPhase.RELEASE:
            cv2.putText(frame, "RELEASE NOW!", 
                        (w//2 - 100, h//2), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 3)
        elif self.shooting_phase == ShootingPhase.PUSH:
            cv2.putText(frame, "READY...", 
                        (w//2 - 60, h//2), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 0), 2)
        
        return frame
        
    def get_shot_type(self):
        """Determine shot type based on motion"""
        if self.motion_type == "fading":
            return "fadeaway"
        elif self.motion_type in ["moving_left", "moving_right"]:
            # Could be stepback or pullup
            if abs(self.velocity[0]) > 8:
                return "stepback"
            else:
                return "pullup"
        elif self.motion_type == "jumping":
            return "normal"
        else:
            return "normal"
            
    def should_release(self):
        """Returns True when optimal release point detected (for no-meter mode)"""
        return self.shooting_phase == ShootingPhase.RELEASE
        
    def close(self):
        """Cleanup"""
        if self.pose:
            self.pose.close()


# ============================================================================
# SHOT ANALYZER - Combines skeleton + meter for comprehensive analysis
# ============================================================================

class ShotAnalyzer:
    """
    Combines skeleton tracking with meter detection for comprehensive shot analysis.
    Supports both meter and no-meter modes.
    """
    
    def __init__(self, skeleton_tracker=None):
        self.skeleton = skeleton_tracker or SkeletonTracker()
        self.mode = "meter"  # "meter" or "no_meter"
        
        # Shot tracking
        self.shots_attempted = 0
        self.shots_made = 0
        self.shot_log = deque(maxlen=100)
        
        # Timing calibration (for no-meter)
        self.release_timing_samples = deque(maxlen=20)
        self.avg_release_time_ms = 0
        
    def process(self, frame, meter_info=None):
        """
        Process frame with both skeleton and meter data.
        
        Args:
            frame: Video frame
            meter_info: Dict from meter detector (optional)
            
        Returns:
            Combined analysis dict
        """
        # Get skeleton data
        skeleton_data = self.skeleton.process(frame)
        
        # Determine shot type from motion
        shot_type = self.skeleton.get_shot_type()
        
        # Combine with meter data if available
        if self.mode == "meter" and meter_info:
            return self._analyze_with_meter(skeleton_data, meter_info, shot_type)
        else:
            return self._analyze_no_meter(skeleton_data, shot_type)
            
    def _analyze_with_meter(self, skeleton_data, meter_info, shot_type):
        """Analysis when using shot meter"""
        # Skeleton provides shot type, meter provides timing
        should_fire = meter_info.get('in_green_zone', False)
        
        return {
            'mode': 'meter',
            'shot_type': shot_type,
            'skeleton': skeleton_data,
            'meter': meter_info,
            'should_fire': should_fire,
            'confidence': 0.9 if meter_info.get('meter_found') else 0.5,
        }
        
    def _analyze_no_meter(self, skeleton_data, shot_type):
        """Analysis for no-meter mode using skeleton only"""
        should_fire = self.skeleton.should_release()
        
        return {
            'mode': 'no_meter',
            'shot_type': shot_type,
            'skeleton': skeleton_data,
            'shooting_phase': skeleton_data.get('shooting_phase'),
            'should_fire': should_fire,
            'confidence': 0.7,  # Lower confidence without meter
        }
        
    def draw_overlay(self, frame, show_skeleton=True, show_analysis=True):
        """Draw all overlays"""
        if show_skeleton:
            frame = self.skeleton.draw_skeleton(frame)
        if show_analysis:
            frame = self.skeleton.draw_shooting_analysis(frame)
        return frame
        
    def set_mode(self, mode):
        """Set analysis mode: 'meter' or 'no_meter'"""
        self.mode = mode
        
    def close(self):
        self.skeleton.close()


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("Skeleton Tracker Test")
    print("="*50)
    
    if not MEDIAPIPE_AVAILABLE:
        print("MediaPipe not installed!")
        print("Install with: pip install mediapipe")
        exit(1)
        
    # Test with webcam
    tracker = SkeletonTracker()
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process
        result = tracker.process(frame)
        
        # Draw
        frame = tracker.draw_skeleton(frame)
        frame = tracker.draw_shooting_analysis(frame)
        
        # Show
        cv2.imshow('Skeleton Tracker Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
