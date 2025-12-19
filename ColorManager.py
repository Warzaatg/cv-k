import cv2
import numpy as np
import json
import os

###############################################################################
# NBA 2K26 METER COLORS - All Available Colors
# BGR format for OpenCV, RGB for display, HSV/LAB for detection
###############################################################################

# All 2K26 meter colors with their detection values
METER_COLORS_2K26 = {
    # =========================================================================
    # PRIMARY COLORS (Most Common)
    # =========================================================================
    "purple": {
        "name": "Purple / Magenta",
        "display_rgb": [186, 42, 146],    # What it looks like
        "bgr": [146, 42, 186],             # OpenCV BGR
        "lower_rgb": [170, 0, 130],
        "upper_rgb": [255, 80, 255],
        "lower_lab": [140, 210, 50],
        "upper_lab": [170, 255, 90],
        "lower_hsv": [140, 100, 100],
        "upper_hsv": [170, 255, 255],
        "tolerance": 45,
    },
    "green": {
        "name": "Green",
        "display_rgb": [0, 255, 0],
        "bgr": [0, 255, 0],
        "lower_rgb": [0, 200, 0],
        "upper_rgb": [70, 255, 70],
        "lower_lab": [145, 30, 160],
        "upper_lab": [220, 80, 220],
        "lower_hsv": [35, 100, 100],
        "upper_hsv": [85, 255, 255],
        "tolerance": 40,
    },
    "red": {
        "name": "Red",
        "display_rgb": [255, 0, 0],
        "bgr": [0, 0, 255],
        "lower_rgb": [200, 0, 0],
        "upper_rgb": [255, 50, 50],
        "lower_lab": [100, 180, 150],
        "upper_lab": [150, 220, 200],
        "lower_hsv": [0, 100, 100],
        "upper_hsv": [10, 255, 255],
        "tolerance": 40,
    },
    "yellow": {
        "name": "Yellow",
        "display_rgb": [255, 255, 0],
        "bgr": [0, 255, 255],
        "lower_rgb": [180, 180, 0],
        "upper_rgb": [255, 255, 80],
        "lower_lab": [200, 100, 170],
        "upper_lab": [255, 140, 220],
        "lower_hsv": [20, 100, 100],
        "upper_hsv": [35, 255, 255],
        "tolerance": 40,
    },
    "blue": {
        "name": "Blue",
        "display_rgb": [0, 140, 255],
        "bgr": [255, 140, 0],
        "lower_rgb": [0, 100, 200],
        "upper_rgb": [80, 180, 255],
        "lower_lab": [160, 110, 70],
        "upper_lab": [180, 130, 90],
        "lower_hsv": [100, 100, 100],
        "upper_hsv": [130, 255, 255],
        "tolerance": 40,
    },
    
    # =========================================================================
    # SECONDARY COLORS
    # =========================================================================
    "orange": {
        "name": "Orange",
        "display_rgb": [255, 165, 0],
        "bgr": [0, 165, 255],
        "lower_rgb": [200, 120, 0],
        "upper_rgb": [255, 200, 80],
        "lower_lab": [180, 140, 190],
        "upper_lab": [210, 170, 225],
        "lower_hsv": [10, 100, 100],
        "upper_hsv": [25, 255, 255],
        "tolerance": 40,
    },
    "pink": {
        "name": "Pink",
        "display_rgb": [255, 105, 180],
        "bgr": [180, 105, 255],
        "lower_rgb": [200, 80, 150],
        "upper_rgb": [255, 150, 220],
        "lower_lab": [170, 180, 120],
        "upper_lab": [200, 220, 160],
        "lower_hsv": [150, 80, 100],
        "upper_hsv": [170, 255, 255],
        "tolerance": 45,
    },
    "cyan": {
        "name": "Cyan",
        "display_rgb": [0, 255, 255],
        "bgr": [255, 255, 0],
        "lower_rgb": [0, 200, 200],
        "upper_rgb": [80, 255, 255],
        "lower_lab": [200, 70, 100],
        "upper_lab": [255, 110, 140],
        "lower_hsv": [85, 100, 100],
        "upper_hsv": [100, 255, 255],
        "tolerance": 40,
    },
    "white": {
        "name": "White",
        "display_rgb": [255, 255, 255],
        "bgr": [255, 255, 255],
        "lower_rgb": [220, 220, 220],
        "upper_rgb": [255, 255, 255],
        "lower_lab": [230, 120, 120],
        "upper_lab": [255, 140, 140],
        "lower_hsv": [0, 0, 220],
        "upper_hsv": [180, 30, 255],
        "tolerance": 30,
    },
    "black": {
        "name": "Black",
        "display_rgb": [30, 30, 30],
        "bgr": [30, 30, 30],
        "lower_rgb": [0, 0, 0],
        "upper_rgb": [50, 50, 50],
        "lower_lab": [0, 120, 120],
        "upper_lab": [50, 140, 140],
        "lower_hsv": [0, 0, 0],
        "upper_hsv": [180, 50, 50],
        "tolerance": 25,
    },
    "gray": {
        "name": "Gray",
        "display_rgb": [128, 128, 128],
        "bgr": [128, 128, 128],
        "lower_rgb": [100, 100, 100],
        "upper_rgb": [180, 180, 180],
        "lower_lab": [100, 120, 120],
        "upper_lab": [180, 140, 140],
        "lower_hsv": [0, 0, 80],
        "upper_hsv": [180, 30, 200],
        "tolerance": 35,
    },
    
    # =========================================================================
    # SPECIAL COLORS (Variations seen in-game)
    # =========================================================================
    "magenta": {
        "name": "Magenta (Bright Purple)",
        "display_rgb": [255, 0, 255],
        "bgr": [255, 0, 255],
        "lower_rgb": [200, 0, 200],
        "upper_rgb": [255, 80, 255],
        "lower_lab": [130, 200, 40],
        "upper_lab": [180, 255, 100],
        "lower_hsv": [145, 100, 100],
        "upper_hsv": [165, 255, 255],
        "tolerance": 45,
    },
    "lime": {
        "name": "Lime Green",
        "display_rgb": [50, 255, 50],
        "bgr": [50, 255, 50],
        "lower_rgb": [30, 220, 30],
        "upper_rgb": [100, 255, 100],
        "lower_lab": [200, 50, 180],
        "upper_lab": [255, 90, 230],
        "lower_hsv": [55, 100, 100],
        "upper_hsv": [75, 255, 255],
        "tolerance": 40,
    },
    "gold": {
        "name": "Gold / Amber",
        "display_rgb": [255, 215, 0],
        "bgr": [0, 215, 255],
        "lower_rgb": [200, 180, 0],
        "upper_rgb": [255, 235, 80],
        "lower_lab": [210, 100, 190],
        "upper_lab": [245, 130, 230],
        "lower_hsv": [25, 100, 100],
        "upper_hsv": [40, 255, 255],
        "tolerance": 40,
    },
    "team_color": {
        "name": "Team Color (Variable)",
        "display_rgb": [128, 128, 128],  # Placeholder - depends on team
        "bgr": [128, 128, 128],
        "lower_rgb": [0, 0, 0],
        "upper_rgb": [255, 255, 255],
        "lower_lab": [0, 0, 0],
        "upper_lab": [255, 255, 255],
        "lower_hsv": [0, 0, 0],
        "upper_hsv": [180, 255, 255],
        "tolerance": 50,
        "note": "Set custom BGR based on your team",
    },
}


class ColorManager:
    """
    ColorManager: Handles all meter color detection for WarzaVision.
    Supports RGB, HSV, and LAB color spaces.
    Includes all NBA 2K26 meter colors.
    """
    
    def __init__(self):
        self.colors = METER_COLORS_2K26.copy()
        self.selected_color = 'purple'  # Default
        self.selected_color_space = 'LAB'  # LAB is most reliable
        self.custom_bgr = None  # For custom color override
        
    def get_all_colors(self):
        """Return list of all available color names."""
        return list(self.colors.keys())
        
    def get_color_info(self, color_name=None):
        """Get full info dict for a color."""
        name = color_name or self.selected_color
        return self.colors.get(name, self.colors['purple'])
        
    def set_color(self, color_name):
        """Set the active detection color."""
        if color_name not in self.colors:
            print(f"Warning: Color '{color_name}' not found. Using purple.")
            color_name = 'purple'
        self.selected_color = color_name
        self.custom_bgr = None  # Clear custom
        return self.colors[color_name]
        
    def set_custom_bgr(self, bgr):
        """Set a custom BGR color for detection."""
        self.custom_bgr = np.array(bgr, dtype=np.uint8)
        
    def set_color_space(self, space):
        """Set color space: RGB, HSV, or LAB"""
        space = space.upper()
        if space not in ['RGB', 'HSV', 'LAB']:
            print(f"Warning: Invalid color space '{space}'. Using LAB.")
            space = 'LAB'
        self.selected_color_space = space
        
    def get_detection_bounds(self, tolerance_override=None):
        """
        Get lower and upper bounds for color detection.
        Returns: (lower_bound, upper_bound) as numpy arrays
        """
        color = self.colors.get(self.selected_color, self.colors['purple'])
        tol = tolerance_override or color.get('tolerance', 40)
        
        # If custom BGR is set, generate bounds from it
        if self.custom_bgr is not None:
            return self._generate_bounds_from_bgr(self.custom_bgr, tol)
            
        # Get bounds based on color space
        cs = self.selected_color_space
        
        if cs == 'LAB':
            lower = np.array(color.get('lower_lab', [100, 100, 100]), dtype=np.uint8)
            upper = np.array(color.get('upper_lab', [200, 200, 200]), dtype=np.uint8)
        elif cs == 'HSV':
            lower = np.array(color.get('lower_hsv', [0, 100, 100]), dtype=np.uint8)
            upper = np.array(color.get('upper_hsv', [180, 255, 255]), dtype=np.uint8)
        else:  # RGB
            lower = np.array(color.get('lower_rgb', [0, 0, 0]), dtype=np.uint8)
            upper = np.array(color.get('upper_rgb', [255, 255, 255]), dtype=np.uint8)
            
        return lower, upper
        
    def _generate_bounds_from_bgr(self, bgr, tolerance):
        """Generate detection bounds from a single BGR color + tolerance."""
        if self.selected_color_space == 'LAB':
            # Convert BGR to LAB
            lab = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0][0]
            lower = np.array([max(0, c - tolerance) for c in lab], dtype=np.uint8)
            upper = np.array([min(255, c + tolerance) for c in lab], dtype=np.uint8)
        elif self.selected_color_space == 'HSV':
            # Convert BGR to HSV
            hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
            lower = np.array([max(0, hsv[0] - tolerance//2), max(0, hsv[1] - tolerance), max(0, hsv[2] - tolerance)], dtype=np.uint8)
            upper = np.array([min(180, hsv[0] + tolerance//2), min(255, hsv[1] + tolerance), min(255, hsv[2] + tolerance)], dtype=np.uint8)
        else:  # RGB (actually BGR in OpenCV)
            lower = np.array([max(0, c - tolerance) for c in bgr], dtype=np.uint8)
            upper = np.array([min(255, c + tolerance) for c in bgr], dtype=np.uint8)
            
        return lower, upper
        
    def get_bgr(self):
        """Get the BGR value for the current color."""
        if self.custom_bgr is not None:
            return self.custom_bgr
        color = self.colors.get(self.selected_color, self.colors['purple'])
        return np.array(color.get('bgr', [255, 0, 255]), dtype=np.uint8)
        
    def get_rgb(self):
        """Get RGB value for display."""
        color = self.colors.get(self.selected_color, self.colors['purple'])
        return color.get('display_rgb', [255, 0, 255])
        
    def create_mask(self, frame, tolerance_override=None):
        """
        Create a detection mask for the current color in the frame.
        Returns: binary mask where white = detected color
        """
        lower, upper = self.get_detection_bounds(tolerance_override)
        
        if self.selected_color_space == 'LAB':
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        elif self.selected_color_space == 'HSV':
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        else:  # RGB (frame is already BGR)
            converted = frame
            
        mask = cv2.inRange(converted, lower, upper)
        return mask
        
    def detect_color_in_region(self, frame, region, min_pixels=100):
        """
        Detect if the current color is present in a region.
        region: (x, y, w, h)
        Returns: (found, pixel_count, fill_percent)
        """
        x, y, w, h = region
        H, W = frame.shape[:2]
        
        # Clamp to frame
        x, y = max(0, x), max(0, y)
        w = min(w, W - x)
        h = min(h, H - y)
        
        roi = frame[y:y+h, x:x+w]
        mask = self.create_mask(roi)
        
        pixel_count = cv2.countNonZero(mask)
        total_pixels = w * h
        fill_pct = (pixel_count / total_pixels * 100) if total_pixels > 0 else 0
        
        return pixel_count >= min_pixels, pixel_count, fill_pct
        
    def to_dict(self):
        """Export current settings to dict."""
        return {
            'selected_color': self.selected_color,
            'selected_color_space': self.selected_color_space,
            'custom_bgr': self.custom_bgr.tolist() if self.custom_bgr is not None else None,
        }
        
    def from_dict(self, data):
        """Load settings from dict."""
        self.selected_color = data.get('selected_color', 'purple')
        self.selected_color_space = data.get('selected_color_space', 'LAB')
        custom = data.get('custom_bgr')
        self.custom_bgr = np.array(custom, dtype=np.uint8) if custom else None
        
    @staticmethod
    def list_all_colors():
        """Print all available colors."""
        print("\n=== NBA 2K26 Meter Colors ===")
        for key, data in METER_COLORS_2K26.items():
            rgb = data.get('display_rgb', [0,0,0])
            print(f"  {key:15} - {data['name']:25} RGB({rgb[0]:3},{rgb[1]:3},{rgb[2]:3})")
        print()


# ============================================================================
# LEGACY SUPPORT - get_bounds function for backward compatibility
# ============================================================================

_color_manager = ColorManager()

def get_bounds():
    """Legacy function - returns current color bounds."""
    return _color_manager.get_detection_bounds()
    
def set_meter_color(color_name):
    """Set the meter color for detection."""
    return _color_manager.set_color(color_name)
    
def get_meter_colors():
    """Get list of all available meter colors."""
    return _color_manager.get_all_colors()


if __name__ == "__main__":
    ColorManager.list_all_colors()
