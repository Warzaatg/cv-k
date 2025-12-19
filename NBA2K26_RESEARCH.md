# NBA 2K26 Shot Meter Research - WarzaVision Pro 7.0

## Overview
This document contains comprehensive research on NBA 2K26 shooting mechanics, shot meters, fadeaways, and game changes for integration into the WarzaVision computer vision tool.

---

## Shot Meter Styles (All Confirmed)

| Style ID | Display Name | Shape Description | Fill Animation |
|----------|--------------|-------------------|----------------|
| arrow_2 | Arrow 2 | Standard arrow, pointed upward | Bottom to top |
| comet | Comet | Head with trailing tail | Bottom to top |
| tusk | Tusk | Curved tusk/horn shape | Bottom to top |
| fang | Fang | Sharp angular fang | Bottom to top |
| rainbow | Rainbow | Curved arc/semi-circle | Arc fill |
| horn | Horn | Thick curved horn | Bottom to top |
| pill | Pill | Elongated oval/capsule | Bottom to top |
| waves | Waves | Undulating wave pattern | Wave fill |
| diamond | Diamond | Geometric diamond | Bottom to top |
| hexagons | Hexagons | Hex cluster pattern | Segment fill |
| tube | Tube | Cylindrical shape | Bottom to top |
| triangles | Triangles | Triangle pattern | Bottom to top |
| shooting_star | Shooting Star | Star with trail | Bottom to top |
| lightning_bolt | Lightning Bolt | Jagged zigzag | Bottom to top |
| fireball | Fireball | Circular flame | Radial fill |
| sword | Sword | Blade shape | Bottom to top |
| curve | Curve | Curved arc (default) | Arc fill |

---

## Shot Meter Colors

### Primary Purple (2K26 Default)
- **RGB**: (186, 42, 146)
- **BGR (OpenCV)**: [146, 42, 186]
- **Hex**: #BA2A92
- **Description**: Vibrant purple with magenta tint

### Color Zones
| Zone | Color | Meaning |
|------|-------|---------|
| Purple | #BA2A92 | Standard timing zone |
| Green | #00FF00 | Optimal release (GREEN WINDOW) |
| Red | #FF0000 | Hot zone indicator |
| Yellow | #FFD000 | Transitional zone |

---

## Shot Meter Position (From Video Analysis)

### "To The Side" Placement (Default)
- **Left Edge**: ~730px from left
- **Right Edge**: ~920px from left  
- **Top Edge**: ~770px from top
- **Bottom Edge**: ~840px from top
- **Width**: ~190px
- **Height**: ~20-30px
- **Aspect Ratio**: ~7:1 to 10:1 (width:height)

### Fill Direction
- Fills from **bottom to top** (or left to right for horizontal meters)
- Green zone is at the **top/end** of the meter

---

## Green Window System (2K26 Mechanics)

### Key Points
1. **Pure Green Window** - NO RNG! Green release = guaranteed make
2. **Meter OFF = Bigger Window** - 48% → 57% boost when meter is disabled
3. **Rhythm Shooting** - Tempo-based timing (stick flick speed matters)
4. **Shot Contest** - Affects green window size dynamically

### Green Zone Threshold
- Approximately top 10% of meter
- Appears at ~90% fill level
- Visual indicator changes when entering green zone

---

## Fadeaway & Specialty Shots

### Shot Types with Timing Adjustments

| Shot Type | Green Window Modifier | Timing Offset (ms) | Detection Cue |
|-----------|----------------------|-------------------|---------------|
| Normal (Standing) | 1.00 (100%) | 0 | Stationary |
| Leaner Fadeaway | 0.85 (85%) | -5 | Lean away motion |
| Step-Back Fadeaway | 0.80 (80%) | -8 | Step-back motion |
| Sombor Shuffle | 0.75 (75%) | -10 | Crossover → step-back |
| Shimmy Fadeaway | 0.80 (80%) | -7 | Side-to-side shake |
| Post Fadeaway | 0.85 (85%) | -5 | Post position |
| Floater | 0.90 (90%) | -3 | Floater arc |

### Fadeaway Mechanics
1. **Leaner**: R2 + lean away + shoot at peak
2. **Step-Back**: Step-back dribble + R2 + shoot at peak
3. **Sombor Shuffle**: Crossover → step-back → shoot as move finishes
4. **Shimmy**: Quick side-to-side + R2 + shoot at peak

---

## 2K26 Game Changes Since Release

### Major Updates

#### Core Shooting Overhaul
- **New Curved Shot Meter** - Replaced dial/arrow/ring system
- **Rhythm Shooting** - Tempo-based timing (stick flick speed matters)
- **ProPLAY Motion Engine** - Enhanced character movement

#### Patch 2.1 (October 17, 2025)
- Improved Rhythm Shooting input reliability
- Slightly tightened rhythm shooting windows
- Increased fadeaway difficulty
- Fixed ghost contest issues
- Updated shot meter for Poster Machine takeover

#### Key Features
- **Signature Go-To Post Shots** - Sombor Shuffle, shimmy fadeaway
- **Mix & Match Layup Styles** - Not locked to packages
- **Rebound Timing Meter** - Green meter for rebounds
- **Quick Protect** - New defensive move
- **No-Dip Catch-and-Shoots** - Faster shooting option

---

## Computer Vision Detection Strategy

### Color Detection (LAB Space Recommended)
```python
# 2K26 Purple in LAB space
target_bgr = [146, 42, 186]
tolerance = 45

# Convert to LAB
lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
target_lab = cv2.cvtColor(np.uint8([[target_bgr]]), cv2.COLOR_BGR2LAB)[0][0]

# Create mask with tolerance
lower = np.array([max(0, c-tolerance) for c in target_lab])
upper = np.array([min(255, c+tolerance) for c in target_lab])
mask = cv2.inRange(lab, lower, upper)
```

### Green Zone Detection
1. Track fill percentage continuously
2. When fill_pct > 80%, start monitoring green zone proximity
3. Green zone starts at ~90% fill
4. Trigger alert/release when in green zone

### Shot Type Detection (Future Enhancement)
- Use motion analysis to detect player movement
- Track velocity and direction changes
- Identify fadeaway patterns based on motion history

---

## Configuration Recommendations

### Default Settings for 2K26
```json
{
    "meter_color": "purple",
    "meter_style": "curve",
    "user_meter_color_bgr": [146, 42, 186],
    "color_distance_threshold": 45,
    "meter_region": {"x": 730, "y": 770, "w": 190, "h": 70},
    "green_zone_alert": true,
    "fadeaway_timing_adjust": true,
    "detect_shot_type": true
}
```

### Meter Region Adjustments
- Position may vary based on:
  - Meter placement setting (above head, side, rim)
  - Screen resolution
  - Game mode (MyCAREER vs Park vs Rec)

---

## References
- Reddit: r/NBA2k shot meter discussions
- NBA 2K26 Official Courtside Report
- NBA2KLab settings guides
- Patch notes from nba.2k.com

---

*Last Updated: November 2025*
*WarzaVision Pro 7.0 - NBA 2K26 Edition*
