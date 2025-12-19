================================================================================
                    WarzaVision Pro 4.0 - Ultimate Edition
                    Real-time NBA 2K CV Engine with GUI
================================================================================

QUICK START
-----------
1. Run INSTALL.bat (one-time setup)
2. Start GTuner with Start_GTuner.bat
3. Load Warzatools2K.py as your GCV script
4. GUI will auto-launch and connect!


FEATURES
--------
* Real-time GUI with live sync (no restart needed)
* Connection monitoring and latency compensation
* Icon Shooting bonuses based on 3PT attribute
* No-Meter Mode for muscle memory shooters
* Roboflow AI detection (94.6% accuracy)
* Color-based meter detection with calibration


GUI CONTROLS
------------
- All settings apply INSTANTLY - no restart needed
- Manual port connection if auto-connect fails
- Live network stats: Ping, Jitter, Loss, Compensation


FILES
-----
* Warzatools2K.py   - Main CV script (load in GTuner)
* WarzaGUI_Qt.py    - PyQt6 GUI application
* config.json       - All your settings (auto-saved)
* Launch_GUI.bat    - Manual GUI launcher
* Start_GTuner.bat  - Proper GTuner launcher
* INSTALL.bat       - Environment setup


TROUBLESHOOTING
---------------
GUI doesn't open:
  1. Run Launch_GUI.bat manually
  2. Check gui_error.log for errors
  3. Make sure PyQt6 is installed: pip install PyQt6

GUI won't connect:
  1. Check the port number shown in GTuner console
  2. Enter that port in the GUI sidebar and click Connect
  3. Default port is 59420

Network stats not showing:
  - The CV script pings 2K servers every 2 seconds
  - Wait a few seconds for stats to populate
  - Make sure Enable Auto-Compensation is checked


SUPPORT
-------
Check warzavision.log for detailed debug information.

================================================================================
