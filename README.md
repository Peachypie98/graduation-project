# Graduation Project (YOLOX)
YOLOX Base Version: https://github.com/Megvii-BaseDetection/YOLOX

## Introduction
My graduation project is a laser-based potato sprout remover machine.
### How it works?
1. Detect a potato's sprout using camera
2. Creates an imaginary circle that surrounds on it. Once imaginary circle is created, use the radius value to determine how long should I move the rod with certain speed.
3. Once a laser on the rod is at the specified location, turn on the laser and rotate the plate that has potato on top of it to begin cutting.

The files that I modified from the base version or added a new file for this project:
1. plate.py (added)
2. rod.py (added)
3. latest_ckpt.pth (added)
4. main.py (added)
5. radius.py (added)
6. assets and YOLOX outputs folder (added)
8. train_log (detailed).txt (added)
9. frame.jpg (added)
10. YOLOX Linux Commands (added)
11. voc.py (modified)
12. voc_classes.py (modified)
13. coco_classes.py (modified)
14. voc_eval.py (modified)
15. yolox_voc_s (modified)

* will update README soon or later
