# Graduation Project
<div align="center"><img src="machine_picture.jpg" width="350"></div>

## Introduction
My graduation project is a laser-based potato sprout remover machine.

### How it works?
* Place a camera that shoot straight down a potato's sprout on top of the plate. 
* Creates an imaginary circle that surrounds on it. Once imaginary circle is created, use its radius value to determine the total distance we need to move the laser. 
* Once the laser on the rod is at the specified location, turn on the laser and rotate the plate to begin cutting the sprout.

## Materials Used
* Jetson Nano 4GB
* 720p Camera
* 3W Motors (3x)
* AA Battery (10x)
* L298N Module
* M6 Nut (2x)
* M6 Bolt 150mm (2x)
* Cardboard
* Line Wires

## Algorithm
### Imaginary Circle
<p align="center" width="100%">
    <img width="32%" img src="Results/3.png", height = "250", width = "350"> 
    <img width="32%" img src="Results/4.png", height = "250", width = "350"> 
    <img width="32%" img src="Results/5.png", height = "250", width = "350"> 
</p>

### Rod
<div align="center"><img src="Results/rod_formula.png" height = "300", width = "300"></div>
<br/>
<div align="center"><img src="https://latex.codecogs.com/svg.image?{\color{Yellow}h3&space;=&space;h1&space;&plus;&space;\frac{h1-h2}{D-R}d}" title="https://latex.codecogs.com/svg.image?{\color{Yellow}h3 = h1 + \frac{h1-h2}{D-R}d}" /></div>  
Once we get the radius value, we insert it to an equation to determine h3's final height.
<br/>

#### Pre-determined settings:
- [ ] h1 = 12.4cm
- [ ] h2 = 7cm (potato's height) + 5.3cm = 12.3cm
- [ ] D = 12.6cm
- [ ] d = 7.8cm
- [ ] h3's initial height is identical to h1

## Accuracy & TensorRT
<div align="center"><img src="Results/accuracy-epoch.jpg" height = "300", width = "800"></div>
<div align="center">AP: 75 at 300 Epochs</div>

### Modified Pre-Trained Yolo-Nano Model
Model |size |mAP<sup>val<br>0.5:0.95 | Params<br>(M) |FLOPs<br>(G)| weights |
| ------        |:---:  |  :---:       |:---:     |:---:  | :---: |
|[YOLOX-Nano](./exps/example/yolox_voc/yolox_voc_s.py) |640  |41.0  | 2.24M |6.93 | [latest_ckpt.pth](./latest.ckpt.pth) |

Total Pictures: 600 Pictures (7 Train : 3 Test)  
Non TensorRT's Inference Time: 0.085s (FPS: ~5)  
TensorRT's Inference Time: 0.042s (FPS: ~10)  
With TensorRT, it is **2.02x faster!** 

### Example Outputs
<p align="center" width="100%">
    <img width="30%" img src="Results/images/4.jpg", height = "280", width = "300"> 
    <img width="30%" img src="Results/images/5.jpg", height = "280", width = "300"> 
    <img width="30%" img src="Results/images/7.jpg", height = "280", width = "300"> 
</p>


## Files Added/Modified
### Added:
* plate.py
* rod.py
* latest_ckpt.pth
* main.py
* radius.py
* train_log.txt
* YOLOX Linux Commands.txt
### Modified:
* voc.py
* voc_classes.py
* coco_classes.py
* voc_eval.py
* yolox_voc_s

YOLOX Base Version: https://github.com/Megvii-BaseDetection/YOLOX
