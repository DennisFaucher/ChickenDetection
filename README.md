# ChickenDetection
Create a custom machine learning model to recognize our six chickens by name
![Two Chickens](https://github.com/DennisFaucher/ChickenDetection/blob/master/Images/Jaja%20Kielyr.png)
## Why
Two Reasons:
1) At work, I am helping multiple customers build machine learning solutions to advance drug discovery and overall subscriber health. Understanding machine learning for object detection at a deeper level will enable me to better guide my customers.
2) I love our chickens
## How
### Parts List
* Chickens🙂🐓: We ordered our chickens as chicks [here](https://www.mypetchicken.com/catalog/Baby-Chicks-c36.aspx)
* Access to an NVIDIA GPU with enough video RAM for training: My laptop does not have an NVIDIA GPU, my NVIDIA Xavier GPU core dumps if you try to train a new model on it.  I rented an AWS g2.2xlarge GPU instance with the Deep Learning AMI (Ubuntu 18.04) Version 29.0 AMI.
* RTSP-Capable camera: I use this [one](https://www.amazon.com/gp/product/B07GSS4ZHK/ref=ppx_yo_dt_b_asin_title_o05_s00?ie=UTF8&psc=1) 
* At least 100 photos of each chicken: I used the VLC "Scene Filter" to split a 2 hour video of the chickens into one JPG for every 150 frames. There is a good tutorial on the scene filter [here](https://www.isimonbrown.co.uk/vlc-export-frames/)
* Darknet OSS neural network software: [Website](https://pjreddie.com/darknet/), GitGub [repository](https://github.com/pjreddie/darknet)
* Image labeling software: I used the program main.py from this [YOLO-Annotation-Tool](https://github.com/ManivannanMurugavel/YOLO-Annotation-Tool) GitHub repository. The python scripts in this repo are very helpful. The video instructions will drive you insane.
* Label conversion software: You need to convert the labels that YOLO-Annotation-Tool creates to the Darknet format. Thankfully, the YOLO-Annotation-Tool repo includes the convert.py Python script to do just that.
### Labeling the Images
![Kielyr Label](https://github.com/DennisFaucher/ChickenDetection/blob/master/Images/Kielyr%20Label.png)
Once you have split your video into hundreds, if not thousands, of frames you need to pick the best 100+ (each class should hav ethe exact number of labeled images) for each image class. In my case, an imge class is a chicken name. I found over time that the most effective images for chicken labeling were either a full side view of the chicken or a full front view of the chicken. Partially obstructed views are not optimal and most chicken butts look the same. 🙂


I happen to have a license for Adobe Lightroom, so I use Lightroom to scan through all the photos and save the best 133 images of each chicken to separate folders named 001 through 006. Any photo viewing software will do including the file manager built in to your operating system.
## Lessons Learned
## Thank you
