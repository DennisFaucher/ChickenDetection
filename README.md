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
* Python🐍
* Git🐙
* Label conversion software: You need to convert the labels that YOLO-Annotation-Tool creates to the Darknet format. Thankfully, the YOLO-Annotation-Tool repo includes the convert.py Python script to do just that.
### Labeling the Images

![Kielyr Label](https://github.com/DennisFaucher/ChickenDetection/blob/master/Images/Kielyr%20Label.png)


Once you have split your video into hundreds, if not thousands, of frames you need to pick the best 100+ (each class should hav ethe exact number of labeled images) for each image class. In my case, an imge class is a chicken name. I found over time that the most effective images for chicken labeling were either a full side view of the chicken or a full front view of the chicken. Partially obstructed views are not optimal and most chicken butts look the same. 🙂


I happen to have a license for Adobe Lightroom, so I use Lightroom to scan through all the photos and save the best 133 images of each chicken to separate folders named 001 through 006. Any photo viewing software will do including the file manager built in to your operating system.


Once your images are in folders, install the YOLO-Annotation-Tool somewhere using the command

````[Javascript]
git clone https://github.com/ManivannanMurugavel/YOLO-Annotation-Tool.git
````


You will notice that YOLO-Annotation-Tool/Images/ includes a 001 and a 002 directory. copy your folders of images in 001 to 006 folders right into YOLO-Annotation-Tool/Images/


Now set a few hours of tedious box drawing aside and fire up "python main.py" from the YOLO-Annotation-Tool directory. When the main.py imaging tool opens, there is a field to enter the number of your image folder, in this case "1", and then you click Load. Your images in that folder will be loaded in seemingly random order.


Draw a box around the object you would like to have the class label of 1 and then click Next. Keep doing this until you have labeled all images in that class. If you still have the stamina, you can enter 2 in the "Image Dir:" field, click Load and move on to the labeling the next class. Keep doing this until all images in all classes are labeled. You will find a corresponding .txt file in Labels/00? for each .jpg in Images/00?


### Converting Labels to Darknet Format

The YOLO Annotation Tool creates labels in this format:
1
167 204 290 403

Darknet expects labels in this format:
0 0.178515625 0.421527777778 0.09609375 0.276388888889

The script convert.py, not surprisingly, converts from one label format to the other and placel the labels in the Output directory.

![convert.py](https://github.com/DennisFaucher/ChickenDetection/blob/master/Images/convert.py.png)

Change the value on lines 26 and 29 to the number of each of your image folders (001-006) one at a time and run. The labels in that Labels folder will be converted and placed in a single Output folder. As you can imagine, it is important that none of your images or labels have the same name before running this step. I changed the prefix of all my images to match the name of my chicken before I labeled them. For instance Labels/001/kielyr_257401.jpg

Now copy all of your labeled iames in this Output directory. That is important for the next major step.

Also decide where the NVIDIA GPU is that you are going to use to train the model. If it is somewhere else, you will need to scp your Output folder there (assuming you have already installed darknet and Yolo-Annotation-Tool there) 

### Splitting Data into Training and Test Groups

The python script, process.py creates a train.txt and a test.txt file in the current directory with a 90/10 split of the full path to all of your images.

![process.py](https://github.com/DennisFaucher/ChickenDetection/blob/master/Images/process.py.png)

On your NVIDIA GPU training machine, change line 8 to be the full path of the Output directory where you have stored all your images and converted label files and run. Very quickly, you will have two new files, train.txt and test.txt. They will look something like this:

````[Javascript]
/Users/faucherd/Documents/Personal/Machine_Learning/YOLO-Annotation-Tool/Labels/output/nai-85.jpg
/Users/faucherd/Documents/Personal/Machine_Learning/YOLO-Annotation-Tool/Labels/output/bo-91.jpg
/Users/faucherd/Documents/Personal/Machine_Learning/YOLO-Annotation-Tool/Labels/output/kielyr-68.jpg
/Users/faucherd/Documents/Personal/Machine_Learning/YOLO-Annotation-Tool/Labels/output/vespyr-40.jpg
/Users/faucherd/Documents/Personal/Machine_Learning/YOLO-Annotation-Tool/Labels/output/rellian-54.jpg
/Users/faucherd/Documents/Personal/Machine_Learning/YOLO-Annotation-Tool/Labels/output/zaja-46.jpg
````

Copy train.txt and test.txt to your darknet directory

### Train that Model!!!

OK. On your NVIDIA GPU machine, in the dakrnet directory, you will need to edit three files, create one directory and download one large file.

1) Create your .data file
The data file tells the training process where to find things. Create a .data file with a prefix of something that makes sense. I used the prefix "6chix" for everything. In a new 6chix.data file, I added these lines:

````[Javascript]
classes= 6 
train  = train.txt  
valid  = test.txt  
names = 6chix.names  
backup = 6chix-backup/
````

Your classes= will need to be changed to match the number of object classes you are training for

2) Create your backup directory
The backup directory specified in your .data file is where the training weights will be stored during training. Create a new directory under the darknet directory that matches the name of the backup variable in your .data file.

3) Create your .names file
The .names file gives human readable names to each of your classes in order. In a new 6chix.names file, I added these lines:

````[Javascript]
kielyr
nai
zaja
bo
vespyr
rellian
````

4) Create your .cfg file
The .cfg file decribes how to build the neural network for your model. The changes you make to the file depends on whether you are training a YOLO model or a YOLO-tiny model. I used YOLO-tiny as it is much lighter on resources and works well enough for my purposes. I will eplain the YOLO-tiny config changes here.

* Copy darknet/cfg/yolov3-tiny.cfg darknet/cfg/6chix.cfg
* Make these edits to your new .cfg file

````[Javascript]
Line 3: batch=24
Line 4: subdivisions=8
Line 127, set filters=(classes + 5)*3, e.g. for 6 classes filters=33
Line 135, set classes=6, the number of custom classes.
Line 171, set filters=(classes + 5)*3, e.g.  for 6 classes filters=33
Line 177, set classes=6, the number of custom classes.
````

5) Download a set of convolutional weights to get started 
To start training, YOLOv3 requires a set of convolutional weights. I used the darknet53 weights from the darknet repo. You can download those weights with this command: 

````[Javascript]
wget https://pjreddie.com/media/files/darknet53.conv.74
````

6) Start training

The standard command line to start the training is this:

````[Javascript]
./darknet detector train 6chix.data cfg/6chix.cfg darknet53.conv.74
````

You let the training run until the average loss gets to be less than 0 and levels out without improvement. For me, that was when the files in my backup directory reached the 40000.weights level. You can read more about when to stop training in [this](https://github.com/AlexeyAB/darknet/blob/master/README.md#when-should-i-stop-training) article.

````[Javascript]
-rw-rw-r-- 1 dennis dennis 34751196 Jun  8 08:43 6chix.backup
-rw-rw-r-- 1 dennis dennis 34751196 Jun  8 08:43 6chix_100.weights
-rw-rw-r-- 1 dennis dennis 34751196 Jun  8 08:43 6chix_200.weights
-rw-rw-r-- 1 dennis dennis 34751196 Jun  8 08:43 6chix_300.weights
-rw-rw-r-- 1 dennis dennis 34751196 Jun  8 08:43 6chix_400.weights
-rw-rw-r-- 1 dennis dennis 34751196 Jun  8 08:43 6chix_500.weights
-rw-rw-r-- 1 dennis dennis 34751196 Jun  8 08:43 6chix_600.weights
-rw-rw-r-- 1 dennis dennis 34751196 Jun  8 08:43 6chix_700.weights
-rw-rw-r-- 1 dennis dennis 34751196 Jun  8 08:43 6chix_800.weights
-rw-rw-r-- 1 dennis dennis 34751196 Jun  8 08:43 6chix_900.weights
-rw-rw-r-- 1 dennis dennis 34751196 Jun  8 08:43 6chix_10000.weights
-rw-rw-r-- 1 dennis dennis 34751196 Jun  8 08:43 6chix_20000.weights
-rw-rw-r-- 1 dennis dennis 34751196 Jun  8 08:43 6chix_30000.weights
-rw-rw-r-- 1 dennis dennis 34751196 Jun  8 08:42 6chix_40000.weights
````

To more easily keep track of 

````[Javascript]
./darknet detector train 6chix.data cfg/6chix.cfg darknet53.conv.74 | grep "avg," > 6chix.out
````

## Lessons Learned
## Thank you
