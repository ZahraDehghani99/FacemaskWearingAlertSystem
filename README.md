# FacemaskWearingAlertSystem
Implementing the paper "Facemask Wearing Alert System Based on Simple Architecture With Low-Computing Devices" using a two-step paradigm, detected faces with Single Shot Detector, and classified into two classes (with/without mask) using a Keras convolutional neural network.



## Install requirements
You should first clone this project:
 ```
 git clone https://github.com/ZahraDehghani99/FacemaskWearingAlertSystem.git
 cd FacemaskWearingAlertSystem
 ```
 Then you should create a conda environmet as follows:
```
conda create -n facemask_classification python==3.7 pip==20.2.4
```
After creating a conda environment, you should activate it and then run the following code to install requirements.

```
pip install -r requirements.txt
```
## How it works?
After installing requirements, you should run `detect_mask_image.py` as follows:
```
python detect_mask_image.py --image examples/1.png
```
**Note** : you should change address of `face_detector`, `mask_detector` base on your local address.

After run this command if the system predicts `without_mask`, the `FaceMask_detection_alert` will be broadcast.

**Note** : In the `MTCC_face_detector` I tried face detection using MTCNN.
