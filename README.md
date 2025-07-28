## Install Instructions
Tensorflow for GPU
~~~
pip install tensorflow[and-cuda]
~~~
Tensorflow for CPU
~~~
pip install tensorflow
~~~

Every other dep
~~~
pip install tensorflow-datasets
pip install matplotlib
pip install scikit-learn
pip install opencv-python
~~~

[Source FaceNet Weights](https://drive.google.com/drive/folders/1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn?usp=drive_link),
 download facenet_keras_weights.h5 and drop it into the source directory

## Run
Three files with different models
* ContrastiveLoss.py
* TripletLoss.py
* FaceNet.py

