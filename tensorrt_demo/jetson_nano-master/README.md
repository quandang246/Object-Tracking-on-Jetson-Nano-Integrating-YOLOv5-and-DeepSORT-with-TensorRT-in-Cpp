# jetson_nano

This repository is a collection of scripts/programs I use to set up the software development environment on my Jetson Nano, TX2, and Xavier NX.

To set Jetson Nano to 10W (MAXN) performance mode ([reference](https://devtalk.nvidia.com/default/topic/1050377/jetson-nano/deep-learning-inference-benchmarking-instructions/)), execute the following from a terminal:

   ```shell
   $ sudo nvpmodel -m 0
   $ sudo jetson_clocks
   ```

These are my blog posts related to the scripts in this repository:

* [JetPack-4.6](https://jkjung-avt.github.io/jetpack-4.6/)
* [JetPack-4.5](https://jkjung-avt.github.io/jetpack-4.5/)
* [Setting up Jetson Xavier NX](https://jkjung-avt.github.io/setting-up-xavier-nx/)
* [JetPack-4.4 for Jetson Nano](https://jkjung-avt.github.io/jetpack-4.4/)
* [JetPack-4.3 for Jetson Nano](https://jkjung-avt.github.io/jetpack-4.3/)
* [Building TensorFlow 2.0.0 on Jetson Nano](https://jkjung-avt.github.io/build-tensorflow-2.0.0/)
* [Installing and Testing SSD Caffe on Jetson Nano](https://jkjung-avt.github.io/ssd-caffe-on-nano/)
* [Building TensorFlow 1.12.2 on Jetson Nano](https://jkjung-avt.github.io/build-tensorflow-1.12.2/)
* [Installing OpenCV 3.4.6 on Jetson Nano](https://jkjung-avt.github.io/opencv-on-nano/)
* [Setting up Jetson Nano: The Basics](https://jkjung-avt.github.io/setting-up-nano/)

And here is a list of TensorFlow versions with the corresponding bazel and protobuf versions:

| tensorflow |  bazel | protobuf |  Tested on  |
|:----------:|:------:|:--------:|:-----------:|
|   1.12.2   | 0.15.2 |   3.6.1  | JetPack-4.2 |
|   1.15.0   | 0.26.1 |   3.8.0  | JetPack-4.3 |
|    2.0.0   | 0.26.1 |   3.8.0  | JetPack-4.3 |
|    2.3.0   |  3.1.0 |   3.8.0  | JetPack-4.4 |
