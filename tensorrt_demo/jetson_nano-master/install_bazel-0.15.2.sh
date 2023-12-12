#!/bin/bash
#
# Reference: https://docs.bazel.build/versions/master/install-ubuntu.html#install-with-installer-ubuntu

set -e

folder=${HOME}/src
mkdir -p $folder

echo "** Install requirements"
sudo apt-get install -y pkg-config zip g++ zlib1g-dev unzip
sudo apt-get install -y openjdk-8-jdk

echo "** Download bazel-0.15.2 sources"
cd $folder
if [ ! -f bazel-0.15.2-dist.zip ]; then
  wget https://github.com/bazelbuild/bazel/releases/download/0.15.2/bazel-0.15.2-dist.zip
fi

echo "** Build and install bazel-0.15.2"
unzip bazel-0.15.2-dist.zip -d bazel-0.15.2-dist
cd bazel-0.15.2-dist
./compile.sh
sudo cp output/bazel /usr/local/bin
bazel help

echo "** Build bazel-0.15.2 successfully"
