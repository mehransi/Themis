#!/bin/bash

path=`realpath $0`;

path=`dirname "$path"`

cp $path/../model_server.py $path

docker build -t mehransi/main:pelastic-video-detector -f $path/detector.Dockerfile $path

rm $path/model_server.py