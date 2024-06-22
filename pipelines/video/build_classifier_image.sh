#!/bin/bash

path=`realpath $0`;

path=`dirname "$path"`

cp $path/../model_server.py $path

image_name="mehransi/main:pelastic-video-classifier"
docker build -t $image_name -f $path/classifier.Dockerfile $path
docker push $image_name

rm $path/model_server.py