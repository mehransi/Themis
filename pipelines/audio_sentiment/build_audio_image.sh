#!/bin/bash

path=`realpath $0`;

path=`dirname "$path"`

cp $path/../model_server.py $path
cp -r $path/../saved_models/facebook $path
image_name="mehransi/main:pelastic-audio-to-text"
docker build -t $image_name -f $path/audio.Dockerfile $path
docker push $image_name

rm $path/model_server.py
rm -rf $path/facebook