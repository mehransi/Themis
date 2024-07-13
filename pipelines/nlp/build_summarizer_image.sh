#!/bin/bash

path=`realpath $0`;

path=`dirname "$path"`

cp $path/../model_server.py $path
cp -r $path/../saved_models/stevhliu $path
image_name="mehransi/main:pelastic-summarizer"
docker build -t $image_name -f $path/summarizer.Dockerfile $path
docker push $image_name

rm $path/model_server.py
rm -rf $path/stevhliu
