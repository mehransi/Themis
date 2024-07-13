#!/bin/bash

path=`realpath $0`;

path=`dirname "$path"`

cp $path/../model_server.py $path
cp -r $path/../saved_models/Mitsua $path
image_name="mehransi/main:pelastic-translation"
docker build -t $image_name -f $path/translation.Dockerfile $path
docker push $image_name

rm $path/model_server.py
rm -rf $path/Mitsua
