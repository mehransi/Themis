#!/bin/bash

path=`realpath $0`;

path=`dirname "$path"`

cp $path/../model_server.py $path
cp -r $path/../saved_models/Souvikcmsa $path
image_name="mehransi/main:pelastic-sentiment-analysis"
docker build -t $image_name -f $path/sentiment.Dockerfile $path
docker push $image_name

rm $path/model_server.py
rm -rf $path/Souvikcmsa