#! /bin/bash

while [[ true ]];
do
  ofc "Write a story for middle schoolers. Do not include a title or any other formatting." | tee ./train/`uuidgen`.md
done

