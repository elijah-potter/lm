#! /bin/bash

while [[ true ]];
do
  ofc -m ministral-3:3b "Write a story for four-year-olds. Do not include a title or any other formatting." | tee ./train/`uuidgen`.md
done

