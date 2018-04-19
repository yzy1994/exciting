#!/bin/bash

for((i=5;i<20;i++))
do
	for((j=1;j<=5;j++))
	do
		python main.py $i $j
	done
done
