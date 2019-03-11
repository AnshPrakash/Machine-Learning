#!/bin/bash
if [[ "$1" == "1" ]]; then
	python3 NaiveBayes.py $2 $3 $4
elif [ "$1" == "2" ]; then
	if [ $4 == 0 ]; then
		if [ $5 == "a" ] || [ $5 == "b" ]; then
			python3 SVM.py $2 $3 $5
		elif [ $5 == "c" ]; then
			python2 LibSvm.py $2 $3 
		fi
	elif [ $4 == 1 ]; then
		if [ $5 == "a" ]; then
			python3 SVM_Multi.py $2 $3
		else
			python2 LibSVM_Multi.py $2 $3 $5
		fi
	fi
fi
