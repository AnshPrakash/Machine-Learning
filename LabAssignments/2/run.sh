#!/bin/bash
if [[ $1 == 1 ]]; then
	if [[ $4 == "a" || $4 == "c" ]]; then
		time python3 NaiveBayes.py $2 $3 $4
	elif [[ $4 == "b" ]]; then
		time python3 RandomPredictionYelp.py $2 $3
	elif [[ $4 == "d" ]]; then
		time python3 NaiveBayesStemming.py $2 $3
	elif [[ $4 == "e" ]]; then
		time python3 naiveBayes_Lemmitization.py $2 $3
		time python3 NAiVEBAYES2_Bigram.py $2 $3
	elif [[ $4 == "g" ]]; then
		time python3 NaiveBayes.py $2 $3 "c"
	fi
elif [[ $1 == 2 ]]; then
	if [[ $4 == 1 ]]; then
		if [[ $5 == "a" ]]; then
			time python3 SVM_Multi.py $2 $3
		elif [[ $5 == "b" ]]; then
			time python3 LibSvm_Multi_redo.py $2 $3
		elif [[ $5 == "c" ]]; then
			time python3 LibSvm_Multi_redo.py $2 $3
		elif [[ $5 == "d" ]]; then
			time python3 SvmGBestC.py $2 $3		
		fi
	elif [[ $4 == 0 ]]; then
		if [[ $5 == "a" ]]; then
			time python3 SVM.py $2 $3
		elif [[ $5 == "b" ]]; then
			time python3 SVM.py $2 $3
		elif [[ $5 == "c" ]]; then
			time python3 LibSvm.py $2 $3
		fi
	fi
fi
