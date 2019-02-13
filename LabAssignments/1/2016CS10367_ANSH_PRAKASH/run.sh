
if [ $1 == 4 ] ; then
	python3 GaussianDiscrimantAnalysis.py 4 $2 $3 $4
elif [[ $1 == 1 ]]; then
	python3 LinearReg.py 1 $2 $3 $4 $5
elif [[ $1 == 2 ]]; then
	python3 LocallyWeightedLR.py 2 $2 $3 $4
elif [[ $1 == 3 ]]; then
	python3  LogisticRegression.py 3 $2 $3
fi
