#!/bin/bash

cd ~/Documents/svm64

# generate models for training files
for i in {0..9}
do
	# linear
    ./svm_learn ./data/Digit${i}_svm.tra ./models/${i}_linear.model

	# gaussian: gamma = 0.005;
	#./svm_learn -g 0.005 -t 2 ./data/Digit${i}_svm.tra ./models/${i}_gaus_gamma0.5.model
done


