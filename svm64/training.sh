#!/bin/bash

#cd ~/Documents/svm64

# generate models for training files
for i in {0..9}
do
	# linear
#	./svm_learn ./data/Digit${i}_svm.tra ./models/${i}_linear.model

	# polynomial
#	./svm_learn -t 1 -d 5 -c 1 ./data/Digit${i}_svm.tra ./models/${i}_polynomial5.model

	# gaussian: gamma = 0.005;
	./svm_learn -g 0.005 -t 2 ./data/Digit${i}_svm.tra ./models/${i}_gaus_gamma0.005.model
#	./svm_learn -g 2 -t 2 ./data/Digit${i}_svm.tra ./models/${i}_gaus_gamma2.model
done


