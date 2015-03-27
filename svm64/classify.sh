#!/bin/bash

cd ~/Documents/svm64

# using models classify the test files
for i in {0..9}
do
	# classify linear egs
	./svm_classify ./data/Digit${i}_svm.tes ./models/${i}_linear.model ./results/${i}_linear_svm.result

	# classify gaussians egs
    #./svm_classify ./data/Digit${i}_svm.tes ./models/${i}_gaus_gamma0.5.model ./results/${i}_svm.result
done
