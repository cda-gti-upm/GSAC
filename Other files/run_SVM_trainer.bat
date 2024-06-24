@echo OFF
title Batch file to automatically run SVM_training.exe
Goto :START

This batch script file runs SVM_training.exe with specified input arguments. SVM_training.exe trains SVM models to
later be used in the Spatial Grid of Foveatic Classifiers detector.
This file has to be in the same folder as SVM_training.exe. Use full paths to every input file/path

The following list includes all the possible options that can be configured. Use --help argument for more information.
The chosen options must be used below, after the :START label

	* General common mandatory params:
		-input <number of folders> <list of folder paths> : input folders containing training images. Indicate total number of folders,
														   and then list of folders (separated by spaces)
		-activations <number of files> <list of file paths> : files with saved activations matrix. Indicate total number of files, and
															 then list of files (separated by spaces)
		-descriptor <HOG|HAAR> : descriptor to extract features
		-num_classifier <number> : first classifier to train (default 0 trains all classifiers) (optional parameter)
	
	* HOG specific params:
		-win_size <width height> : window size to compute features
		-win_step <width height> : sliding step of previous window
	
	* HAAR specific params:
		-win_size <width height> : original window size to compute features
		-win_step <width height> : sliding step of previous window
		-haar_final_win <width height> : final window to compute features (resized from original window size)"			
	

:: The program starts here! Choose your arguments, and concatenate them using the ^ symbol
:START
@echo on
START "SVM training for Spatial Grid of Foveatic Classifiers Detector (SVM_training.exe)" /wait SVM_training.exe ^
	-input 2 F:/sequences/sequence2_training F:/sequeces/sequence2_training_center ^ 
	-activations 2 F:/sequences/activations1.csv F:/sequences/activations2.csv ^
	-descriptor HAAR ^
	-win_size 96 72 ^
	-win_step 64 64 ^
	-haar_final_win 24 24
	
:: Other example
REM START "SVM training for Spatial Grid of Foveatic Classifiers Detector (SVM_training.exe)" /wait SVM_training.exe ^
	REM -input 1 F:/sequences/sequence4_training ^ 
	REM -activations 1 F:/sequences/activations.csv ^
	REM -descriptor HOG ^
	REM -win_size 24 24 ^
	REM -win_step 8 8

	