@echo OFFtitle Batch file to automatically run Detector.exeGoto :STARTThis batch script file runs Detector.exe with specified input arguments. Detector.exe runs a Spatial Grid of Foveatic Classifiers detector.This file has to be in the same folder as Detector.exe. Use full paths to every input file/pathThe following list includes all the possible options that can be configured. Use --help argument for more information.
The chosen options must be used below, after the :START label	* General common for all modes params:		-input <video file|folder path|camera id|camera ip address> : input video
			:: -input C:/myVideo.avi																				//example for video file
			:: -input C:/myFolder																					//example for folder with images
			:: -input http://user:password@138.4.32.13/axis-cgi/mjpg/video.cgi?fps=10^&resolution=800x600^&.mjpg	//example for Axis IP camera		-frame_size <width height> : frame size of input file (default: 800 600)		-mode <g|d|t> : running mode (generation, detection [default], or test)		-delay <ms> : minimum processing delay (in ms) between two frames (default: 100)		-output <output_video_filename (.avi)> : optional output video filename		-displayBgFg : dispaly foreground mask (if BgSub+Haar features)		* Generation mode specific params:		-grid <q|r|c> : grid pattern (quincunx [default], rectangular, or circular)		-rows <#rows> : number of grid rows (default: 25)		-columns <#columns> : number of grid columns (default: 33)
		-gtfile <file path> : file containing ground truth annotations		* Detection mode specific params:		-trainedModels <path> : path to folder containing models		-minThreshold <th> : minimum classifier neighborhood value (default: 0.0)		-hyperplaneThreshold <th> : loss function value	(default: 0.0)		-showGrid : show grid points		-showNotTrained : show not trained (inactive) grid classifiers		-showActivations : show individual grid members activations		-showActivationsAndConfidence : show activations and associated confidence		* Test mode specific params, additional to detection mode ones:		-gtfile <file path> : file containing ground truth annotations		-activations <file path> : file containing the training activations:: The program starts here! Choose your arguments, and concatenate them using the ^ symbol:START@echo onSTART "Spatial Grid of Foveatic Classifiers detector (Detector.exe)" /wait Detector.exe ^	-input F:/sequences/sequence2_test1 ^ 	-mode d ^
	-trainedModels F:/sequences/trained_2_16x16windows ^	-minThreshold 0.1 ^
	-hyperplaneThreshold 0.4 ^	-delay 100 ^	-showActivations
	
:: Example for generation
REM START "Detector based on array of classifiers (Detector.exe)" /wait Detector.exe ^
	REM -input F:/sequences/sequence2_training ^ 
	REM -mode g ^
	REM -gtfile F:/sequences/groundTruth_sequence2_training ^
	REM -grid r ^
	REM -rows 30 ^
	REM -columns 42 ^
	REM -delay 1 	