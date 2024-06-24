@echo OFF
The chosen options must be used below, after the :START label
			:: -input C:/myVideo.avi																				//example for video file
			:: -input C:/myFolder																					//example for folder with images
			:: -input http://user:password@138.4.32.13/axis-cgi/mjpg/video.cgi?fps=10^&resolution=800x600^&.mjpg	//example for Axis IP camera
		-gtfile <file path> : file containing ground truth annotations
	-trainedModels F:/sequences/trained_2_16x16windows ^
	-hyperplaneThreshold 0.4 ^
	
:: Example for generation
REM START "Detector based on array of classifiers (Detector.exe)" /wait Detector.exe ^
	REM -input F:/sequences/sequence2_training ^ 
	REM -mode g ^
	REM -gtfile F:/sequences/groundTruth_sequence2_training ^
	REM -grid r ^
	REM -rows 30 ^
	REM -columns 42 ^
	REM -delay 1 