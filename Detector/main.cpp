/************************************************************************************************** 
* Copyright (C) 2015-2016 Lorena García de Lucas (lgl@gti.ssr.upm.es)
* Grupo de Tratamiento Digital de Imágenes, Universidad Politécnica de Madrid (www.gti.ssr.upm.es)
*
* This file is part of the project "Spatial Grid of Foveatic Classifiers Detector"
*
* This software is to be used only for research and educational purposes.
* Any reproduction or use for commercial purposes is prohibited
* without the prior express written permission from the author. 
*
* Check included file license.txt for more details.
**************************************************************************************************/
#pragma once 

#include "Processor.h"
#include "Detector.h"

using namespace cv;

static void help(){
	cout
		<< "**************************************************************************\n"
		<< "* PEOPLE DETECTION USING SPATIAL GRID OF FOVEATIC CLASSIFIERS\n"
		<< "*\n"
		<< "* The implemented method is suitable for detection in videos\n"
		<< "* recorded with both conventional and omnidirectional cameras\n"
		<< "*\n"
		<< "* This program has been created as part of the following Master Thesis:\n"
		<< "*  - Title: Development of an algorithm for people detection\n"
		<< "*           using onmidirectional cameras\n"
		<< "*  - Author: Lorena Garcia (lgl@gti.ssr.upm.es/lorena.gdelucas@gmail.com)\n"
		<< "*  - Grupo de Tratamiento de Imagenes, Universidad Politecnica de Madrid\n"
		<< "*    (GTI-UPM www.gti.ssr.upm.es)\n"
		<< "*\n"
		<< "* For help about program usage, type --help\n"
		<< "**************************************************************************\n"
		<< endl;
}

static void help2(){
	cout
		<< "**************************************************************************\n"
		<< "* List of input parameters\n"
		<< "*\n"
		<< "** General common params:\n"
		<< "*    -input <video file|folder path|camera id|camera ip address> : input video\n"
		<< "*    -frame_size <width height> : frame size of input file\n"
		<< "*    -mode <g|d|t> : running mode (generation, detection, or test)\n"
		<< "*    -delay <ms> : minimum processing delay (in ms) between two frames\n"
		<< "*    -output <output_video_filename (.avi)> : optional output video filename\n"
		<< "*    -displayBgFg : dispaly foreground mask (if BgSub+Haar features)\n"
		<< "*\n"
		<< "** Generation mode specific params:\n"
		<< "*    -grid <q|r|c> : grid pattern (quincunx, rectangular, or circular)\n"
		<< "*    -rows <#rows> : number of grid rows\n"
		<< "*    -columns <#columns> : number of grid columns\n"
		<< "*    -gtfile <file path> : file containing ground truth annotations\n"
		<< "*\n"
		<< "** Detection mode specific params:\n"
		<< "*    -trainedModels <path> : path to folder containing models\n"
		<< "*    -minThreshold <th> : minimum classifier neighborhood value\n"
		<< "*    -hyperplaneThreshold <th> : loss function value\n"
		<< "*    -showGrid : show grid points\n"
		<< "*    -showNotTrained : show not trained (inactive) grid classifiers\n"
		<< "*    -showActivations : show individual grid members activations\n"
		<< "*    -showActivationsAndConfidence : show activations and associated confidence\n"
		<< "*\n"
		<< "** Test mode specific params, additional to detection mode ones:\n"
		<< "*    -gtfile <file path> : file containing ground truth annotations\n"
		<< "*    -activations <file path> : file containing the training activations\n"
		<< "*\n\n"
		<< " Example of usage:\n"
		<< "   detector.exe -input C:/myvideo.avi -mode d -trainedModels C:/trained_models\n"
		<< "   -minThreshold 0.5 -hyperplaneThreshold 0.1 -showNotTrained -showActivations\n\n"
		<< "*******************************************************************************\n"
		<< endl;
}

void main(int argc, char* argv[])
{
	help();	//show program information/help

	//-------------------------------------
	//-- Define basic parameters and default values (may be overwritten through command line arguments, see below)
	//string inputFile = "http://138.4.32.13/axis-cgi/mjpg/video.cgi?fps=10&resolution=800x600&.mjpg";	//example for IP camera stream
	string inputFile = "F:/sequences/sequence2_test1";
	string trainedModels = "F:/sequences/trained_2_001_16x16windows";
	double numRowsClassifier = 25;	//just for generation mode
	double numColsClassifier = 33;	//just for generation mode
	bool showGrid = false;
	bool showActivations = false;
	bool showActivationsAndConfidence = false;
	bool showNotTrained = false;
	Detector::DescriptorType descriptor = Detector::DescriptorType::HOG;	//unused (takes descriptor used during training)
	Detector::GridType grid = Detector::GridType::QUINCUNX;
	Detector::Mode mode = Detector::Mode::DETECTION;
	string groundTruthFile = "F:/sequences/groundTruth_sequence2_test1.csv";			//just for test mode
	string activationsFile = "F:/sequences/sequence2_activations7q_v4_825_test1.csv";	//just for test mode
	bool useMasks = false;		//experimental for bgsub
	bool showCentroids = false;	//experimental for bgsub
	bool displayBgFg = false;
	int delay = 100;
	float minGroupThreshold = 0.0;
	float hyperplaneThreshold = 0.0;

	string output_video_filename = "";	//set to save output video
	Size frame_size = Size(800, 600);	//default frame size (to change properly via CL)

	//-------------------------------------
	//-- Parse command line input arguments
	for (int i = 1; i < argc; i++)
	{
		if (!strcmp(argv[i], "--help"))
		{
			help2();
			return;
		}
		if (!strcmp(argv[i], "-input"))
		{
			inputFile = string(argv[i + 1]);
		}
		else if (!strcmp(argv[i], "-trainedModels"))
		{
			trainedModels = string(argv[i + 1]);
		}
		else if (!strcmp(argv[i], "-descriptor"))
		{
			if (string(argv[i + 1]) == "HAAR"){
				descriptor = Detector::DescriptorType::HAAR;
			}
			else if (string(argv[i + 1]) == "HOG"){
				descriptor = Detector::DescriptorType::HOG;
			}
		}
		else if (!strcmp(argv[i], "-rows"))
		{
			numRowsClassifier = stod(argv[++i]);
		}
		else if (!strcmp(argv[i], "-cols"))
		{
			numColsClassifier = stod(argv[++i]);
		}
		else if (!strcmp(argv[i], "-showGrid"))
		{
			showGrid = true;
		}
		else if (!strcmp(argv[i], "-showActivations"))
		{
			showActivations = true;
		}
		else if (!strcmp(argv[i], "-showActivationsAndConfidence"))
		{
			showActivationsAndConfidence = true;
		}
		else if (!strcmp(argv[i], "-showNotTrained"))
		{
			showNotTrained = true;
		}
		else if (!strcmp(argv[i], "-grid"))
		{
			if (string(argv[i + 1]) == "r"){
				grid = Detector::GridType::RECTANGULAR;
			}
			else if (string(argv[i + 1]) == "q"){
				grid = Detector::GridType::QUINCUNX;
			}
			else if (string(argv[i + 1]) == "c"){
				grid = Detector::GridType::CIRCULAR;
			}
		}
		else if (!strcmp(argv[i], "-mode"))
		{
			if (string(argv[i + 1]) == "t"){
				mode = Detector::Mode::TEST;
			}
			else if (string(argv[i + 1]) == "g"){
				mode = Detector::Mode::GENERATION;
			}
			else if (string(argv[i + 1]) == "d"){
				mode = Detector::Mode::DETECTION;
			}
		}
		else if (!strcmp(argv[i], "-gtfile"))
		{
			groundTruthFile = string(argv[i + 1]);
		}
		else if (!strcmp(argv[i], "-activations"))
		{
			activationsFile = string(argv[i + 1]);
		}
		else if (!strcmp(argv[i], "-useMasks"))
		{
			useMasks = true;
		}
		else if (!strcmp(argv[i], "-showCentroids"))
		{
			showCentroids = true;
		}
		else if (!strcmp(argv[i], "-minThreshold"))
		{
			minGroupThreshold = stod(argv[++i]);
		}
		else if (!strcmp(argv[i], "-hyperplaneThreshold"))
		{
			hyperplaneThreshold = stod(argv[++i]);
		}
		else if (!strcmp(argv[i], "-displayBgFg"))
		{
			displayBgFg = true;
		}
		else if (!strcmp(argv[i], "-delay"))
		{
			delay = stod(argv[++i]);
		}
		else if (!strcmp(argv[i], "-output"))
		{
			output_video_filename = string(argv[++i]);
		}
		else if (!strcmp(argv[i], "-frame_size"))
		{
			int width = (int)stod(argv[++i]);
			int height = (int)stod(argv[++i]);
			frame_size = Size(width, height);
		}
	}

	//-------------------------------------
	//Create video processor instance
	Processor processor;

	//Set frameProcessor instance
	Detector detector;
	processor.setFrameProcessor(&detector);

	//Set basic detector variables
	detector.setFrameSize(frame_size);
	detector.setDimensions(numRowsClassifier, numColsClassifier);
	detector.showGrid = showGrid;
	detector.showActivations = showActivations;
	detector.showActivationsAndConfidence = showActivationsAndConfidence;
	detector.showNotTrained = showNotTrained;
	detector.setGridType(grid);
	detector.setMode(mode);
	detector.setGroundTruthFilename(groundTruthFile);
	detector.setActivationsFilename(activationsFile);
	detector.setTrainedModelsFolder(trainedModels);
	detector.useMasks = useMasks;
	detector.showCentroids = showCentroids;
	detector.setMinGroupThreshold(minGroupThreshold);
	detector.setHyperplaneThreshold(hyperplaneThreshold);
	detector.setDescriptorType(descriptor);
	detector.writeOutputVideo(output_video_filename);


	//Configures the processor of the images
	processor.configure();	//runs detector.preprocess()

	//Declare a window to display the video
	processor.displayInput("Original video + detections");

	//Declare a window to display the foreground masks video (if BgSub+HAAR used)
	if (displayBgFg) { processor.displayOutput("Computed foreground mask"); }

	//Play the video at set delay
	processor.setDelay(delay);		//minimum wait between images

	//Set output filenames to save processed frames
	//processor.setOutput("Detector_mask", ".pgm", 5, 0);

	//Open video file
	processor.setInput(inputFile);

	//Start the process
	processor.run();		//runs detector.process() in an infinite loop

}