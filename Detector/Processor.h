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
#if !defined VIDEO_PROCESSOR
#define VIDEO_PROCESSOR

#include <iostream>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "frameProcessor.h"

using namespace std;
using namespace cv;

class Processor {

private:

	// -- Reading input -- //
	VideoCapture capture;
	void(*process)(Mat&, Mat&);		//callback function to process each frame
	FrameProcessor *frameProcessor;			//pointer to processing interface
	
	vector<string> images;				//vector of input image filenames
	vector<string>::const_iterator itImg;	//images iterator
	bool readNextFrame(Mat& frame);		//read next frame from folder, video, camera
	
	// -- Display -- //
	string input_window_name;		//input display window name
	string output_window_name;		//output display window name

	// -- Control variables -- //
	int wantedDelay;					//minimum wanted delay between frame processing
	int delay;							//aux delay variable
	bool stop;							//bool to stop processing

	// -- Writing output images -- //
	VideoWriter writer;
	string outputFile;
	int currentIndex;					//current index for output images
	int digits;							//digits in output filenames (padded with zero is necessary)
	string extension;					//output images extension (.jpg, .png...)
	void writeNextFrame(Mat& frame);	//write frame to output stream/folder

public:

	Processor();	//default constructor

	// -- Input setting -- //
	bool setInput(string filename);				//video file
	bool setInput(int id);						//camera stream (dafault id = 0)
	bool setInput(const vector<string>& imgs);	//vector of images (read from folder)
	void readDirectory(const string& directoryName, vector<string>& filenames);		//read filename in folder
	bool isOpened();		//check is input is opened
	
	// -- Output setting -- //
	bool setOutput(const string &filename, Size frameSize, int codec, double framerate, bool isColor);	//video file
	bool setOutput(const string &filename, const string &ext, int numberOfDigits, int startIndex);		//image series

	// -- Processing -- //
	void setFrameProcessor(FrameProcessor* frameProcessorPtr);				//set frame processor interface instance
	void setFrameProcessor(void(*frameProcessingCallback)(Mat&, Mat&));		//set callback function
	void configure();														//configure frame processor instance
	void run();																//grab and process frames

	// -- Control functions -- //
	void setDelay(int d);	//set wanted delay between each frame processing (default = 1)
	void stopIt();			//stop processing (for certain time specified in delay)
	bool isStopped();		//check is process stopped

	// -- Display -- //
	void displayInput(string wn);		//display input frames
	void displayOutput(string wn);		//display output frames

};

#endif
