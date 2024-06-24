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

using namespace std;
using namespace cv;

// Default constructor
Processor::Processor() : delay(-1), stop(false), digits(0), process(0), frameProcessor(0) {}

//Read filenames in folder
void Processor::readDirectory(const string& directoryName, vector<string>& filenames){
	glob(directoryName, filenames, false);
}

// Set input (vector of image filenames)
bool Processor::setInput(const vector<string>& imgs) {

	// Release previous instances
	capture.release();
	images.clear();

	// the input will be this vector of images
	images = imgs;
	itImg = images.begin();

	return true;
}

// Set input (video or image directory)
bool Processor::setInput(string filename) {

	// Release previous instances
	capture.release();
	images.clear();

	vector<string> filenames;
	// try-catch for IP camera inputs
	try	{
		readDirectory(filename, filenames);
	}
	catch (exception e){}

	if (filenames.size() > 1)          //image directory
		return setInput(filenames);
	else                                //video file
		return capture.open(filename);
}

// Set input (camera)
bool Processor::setInput(int id) {

	// Release previous instances
	capture.release();
	images.clear();

	return capture.open(id);
}

// Check if input set and opened
bool Processor::isOpened() {
	return capture.isOpened() || !images.empty();
}

//Read next frame
bool Processor::readNextFrame(Mat& frame) {
	if (images.size() == 0)				//video or camera
		return capture.read(frame);
	else if (itImg != images.end()) {	//folder of images
		frame = imread(*itImg);
		itImg++;
		return frame.data != 0;
	}
}

// Set output video writer
bool Processor::setOutput(const string &filename, Size frameSize, int codec, double framerate, bool isColor = true) {
	return writer.open(filename, codec, framerate, frameSize, isColor);
}

// Set output image writer
bool Processor::setOutput(const string &filename, const string &ext, int numberOfDigits = 4, int startIndex = 0) {

	//number of digits must be positive
	if (numberOfDigits < 0)
		return false;

	//filenames and their common extension
	outputFile = filename;
	extension = ext;

	
	digits = numberOfDigits;	//number of digits in the file numbering scheme
	currentIndex = startIndex;	//start numbering at this index

	return true;
}

// Write frame to output stream
void Processor::writeNextFrame(Mat& frame) {
	if (extension.length()) {	// if extension set, then we write images
		stringstream ss;
		ss << outputFile << setfill('0') << setw(digits) << currentIndex++ << extension;
		imwrite(ss.str(), frame, { 0 });
	}
	else { // then write video file
		writer.write(frame);
	}
}

// Set callback function (called at each frame)
void Processor::setFrameProcessor(void(*frameProcessingCallback)(Mat&, Mat&)) {

	frameProcessor = 0;					//invalidate frame processor interface instance
	process = frameProcessingCallback;	//set function to call
}

// Set frame processor interface instance
void Processor::setFrameProcessor(FrameProcessor* frameProcessorPtr) {

	process = 0;							//invalidate callback function
	frameProcessor = frameProcessorPtr;		//set frame processor interface instance
}

// Configure frame processor instance
void Processor::configure(){
	if (!frameProcessor){
		cout << "Error - processor not loaded yet" << endl; exit(-1);
	}
	frameProcessor->preprocess();
}

// Set input display params
void Processor::displayInput(string wn) {
	input_window_name = wn;
	namedWindow(input_window_name);
}

// Set output display params
void Processor::displayOutput(string wn) {
	output_window_name = wn;
	namedWindow(output_window_name);
}

// Set minimum wante dealy between each frame
void Processor::setDelay(int d) {
	wantedDelay = d;
}

// Stop processing
void Processor::stopIt() {
	stop = true;
}

// Check is process stopped
bool Processor::isStopped() {
	return stop;
}

// Grab frames and process them
void Processor::run() {

	Mat frame;		//frame being processed
	Mat output;		//output frame

	//check is input set and opened
	if (!isOpened())
		return;

	stop = false;

	//processing loop
	while (!isStopped()) {					//wait until processing resumed
		double duration = getTickCount();	//to set delay

		//read next frame
		if (!readNextFrame(frame))
			break;

		//call processing function
		frameProcessor->process(frame, output);

		//display input frame (if window name set)
		if (input_window_name.length() != 0)
			imshow(input_window_name, frame);

		//display output frame (if window name set)
		if (output_window_name.length() != 0)
			imshow(output_window_name, output);

		//write output sequence (if output filename set)
		if (outputFile.length() != 0)
			writeNextFrame(output);

		//compute delay (given wanted delay, and elapsed processing time)
		double duration1 = static_cast<double>(getTickCount()) - duration;
		duration1 /= (getTickFrequency() / 1000);	//duration of processing in ms
		if (duration1 < wantedDelay){
			delay = wantedDelay - (int)duration1;	//wait if minimum delay not achieved
		}
		else{ delay = 1; }							//continue as fast as possible is minimum delay achieved


		//set delay and pause delay
		if (delay >= 0 && waitKey(delay) >= 0)
			stopIt();

		//double duration2 = static_cast<double>(getTickCount()) - duration;
		//duration2 /= (getTickFrequency() / 1000);	//duration of processing in ms
		//cout << "Total time between frames: " << duration2 << endl;

	}
}