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

#include <fstream>
#include <iomanip>
#include "Detector.h"

using namespace std;
using namespace cv;

/**Constructor setting the default values*/
Detector::Detector() {
	saveVideo = false;	//don't save video
	totalFrames = -1;
	trainedClassifiers = 0;
	totalDuration = 0;
	totalBgsubDuration = 0;
	totalPredictionDuration = 0;
	totalFeatExtractionDurationHOG = 0;
	totalFeatExtractionDurationHAAR = 0;
	verbose = false;
}

/**Set number of rows and columns of classifiers' grid*/
void Detector::setDimensions(int numRows, int numCols){
	numRowsClassifier = numRows;
	numColsClassifier = numCols;
}

/**Set frame size*/
void Detector::setFrameSize(Size size){
	frame_size = size;
}

/**Set vector of SVM pointers (depending on number of classifiers in grid)*/
void Detector::setSVMs(Mat &grid){
	svm_ptr.resize(grid.rows);
	for (int i = 0; i < svm_ptr.size(); i++){
		svm_ptr[i] = new SVM;
	}
}

/**Set processing mode (detection, test or generation)*/
void Detector::setMode(Mode chosenMode){
	mode = chosenMode;
}

/**Set grid type (rectangular, quincunx, circular)*/
void Detector::setGridType(GridType type){
	gridType = type;
}

/**Set descriptor type (HOG or HAAR)*/
void Detector::setDescriptorType(DescriptorType type){
	descriptorType = type;
}

/**Set ground truth filename*/
void Detector::setGroundTruthFilename(string filename){
	groundTruthFilename = filename;
}

/**Set activations matrix filename*/
void Detector::setActivationsFilename(string filename){
	activationsFile = filename;
}

/**Set trained models folder*/
void Detector::setTrainedModelsFolder(string folder){
	trainedModelsFolder = folder;
}

/**Set minimum confidence threshold for group of classifiers*/
void Detector::setMinGroupThreshold(float thresh){
	minGroupThreshold = thresh;
}

/**Set hyperplane threshold translation*/
void Detector::setHyperplaneThreshold(float thresh){
	hyperplane_threshold = thresh;
}

/**Set true to save output video, and set output video filename (default = "output_video.avi")*/
void Detector::writeOutputVideo(string filename){
	if (filename == ""){
		saveVideo = false;
		return;
	}
	videoName = filename;
	saveVideo = true;
}

/**Save main grid variables in a .yml file to later reuse them*/
void Detector::saveMainVariables(Point center = Point(0, 0), int num_circles = 0){
	FileStorage fs("gridParams.yml", FileStorage::WRITE);
	switch (gridType){
		case 0:
			fs << "Grid_pattern" << "Rectangular";
			break;
		case 1:
			fs << "Grid_pattern" << "Quincunx";
			break;
		case 2:
			fs << "Grid_pattern" << "Circular";
			break;
	}
	fs << "Grid_pattern_id" << gridType;
	fs << "Grid_params";
	if (gridType == CIRCULAR){
		fs << "{";
		fs << "Number_circles" << num_circles;
		fs << "Center" << center;
		fs << "Total_grid_points" << grid.rows;
		fs << "}";
	}
	else {
		fs << "{";
		fs << "Num_rows" << numRowsClassifier;
		fs << "Num_cols" << numColsClassifier;
		fs << "Total_grid_points" << grid.rows;
		fs << "}";
	}
	fs << "Num_neighbors" << numNeighbors;

	fs.release();
}

/**Load previously saved main grid variables/configuration*/
void Detector::loadMainVariables(string filename){
	FileStorage fs(filename, FileStorage::READ);
	int type = (int)fs["Grid_pattern_id"];
	gridType = static_cast<GridType>(type);
	FileNode gridParams = fs["Grid_params"];
	if (gridType == CIRCULAR){
		int num_circles = (int)gridParams["Number_circles"];
		Point center; gridParams["Number_circles"] >> center;
	}
	else{
		gridParams["Num_rows"] >> numRowsClassifier;
		gridParams["Num_cols"] >> numColsClassifier;
	}
	fs["Num_neighbors"] >> numNeighbors;

	fs.release();
}

/**Read info from a .csv file and store it in a matrix*/
void Detector::loadCsv(string filename, Mat &data){
	if (csvFile.read_csv(filename.c_str()) == -1){
		cout << "Error - Info couldn't be loaded from .csv file" << endl; return;
	};
	const CvMat* data_ptr = csvFile.get_values();	//matrix with ground truth data
	data = Mat(data_ptr);
}

/**Save a Mat into a .csv file*/
void Detector::saveMatToCsv(Mat& matrix, string filename){
	ofstream outputFile(filename);
	outputFile << format(matrix, "CSV") << endl;
	outputFile.close();
}

/**Create a grid of points given a number of rows and columns (rectangular pattern)*/
void Detector::createGrid(double numRows, double numCols, Size frameSize, Mat& grid){
	int height_step = floor(frameSize.height / ((double)numRows - 1));	//step between rows
	int width_step = floor(frameSize.width / ((double)numCols - 1));	//step between columns
	int xc = -height_step;		//initial x_coordinate
	int yc = -width_step;		//initial y_coordinate
	grid = Mat(numCols*numRows, 2, CV_32FC1, Scalar(0));	//matrix to store grid coordinates
	// obtain iterator at initial position
	cv::Mat_<float>::iterator it = grid.begin<float>();

	// fill grid coordinates
	for (int i = 0; i < (numRows); i++) {
		for (int j = 0; j < (numCols); j++){
			(*it) = xc + height_step;
			(*++it) = yc + width_step;
			it++;
			yc += width_step;
		}
		xc += height_step;
		yc = -width_step;
	}

	//reshape grid for easier/later processing: numRows x numCols x 2 channels (1st channel x_coordinate, 2nd channel y_coordinate)
	positions = grid.reshape(2, numRows);
}

/**Create a grid of points given a number of rows and columns (quincunx pattern)*/
void Detector::createQuincunxGrid(double numRows, double numCols, Size frameSize, Mat& grid){
	int height_step = floor(frameSize.height / ((double)numRows - 1));	//step between rows
	int width_step = floor(frameSize.width / ((double)numCols - 1));	//step between columns
	int xc = -height_step;		//initial x_coordinate
	int yc = -width_step;		//initial y_coordinate
	Mat temp_grid(numCols*numRows, 2, CV_32FC1, Scalar(0));	//matrix to store grid coordinates
	// obtain iterator at initial position
	cv::Mat_<float>::iterator it = temp_grid.begin<float>();

	// fill grid coordinates - regular (rectangular) grid
	for (int i = 0; i < (numRows); i++) {
		for (int j = 0; j < (numCols); j++){
			(*it) = xc + height_step;
			(*++it) = yc + width_step;
			it++;
			yc += width_step;
		}
		xc += height_step;
		yc = -width_step;
	}
	// add offset to alternate lines
	for (int i = 0; i < temp_grid.rows; i = i + 2 * numCols){
		for (int j = 0; j < numCols; j++){
			temp_grid.at<float>(i + j, 1) = temp_grid.at<float>(i + j, 1) + width_step / 2;
		}
	}
	grid.push_back(temp_grid);

	//reshape grid for easier/later processing: numRows x numCols x 2 channels (1st channel x_coordinate, 2nd channel y_coordinate)
	positions = grid.reshape(2, numRows);
}

/**Create a grid of points (circular pattern)*/
void Detector::createCircularGrid(Size frameSize, Mat& grid, int center_x_offset = 0, int center_y_offset = 0, double num_circles = 12, double num_min_points = 4){

	Point2f center = Point2f(frameSize.width / 2, frameSize.height / 2);
	center.y = center.y + center_y_offset;
	center.x = center.x + center_x_offset;

	//Set radius for all circles in grid
	double max_radius = center.y + 8;		//max radius is larger than image height (cropped omnidirectional image)
	double number_circles = num_circles;
	vector<double> radius(number_circles);
	for (int i = 1; i <= number_circles; i++){
		radius[i - 1] = max_radius*i / number_circles;
		//radius[i - 1] = it.count*log(i) / log(number_circles);		//alternative logarithmic grid
		//circle(image, center, radius[i - 1], Scalar(255, 0, 0));
	}

	//Always include center point in the grid
	Mat center_mat = (Mat_<float>(1, 2) << center.y, center.x);
	grid.push_back(center_mat);

	//Set points in each circle of the grid
	double min_number_points = num_min_points;		//points in the smaller inner circle
	double total_points = 0;

	for (int j = 0; j < number_circles; j++){
		vector<double>angles;
		double num_points = min_number_points * (2 * (j + 1));		//each circle has double points than previous inner one
		for (int i = 0; i < num_points; i++){
			angles.push_back(360 * (i) / num_points);				//get angles (in polar coordinates)
			Mat x; Mat y;
			polarToCart(radius[j], angles[i], x, y, true);

			Mat point(1, 2, CV_32F);
			point.at<float>(1) = (x.at<double>(0) + center.x);
			point.at<float>(0) = (y.at<double>(0) + center.y);
			grid.push_back(point);									//save final cartesian coordinates
		}
		total_points += num_points;
	}
}

/**Divide image in sectors (circular pattern, meant for omnidirectional images)*/
void Detector::getSectorsCirc(vector<vector<Point>> &sectors, Size frameSize, int center_x_offset = 0, int center_y_offset = 0, double num_circles = 12, double num_min_points = 4){

	Mat image(frameSize.height, frameSize.width, CV_8UC1, Scalar(0));

	Point center = Point(frameSize.width / 2, frameSize.height / 2);
	center.y = center.y + center_y_offset;
	center.x = center.x + center_x_offset;

	double max_radius = center.y + 8;		//max radius is larger than image height (cropped omnidirectional image)
	double number_circles = num_circles;
	vector<double> radius(number_circles);

	//Draw circles outer to inner
	for (int j = number_circles; j >= 1; j--){
		radius[j - 1] = max_radius*j / number_circles;
		//radius[i - 1] = it.count*log(i) / log(number_circles);	//alternative logarithmic grid
		circle(image, center, radius[j - 1], Scalar(0), -1);		//draw filled black circle
		circle(image, center, radius[j - 1], Scalar(255), 1);		//draw white circumference line

		double min_number_points = num_min_points;
		vector<double>angles;
		double num_points = min_number_points * (2 * (j + 1));
		for (int i = 1; i <= num_points; i++){
			angles.push_back(360 * (i) / num_points);
			Mat x; Mat y;
			polarToCart(radius[j - 1], angles[i - 1], x, y, true);
			Point new_point = Point(x.at<double>(0), y.at<double>(0)) + center;
			line(image, Point2i(center), Point2i(new_point), Scalar(255), 1);		//draw radius from points
		}
	}

	//Extend border of original image to find contours touching image limits
	Mat image_extended = image.clone();
	copyMakeBorder(image_extended, image_extended, 2, 2, 0, 0, BORDER_CONSTANT, Scalar(255));

	//Find contours (contours), and apply offset to balance previous border extension
	vector<vector<Point>> contours;
	findContours(image_extended, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, -2));

	////Draw contours with random colors to make sure every one is well detected
	//RNG rng(12345);
	//Mat image_bgr = image.clone();
	//cvtColor(image_bgr, image_bgr, CV_GRAY2BGR);
	//Mat image_bgr_lines = image_bgr.clone();

	for (int i = 0; i < contours.size(); i++)
	{
		if (contourArea(contours[i]) < 5000){	//skip outer contours
			sectors.push_back(contours[i]);
			//Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			//drawContours(image_bgr, contours, i, color, -1);
		}
	}
	/*imshow("Colored sectors", image_bgr);
	waitKey();
	add(image_bgr, image_bgr_lines, image_bgr);
	imshow("Colored sectors with lines", image_bgr);
	waitKey();*/
}

/**Divide image in sectors (rectangular pattern)*/
void Detector::getSectorsRect(vector<vector<Point>> &sectors, Size frameSize, double numRows, double numCols){

	Mat image(frameSize.height, frameSize.width, CV_8UC1, Scalar(0));

	double step_width = frameSize.width / (numCols - 1);
	double step_height = frameSize.height / (numRows - 1);

	//Draw vertical and horizontal lines of the grid
	for (int i = 0; i <= frameSize.width; i += step_width){		//vertical
		Point top = Point(i, 0);
		Point bottom = Point(i, frameSize.height);
		line(image, top, bottom, Scalar(255), 1);
	}
	for (int i = 0; i <= frameSize.height; i += step_height){	//horizontal
		Point top = Point(0, i);
		Point bottom = Point(frameSize.width, i);
		line(image, top, bottom, Scalar(255), 1);
	}

	//Extend border of original image to find contours touching image limits
	Mat image_extended = image.clone();
	copyMakeBorder(image_extended, image_extended, 2, 2, 2, 2, BORDER_CONSTANT, Scalar(255));

	//Find contours (contours), and apply offset to balance previous border extension
	vector<vector<Point>> contours;
	findContours(image_extended, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(-2, -2));

	////Draw contours with random colors to make sure every one is well detected
	//RNG rng(12345);
	//Mat image_bgr = image.clone();
	//cvtColor(image_bgr, image_bgr, CV_GRAY2BGR);
	//Mat image_bgr_lines = image_bgr.clone();

	for (int i = 0; i < contours.size(); i++)
	{
		if (contourArea(contours[i]) < 5000){	//skip outer contours
			sectors.push_back(contours[i]);
			//Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			//drawContours(image_bgr, contours, i, color, -1);
		}
	}
	//imshow("Colored sectors", image_bgr);
	//waitKey();
	//add(image_bgr, image_bgr_lines, image_bgr);
	//imshow("Colored sectors with lines", image_bgr);
	//waitKey();
}

/**Divide image in sectors (quincunx pattern)*/
void Detector::getSectorsQuinc(vector<vector<Point>> &sectors, Mat &grid, Size frameSize, double numRows, double numCols){

	Mat image(frameSize.height, frameSize.width, CV_8UC1, Scalar(0));

	double step_width = frameSize.width / (numCols - 1);
	double step_height = frameSize.height / (numRows - 1);

	//Draw horizontal and diagonal lines of the grid
	for (int i = 0; i <= frameSize.height; i += step_height){	//horizontal
		Point top = Point(0, i);
		Point bottom = Point(frameSize.width, i);
		line(image, top, bottom, Scalar(255), 1);
	}

	Mat temp_grid(grid.rows, grid.cols, CV_32F, Scalar(0));		//diagonal
	grid.col(0).copyTo(temp_grid.col(1));
	grid.col(1).copyTo(temp_grid.col(0));
	for (int i = 0; i < numRows*numCols - numCols - 1; i += numCols){
		Mat row1 = temp_grid.rowRange(i, i + numCols);
		Mat row2 = temp_grid.rowRange(i + numCols, i + 2 * numCols);
		if (!(i % 2)){										//even rows
			for (int j = 0; j < (row1.rows) - 1; j++){
				Point top = Point(row1.row(j));
				Point bottom1 = Point(row2.row(j));
				line(image, top, bottom1, Scalar(255), 1);
				if (j != row1.rows - 1){
					Point bottom2 = Point(row2.row(j + 1));
					line(image, top, bottom2, Scalar(255), 1);
				}
			}
		}
		else{												//odd rows
			for (int j = 0; j < (row1.rows); j++){
				Point top = Point(row1.row(j));
				if (j != row1.rows){
					Point bottom2 = Point(row2.row(j));
					line(image, top, bottom2, Scalar(255), 1);
				}
				if (j != 0){
					Point bottom1 = Point(row2.row(j - 1));
					line(image, top, bottom1, Scalar(255), 1);
				}
			}
		}
	}

	//Extend border of original image to find contours touching image limits
	Mat image_extended = image.clone();
	copyMakeBorder(image_extended, image_extended, 2, 2, 2, 2, BORDER_CONSTANT, Scalar(255));

	//Find contours (contours), and apply offset to balance previous border extension
	vector<vector<Point>> contours;
	findContours(image_extended, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(-2, -2));

	////Draw contours with random colors to make sure every one is well detected
	//RNG rng(12355);
	//Mat image_bgr = image.clone();
	//cvtColor(image_bgr, image_bgr, CV_GRAY2BGR);
	//Mat image_bgr_lines = image_bgr.clone();

	for (int i = 0; i < contours.size(); i++)
	{
		if (contourArea(contours[i]) < 5000){	//skip outer contours
			sectors.push_back(contours[i]);
			//Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			//drawContours(image_bgr, contours, i, color, -1);
		}
	}
	//imshow("Colored sectors", image_bgr);
	//waitKey();
	//add(image_bgr, image_bgr_lines, image_bgr);
	//imshow("Colored sectors with lines", image_bgr);
	//waitKey();
}

/**Find which sector(s) of the set contains the given point*/
vector<int> Detector::findSector(vector<vector<Point>> sectors, Point point){
	vector<int> indexes;
	for (int i = 0; i < sectors.size(); i++)
	{
		int check = pointPolygonTest(sectors[i], point, false);		//check if point is inside sector
		if (check >= 0){
			indexes.push_back(i);	//if it is, save index
		}
	}
	return indexes;
}

/**Given a grid of points and a specific external point, computes the nearest numNeighbors grid points (i.e. classifiers) to it
	@params:
	- (in) Mat grid: numClassifiers x 2 matrix containing coordinates of grid's points positions
	- (in) Point point: external point for which the neighbors are searched
	- (out) vector<int> numNearestClassifier: vector to save numbers of neighbors
	- (in) int numNeighbors: number of neighbors to retrieve
*/
void Detector::getNearestClassifiers(Mat &grid, Point &point, vector<int> &numNearestClassifier, int &numNeighbors){
	cv::Mat_<float>::iterator it = grid.begin<float>();
	cv::Mat_<float>::iterator itend = grid.end<float>();
	vector <double> magnitude(numNeighbors);						//vector of Euclidean distances between point and neighbors
	std::fill(magnitude.begin(), magnitude.end(), INFINITY);		//initialize distances to infinity
	int numClassifier = 0;											//to update associated classifier number
	std::fill(numNearestClassifier.begin(), numNearestClassifier.end(), 0);		//initialize all neighbors to zero
	vector<double> current_point = { double(point.y), double(point.x) };
	for (; it != itend; it++) {													//for each grid point (classifier)
		vector<double> grid_point = { double(*(it)), double(*(++it)) };
		double new_magnitude = norm(current_point, grid_point, NORM_L2);		//compute norm (distance) between vectors
		for (int i = 0; i < magnitude.size(); i++){
			if (new_magnitude <= magnitude[i]){										//if distance is the lowest
				for (int j = magnitude.size() - 1; j > i; j--){
					magnitude[j] = magnitude[j - 1];								//update all the magnitudes and classifiers' numbers
					numNearestClassifier[j] = numNearestClassifier[j - 1];
				}
				magnitude[i] = new_magnitude;
				numNearestClassifier[i] = numClassifier;
				break;
			}
		}
		numClassifier += 1;
	}
}

/**Draw a line between a given point and a classifier (defined by a point in a grid)*/
void Detector::drawLine(Mat &img, Point pt1, int classifier){
	cv::Mat_<float>::iterator it = grid.begin<float>();
	it += 2 * (classifier);			//move iterator to find point in grid defining the classifier
	Point pt2;
	pt2.y = (*it);
	pt2.x = (*++it);
	line(img, pt1, pt2, Scalar(255, 255, 0), 2);	//draw a cyan line between classifier and point
}

/**Given a ground truth matrix for whole sequence, found for every frame all the associated classifiers to be activated
	@params:
	- (in/out) Mat frame: frame to draw lines between each ground truth point and associated classifiers
	- (in) int numNeighbors: number of neighbors to retrieve
*/
void Detector::associateGT(Mat &frame, int numNeighbors){
	Mat frameGT = groundTruth(cv::Rect(0, numFrame, groundTruth.cols, 1));	//ground truth of the current frame
	cv::Mat_<float>::iterator it = frameGT.begin<float>();					// obtain iterator at initial position
	cv::Mat_<float>::iterator itend = frameGT.end<float>();					// obtain iterator at end position
	it++;		//skip first number (first number in ground truth data is always frame number)
	
	// loop over all pixels and get ground truth points
	vector <int> Classifier(numNeighbors);
	for (; it != itend; ++it) {
		// process each pair of ground truth points
		Point point;
		point.x = (*it);
		point.y = (*++it);
		if ((point.x != 0) || (point.y != 0)){					//don't consider (0,0) points (dummy numbers when there's no one in scene)
			getNearestClassifiers(grid, point, Classifier, numNeighbors);	//get nearest classifiers for each GT point
			for (int i = 0; i < Classifier.size(); i++){
				drawLine(frame, point, Classifier[i]);						//draw line connecting point with nearest classifiers
				activations.at<uchar>(numFrame, Classifier[i]) = 255;		//mark as active such classifiers within current frame
			}
		}
	}
}

/**Given ground truth data for whole sequence, retrieve GT points for specific frame and draw them
	@params:
	- (in/out) Mat frame: frame to draw ground truth points
	- (out) vector<Point> points: vector of retrieved ground truth points for the current frame
*/
void Detector::drawGT(Mat &frame, vector<Point> &points){
	Mat frameGT = groundTruth(cv::Rect(0, numFrame, groundTruth.cols, 1));	//ground truth of the current frame
	cv::Mat_<float>::iterator it = frameGT.begin<float>();		// obtain iterator at initial position
	cv::Mat_<float>::iterator itend = frameGT.end<float>();		// obtain end position
	it++;	//skip first number (first number in ground truth data is always frame number)
	// loop over all pixels
	for (; it != itend; ++it) {
		// process each pair of ground truth points
		Point gt_point;
		gt_point.x = (*it);
		gt_point.y = (*++it);
		if ((gt_point.x != 0) || (gt_point.y != 0)){				//don't draw (0,0) points
			line(frame, gt_point, gt_point, Scalar(0, 0, 255), 4);	//draw the point on output image (red)
			points.push_back(gt_point);								//save points
		}
	}
}

/**Draw a given grid of points*/
void Detector::drawGrid(Mat& grid, Mat& frame, Scalar color, bool showSVMnumber = false){
	cv::Mat_<float>::iterator it = grid.begin<float>();		// obtain iterator at initial position
	cv::Mat_<float>::iterator itend = grid.end<float>();	// obtain end position
	for (int i = 0; it != itend; ++it, i++) {
		// get grid points
		Point point;
		point.y = (*it);
		point.x = (*++it);
		line(frame, point, point, color, 4);	//draws the point on output image
		if (showSVMnumber){		//indicate associated SVM number
			putText(frame, to_string(i), Point(point.x+4, point.y), 1, 0.75, Scalar(0, 255, 255));
		}
	}
}

/**Draw specific point of a grid of points (given the number position of such point in the grid)*/
void Detector::drawGridPoint(Mat &grid, int number, Mat& frame, Scalar color){
	//obtain iterator at initial position
	cv::Mat_<float>::iterator it = grid.begin<float>();
	it += 2 * number;		//get coordinates of the point
	Point point;
	point.y = (*it);
	point.x = (*++it);
	line(frame, point, point, color, 8);	//draws the point on output image
}

/**Draw specific point of a grid of points (given the number position of such point in the grid) and an associated value (confidence)*/
void Detector::drawGridPointAndConfidence(Mat &grid, int number, Mat &frame, Scalar color, float &confidence){
	cv::Mat_<float>::iterator it = grid.begin<float>();		//obtain iterator at initial position
	it += 2 * number;		//get coordinates of the point
	Point point;
	point.y = (*it);
	point.x = (*++it);
	line(frame, point, point, color, 8);	//draws the point on output image
	char confidencestring[255];
	sprintf(confidencestring, "%.2f", abs(confidence));		//only two decimals on confidence score
	putText(frame, confidencestring, Point(point.x, point.y - 4), 1, 0.7, Scalar(0, 255, 0));	//draw confidence above point
}

/**Draw specific given point and an associated value (confidence)*/
void Detector::drawPointAndConfidence(Point point, Mat &frame, Scalar color, float &confidence){
	char confidencestring[255];
	sprintf(confidencestring, "%.2f", abs(confidence));		//only two decimals on confidence score
	int baseline = 0;
	cv::Size text = cv::getTextSize(confidencestring, 1, 1.5, 2, &baseline);
	Point or = Point(point.x, point.y-8);
	cv::rectangle(frame, or + cv::Point(0, baseline), or + cv::Point(text.width, -text.height-4), Scalar(0,0,0), CV_FILLED);
	putText(frame, confidencestring, or, 1, 1.5, Scalar(0, 255, 0), 2);	//draw confidence above point
	line(frame, point, point, color, 8);	//draws the point on output image
}

/**Save in a matrix the number of positive and negative samples associated with each classifier (for training)
	@params:
	- Mat activations: rows = total number of frames in dataset, columns = number of classifiers; (255 if activated, 0 otherwise)
	- Mat sampleDistribution: rows=numClassifiers; 3 columns (numClassifier; positive samples; negative samples)
	*/
void Detector::getSamplesDistribution(Mat &activations, Mat &sampleDistribution){
	sampleDistribution = Mat(activations.cols, 3, CV_32FC1, Scalar(0));
	cv::Mat_<uchar>::iterator it = activations.begin<uchar>();
	cv::Mat_<uchar>::iterator itend = activations.end<uchar>();
	for (int i = 0; i < activations.cols; i++){
		Mat classifierAct = activations.col(i); //take i-th column
		int numPos = std::count(classifierAct.begin<uchar>(), classifierAct.end<uchar>(), 255);
		int numNeg = std::count(classifierAct.begin<uchar>(), classifierAct.end<uchar>(), 0);
		if ((numPos + numNeg) != activations.rows){
			cout << "Positive samples + negative samples do not equal total samples" << endl;
			break;
		}
		sampleDistribution.at<float>(i, 0) = i;			//first number: classifier number
		sampleDistribution.at<float>(i, 1) = numPos;	//second number: number of assigned positive samples
		sampleDistribution.at<float>(i, 2) = numNeg;	//third number: number of assigned negative samples
	}
}

/**Add colorbar to a previously converted image using COLORMAP_HOT*/
void Detector::addColorbar(Mat &image, double min, double max, bool sideShow = true, bool separateShow = false, double x_offset = 10, double y_offset = 10){

	//Define 5 different values to show along colobar y-axis
	string maxstr = format("%.2f", max);
	string minstr = format("%.2f", min);
	string midstr = format("%.2f", (max - min) / 2);
	string topmidstr = format("%.2f", (max - min) * 3 / 4);
	string bottommidstr = format("%.2f", (max - min) / 4);

	//Initialize colobar dimensions and content
	Mat colorbar_gray(256, 65, CV_8U, Scalar(0));
	for (int i = 0; i < colorbar_gray.rows; i++){
		colorbar_gray.row(i) = colorbar_gray.rows - i;
	}

	//Apply colormap and show values
	Mat colorbar;
	applyColorMap(colorbar_gray, colorbar, COLORMAP_HOT);		//apply colormap
	colorbar.colRange(23, 65) = 0;								//set right zone to zero to show values
	rectangle(colorbar, Rect(0, 0, 23, 256), Scalar(255, 255, 255));
	putText(colorbar, maxstr, Point(25, 8), 1, 0.7, Scalar(255, 255, 255));
	putText(colorbar, topmidstr, Point(25, colorbar.rows / 4), 1, 0.7, Scalar(255, 255, 255));
	putText(colorbar, midstr, Point(25, colorbar.rows / 2), 1, 0.7, Scalar(255, 255, 255));
	putText(colorbar, bottommidstr, Point(25, colorbar.rows * 3 / 4), 1, 0.7, Scalar(255, 255, 255));
	putText(colorbar, minstr, Point(25, colorbar.rows - 1), 1, 0.7, Scalar(255, 255, 255));

	//Show colobar in separate window or over original window
	if (sideShow){
		copyMakeBorder(image, image, 0, 0, colorbar.cols + x_offset, 0, BORDER_CONSTANT, Scalar(0));
		addWeighted(image(Rect(x_offset, y_offset, colorbar.cols, colorbar.rows)), 0, colorbar, 1, 0, image(Rect(x_offset, y_offset, colorbar.cols, colorbar.rows)));
	}
	else if (separateShow){
		//namedWindow("Colorbar", CV_WINDOW_NORMAL);
		imshow("Colorbar", colorbar);
	}
	else{
		addWeighted(image(Rect(x_offset, y_offset, colorbar.cols, colorbar.rows)), 0, colorbar, 1, 0, image(Rect(x_offset, y_offset, colorbar.cols, colorbar.rows)));
	}
}

/**Experimental function to get bounding boxes around head center point*/
void Detector::getBBox(Point foundPoint, Point2f center, Point2f offset, vector<Point2f> &bbox){

	center = center + offset;
	double radius = norm((Mat)((Point2f)foundPoint), (Mat)center);
	double width;
	if (radius > 205){
		//width = -89.64996037 * log(radius) + 519.4551902; //totally empirical formula... original!
		width = -89.64996037 * log(radius) + 530; // bigger, to test
	}
	else{
		width = 45;
	}
	double angle = atan2((foundPoint.y - center.y), (foundPoint.x - center.x));
	RotatedRect rect(foundPoint, Size(width, width), angle * 180 / CV_PI);
	Point2f vertices[4];
	rect.points(vertices);
	for (int i = 0; i < 4; i++){
		bbox.push_back(vertices[i]);
		//line(frame, vertices[i], vertices[(i + 1) % 4], Scalar(255, 0, 0), 2);
	}
}

/**Load previously saved main descriptor variables/configuration*/
void Detector::loadDescriptorVariables(string filename){
	FileStorage fs(filename, FileStorage::READ);
	string descriptor = fs["Descriptor"];
	if (descriptor == "HAAR"){
		descriptorType = DescriptorType::HAAR;
		FileNode haarParams = fs["HAAR_params"];
		haarParams["Sliding_window_width"] >> slidingWin_width;
		haarParams["Sliding_window_height"] >> slidingWin_height;
		haarParams["Sliding_window_stride_step_height"] >> sliding_step_h;
		haarParams["Sliding_window_stride_step_width"] >> sliding_step_w;
		haarParams["Final_ROI_size"] >> finalROI_size;
	}
	else if (descriptor == "HOG"){
		descriptorType = DescriptorType::HOG;
		FileNode hogParams = fs["HOG_params"];
		hogParams["Cell_size"] >> hog.cellSize;
		hogParams["Block_size"] >> hog.blockSize;
		hogParams["Block_stride"] >> hog.blockStride;
		hogParams["Number_bins"] >> hog.nbins;
		hogParams["Window_width"] >> hog.winSize.width;
		hogParams["Window_height"] >> hog.winSize.height;
		hogParams["Window_horizontal_stride"] >> step_w;
		hogParams["Window_vertical_stride"] >> step_h;
	}

	fs.release();
}

/**Computes Haar features over image*/
void Detector::computeHaarFeatures(Mat &frame, vector<float> &descriptors, int step_w, int step_h, int overlapping_step_h, int overlapping_step_w){

	for (int i = 0; i < frame.rows - step_h + 1; i += overlapping_step_h)
	{
		for (int j = 0; j < frame.cols - step_w + 1; j += overlapping_step_w)
		{
			Mat imageROI = frame(cv::Rect(j, i, step_w, step_h));	//select image ROI (window size)
			resize(imageROI, imageROI, finalROI_size);				//resize to evaluation window size
			eval.setImage(imageROI, 0, 0);							//set evaluator
			for (int i = 0; i < eval.features.size(); i++){
				float result = eval.operator()(i, 0);
				descriptors.push_back(result);						//save vector of features
			}
		}
	}
}

/**Computes HOG features over image*/
void Detector::computeHOGFeatures(Mat &frame, vector<float> &descriptors, int step_w, int step_h){

	int scale = 1;        //parameter to resize image (pyramid) - first set to 1 (original size)
	int max_scale = 4;    //max scale - resizes original imagen until this maximun is reached
	int scale_factor = 2; //scale factor (scale gets multiplied by this factor); minimum = 2

	do {
		vector<float>winDescriptor;
		hog.compute(frame, winDescriptor, Size(step_w, step_h));
		descriptors.insert(descriptors.end(), winDescriptor.begin(), winDescriptor.end());

		scale = scale * scale_factor;      //update of scale parameter
		pyrDown(frame, frame, Size((frame.cols + 1) / scale_factor, (frame.rows + 1) / scale_factor));     //creation of multiscale pyramid (default = half width, half height)
	} while (scale <= max_scale);
}

/** Given a vector of feature descriptors, compute prediction of array of classifiers and find final detection position
	(based on confidences and location of activated classifiers). Performs non maxima suppresion.
	@params:
	- Mat frame: image to draw detection(s) over
	- vector<float> descriptors: features vector to input into the classifiers
*/
void Detector::detect(Mat &frame, vector<float> descriptors){

	//Get which classifiers output a positive prediction and save confidences
	Mat confidences(1, svm_ptr.size(), CV_32F, Scalar(0));				//row vector to contain confidences

	for (int i = 0; i < svm_ptr.size(); i++){							//for each classifier in the array
		double duration3 = static_cast<double>(getTickCount());
		SVMParams params = (svm_ptr[i])->get_params();
		if (params.svm_type == 102){									//if ONE_CLASS classifier (dummy trained, not used)
			if (showNotTrained == true)
				drawGridPoint(grid, i, frame, Scalar(0, 255, 255));		//draw yellow point
		}
		else {															//if 2-class classifier
			float predicted_value = (svm_ptr[i])->predict((Mat)descriptors, true);		//predicted signed value = confidence
			duration3 = static_cast<double>(cv::getTickCount()) - duration3;
			duration3 /= (cv::getTickFrequency() / 1000);	//duration of processing in ms
			totalPredictionDuration += duration3;
			if (predicted_value < hyperplane_threshold){							//if < hyperplane_threshold, positive prediction
				if (predicted_value < 0){
					confidences.at<float>(i) = abs(predicted_value) + hyperplane_threshold;	//save confidence
				}
				else{
					confidences.at<float>(i) = hyperplane_threshold - abs(predicted_value);	//save confidence
				}
				if (showActivations){
					drawGridPoint(grid, i, frame, Scalar(255, 0, 255));		//draw activated classifier - magenta point
				}
				if (showActivationsAndConfidence){
					drawGridPointAndConfidence(grid, i, frame, Scalar(255, 0, 255), predicted_value);		//alternative, draw point and confidence
				}
				if (mode == TEST){
					realActivated.at<uchar>(numFrame, i) = 255;				//save info of real activated classifiers
				}
			}
		}
	}

	//Generate probability map: matrix containing group confidences of consecutive groups of numNeighbors
	Size nSize = Size(3, 3);															//size of neighborhood
	Mat probMap(numRowsClassifier - nSize.height + 1, numColsClassifier - nSize.width + 1, CV_32FC1, Scalar(0));		//probability map
	confidences = confidences.reshape(0, numRowsClassifier);							//reshape to have numRows x numCols matrix

	for (int i = 0; i < probMap.rows; i++){
		for (int j = 0; j < probMap.cols; j++){
			Mat confROI = confidences(cv::Rect(j, i, nSize.width, nSize.height));        //select image ROI (3x3 submatrix, optimum for 9 neighbors)
			float confSum = sum(confROI)[0];						//confidence sum over neighboorhood	(group confidence)	
			probMap.at<float>(i, j) = confSum;
		}
	}

	//Search local maxima locations over probability map
	threshold(probMap, probMap, minGroupThreshold, 0, THRESH_TOZERO);	//delete group confidences below threshold
	Mat maxMap = probMap.clone();
	dilate(maxMap, maxMap, Mat());										//dilate and subtract to get local maxima
	maxMap = maxMap - probMap;
	threshold(maxMap, maxMap, 0, 255, THRESH_BINARY_INV);				//invert local maxima values
	Mat binaryProbMap;
	threshold(probMap, binaryProbMap, 0, 255, THRESH_BINARY);
	bitwise_and(maxMap, binaryProbMap, maxMap);							//restrict local maxima to appear within probMap

	//Compute final detection position based on local maxima confidences
	float x_position = 0;
	float y_position = 0;
	vector<Point>frameFoundPositions;		//to store all position found in current frame
	float groupConfidence;
	for (int i = 0; i < maxMap.rows; i++){
		for (int j = 0; j < maxMap.cols; j++){
			if (maxMap.at<float>(i, j) == 255){							//for each group confidence local maximum
				Mat confROI = confidences(cv::Rect(j, i, 3, 3));		//retrieve individual confidences of group of classifiers
				groupConfidence = sum(confROI)[0];					
				Mat positionsROI = positions(cv::Rect(j, i, 3, 3));		//retrieve positions of each classifier of the group
				vector<Mat> vecPos;
				split(positionsROI, vecPos);							//split x and y coordinates to different matrices
				Mat weightedXPositions;
				Mat weightedYPositions;

				multiply(confROI, vecPos[0], weightedXPositions);		//weight positions according to each confidence
				multiply(confROI, vecPos[1], weightedYPositions);

				x_position = (sum(weightedXPositions) / sum(confROI))[0];	//find final values
				y_position = (sum(weightedYPositions) / sum(confROI))[0];

				//store and draw locations
				if ((x_position > 0) || (y_position > 0)){		//found points must be different from (0,0)
					Point foundPoint = Point(y_position, x_position);
					if (std::find(std::begin(frameFoundPositions), std::end(frameFoundPositions), foundPoint) == std::end(frameFoundPositions)){
						frameFoundPositions.push_back(foundPoint);	//only save point if it has NOT been previously saved
					}
					line(frame, foundPoint, foundPoint, Scalar(255, 127, 0), 8);	//draws the point on output image
					//drawPointAndConfidence(foundPoint, frame, Scalar(255, 127, 0), groupConfidence);	//optionally show group confidence

					////alternative - get and draw bounding box around detected heads
					//vector<Point2f> bbox;
					//getBBox(foundPoint, Point2f(frame.cols / 2, frame.rows / 2), Point2f(5, 10), bbox);
					//for (int i = 0; i < bbox.size(); i++)
					//	line(frame, bbox[i], bbox[(i + 1) % bbox.size()], Scalar(255,0,0), 2);
				}
			}
		}
	}

	//If in TEST mode, save per frame detections to a global matrix, for later comparison with ground truth info
	if (mode == TEST){
		if (frameFoundPositions.empty()){	//save dummy point to frame detections if no point was found
			float zeroNum = 0;
			frameFoundPositions.push_back(Point(zeroNum, zeroNum));
		}
		//Include frame detected points in global matrix (foundPositions), making sure the dimensions are appropiate
		Mat temp = (Mat)frameFoundPositions;
		temp = temp.t();
		if (!foundPositions.empty() && temp.cols > foundPositions.cols){
			Mat emptyColumns(foundPositions.rows, temp.cols - foundPositions.cols, CV_32SC(2), Scalar(0));
			hconcat(foundPositions, emptyColumns, foundPositions);
		}
		if (temp.cols < foundPositions.cols){
			Mat emptyColumns(1, foundPositions.cols - temp.cols, CV_32SC(2), Scalar(0));
			hconcat(temp, emptyColumns, temp);
		}
		foundPositions.push_back(temp);		//save detections in global matrix
	}
}

/**Alternative to previous function to fix computation with circular grid - not optimized (further improvement is possible) - TO REVIEW*/
void Detector::detect_circular(Mat &frame, vector<float> descriptors){

	//Get which classifiers output a positive prediction
	Mat confidences(frame.rows, frame.cols, CV_32F, Scalar(0));			//matrix to contain confidences

	for (int i = 0; i < svm_ptr.size(); i++){
		CvSVMParams params = (svm_ptr[i])->get_params();
		if (params.svm_type == 102){					//if ONE_CLASS classifier
			if (showNotTrained == true)
				drawGridPoint(grid, i, frame, Scalar(0, 255, 255));		//draw yellow point
		}
		else {															//if 2-class classifier
			float predicted_value = (svm_ptr[i])->predict((Mat)descriptors, true);		//predicted signed value = confidence
			if (predicted_value < hyperplane_threshold){							//if < 0, positive prediction
				if (predicted_value < 0){
					float svm_x_coord = grid.at<float>(i, 1);
					float svm_y_coord = grid.at<float>(i, 0);
					if ((svm_y_coord < confidences.rows) && (svm_x_coord < confidences.cols))
						confidences.at<float>(svm_y_coord, svm_x_coord) = abs(predicted_value) + hyperplane_threshold;	//save confidence
				}
				else{
					confidences.at<float>(i) = hyperplane_threshold - abs(predicted_value);
				}
				if (showActivations){
					drawGridPoint(grid, i, frame, Scalar(255, 0, 255));		//draw activated classifier - magenta point
				}
				if (showActivationsAndConfidence){
					drawGridPointAndConfidence(grid, i, frame, Scalar(255, 0, 255), predicted_value);		//alternative, draw point and confidence
				}
				if (mode == TEST){
					realActivated.at<uchar>(numFrame, i) = 255;				//save info of real activated classifiers
				}
			}
		}
	}

	//Generate probability map: matrix containing group confidences of consecutive groups of numNeighbors
	Size box_size = Size(100, 100);		//size of the box used to group neighbors
	Mat probMap(confidences.rows - box_size.width + 1, confidences.cols - box_size.height + 1, CV_32FC1, Scalar(0));		//probability map

	for (int i = 0; i < probMap.rows; i += 10){
		for (int j = 0; j < probMap.cols; j += 10){
			Mat confROI = confidences(cv::Rect(j, i, box_size.width, box_size.height));        //select image ROI
			float confSum = sum(confROI)[0];									//confidence sum over neighboorhood	(group confidence)	
			probMap.at<float>(i, j) = confSum;
		}
	}

	//Search local maxima locations over probability map
	threshold(probMap, probMap, minGroupThreshold, 0, THRESH_TOZERO);	//delete group confidences below threshold
	Mat maxMap = probMap.clone();
	dilate(maxMap, maxMap, Mat(box_size.width, box_size.height, CV_32F));	//dilate and subtract to get local maxima
	maxMap = maxMap - probMap;
	threshold(maxMap, maxMap, 0, 255, THRESH_BINARY_INV);				//invert local maxima values
	Mat binaryProbMap;
	threshold(probMap, binaryProbMap, 0, 255, THRESH_BINARY);
	bitwise_and(maxMap, binaryProbMap, maxMap);							//restrict local maxima to appear within probMap

	//Compute final detection position based on local maxima confidences
	float x_position = 0;
	float y_position = 0;
	vector<Point>frameFoundPositions;
	float groupConfidence;
	for (int i = 0; i < maxMap.rows; i++){
		for (int j = 0; j < maxMap.cols; j++){
			if (maxMap.at<float>(i, j) == 255){							//for each group confidence local maximum
				Mat ROI = confidences(cv::Rect(j, i, box_size.width, box_size.height));		//retrieve individual confidences of group of classifiers
				Mat confidences2(confidences.rows, confidences.cols, CV_32F, Scalar(0));
				Mat confROI2 = confidences2(cv::Rect(j, i, box_size.width, box_size.height));
				addWeighted(ROI, 1, confROI2, 0, 0, confROI2);
				groupConfidence = sum(ROI)[0];

				threshold(confidences2, confidences2, 0, 255, THRESH_BINARY);	//convert to binary image
				confidences2.convertTo(confidences2, CV_8U);
				Mat confCoordinates;
				findNonZero(confidences2, confCoordinates);
				float weightedXPositions = 0;
				float weightedYPositions = 0;
				for (int i = 0; i < confCoordinates.total(); i++) {
					float x_coord = confCoordinates.at<Point>(i).x;
					float y_coord = confCoordinates.at<Point>(i).y;
					weightedXPositions += x_coord * confidences.at<float>(y_coord, x_coord);
					weightedYPositions += y_coord * confidences.at<float>(y_coord, x_coord);
				}

				x_position = (weightedXPositions) / (sum(ROI)[0]);	//find final values
				y_position = (weightedYPositions) / (sum(ROI)[0]);

				//store and draw locations
				if ((x_position > 0) || (y_position > 0)){
					Point foundPoint = Point(x_position, y_position);
					if (std::find(std::begin(frameFoundPositions), std::end(frameFoundPositions), foundPoint) == std::end(frameFoundPositions)){
						frameFoundPositions.push_back(foundPoint);
					}
					line(frame, foundPoint, foundPoint, Scalar(255, 127, 0), 8);	//draws the point on output image
					//drawPointAndConfidence(foundPoint, frame, Scalar(255, 127, 0), groupConfidence);
				}
			}
		}
	}
	//If in TEST mode, save per frame detections to a global matrix, for latter comparison with ground truth info
	if (mode == TEST){
		if (frameFoundPositions.empty()){
			float zeroNum = 0;
			frameFoundPositions.push_back(Point(zeroNum, zeroNum));
		}
		Mat temp = (Mat)frameFoundPositions;
		temp = temp.t();
		if (!foundPositions.empty() && temp.cols > foundPositions.cols){
			Mat emptyColumns(foundPositions.rows, temp.cols - foundPositions.cols, CV_32SC(2), Scalar(0));
			hconcat(foundPositions, emptyColumns, foundPositions);
		}
		if (temp.cols < foundPositions.cols){
			Mat emptyColumns(1, foundPositions.cols - temp.cols, CV_32SC(2), Scalar(0));
			hconcat(temp, emptyColumns, temp);
		}
		foundPositions.push_back(temp);
	}
}

/**Retrieve connected components (countours) given a binary input image*/
void Detector::getConnectedComp(Mat &image, vector<vector<Point>> &contours){

	//realize morphological operations to reduce noise and get bigger connected components
	Mat image2 = image.clone();		//clone input image to preserve it
	morphologyEx(image2, image2, MORPH_OPEN, Mat(), Point(-1, -1), 3);	//opening
	morphologyEx(image2, image2, MORPH_CLOSE, Mat(), Point(-1, -1), 3);	//closing

	//find contours
	findContours(image2, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);	//only external contours (not nested ones)
}

/**Given a set of contours, get a series of matrices which contain a mask for each contour
	@params:
	- (in) Size frame_size: size of original image (where contours were retrieved)
	- (in) vector<vector<Point>> countours: previously retrieved contours
	- (out) vector<Mat> seg_masks: collection of matrices, each containing a mask corresponding to one of the contours
*/
void Detector::getSegmentationMasks(Size frame_size, vector<vector<Point> > &contours, vector<Mat> &seg_masks){

	if (!contours.empty()){
		// iterate through all the top-level contours,
		// choose only biggest ones, and fill them to create individual segmentation masks per connected component
		for (int idx = 0; idx < contours.size(); idx++){
			if (contourArea(contours[idx]) > 50){			//only take into account contours bigger than 50 (manually chosen threshold)
				Mat mask = Mat::zeros(frame_size, CV_8U);
				drawContours(mask, contours, idx, Scalar(255), CV_FILLED, 8);
				seg_masks.push_back(mask);
			}
		}
	}
}

/**Given a set of contours, get and draw the centroid of each one*/
void Detector::getCentroids(Size frame_size, vector<vector<Point> > &contours){
	if (!contours.empty()){
		Mat mask = Mat::zeros(frame_size, CV_8UC3);

		// Get the moments for contours greater than threshold
		vector<Moments> mu;
		for (int idx = 0; idx < contours.size(); idx++){
			if (contourArea(contours[idx]) > 250){		//threshold = 250, manually chosen
				mu.push_back(moments(contours[idx]));
				drawContours(mask, contours, idx, Scalar(255, 255, 255), CV_FILLED, 8);
			}
		}

		// Get the mass centroids
		vector<Point> centroids(mu.size());
		for (int i = 0; i < mu.size(); i++){
			centroids[i] = Point(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
		}

		// Draw centroids
		for (int i = 0; i < centroids.size(); i++){
			line(mask, centroids[i], centroids[i], Scalar(0, 0, 255), 6);
		}
		imshow("Centroids", mask);
	}
}

/**Get the number of true/false positives/negatives detected (based on individual SVM metrics)*/
void Detector::getMetrics(Mat groundTruthActivations, Mat realActivations, int &TP, int& FP, int& TN, int& FN){
	if (groundTruthActivations.size() != realActivations.size()){
		cout << "Error - ground truth info size does not match real detected info size" << endl; exit(-1);
	}
	TP = 0;
	FP = 0;
	TN = 0;
	FN = 0;
	cv::Mat_<float>::iterator it = groundTruthActivations.begin<float>();
	cv::Mat_<float>::iterator itend = groundTruthActivations.end<float>();
	cv::Mat_<uchar>::iterator it2 = realActivations.begin<uchar>();
	for (; it != itend; ++it){
		if (((*it) == 255) && ((*it2) == 255)){ TP += 1; }		//true positive
		else if (((*it) == 0) && ((*it2) == 255)) { FP += 1; }	//false positive
		else if (((*it) == 0) && ((*it2) == 0)) { TN += 1; }	//true negative
		else if (((*it) == 255) && ((*it2) == 0)) { FN += 1; }	//false negative
		++it2;
	}
}

void getDistanceHistogram(Mat &distances){
	Mat max_distance(1,1,CV_32FC1);
	reduce(distances, max_distance, 0, CV_REDUCE_MAX, CV_32FC1);
	
	/// Establish the number of bins
	int histSize = 100;

	/// Set the range
	float range[] = { 0, (max_distance.at<float>(0))+2 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat hist;

	/// Compute the histograms:
	calcHist(&distances, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

	// Draw the histogram
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i), hist_h),
			Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
			Scalar(0, 0, 255), 1, CV_AA, 0);
	}

	//Add axes to image
	copyMakeBorder(histImage, histImage, 20, 40, 40, 20, BORDER_CONSTANT, Scalar(255, 255, 255));
	line(histImage, Point(35, histImage.rows - 41), Point(histImage.cols-20, histImage.rows - 41), Scalar(0, 0, 0), 1);
	line(histImage, Point(40, 20), Point(40, histImage.rows-35), Scalar(0, 0, 0), 1);

	putText(histImage, "0", Point(35, histImage.rows-15), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 0), 1);
	putText(histImage, to_string(max_distance.at<float>(0)), Point(histImage.cols - 45, histImage.rows-15), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 0), 1);

	/// Display
	namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
	imshow("calcHist Demo", histImage);
	waitKey(1);
}

/**Old function to get error based in distances - still here just for convenience*/
void Detector::getError2(string gtfilename, Mat positionsFound){
	int notDetectedPoints = 0;		//false negative detections
	int extraDetectedPoints = 0;	//false positive detections	
	Mat temp;
	loadCsv(gtfilename, temp);
	Mat gtposition = temp.colRange(1, temp.cols);		//eliminate first column (frame number)
	gtposition = gtposition.reshape(2);					//x and y coordinates in two separate channels
	Mat distancesError;

	//Variables for error statistics per sector
	vector<float> error_per_sector;				//to save error per sector
	error_per_sector.resize(sectors.size(), 0);
	vector<int> count;
	count.resize(error_per_sector.size(), 1);	//to count how many errors are inputted per sector (to later calculate mean)
	vector<float> notDetected_per_sector;
	notDetected_per_sector.resize(sectors.size(), 0);	//to save points not detected per sector

	//Compute error between ground truth and real detected points
	for (int i = 0; i < gtposition.rows; i++){			//take each gt row (gt for each frame)
		Mat frameGtPositions = gtposition.row(i);			//ground truth
		Mat frameFoundPositions = positionsFound.row(i);	//found detections
		Mat res1 = frameGtPositions.reshape(1); Mat res2 = frameFoundPositions.reshape(1);
		int numGtPoints = ceil(countNonZero(res1) / 2);
		int numFoundPoints = ceil(countNonZero(res2) / 2);
		notDetectedPoints += (numGtPoints - numFoundPoints) > 0 ? numGtPoints - numFoundPoints : 0;
		extraDetectedPoints += (numGtPoints - numFoundPoints) < 0 ? numFoundPoints - numGtPoints : 0;
		cv::Mat_<Vec2f>::iterator it = frameGtPositions.begin<Vec2f>();
		cv::Mat_<Vec2f>::iterator itend = frameGtPositions.end<Vec2f>();
		cv::Mat_<Vec2i>::iterator it2 = frameFoundPositions.begin<Vec2i>();
		cv::Mat_<Vec2i>::iterator itend2 = frameFoundPositions.end<Vec2i>();
		vector<float>frameDistancesError;
		for (; it != itend; ++it){
			double distance = INFINITY;
			if ((*it)[0] != 0 || (*it)[1] != 0)		{	//if not (0,0) in ground truth
				for (; it2 != itend2; ++it2){
					if ((*it2)[0] != 0 || (*it2)[1] != 0){	//if not (0,0) in found positions
						double new_distance;
						vector<float> first_operand = { (*it)[0], (*it)[1] };
						vector<float> second_operand = { (float)(*it2)[0], (float)(*it2)[1] };
						new_distance = norm(first_operand, second_operand, NORM_L2);
						if (new_distance < distance){
							distance = new_distance;
						}
					}
				}
				if (distance != INFINITY){
					frameDistancesError.push_back(distance);
					vector<int> indexes = findSector(sectors, Point((*it)[0], (*it)[1]));
					for (int j = 0; j < indexes.size(); j++){
						error_per_sector[indexes[j]] += distance;
						count[indexes[j]] += 1;
					}
				}
				else {		//save not detected points - ONLY VALID WHEN THERE'S JUST ONE PERSON IN SCENE!
					vector<int> indexes = findSector(sectors, Point((*it)[0], (*it)[1]));
					for (int j = 0; j < indexes.size(); j++){
						notDetected_per_sector[indexes[j]] += 1;
					}
				}
			}
		}
		distancesError.push_back((Mat)frameDistancesError);
	}


	//Per sector statistics (error distances)
	for (int i = 0; i < error_per_sector.size(); i++){	//take mean of error in every sector
		error_per_sector[i] = error_per_sector[i] / count[i];
	}
	double min; double max;
	minMaxLoc(error_per_sector, &min, &max);
	cout << "Minimum mean error in a sector: " << min << " / Maximum mean error in a sector: " << max << endl;

	normalize(error_per_sector, error_per_sector, 0, 255, NORM_MINMAX);
	Mat sectorimg(frame_size.height, frame_size.width, CV_8U, Scalar(0));
	for (int i = 0; i < sectors.size(); i++){
		drawContours(sectorimg, sectors, i, Scalar(error_per_sector[i]), -1);
		drawContours(sectorimg, sectors, i, Scalar(255), 1);
	}
	Mat sectorimgcolor;
	applyColorMap(sectorimg, sectorimgcolor, COLORMAP_HOT);
	addColorbar(sectorimgcolor, min, max);
	imshow("Error per sector", sectorimgcolor);
	imwrite("error_per_sector.png", sectorimgcolor);
	waitKey();

	//not detected per sector
	minMaxLoc(notDetected_per_sector, &min, &max);
	normalize(notDetected_per_sector, notDetected_per_sector, 0, 255, NORM_MINMAX);
	Mat notDetsectorimg(frame_size.height, frame_size.width, CV_8U, Scalar(0));
	for (int i = 0; i < sectors.size(); i++){
		drawContours(notDetsectorimg, sectors, i, Scalar(notDetected_per_sector[i]), -1);
		drawContours(notDetsectorimg, sectors, i, Scalar(255), 1);
	}
	Mat notDetsectorimgcolor;
	applyColorMap(notDetsectorimg, notDetsectorimgcolor, COLORMAP_HOT);
	addColorbar(notDetsectorimgcolor, min, max);
	imshow("Not detected per sector", notDetsectorimgcolor);
	imwrite("notDetected_per_sector.png", notDetsectorimgcolor);
	waitKey();


	//Global statistics
	Mat meanErrorDistance; Mat standardDev;
	meanStdDev(distancesError, meanErrorDistance, standardDev);
	cout << "Mean error distance of all positions (between ground truth and found point): " 
		 << std::setprecision(4) << meanErrorDistance.at<float>(0) << endl;
	cout << "Standard deviation of the error distance of all positions: " << standardDev.at<float>(0) << endl;

	cout << "Total number of detected positions (True positives): " << distancesError.rows << endl;
	cout << "Total number of wrong extra detected positions (False positives): " << extraDetectedPoints << endl;
	cout << "Total number of undetected positions (False negatives): " << notDetectedPoints << endl;
}

/**Alternative function to get error based on overlapped bounding boxes*/
void Detector::getError3(string gtfilename, Mat positionsFound){
	int truePositives = 0;
	int notDetectedPoints = 0;		//false negative detections
	int extraDetectedPoints = 0;	//false positive detections	
	Mat temp;
	loadCsv(gtfilename, temp);
	Mat gtposition = temp.colRange(1, temp.cols);		//eliminate first column (frame number)
	gtposition = gtposition.reshape(2);					//x and y coordinates in two separate channels
	Mat distancesError;

	//Variables for error statistics per sector
	vector<float> error_per_sector;				//to save error per sector
	error_per_sector.resize(sectors.size(), 0);
	vector<int> count;
	count.resize(error_per_sector.size(), 1);	//to count how many errors are inputted per sector (to later calculate mean)
	vector<float> notDetected_per_sector;
	notDetected_per_sector.resize(sectors.size(), 0);	//to save points not detected per sector

	//Compute error between ground truth and real detected points
	for (int i = 0; i < gtposition.rows; i++){			//take each gt row (gt for each frame)
		Mat frameGtPositions = gtposition.row(i);			//ground truth
		Mat frameFoundPositions = positionsFound.row(i);	//found detections

		cv::Mat_<Vec2f>::iterator it = frameGtPositions.begin<Vec2f>();
		cv::Mat_<Vec2i>::iterator it2 = frameFoundPositions.begin<Vec2i>();

		//count number of gt points and found points per frame
		Mat sumGT(1, frameGtPositions.cols, CV_32FC1, Scalar(0));

		cv::Mat_<float>::iterator it3 = sumGT.begin<float>();
		cv::Mat_<float>::iterator it3end = sumGT.end<float>();

		Mat sumFound(1, frameFoundPositions.cols, CV_32FC1, Scalar(0));

		cv::Mat_<float>::iterator it4 = sumFound.begin<float>();
		cv::Mat_<float>::iterator it4end = sumFound.end<float>();

		for (; it3 != it3end; ++it3){
			(*it3) = (*it)[0] + (*it)[1];
			++it;
		}
		int numGtPoints = countNonZero(sumGT);

		for (; it4 != it4end; ++it4){
			(*it4) = (*it2)[0] + (*it2)[1];
			++it2;
		}
		int numFoundPoints = countNonZero(sumFound);

		//notDetectedPoints += (numGtPoints - numFoundPoints) > 0 ? numGtPoints - numFoundPoints : 0;
		//extraDetectedPoints += (numGtPoints - numFoundPoints) < 0 ? numFoundPoints - numGtPoints : 0;


		//match GT bounding boxes and detected bounding boxes
		vector<int> matchedGTPoints; matchedGTPoints.resize(numGtPoints, 0);
		vector<int> matchedFoundPoints; matchedFoundPoints.resize(numFoundPoints, 0);

		it = frameGtPositions.begin<Vec2f>();
		cv::Mat_<Vec2f>::iterator itend = frameGtPositions.end<Vec2f>();
		it2 = frameFoundPositions.begin<Vec2i>();
		cv::Mat_<Vec2i>::iterator itend2 = frameFoundPositions.end<Vec2i>();
		double min_ratio = 0.5;	//0.5
		int FoundIdx = 0;
		for (; it2 != itend2; ++it2){
			if ((*it2)[0] != 0 || (*it2)[1] != 0)		{	//if not (0,0) in found positions
				double ratio = 0;
				vector<Point2f>FoundBbox;
				int GtIdx = 0; int previousGtIdx = -1;
				getBBox(Point((*it2)[0], (*it2)[1]), Point2f(frame_size.width / 2, frame_size.height / 2), Point2f(5, 10), FoundBbox);
				for (; it != itend; ++it){
					if ((*it)[0] != 0 || (*it)[1] != 0){	//if not (0,0) in ground truth
						vector<Point2f>GtBbox;
						getBBox(Point((*it)[0], (*it)[1]), Point2f(frame_size.width / 2, frame_size.height / 2), Point2f(5, 10), GtBbox);

						vector<Point2f> intersection;
						intersectConvexConvex(GtBbox, FoundBbox, intersection);
						double gtarea = contourArea(GtBbox);
						double foundarea = contourArea(FoundBbox);
						double intersectarea;
						if (!intersection.empty()){
							intersectarea = contourArea(intersection);
						}
						else{
							intersectarea = 0;
						}
						double unionarea = gtarea + foundarea - intersectarea;
						double temp_ratio = intersectarea / unionarea;
						if ((temp_ratio > ratio) && (temp_ratio > min_ratio) && (matchedGTPoints[GtIdx] == 0)){
							if (previousGtIdx != -1){
								matchedGTPoints[previousGtIdx] = 0;
							}
							matchedGTPoints[GtIdx] = 1;
							//matchedFoundPoints[FoundIdx] = 1;
							previousGtIdx = GtIdx;
							ratio = temp_ratio;
						}
					}
					GtIdx += 1;
				}
			}
			FoundIdx += 1;
		}
		truePositives += countNonZero(matchedGTPoints);
		notDetectedPoints += matchedGTPoints.size() - countNonZero(matchedGTPoints);
		if ((matchedFoundPoints.size() - countNonZero(matchedGTPoints)) > 0){
			extraDetectedPoints += matchedFoundPoints.size() - countNonZero(matchedGTPoints);
		}
	}

	cout << "Total number of detected positions (True positives): " << truePositives << endl;
	cout << "Total number of wrong extra detected positions (False positives): " << extraDetectedPoints << endl;
	cout << "Total number of undetected positions (False negatives): " << notDetectedPoints << endl;
}

/**Current used function to get error - based on distances, every gt point and found point can be matched once at most*/
void Detector::getError4(string gtfilename, Mat positionsFound, int &truePositives, int &extraDetectedPoints, int &notDetectedPoints){
	double distanceThreshold = 15;	//global threshold for defining which found points are true positives and which are false positives
	truePositives = 0;			//true positive detections
	notDetectedPoints = 0;		//false negative detections
	extraDetectedPoints = 0;	//false positive detections	
	Mat temp;
	loadCsv(gtfilename, temp);
	Mat gtposition = temp.colRange(1, temp.cols);		//eliminate first column (frame number)
	gtposition = gtposition.reshape(2);					//x and y coordinates in two separate channels
	Mat distancesError;

	//Variables for error statistics per sector
	vector<float> error_per_sector;				//to save error per sector
	error_per_sector.resize(sectors.size(), 0);
	vector<float> count;
	count.resize(error_per_sector.size(), 1);	//to count how many errors are inputted per sector (to later calculate mean error)
	vector<float> notDetected_per_sector;
	notDetected_per_sector.resize(sectors.size(), 0);	//to save points not detected per sector
	vector<float> extraDetected_per_sector;
	extraDetected_per_sector.resize(sectors.size(), 0);	//to save points extra detected per sector

	//Compute error between ground truth and real detected points
	for (int i = 0; i < gtposition.rows; i++){			//take each gt row (gt for each frame)
		Mat frameGtPositions = gtposition.row(i);			//ground truth
		Mat frameFoundPositions = positionsFound.row(i);	//found detections

		//first, count number of gt points and found points per frame (ignore (0,0) points)
		cv::Mat_<Vec2f>::iterator it = frameGtPositions.begin<Vec2f>();
		cv::Mat_<Vec2i>::iterator it2 = frameFoundPositions.begin<Vec2i>();

		Mat sumGT(1, frameGtPositions.cols, CV_32FC1, Scalar(0));
		cv::Mat_<float>::iterator it3 = sumGT.begin<float>();
		cv::Mat_<float>::iterator it3end = sumGT.end<float>();
		for (; it3 != it3end; ++it3){
			(*it3) = (*it)[0] + (*it)[1];
			++it;
		}
		int numGtPoints = countNonZero(sumGT);

		Mat sumFound(1, frameFoundPositions.cols, CV_32FC1, Scalar(0));
		cv::Mat_<float>::iterator it4 = sumFound.begin<float>();
		cv::Mat_<float>::iterator it4end = sumFound.end<float>();
		for (; it4 != it4end; ++it4){
			(*it4) = (*it2)[0] + (*it2)[1];
			++it2;
		}
		int numFoundPoints = countNonZero(sumFound);

		//match GT points and detected points
		vector<int> matchedGTPoints; matchedGTPoints.resize(numGtPoints, 0);
		vector<int> matchedFoundPoints; matchedFoundPoints.resize(numFoundPoints, 0);

		it = frameGtPositions.begin<Vec2f>();		//reset iterator position to begin()
		it2 = frameFoundPositions.begin<Vec2i>();	//reset iterator position to begin()
		cv::Mat_<Vec2f>::iterator itend = frameGtPositions.end<Vec2f>();
		cv::Mat_<Vec2i>::iterator itend2 = frameFoundPositions.end<Vec2i>();

		vector<Point>GtNonZeroPoints;		//vector with frame GT points coordinates different from zero
		for (; it != itend; ++it){
			if ((*it)[0] != 0 || (*it)[1] != 0){	//if not (0,0) in ground truth
				GtNonZeroPoints.push_back(Point((*it)[0], (*it)[1]));
			}
		}

		int FoundIdx = 0;
		vector<float>frameDistancesError;
		for (; it2 != itend2; ++it2){
			double distance = INFINITY;
			if ((*it2)[0] != 0 || (*it2)[1] != 0)		{	//if not (0,0) in found positions
				int GtIdx = 0; int previousGtIdx = -1;
				it = frameGtPositions.begin<Vec2f>();		//reset iterator position to begin()
				Point associatedGtPoint = Point((*it)[0], (*it)[1]);
				for (; it != itend; ++it){
					if ((*it)[0] != 0 || (*it)[1] != 0){	//if not (0,0) in ground truth
						double new_distance;
						vector<float> first_operand = { (*it)[0], (*it)[1] };
						vector<float> second_operand = { (float)(*it2)[0], (float)(*it2)[1] };
						new_distance = norm(first_operand, second_operand, NORM_L2);

						if ((new_distance < distance) && (new_distance < distanceThreshold) && (matchedGTPoints[GtIdx] == 0)){
							if (previousGtIdx != -1){
								matchedGTPoints[previousGtIdx] = 0;
							}
							matchedGTPoints[GtIdx] = 1;
							previousGtIdx = GtIdx;
							distance = new_distance;
							associatedGtPoint = Point((*it)[0], (*it)[1]);
						}
					}
					GtIdx += 1;
				}
				if (distance != INFINITY){
					frameDistancesError.push_back(distance);
					vector<int> indexes = findSector(sectors, associatedGtPoint);
					for (int j = 0; j < indexes.size(); j++){
						error_per_sector[indexes[j]] += distance;
						count[indexes[j]] += 1;
					}
				}
				else {		//save extra detected points
					vector<int> indexes = findSector(sectors, Point((*it2)[0], (*it2)[1]));
					for (int j = 0; j < indexes.size(); j++){
						extraDetected_per_sector[indexes[j]] += 1;
					}
				}
			}
		}
		//save not detected points
		for (int i = 0; i < matchedGTPoints.size(); i++){
			if (matchedGTPoints[i] == 0){
				Point notDetectedGtPoint = GtNonZeroPoints[i];
				vector<int> indexes = findSector(sectors, notDetectedGtPoint);
				for (int j = 0; j < indexes.size(); j++){
					notDetected_per_sector[indexes[j]] += 1;
				}
			}
		}


		//numerical values for TP, FP, FN
		truePositives += countNonZero(matchedGTPoints);
		notDetectedPoints += matchedGTPoints.size() - countNonZero(matchedGTPoints);
		if ((matchedFoundPoints.size() - countNonZero(matchedGTPoints)) > 0){
			extraDetectedPoints += matchedFoundPoints.size() - countNonZero(matchedGTPoints);
		}
		distancesError.push_back((Mat)frameDistancesError);
	}

	//Find sectors with at least one ground truth annotation (1TP or 1FN)
	Mat mask1 = ((Mat)count != 1);
	Mat mask2 = ((Mat)notDetected_per_sector != 0);
	Mat groundtruth = mask1 | mask2;
	bitwise_not(groundtruth, groundtruth);
	Mat	off_sectors(frame_size.height, frame_size.width, CV_8UC3, Scalar(0));
	for (int i = 0; i < sectors.size(); i++){
		drawContours(off_sectors, sectors, i, Scalar((groundtruth.at<uchar>(i)),(groundtruth.at<uchar>(i)) / 6, (groundtruth.at<uchar>(i)) / 6), -1);
		drawContours(off_sectors, sectors, i, Scalar(255, 255, 255), 1);
	}
	//imshow("inactive sectors", off_sectors);
	//waitKey(1);

	//Per sector statistics (error distances)
	for (int i = 0; i < error_per_sector.size(); i++){	//take mean of error in every sector
		error_per_sector[i] = error_per_sector[i] / count[i];
	}
	double min; double max;
	minMaxLoc(error_per_sector, &min, &max);
	cout << "\n* Sectors statistics ***************************" << endl;
	cout << "* - Minimum mean error in a sector: " << min << endl;
	cout << "* - Maximum mean error in a sector: " << max << endl;

	normalize(error_per_sector, error_per_sector, 0, 255, NORM_MINMAX);
	Mat sectorimg(frame_size.height, frame_size.width, CV_8U, Scalar(0));
	for (int i = 0; i < sectors.size(); i++){
		drawContours(sectorimg, sectors, i, Scalar(error_per_sector[i]), -1);
		drawContours(sectorimg, sectors, i, Scalar(255), 1);
	}
	Mat sectorimgcolor;
	applyColorMap(sectorimg, sectorimgcolor, COLORMAP_HOT);
	add(off_sectors, sectorimgcolor, sectorimgcolor);
	addColorbar(sectorimgcolor, min, max);
	imshow("Error per sector", sectorimgcolor);
	imwrite("error_per_sector.png", sectorimgcolor);
	waitKey(1);

	//not detected per sector (FN)
	minMaxLoc(notDetected_per_sector, &min, &max);
	normalize(notDetected_per_sector, notDetected_per_sector, 0, 255, NORM_MINMAX);
	Mat notDetsectorimg(frame_size.height, frame_size.width, CV_8U, Scalar(0));
	for (int i = 0; i < sectors.size(); i++){
		drawContours(notDetsectorimg, sectors, i, Scalar(notDetected_per_sector[i]), -1);
		drawContours(notDetsectorimg, sectors, i, Scalar(255), 1);
	}
	Mat notDetsectorimgcolor;
	applyColorMap(notDetsectorimg, notDetsectorimgcolor, COLORMAP_HOT);
	addColorbar(notDetsectorimgcolor, min, max);
	imshow("Not detected per sector", notDetsectorimgcolor);
	imwrite("notDetected_per_sector.png", notDetsectorimgcolor);
	waitKey(1);

	//extra detected per sector (FP)
	minMaxLoc(extraDetected_per_sector, &min, &max);
	normalize(extraDetected_per_sector, extraDetected_per_sector, 0, 255, NORM_MINMAX);
	Mat extraDetsectorimg(frame_size.height, frame_size.width, CV_8U, Scalar(0));
	for (int i = 0; i < sectors.size(); i++){
		drawContours(extraDetsectorimg, sectors, i, Scalar(extraDetected_per_sector[i]), -1);
		drawContours(extraDetsectorimg, sectors, i, Scalar(255), 1);
	}
	Mat extraDetsectorimgcolor;
	applyColorMap(extraDetsectorimg, extraDetsectorimgcolor, COLORMAP_HOT);
	addColorbar(extraDetsectorimgcolor, min, max);
	imshow("Extra detected per sector", extraDetsectorimgcolor);
	imwrite("extraDetected_per_sector.png", extraDetsectorimgcolor);
	waitKey(1);

	//recall per sector (TP/(TP+FN))
	Mat recall = ((Mat)count-1) / (((Mat)count-1) + (Mat)notDetected_per_sector);
	minMaxLoc(recall, &min, &max);
	normalize(recall, recall, 0, 255, NORM_MINMAX);
	Mat recallsectorimg(frame_size.height, frame_size.width, CV_8U, Scalar(0));
	for (int i = 0; i < sectors.size(); i++){
		drawContours(recallsectorimg, sectors, i, Scalar(recall.at<float>(i)), -1);
		drawContours(recallsectorimg, sectors, i, Scalar(255), 1);
	}
	Mat recallsectorimgcolor;
	applyColorMap(recallsectorimg, recallsectorimgcolor, COLORMAP_HOT);
	add(off_sectors, recallsectorimgcolor, recallsectorimgcolor);
	addColorbar(recallsectorimgcolor, min, max);
	imshow("Recall per sector", recallsectorimgcolor);
	imwrite("recall_per_sector.png", recallsectorimgcolor);
	waitKey(1);

	//precision per sector (TP/(TP+FP))
	Mat precision = ((Mat)count-1) / (((Mat)count-1) + (Mat)extraDetected_per_sector);
	minMaxLoc(precision, &min, &max);
	normalize(precision, precision, 0, 255, NORM_MINMAX);
	Mat precisionsectorimg(frame_size.height, frame_size.width, CV_8U, Scalar(0));
	for (int i = 0; i < sectors.size(); i++){
		drawContours(precisionsectorimg, sectors, i, Scalar(precision.at<float>(i)), -1);
		drawContours(precisionsectorimg, sectors, i, Scalar(255), 1);
	}
	Mat precisionsectorimgcolor;
	applyColorMap(precisionsectorimg, precisionsectorimgcolor, COLORMAP_HOT);
	add(off_sectors, precisionsectorimgcolor, precisionsectorimgcolor);
	addColorbar(precisionsectorimgcolor, min, max);
	imshow("Precision per sector", precisionsectorimgcolor);
	imwrite("precision_per_sector.png", precisionsectorimgcolor);
	waitKey(1);

	//Global statistics
	cout << "\n* Global statistics *******************************************************" << endl;
	Mat meanErrorDistance; Mat standardDev;
	meanStdDev(distancesError, meanErrorDistance, standardDev);
	cout << "* - Error distance between ground truth and associated found point" << endl;
	cout << "* -- Mean error distance of all positions: " << std::fixed << std::setprecision(4) << meanErrorDistance.at<double>(0) << endl;
	cout << "* -- Standard deviation of the error distance of all positions: " << std::fixed << std::setprecision(4) << standardDev.at<double>(0) << endl;

	cout << "* - Total number of detected positions (TP): " << truePositives << endl;
	cout << "* - Total number of wrong extra detected positions (FP): " << extraDetectedPoints << endl;
	cout << "* - Total number of undetected positions (FN): " << notDetectedPoints << endl;

	double _precision = (double)truePositives / (double)(truePositives + extraDetectedPoints);
	double _recall = (double)truePositives / (double)(truePositives + notDetectedPoints);
	double _fscore = (2*_precision*_recall) / (_precision + _recall);

	cout << "* - Precision: " << _precision << endl;
	cout << "* - Recall: " << _recall << endl;
	cout << "* - F1-Score: " << _fscore << endl;

	//Distance histogram (half-done yet)
	//getDistanceHistogram(distancesError.clone());
}

/**Generation Preprocess: creation of grid, load of ground truth info, association of ground truth and classifiers*/
void Detector::preprocessGeneration(){

	//-- Load ground truth info of all input frames
	loadCsv(groundTruthFilename, groundTruth);
	totalFrames = groundTruth.rows;

	//-- Create classifiers' grid
	if (gridType == RECTANGULAR){
		createGrid(numRowsClassifier, numColsClassifier, frame_size, grid);
		numNeighbors = 9;
	}
	else if (gridType == QUINCUNX){
		createQuincunxGrid(numRowsClassifier, numColsClassifier, frame_size, grid);
		numNeighbors = 7;
	}
	else if (gridType == CIRCULAR){
		createCircularGrid(frame_size, grid, 5, 10);	//center offset is a tentative value (camera calibration needed for exact value)
		numNeighbors = 8;
	}

	//-- Prepare division of image in sectors, to compute number of ground truth points per sector (distribution statistics)
	getSectorsCirc(sectors, frame_size, 5, 10);		//circular division with predefined number of sectors (set inside function)
	//getSectorsQuinc(sectors, grid, frame_size, numRowsClassifier, numColsClassifier);		//alternative division with quincuncial pattern
	//getSectorsRect(sectors, frame_size, numRowsClassifier, numColsClassifier);			//alternative division with rectangular pattern
	gt_per_sector.resize(sectors.size());			//resize vector to number of sectors

	//-- Save main variables (pattern, #classifiers...) for later use (not fully implemented yet)
	saveMainVariables();

	//-- Initialize activations matrix (matrix to store which classifiers should be active in each frame)
	activations = Mat(totalFrames, grid.rows, CV_8UC1, Scalar(0));
}

/**Generation Process (per frame processing)*/
void Detector::processGeneration(cv::Mat &frame, cv::Mat &output){

	drawGrid(grid, frame, Scalar(0, 255, 0));	//draw grid of classifiers in green
	vector<Point> gt_points;
	drawGT(frame, gt_points);					//draw ground truth on original frame and save gt points
	associateGT(frame, numNeighbors);			//find nearest neighbors, and save them in activations matrix

	//get distribution of GT points over sectors (to show diagram)
	for (int i = 0; i < gt_points.size(); i++){
		vector<int> indexes = findSector(sectors, gt_points[i]);
		for (int j = 0; j < indexes.size(); j++){
			gt_per_sector[indexes[j]] += 1;
		}
	}

	//if last frame of sequence
	if (numFrame == totalFrames - 1){

		//Show distribution of ground truth points over image sectors
		double min; double max;
		minMaxLoc(gt_per_sector, &min, &max);

		normalize(gt_per_sector, gt_per_sector, 0, 7500, NORM_MINMAX);			//normalize, with saturation of higher values
		Mat sectorimg(frame_size.height, frame_size.width, CV_8U, Scalar(0));
		for (int i = 0; i < sectors.size(); i++){
			drawContours(sectorimg, sectors, i, Scalar(gt_per_sector[i]), -1);	//draw sector values
			drawContours(sectorimg, sectors, i, Scalar(255), 1);
		}
		Mat sectorimgcolor;
		applyColorMap(sectorimg, sectorimgcolor, COLORMAP_HOT);					//apply colormap for better visualization
		addColorbar(sectorimgcolor, min, max);									//add colobar
		imshow("GT per sector", sectorimgcolor);								//show and save results
		imwrite("gt_per_sector.png", sectorimgcolor);
		waitKey(0);

		//save activations matrix ( positive/negative samples per classifier)
		saveMatToCsv(activations, "activations.csv");			//save activations matrix

		//get number of positive and negative samples per classifier for statistical purposes
		Mat sampleDistribution;
		getSamplesDistribution(activations, sampleDistribution);
		saveMatToCsv(sampleDistribution, "sampleDistribution.csv");	//save sampleDistribution matrix
		exit(0);	//exit program
	}
}

/** Detection Preprocess: setting of chosen feature extractor, creation of grid, load array of previously saved classifiers*/
void Detector::preprocessDetection(){

	//load and save main parameters of descriptor extractor (as set during training)
	string descriptorParamsFile = trainedModelsFolder + "/descriptorParams.yml";
	loadDescriptorVariables(descriptorParamsFile);

	//initialize parameters depending of chosen descriptor
	if (descriptorType == HAAR){
		//Setting of Background Subtractor - we want a "bad" subtraction with integrated shadows
		int history = 30;
		float threshold = 16;
		bool bShadowDetection = false;	//don't realize shadow detection
		bg = BackgroundSubtractorMOG2(history, threshold, bShadowDetection);
		//bg.setInt("nShadowDetection", 0);		//uncomment to perform shadow detection (shadows as black pixels)

		//Setting of feature extractor - HAAR
		CvFeatureParams f;
		Ptr<CvFeatureParams> haarParams = f.create(0);
		eval.init(haarParams, 1, finalROI_size);
	}
	else if (descriptorType == HOG){		//normalization is only done with HOG descriptors (HAAR descriptors are internally normalized)
		Mat meansigma;
		string file = trainedModelsFolder + "/meansigma.csv";
		loadCsv(file, meansigma);
		means = meansigma.col(0).clone();
		sigmas = meansigma.col(1).clone();
	}

	//Load main grid params (saved on generation)
	string gridParamsFile = trainedModelsFolder + "/gridParams.yml";
	loadMainVariables(gridParamsFile);

	//Create classifiers' grid
	if (gridType == RECTANGULAR){
		createGrid(numRowsClassifier, numColsClassifier, frame_size, grid);
	}
	else if (gridType == QUINCUNX){
		createQuincunxGrid(numRowsClassifier, numColsClassifier, frame_size, grid);
	}
	else if (gridType == CIRCULAR){
		createCircularGrid(frame_size, grid, 5, 10);
	}

	//initialize array of classifiers
	setSVMs(grid);

	//Load each previously trained classifier
	string filename = trainedModelsFolder + "/svmNum";
	for (int i = 0; i < svm_ptr.size(); i++){
		string model_name = filename + to_string(i) + ".yml";		//full name of every trained model
		std::ifstream infile(model_name);
		//check if SVM has been trained (filename exists). If not, load a default dummy one_class SVM model (not used later)
		if (infile.good()){
			(svm_ptr[i])->load(model_name.c_str());
			trainedClassifiers += 1;
		}
		else{
			string one_class = filename + "_oneclass.yml";			//model named "svmNum_oneclass.yml"
			(svm_ptr[i])->load(one_class.c_str());
		}
		cout << "Loading SVM classifiers from pre-trained models... " << i << "/" << svm_ptr.size() << "\r" << std::flush;
	}
	cout << "Loading SVM classifiers from pre-trained models... Done!\t\t" << endl;
}

/** Detection Process (per frame processing)*/
void Detector::processDetection(cv::Mat &frame, cv::Mat &output){
	double duration = static_cast<double>(cv::getTickCount());
	double duration2 = duration;

	//for HAAR descriptors
	if (descriptorType == HAAR){
		//do background subtraction
		float learningRate = 0.01;							//set to 0.01 to avoid flickering
		bg.operator ()(frame, output, learningRate);	//computes foreground image
		//imwrite("bgmask_orig" + to_string(numFrame) + ".png", output);

		duration2 = static_cast<double>(cv::getTickCount()) - duration2;
		duration2 /= (cv::getTickFrequency() / 1000);	//duration of processing in ms
		totalBgsubDuration += duration2;

		//Optionally show connected components' centroids baseline
		if (showCentroids){
			vector<vector<Point> > contours2;
			getConnectedComp(output, contours2);
			getCentroids(output.size(), contours2);
		}

		//processing if segmentation masks are not used
		if (!useMasks) {
			double duration5 = static_cast<double>(getTickCount());
			vector< float > descriptors;
			computeHaarFeatures(output, descriptors, slidingWin_width, slidingWin_height, sliding_step_h, sliding_step_w);
			duration5 = static_cast<double>(cv::getTickCount()) - duration5;
			duration5 /= (cv::getTickFrequency() / 1000);	//duration of processing in ms
			totalFeatExtractionDurationHAAR += duration5;

			if (gridType == CIRCULAR){
				detect_circular(frame, descriptors);
			}
			else{
				detect(frame, descriptors);
			}
		}
		//processing if segmentation masks are used
		else{
			vector<vector<Point> > contours;
			vector<Mat> seg_masks;
			getConnectedComp(output, contours);
			getSegmentationMasks(output.size(), contours, seg_masks);
			for (int i = 0; i < seg_masks.size(); i++){
				Mat temp_output;
				output.copyTo(temp_output);							//temporarily replicate output (foreground mask)
				bitwise_and(output, seg_masks[i], temp_output);		//and filter with each of the masks

				//detect over each of the new filtered foreground masks
				vector< float > descriptors;
				computeHaarFeatures(output, descriptors, slidingWin_width, slidingWin_height, sliding_step_h, sliding_step_w);

				if (gridType == CIRCULAR){
					detect_circular(frame, descriptors);
				}
				else{
					detect(frame, descriptors);
				}
			}
		}
	}
	//for HOG descriptors
	else{
		double duration4 = static_cast<double>(getTickCount());
		vector< float > descriptors;
		Mat gray = frame.clone();
		cvtColor(gray, gray, CV_BGR2GRAY);
		computeHOGFeatures(gray, descriptors, step_w, step_h);

		//normalize descriptors prior to classification
		for (int idx = 0; idx < descriptors.size(); idx++){
			float mean = means.at<float>(idx);
			float sigma = sigmas.at<float>(idx);
			descriptors[idx] = (descriptors[idx] - mean) / sigma;
		}
		duration4 = static_cast<double>(cv::getTickCount()) - duration4;
		duration4 /= (cv::getTickFrequency() / 1000);	//duration of processing in ms
		totalFeatExtractionDurationHOG += duration4;

		if (gridType == CIRCULAR){
			detect_circular(frame, descriptors);
		}
		else{
			detect(frame, descriptors);
		}
	}
	

	//Optionally draw classifiers grid
	if (showGrid){
		drawGrid(grid, frame, Scalar(0, 255, 0));			//draw grid of classifiers in green
	}

	//Print execution time
	duration = static_cast<double>(cv::getTickCount()) - duration;
	duration /= (cv::getTickFrequency() / 1000);	//duration of processing in ms
	totalDuration += duration;
	if(verbose)
		cout << "Frame "<< numFrame << " - Elapsed time: " << duration << " ms" << endl;
}

/**Testing Preprocess: load theorical activations and real activations per frame, get sectors to compute statistics*/
void Detector::preprocessTest(){

	//load of activations matrix previously saved
	loadCsv(activationsFile, activations);
	//initialize matrix to save truly activated classifiers per frame
	realActivated = Mat(activations.rows, activations.cols, CV_8UC1, Scalar(0));

	totalFrames = activations.rows;

	//get sectors to later saved statistics
	getSectorsCirc(sectors, frame_size, 5, 10);
	//getSectorsRect(sectors, frame_size, numRowsClassifier, numColsClassifier);

	totalTP = 0; totalFP = 0; totalTN = 0; totalFN = 0;		//initialize metrics variales
}

/**Testing Process (per frame processing)*/
void Detector::processTest(Mat &frame, Mat &output){

	//update evaluation metrics (based on individual classifiers performance)
	int TP, FP, TN, FN;
	Mat frameGT = activations.row(numFrame);
	Mat frameRealAct = realActivated.row(numFrame);
	getMetrics(frameGT, frameRealAct, TP, FP, TN, FN);

	totalTP += TP;	totalFP += FP; totalTN += TN; totalFN += FN;

	//if last frame of sequence
	if (numFrame == totalFrames - 1){		
		//Processing time
		cout << "\n\n* Time profiling (ms) ********************************" << endl;
		double averageTime = totalDuration / totalFrames;
		cout << "* - Average frame processing time: " << averageTime << endl;
		double averageBgsubTime = totalBgsubDuration / totalFrames;
		cout << "* -- Average BgSub frame processing time: " << averageBgsubTime << endl;
		if (descriptorType == DescriptorType::HAAR){
			double averageFeatExtTime = totalFeatExtractionDurationHAAR / totalFrames;
			cout << "* -- Average Feature extraction processing time: " << averageFeatExtTime << endl;
		}
		else if (descriptorType == DescriptorType::HOG){
			double averageFeatExtTime = totalFeatExtractionDurationHOG / totalFrames;
			cout << "* -- Average Feature extraction processing time: " << averageFeatExtTime << endl;
		}
		double averagePredictionTime = totalPredictionDuration / totalFrames;
		cout << "* -- Average prediction frame processing time: " << averagePredictionTime << endl;
		cout << "* --- Average prediction time per classifier: " << averagePredictionTime/(double)trainedClassifiers << endl;
		
		//Get and show error based on distances between ground truth points and detected points
		int truePositives, falsePositives, falseNegatives;
		getError4(groundTruthFilename, foundPositions, truePositives, falsePositives, falseNegatives);

		
		////Optional - Show global metrics based on individual performance of classifier
		//cout << "* Classifiers metrics (based on individual activations) *" << endl;
		//cout << "True positives: " << totalTP << endl;
		//cout << "False positives: " << totalFP << endl;
		//cout << "True negatives: " << totalTN << endl;
		//cout << "False negatives: " << totalFN << endl;
		//double precision, recall;
		//if (totalTP == 0){
		//	precision = 0;
		//	recall = 0;
		//}
		//else{
		//	precision = double(totalTP) / double(totalTP + totalFP);
		//	recall = double(totalTP) / double(totalTP + totalFN);
		//}
		//cout << "Precision: " << precision << endl;
		//cout << "Recall: " << recall << endl;
		//cout << "F1-Score: " << 2 * precision*recall / (precision + recall) << endl;

		////Optional - calculate and save performance metrics of each of the classifiers
		//	Mat svmMetrics(svm_ptr.size(), 3, CV_32FC1);
		//	for (int i = 0; i < svm_ptr.size(); i++){
		//		int TP2 = 0;
		//		int FP2 = 0;
		//		int TN2 = 0;
		//		int FN2 = 0;
		//		Mat svmGT = activations.col(i);
		//		Mat svmRealAct = realActivated.col(i);
		//		getMetrics(svmGT, svmRealAct, TP2, FP2, TN2, FN2);
		//		double precision = double(TP2) / double(TP2 + FP2);
		//		double recall = double(TP2) / double(TP2 + FN2);
		//		svmMetrics.at<float>(i, 0) = i;
		//		svmMetrics.at<float>(i, 1) = precision;
		//		svmMetrics.at<float>(i, 2) = recall;
		//		saveMatToCsv(svmMetrics, "metricsPerSVM.csv");
		//		/*cout << "SVM #" << i << endl;
		//		cout << "True positives: " << TP << endl;
		//		cout << "False positives: " << FP << endl;
		//		cout << "True negatives: " << TN << endl;
		//		cout << "False negatives: " << FN << endl;
		//		cout << "Precision: " << precision << endl;
		//		cout << "Recall: " << recall << endl;*/
		//}

		////Optionally save stats in .csv, allowing batch testing
		//Mat savedStats;
		//string filename = "savedStats.csv";
		//loadCsv(filename, savedStats);
		//float stats[] = { hyperplane_threshold, minGroupThreshold, truePositives, falsePositives, falseNegatives };
		//vector<float> v_stats(stats, stats + sizeof(stats) / sizeof(float));
		//Mat m_stats = (Mat)v_stats;
		//m_stats = m_stats.t();
		//savedStats.push_back(m_stats);
		//saveMatToCsv(savedStats, filename);

	}
}

/**Preprocess (general, before frame by frame processing): loading and configuring parameters*/
void Detector::preprocess(){
	cout << "Configuring parameters..." << endl;
	if (mode == DETECTION){
		preprocessDetection();
	}
	else if (mode == TEST){
		preprocessDetection();
		preprocessTest();
	}
	else if (mode == GENERATION){
		preprocessGeneration();
	}
	numFrame = 0; //initialize frame counter
	if (saveVideo){
		writer.open(videoName, CV_FOURCC('X', 'V', 'I', 'D'), 10, frame_size);
		if (!writer.isOpened()){
			cout << "Error opening output video file! Video will not be saved" << endl;
		}
	}
	cout << "Frame by frame processing started!" << endl;
}

/**Process (general, per frame processing method)*/
void Detector::process(cv::Mat &frame, cv::Mat &output) {
	if (mode == DETECTION){
		processDetection(frame, output);
	}
	else if (mode == TEST){
		processDetection(frame, output);
		if (numFrame > 75){					//don't start test functions immediately due to possible background subtraction initialization
			processTest(frame, output);
		}
	}
	else if (mode == GENERATION){
		processGeneration(frame, output);
	}

	//Save video (if set)
	if (saveVideo){
		writer << frame;
	}
	
	//Update frame number and exit program if last frame (just for generation and test modes, detection runs infinitely)
	if (numFrame == totalFrames-1){
		cout << "\nPress any key to exit...";
		waitKey(0);
		exit(0);
	}

	//Print frame number (if not printed before)
	if (!verbose){
		cout << "Frame " << numFrame << "\r";
	}
	numFrame += 1;
}