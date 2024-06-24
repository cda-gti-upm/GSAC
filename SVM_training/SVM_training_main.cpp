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
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>

#include <fstream>

#include "haar_source/haarfeatures.h"
#include "haar_source/traincascade_features.h"

using namespace cv;
using namespace std;

CvMLData csv;	//structure to load from .csv file

static void help(){
	cout
		<< "**************************************************************************\n"
		<< "* SPATIAL GRID OF FOVEATIC CLASSIFIERS - SVM MODELS TRAINER\n"
		<< "*\n"
		<< "* Trainer for the Spatial Grid of Foveatic Classifiers detector.\n"
		<< "* Requires an \"activations\" file previously generated with main application.\n"
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
		<< "** General common mandatory params:\n"
		<< "*    -input <number of folders> <list of folder paths> : input folders\n"
		<< "*           containing training images. Indicate total number of folders,\n"
		<< "*           and then list of folders (separated by spaces)\n"
		<< "*    -activations <number of files> <list of file paths> : files with saved\n"
		<< "*                 activations matrix. Indicate total number of files, and\n"
		<< "*                 then list of files (separated by spaces)\n"
		<< "*    -descriptor <HOG|HAAR> : descriptor to extract features\n"
		<< "*    -num_classifier <number> : first classifier to train (default 0 trains\n"
		<< "*                    all classifiers) (optional parameter)\n"
		<< "*\n"
		<< "** HOG specific params:\n"
		<< "*    -win_size <width height> : window size to compute features\n"
		<< "*    -win_step <width height> : sliding step of previous window\n"
		<< "*\n"
		<< "** HAAR specific params:\n"
		<< "*    -win_size <width height> : original window size to compute features\n"
		<< "*    -win_step <width height> : sliding step of previous window\n"
		<< "*    -haar_final_win <width height> : final window to compute features (resized)"			
		<< "*\n\n"
		<< " Example of usage:\n"
		<< "   SVM_training.exe -input 2 F:/sequence_training1 F:/sequence_training2\n"
		<< "   -activations 2 F:/activations1.csv F:/activations2.csv -descriptor HOG\n"
		<< "   -win_size 24 24 -win_step 8 8\n\n"
		<< "*******************************************************************************\n"
		<< endl;
}


void readDirectory(const string& directoryName, vector<string>& filenames);
void compute_hog(Mat &img, Mat& descriptor, HOGDescriptor hog, int step_w, int step_h);
void compute_haar(Mat& img, Mat& descriptor, CvHaarEvaluator eval, int step_w, int step_h, int overlapping_step, Size finalROI_size);
void train_svm(const vector< Mat > & gradient_lst, const vector< int > & labels, const string& output_model_name);
void loadCsv(string filename, Mat &data);
void saveMatToCsv(Mat &matrix, string filename);

/**Read and store filenames from directoryName folder*/
void readDirectory(const string& directoryName, vector<string>& filenames)
{
	glob(directoryName, filenames, false);
}

/**Computes HOG descriptors over whole image img*/
void compute_hog(Mat &img, Mat& descriptor, HOGDescriptor hog, int step_w, int step_h){

	Mat image = img.clone();
	int scale = 1;					//parameter to resize image (pyramid) - first set to 1 (original size)
	int max_scale = 4;				//max scale - resizes original imagen until this maximum is reached
	int scale_factor = 2;			//scale factor (scale gets multiplied by this factor); minimum = 2
	vector< float > descriptors;
	do {
		vector<float>winDescriptor;
		hog.compute(image, winDescriptor, Size(step_w, step_h));   //computes HOG descriptor
		descriptors.insert(descriptors.end(), winDescriptor.begin(), winDescriptor.end());

		scale = scale * scale_factor;      //update of scale parameter
		pyrDown(image, image, Size((image.cols + 1) / scale_factor, (image.rows + 1) / scale_factor));     //creation of multiscale pyramid (default = half width, half height)
	} while (scale <= max_scale);

	descriptor.push_back(Mat(descriptors).clone());
}

/**Computes HAAR descriptors over whole image img*/
void compute_haar(Mat& img, Mat& descriptor, CvHaarEvaluator eval, int step_w, int step_h, int overlapping_step, Size finalROI_size){

	Mat gray = img.clone();
	vector< float > descriptors;

	for (int i = 0; i < gray.rows - step_h + 1; i += overlapping_step)
	{
		for (int j = 0; j < gray.cols - step_w + 1; j += overlapping_step)
		{
			Mat imageROI = gray(cv::Rect(j, i, step_w, step_h));  //select image ROI (window size)
			resize(imageROI, imageROI, finalROI_size);
			eval.setImage(imageROI, 0, 0);
			for (int i = 0; i < eval.features.size(); i++){
				float result = eval.operator()(i, 0);
				descriptors.push_back(result);
			}
		}
	}

	descriptor.push_back(Mat(descriptors).clone());
}

/**Train SVM classifier according to descriptors and labels, and saves model*/
void train_svm(int type, Mat& gradient_lst, vector< int > labels, const string& output_model_name){

	clog << "Start training...";
	SVM svm;
	CvSVMParams params;
	/* Default values to train SVM */
	//params.coef0=0.0;
	//params.degree=0;
	//params.term_crit = TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 20000, 1e-7);
	//params.gamma = 0;
	params.kernel_type = SVM::LINEAR;
	params.nu = 0.5;
	//params.p = 0.1; // for EPSILON_SVR, epsilon in loss function?
	params.svm_type = type; // C_SVC=100 //NU_SVC=101; ONE_CLASS=102; // EPSILON_SVR=103; // may be also NU_SVR=104; // do regression task

	//// -- Use weights to optimize unbalanced datasets (much more negatives than positive samples)
	//double positives = countNonZero(labels);	//count positive samples
	//double negatives = labels.size() - positives;	//count negative samples
	//double neg_weight = positives / labels.size();
	//double pos_weight = 1 - neg_weight;
	//vector<double> weights{ neg_weight, pos_weight };
	//vector<double> weights{ 0.005, 0.995 };
	//CvMat class_weights = Mat(weights);
	//params.class_weights = &class_weights;

	params.C = 0.01; // From paper, soft classifier C=0.01
	svm.train(gradient_lst, (Mat)labels, Mat(), Mat(), params);

	//// -- adaptive C with cross-validation
	//ParamGrid Cgrid = ParamGrid(0.01, 1, (10 / 3));
	//double positives = countNonZero(labels);	//count positive samples
	//int k_fold = round(sqrt(positives));
	//if (k_fold < 2){ k_fold = 2; }

	//svm.train_auto(gradient_lst, (Mat)labels, Mat(), Mat(), params, k_fold, Cgrid, CvSVM::get_default_grid(CvSVM::GAMMA), CvSVM::get_default_grid(CvSVM::P), CvSVM::get_default_grid(CvSVM::NU), CvSVM::get_default_grid(CvSVM::COEF), CvSVM::get_default_grid(CvSVM::DEGREE), true);

	clog << "...[done]" << endl;

	svm.save(output_model_name.c_str());
}

/**Load .csv contents to a Mat*/
void loadCsv(string filename, Mat &data){
	if (csv.read_csv(filename.c_str()) == -1){
		cout << "Error - .csv file couldn't be loaded" << endl; exit(-1);
	};
	const CvMat* data_ptr = csv.get_values();  //matrix with .csv data
	data = Mat(data_ptr);
}

/**Save a Mat into a .csv file*/
void saveMatToCsv(Mat &matrix, string filename){
	ofstream outputFile(filename);
	outputFile << format(matrix, "CSV") << endl;
	outputFile.close();
}

/**Save main HOG descriptor variables in a .yml to later reuse them*/
void saveDescriptorVariablesHOG(HOGDescriptor hog, int step_w, int step_h){

	FileStorage fs("descriptorParams.yml", FileStorage::WRITE);
	fs << "Descriptor" << "HOG";
	fs << "HOG_params";
	fs << "{";
	fs << "Cell_size" << hog.cellSize;
	fs << "Block_size" << hog.blockSize;
	fs << "Block_stride" << hog.blockStride;
	fs << "Number_bins" << hog.nbins;
	fs << "Window_width" << hog.winSize.width;
	fs << "Window_height" << hog.winSize.height;
	fs << "Window_horizontal_stride" << step_w;
	fs << "Window_vertical_stride" << step_h;
	fs << "}";

	fs.release();
}

/**Save main HAAR descriptor variables in a .yml to later reuse them*/
void saveDescriptorVariablesHAAR(int slidingWin_width, int slidingWin_height, int sliding_step, Size finalROI_size){

	FileStorage fs("descriptorParams.yml", FileStorage::WRITE);
	fs << "Descriptor" << "HAAR";
	fs << "HAAR_params";
	fs << "{";
	fs << "Sliding_window_width" << slidingWin_width;
	fs << "Sliding_window_height" << slidingWin_height;
	fs << "Sliding_window_stride_step" << sliding_step;
	fs << "Final_ROI_size" << finalROI_size;
	fs << "}";

	fs.release();
}

int main(int argc, char** argv)
{
	help();		//show program info

	double duration = static_cast<double>(cv::getTickCount());

	// -- Main input parameters

	int firstClassifier = 0;	//first classifier to train (set to a higher number to resume or train specific classifiers)

	//Input folders containing training images
	vector<string> inputFolders = { "F:/sequences/sequence2_training",
									"F:/sequences/sequence2_training_center" };

	//Files with activations matrices, previously computed with GENERATION mode
	vector<string> activationsFiles = { "F:/sequences/sequence2_activations7q_v4_825_training.csv",
										"F:/sequences/sequence2_activations7q_v4_825_training_center.csv"};

	//Descriptor extractor (HOG or HAAR) and its properties (with default values)
	string descriptorType = "HOG";

	//- for HOG -
	int hog_window_width = 16;		//hog window width
	int hog_window_height = 16;		//hog window height
	int step_w = hog_window_width;		//horizontal stride of hog window
	int step_h = hog_window_height;	//vertical stride of hog window

	//- for HAAR -
	int haar_slidingWin_width = 96;		//haar computing window width
	int haar_slidingWin_height = 72;	//haar conputing window height
	int haar_sliding_step = 64;			//haar computing sliding step
	Size haar_finalROI_size = Size(32, 24);		//haar final ROI size (resized from computing size)



	//-- Command line arguments parsing
	for (int i = 1; i < argc; i++)
	{
		if (!strcmp(argv[i], "--help"))
		{
			help2();
			cout << "Press enter to exit...";
			cin.ignore();
			return 0;
		}
		if (!strcmp(argv[i], "-input"))
		{
			inputFolders.clear();
			int number_inputs = stoi(argv[++i]);	//number of input folder (if training images are split in several folders)
			for (int n = 0; n < number_inputs; n++){
				inputFolders.push_back(argv[++i]);
			}
		}
		else if (!strcmp(argv[i], "-activations"))
		{
			activationsFiles.clear();
			int number_activations = stoi(argv[++i]);	//number of activations files (if training images are split in several folders)
			for (int n = 0; n < number_activations; n++){
				activationsFiles.push_back(argv[++i]);
			}
		}
		else if (!strcmp(argv[i], "-descriptor"))
		{
			descriptorType = string(argv[i+1]);
		}
		else if (!strcmp(argv[i], "-win_size"))
		{
			if (descriptorType == "HOG"){
				hog_window_width = stoi(argv[++i]);
				hog_window_height = stoi(argv[++i]);
			}
			else if (descriptorType == "HAAR"){
				haar_slidingWin_width = stoi(argv[++i]);
				haar_slidingWin_height = stoi(argv[++i]);
			}
		}
		else if (!strcmp(argv[i], "-win_step"))
		{
			if (descriptorType == "HOG"){
				step_w = stoi(argv[++i]);
				step_h = stoi(argv[++i]);
			}
			else if (descriptorType == "HAAR"){
				haar_sliding_step = stoi(argv[++i]);
			}
		}
		else if (!strcmp(argv[i], "-haar_final_win"))
		{
			if (descriptorType == "HAAR"){
				int w = stoi(argv[++i]);
				int h = stoi(argv[++i]);
				haar_finalROI_size = Size(w, h);
			}
		}
		else if (!strcmp(argv[i], "-num_classifier"))
		{
			firstClassifier = stoi(argv[++i]);
		}
	}

	// -- Load input training images from folders
	vector<string> filenames;
	for (int i = 0; i < inputFolders.size(); i++){
		vector<string> temp_filenames;
		readDirectory(inputFolders[i], temp_filenames);
		for (int j = 0; j < temp_filenames.size(); j++){
			filenames.push_back(temp_filenames[j]);
		}
	}

	// -- Load activation files
	Mat activations;

	for (int i = 0; i < activationsFiles.size(); i++){
		Mat temp_activations;
		loadCsv(activationsFiles[i], temp_activations);
		activations.push_back(temp_activations);
	}


	// -- Initialize descriptor extractor (HOG or HAAR)

		//for HOG -----------------------------------------------------------------------------------
		//HOGDescriptor hog(Size(), Size(16,16), Size(8,8), Size(8,8), 9);	//alternative constructor
		HOGDescriptor hog;			//hog instance
		hog.winSize = Size(hog_window_width, hog_window_height);

		//for HAAR ----------------------------------------------------------------------------------
		CvFeatureParams f;
		Ptr<CvFeatureParams> haarParams = f.create(0);		//pointer to HaarFeatureParams object
		CvHaarEvaluator eval;
		eval.init(haarParams, 1, haar_finalROI_size);

		int history = 30;
		float threshold = 16;
		bool bShadowDetection = false;	//don't realize shadow detection
		BackgroundSubtractorMOG2 bg = BackgroundSubtractorMOG2(history, threshold, bShadowDetection);
		float learningRate = 0.01;							//set to 0.01 to avoid flickering

	if (descriptorType == "HOG") { saveDescriptorVariablesHOG(hog, step_w, step_h); }
	else if (descriptorType == "HAAR") { saveDescriptorVariablesHAAR(haar_slidingWin_width, haar_slidingWin_height, haar_sliding_step, haar_finalROI_size); }


	// -- Read training images and compute descriptors
	Mat train_features;		//matrix containing descriptors (each row = one image descriptor)

	for (int i = 0; i < filenames.size(); i++){
		Mat read_image = imread(filenames[i], 0);				//read grayscale image
		Mat descriptors;
		if (descriptorType == "HOG"){
			compute_hog(read_image, descriptors, hog, step_w, step_h);	//compute and save HOG descriptors
		}
		else{
			bg.operator ()(read_image, read_image, learningRate);	//computes foreground image
			compute_haar(read_image, descriptors, eval, haar_slidingWin_width, haar_slidingWin_height, haar_sliding_step, haar_finalROI_size);		//compute and save HAAR descriptors
		}
		if (i == 0){
			train_features = Mat((int)filenames.size(), descriptors.rows, CV_32FC1);	//reserve size of matrix in first iteration
		}
		Mat tmp = descriptors.t();				//transpose to get row vector
		tmp.copyTo(train_features.row(i));

		cout << "Reading frame " << i << "\r";
	}

	// -- Normalization of data (only for HOG descriptors - HAAR descriptors are internally normalized)
	if (descriptorType == "HOG"){
		cout << "\nNormalizing data..." << endl;
		Mat means;
		Mat sigmas;
		for (int i = 0; i < train_features.cols; i++){
			Mat mean; Mat sigma;
			meanStdDev(train_features.col(i), mean, sigma);
			means.push_back(mean);
			if (countNonZero(sigma) < 1){
				sigma = 1;		//to prevent division by zero
			}
			sigmas.push_back(sigma);
			train_features.col(i) = (train_features.col(i) - mean) / sigma;
			cout << "Normalizing feature " << i << "\r";
		}
		Mat meansigma;
		hconcat(means, sigmas, meansigma);
		cout << "\nSaving mean and sigma parameters in file...";
		saveMatToCsv(meansigma, "meansigma.csv");
		cout << "Done!" << endl;
	}


	// -- Assign training labels and train
	// --- Get info from activations matrix (previously computed in GENERATION mode of main application)
	int numTotalSamples = activations.rows;
	int numClassifiers = activations.cols;

	// --- For each of the SVM classifiers in activations matrix, assign positive and negative samples and train
	for (int i = firstClassifier; i < numClassifiers; i++){
		//assign corresponding labels
		vector< int > labels(numTotalSamples, 0);
		for (int j = 0; j < numTotalSamples; j++){
			if (activations.at<float>(j, i) == 0){		//sample is negative for current classifier
				labels[j] = 0;
			}
			else{										//sample is positive for current classifier
				labels[j] = +1;
			}
		}
		// --- Train classifier
		string output_model_name = "svmNum" + to_string(i) + ".yml";
		cout << "SVM #" << i << endl;
		int type;
		if (std::find(labels.begin(), labels.end(), 1) != labels.end()) {
			type = 100;		//if both positive and negative samples, use C_SVC type for training
			train_svm(type, train_features, labels, output_model_name);
		}
		else {
			type = 102;		//if only negative samples, use ONE_CLASS classifier - not trained!
		}
	}


	// -- Print execution time
	duration = static_cast<double>(cv::getTickCount()) - duration;
	duration /= (cv::getTickFrequency() / 1000);	//duration of processing in ms
	cout << "Frame processing elapsed time: " << duration << " ms" << endl;
	cin.ignore();

	return 0;
}

