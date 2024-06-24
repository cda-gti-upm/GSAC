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
#if !defined FRAME_PROCESSOR
#define FRAME_PROCESSOR

// The frame processor interface
class FrameProcessor {

public:
	//configuration of FrameProcessor instace before frame by frame processing
	virtual void preprocess() = 0;

	// processing method
	virtual void process(cv::Mat &input, cv::Mat &output) = 0;

};

#endif