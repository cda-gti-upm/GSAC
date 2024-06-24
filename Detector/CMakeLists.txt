#---------------------------------------------------#
#          CMake Project Configuration              #
#---------------------------------------------------#
# CMake will issue a warning if you don't set this
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

if(COMMAND cmake_policy)
     cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

SET( PROJECT_NAME "SGFC_detector" CACHE STRING "Spatial Grid of Foveatic Classifiers detector")
PROJECT( ${PROJECT_NAME} )

set_property( GLOBAL PROPERTY USE_FOLDERS On)

#---------------------------------------------------#
#              OpenCV Configuration                 #
#---------------------------------------------------#
# Automatically find OpenCV files
FIND_PACKAGE( OpenCV 2.4.8 REQUIRED )	#recommended minimum version = 2.4.8

#---------------------------------------------------#
#              Project Configuration                #
#---------------------------------------------------#
SET ( CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules )
SET ( CMAKE_DEBUG_POSTFIX d )
SET ( LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin/ CACHE PATH "Single output directory for building all libraries." FORCE )
SET ( EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin/ CACHE PATH "Single output directory for building all executables." FORCE )
SET ( CMAKE_CONFIGURATION_TYPES "Debug;Release;" CACHE STRING "" FORCE )

INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS})

#---------------------------------------------------#
#              Control version                 		#
#---------------------------------------------------#
SET(APP_VERSION_MAJOR 1)	# 1: First version
SET(APP_VERSION_MINOR 0)	# 0: First version
SET(APP_VERSION_PATCH 0)	# 0: First version
SET(APP_VERSION_STATUS "Alpha")

ADD_DEFINITIONS(-DAPP_VERSION_MAJOR=${APP_VERSION_MAJOR} 
						    -DAPP_VERSION_MINOR=${APP_VERSION_MINOR}
							-DAPP_VERSION_PATCH=${APP_VERSION_PATCH}
							-DAPP_VERSION_STATUS="${APP_VERSION_STATUS}" )

#---------------------------------------------------#
#    		Project files     		  				#
#---------------------------------------------------#
SET( PROJECT_SRCS
main.cpp
Detector.h
Detector.cpp
haar_source/features.cpp
haar_source/haarfeatures.cpp
frameProcessor.h
Processor.h
Processor.cpp
)
#Optional
SET ( PROJECT_RCS
Detector.rc 	#resource for icon
icon.ico		#icon
)

#---------------------------------------------------#
# 		EXECUTABLE AND LIBRARIES	    			#
#---------------------------------------------------#
ADD_EXECUTABLE( ${PROJECT_NAME} ${PROJECT_SRCS} ${PROJECT_RCS})#src/icon_resource.rc)
TARGET_LINK_LIBRARIES( ${PROJECT_NAME} ${OpenCV_LIBS})
