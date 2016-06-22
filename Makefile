
TEEM_BUILD_PATH=/Users/johnsonhj/src/Slicer-build/teem-build
MEX_PATH=/Applications/MATLAB_R2013b.app/bin/mex

all:
	$(MEX_PATH) nrrdLoadWithMetadata.c -I$(TEEM_BUILD_PATH)/include -L$(TEEM_BUILD_PATH)/bin -lteem
	$(MEX_PATH) nrrdSaveWithMetadata.c -I$(TEEM_BUILD_PATH)/include -L$(TEEM_BUILD_PATH)/bin -lteem
