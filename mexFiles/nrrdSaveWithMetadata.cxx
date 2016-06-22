#include "nrrdCommon.h"

void nrrdSaveWithMetadata(int /*nlhs */, mxArray * /*plhs*/ [],
                          int nrhs, const mxArray *prhs[])
{
  const char            me[] = "nrrdSaveWithMetadata";
  char                  errBuff[NRRD_MAX_ERROR_MSG_SIZE] = { '\0' };
  const mxArray * const filenameMx = prhs[0];

  if( !(2 == nrhs && mxIsChar(filenameMx) ) )
    {
    snprintf(errBuff, NRRD_MAX_ERROR_MSG_SIZE, "%s: requires two args: one string, one struct", me);
    mexErrMsgTxt(errBuff);
    return;
    }
  const MatlabStructManager msm(prhs[1]);

  const int filenameLen = mxGetM(filenameMx) * mxGetN(filenameMx) + 1;
  /* managed by Matlab */
  char * const filename = static_cast<char *>(mxCalloc(filenameLen, sizeof(mxChar) ) );
  mxGetString(filenameMx, filename, filenameLen);

  if( !( 1 <= msm.GetNumberOfDimensions("data") && msm.GetNumberOfDimensions("data") <= NRRD_DIM_MAX ) )
    {
    snprintf(errBuff, NRRD_MAX_ERROR_MSG_SIZE, "%s: number of array dimensions %u outside range [1,%d]",
             me, msm.GetNumberOfDimensions("data"), NRRD_DIM_MAX);
    mexErrMsgTxt(errBuff);
    }

  Nrrd * const     nrrd = nrrdNew();
  airArray * const mop = airMopNew();
  airMopAdd(mop, nrrd, (airMopper)nrrdNix, airMopAlways);

  /** space **/
  nrrd->dim = msm.GetNumberOfDimensions("data");

  if( msm.isDWIdata() )
    {
    // One of the directions is the gradient list instead of a spaceDim
    nrrd->spaceDim = msm.GetNumberOfDimensions("data") - 1;
      { // Get the DWI specific fields.
      const mxArray * const gradientdirections = msm.GetField("gradientdirections");
      if( gradientdirections != 0 )
        {
        std::stringstream ss;
        ss.precision(10);
        nrrdKeyValueAdd(nrrd, "modality", "DWMRI");
        const unsigned numGradients = mxGetNumberOfElements(gradientdirections)/3;

        const double * const bvalue = static_cast<double *>(mxGetData(msm.GetField("bvalue")));
        ss << *bvalue;
        nrrdKeyValueAdd(nrrd, "DWMRI_b-value", ss.str().c_str());

        if( numGradients > 0)
          {
          const double * const gradients = static_cast<double *>(mxGetData(gradientdirections) );
          for( unsigned i = 0; i < numGradients; ++i )
            {
            ss.str(std::string() ); // reset
            ss << gradients[i] << " "
               << gradients[i + numGradients] << " "
               << gradients[i + (2 * numGradients)];

            std::stringstream gradName;
            gradName << GRADIENT_PREFIX << std::setw(4) << std::setfill('0') << i;
            nrrdKeyValueAdd(nrrd, gradName.str().c_str(), ss.str().c_str() );
            }
          }
        }
      // OK
      }
    }
  else
    {
    nrrd->spaceDim = msm.GetNumberOfDimensions("data");
    }

  nrrd->space = *( (int *)mxGetData( msm.GetField("space") ) );

  // OK
  /** centerings **/
  const int *centerings_temp = (int *)mxGetData( msm.GetField("centerings") );
  for( unsigned int axIdx = 0; axIdx < nrrd->dim; ++axIdx )
    {
    nrrd->axis[axIdx].center = centerings_temp[axIdx];
    }
  /** kinds **/
  const int *kinds_temp = (int *)mxGetData( msm.GetField("kinds") );
  for( unsigned int axIdx = 0; axIdx < nrrd->dim; ++axIdx )
    {
    nrrd->axis[axIdx].kind = kinds_temp[axIdx];
    }
  /** spaceunits **/
  for( unsigned int sdIdx = 0; sdIdx < nrrd->spaceDim; sdIdx++ )
    {
    nrrd->spaceUnits[sdIdx] = static_cast<char *>(malloc(200) );
    const char * const myvalue =
      mxArrayToString( mxGetCell( msm.GetField("spaceunits"), sdIdx) );
    snprintf( nrrd->spaceUnits[sdIdx],  200, "%s", myvalue );
    }
  /** spacedirections **/
  const double *spacedirections_temp = (double *)mxGetData( msm.GetField("spacedirections") );
    {
    unsigned int count = 0;
    for( unsigned int axIdx = 0; axIdx < nrrd->dim; ++axIdx )
      {
      if( isGradientAxis( nrrd->axis[axIdx].kind ) )
        {
        // Do not fill out if it is a gradient direction.
        continue;
        }
      for( unsigned int sdIdx = 0; sdIdx < nrrd->spaceDim; sdIdx++ )
        {
        nrrd->axis[axIdx].spaceDirection[sdIdx] = spacedirections_temp[count];
        count++;
        }
      }
    }
  /** spaceorigin **/
  const double *spaceorigin_temp = (double *)mxGetData( msm.GetField("spaceorigin") );
  for( unsigned int sdIdx = 0; sdIdx < nrrd->spaceDim; sdIdx++ )
    {
    nrrd->spaceOrigin[sdIdx] = spaceorigin_temp[sdIdx];
    }
  /****** TODO:  Need FIELDNAME_INDEX_spacedefinition ***/
  /** measurementframe **/
  const double *measurementframe_temp = (double *)mxGetData( msm.GetField("measurementframe") );
  for( unsigned int axIdx = 0, count = 0; axIdx < nrrd->spaceDim; ++axIdx )
    {
    for( unsigned int sdIdx = 0; sdIdx < nrrd->spaceDim; sdIdx++ )
      {
      nrrd->measurementFrame[axIdx][sdIdx] = measurementframe_temp[count];
      count++;
      }
    }
    {
    NrrdIoState * const nio = nrrdIoStateNew();
    nio->encoding = nrrdEncodingGzip;
    size_t sizeZ[NRRD_DIM_MAX];
      {
      const mwSize * const mSize = msm.GetDimensions("data");
      for( unsigned int axIdx = 0; axIdx < msm.GetNumberOfDimensions("data"); ++axIdx )
        {
        sizeZ[axIdx] = mSize[axIdx];
        }
      }
    if( nrrdWrap_nva(nrrd, mxGetPr(msm.GetField("data") ), msm.GetDataType(), msm.GetNumberOfDimensions("data"), sizeZ)
        || nrrdSave(filename, nrrd, nio) )
      {
      char *errPtr = biffGetDone(NRRD);
      airMopAdd(mop, errPtr, airFree, airMopAlways);
      snprintf(errBuff, NRRD_MAX_ERROR_MSG_SIZE, "%s: error saving NRRD:\n%s", me, errPtr);
      airMopError(mop);
      mexErrMsgTxt(errBuff);
      }
    }
  airMopOkay(mop);
  return;
}

void mexFunction(int /*nlhs */, mxArray * /*plhs*/ [],
                 int nrhs, const mxArray *prhs[])
{
  try
    {
    nrrdSaveWithMetadata(0, 0, nrhs, prhs);
    }
  catch( ... )
    {
    printf("Exception in nrrdSaveWithMetaData\n");
    mexErrMsgTxt("Exception in nrrdSaveWithMetaData");
    }

}
