#include "nrrdCommon.h"

// #define DEBUG(x) { x }
#define DEBUG(x)

bool debug = false;

void myMexPrintf( std::string msg){
  if(debug)
  mexPrintf(msg.c_str());
}
void myMexPrintf( std::string msg, int value){
  if(debug)
  mexPrintf(msg.c_str(), value);
}
void myMexPrintf( std::string msg, std::string value){
  if(debug)
  mexPrintf(msg.c_str(), value.c_str());
}
void myMexPrintf(std::string msg, double value){
  if(debug)
  mexPrintf(msg.c_str(), value);
}
void myMexPrintf(std::string msg, unsigned int value){
  if(debug)
  mexPrintf(msg.c_str(), value);
}

void nrrdLoadWithMetadata(int, mxArray *plhs[],
                          int nrhs, const mxArray *prhs[])
{
  // NOTE: Useful for "printf" debugging
  // snprintf(errBuff, NRRD_MAX_ERROR_MSG_SIZE, "HERE %d: %s ", __LINE__,
  // __FILE__);
  // mexWarnMsgTxt(errBuff);

  const char me[] = "nrrdLoadWithMetadata";
  char       errBuff[NRRD_MAX_ERROR_MSG_SIZE];
  /* Meta data stuff */

  const mxArray * const filenameMx = prhs[0];

  if( !(1 == nrhs && mxIsChar(filenameMx) ) )
    {
    snprintf(errBuff, NRRD_MAX_ERROR_MSG_SIZE, "%s: requires one string argument (the name of the file)", me);
    mexErrMsgTxt(errBuff);
    }

  airArray * const mop = airMopNew();
  const int        filenameLen = mxGetM(prhs[0]) * mxGetN(prhs[0]) + 1;
  /* managed by Matlab */
  char *filename = static_cast<char *>(mxCalloc(filenameLen, sizeof(mxChar) ) );
  mxGetString(prhs[0], filename, filenameLen);

  myMexPrintf(  "filename = %s\n", filename);

  Nrrd * const nrrd = nrrdNew();
  airMopAdd(mop, nrrd, (airMopper)nrrdNix, airMopAlways);
  NrrdIoState * const nio = nrrdIoStateNew();
  airMopAdd(mop, nio, (airMopper)nrrdIoStateNix, airMopAlways);
  nrrdIoStateSet(nio, nrrdIoStateSkipData, AIR_TRUE);

  /* read header, but no data */
  if( nrrdLoad(nrrd, filename, nio) )
    {
    char * errPtr = biffGetDone(NRRD);
    airMopAdd(mop, errPtr, airFree, airMopAlways);
    snprintf(errBuff, NRRD_MAX_ERROR_MSG_SIZE, "%s: trouble reading NRRD header:\n%s", me, errPtr);
    airMopError(mop);
    mexErrMsgTxt(errBuff);
    }

  myMexPrintf(  "after first nrrdLoad\n");

  const mxClassID mtype = typeNtoM(nrrd->type);
  if( mxUNKNOWN_CLASS == mtype )
    {
    snprintf(errBuff, NRRD_MAX_ERROR_MSG_SIZE, "%s: sorry, can't handle type %s (%d)", me,
             airEnumStr(nrrdType, nrrd->type), nrrd->type);
    airMopError(mop);
    mexErrMsgTxt(errBuff);
    }

  /* Create a MATLAB Struct and Set All the Fields */
  /** Setup the Fieldnames **/
    /** TODO - struct field names which coorespond with ITK image field names and organization **/
  const char * *fieldnames;  /* matlab struct field names*/
  if( NULL != nrrd->kvp )
    {
    fieldnames = static_cast<const char * *>(mxCalloc(FIELDNAME_INDEX_MAXFEILDS, sizeof(*fieldnames) ) );
    }
  else
    {
    fieldnames = static_cast<const char * *>(mxCalloc(FIELDNAME_INDEX_MANDATORYFEILDS, sizeof(*fieldnames) ) );
    }
  fieldnames[FIELDNAME_INDEX_data] = "data";
  fieldnames[1] = "space";
  fieldnames[2] = "spacedirections";
  fieldnames[3] = "centerings";
  fieldnames[4] = "kinds";
  fieldnames[5] = "spaceunits";
  fieldnames[6] = "spacedefinition";
  fieldnames[7] = "spaceorigin";
  fieldnames[8] = "measurementframe";
  mxArray * structMx;
  if( NULL != nrrd->kvp )
    {
    fieldnames[9] = "modality";
    fieldnames[10] = "bvalue";
    fieldnames[11] = "gradientdirections";
    /** Create the Struct **/
    structMx = mxCreateStructMatrix( 1, 1, FIELDNAME_INDEX_MAXFEILDS, fieldnames );
    }
  else
    {
    structMx = mxCreateStructMatrix( 1, 1, FIELDNAME_INDEX_MANDATORYFEILDS, fieldnames );
    }
  mxFree( (void *)fieldnames);

  myMexPrintf(  "Created Field Names");

  mwSize sizeI[NRRD_DIM_MAX];
  /** data **/
  for( unsigned int axIdx = 0; axIdx < nrrd->dim; axIdx++ )
    {
    sizeI[axIdx] = nrrd->axis[axIdx].size;
    }
  DEBUG(printf("dim %d ", nrrd->dim);
        for( unsigned i = 0; i < nrrd->dim; ++i )
          {
          printf("%d", sizeI[i]);
          if( i < nrrd->dim - 1 )
            {
            printf(" ");
            }
          else
            {
            printf("\n");
            }
          }
        );

  mxArray *data = mxCreateNumericArray( nrrd->dim, sizeI, mtype, mxREAL );

  nrrd->data = mxGetPr(data);
  mxSetFieldByNumber( structMx, 0, FIELDNAME_INDEX_data, data );

  myMexPrintf(  "after setting data\n");

  /** space **/
  if( NULL != &nrrd->space )
    {
    const mwSize space_size = 1;
    mxArray *    space = mxCreateNumericArray( 1, &space_size, mxINT32_CLASS, mxREAL );
    int *        space_temp = (int *)mxGetData(space);
    *space_temp = nrrd->space;
    mxSetFieldByNumber( structMx, 0, FIELDNAME_INDEX_space, space );
    }
  myMexPrintf(  "after space\n");

  /** centerings **/
  const mwSize centerings_size = nrrd->dim;
  mxArray *    centerings = mxCreateNumericArray( 1, &centerings_size, mxINT32_CLASS, mxREAL );
  int * const  centerings_temp = (int *)mxGetData(centerings);
  for( unsigned int axIdx = 0; axIdx < nrrd->dim; axIdx++ )
    {
      centerings_temp[axIdx] = axIdx < 3 ? nrrdCenterCell : nrrd->axis[axIdx].center;//adjusted, is this okay??
      myMexPrintf(  "nrrdLoadCenterings: %d\n", centerings_temp[axIdx]);
    }
  mxSetFieldByNumber( structMx, 0, FIELDNAME_INDEX_centerings, centerings );
  myMexPrintf(  "after centering\n");

  /** kinds **/
  const mwSize    mxNrrdDim = nrrd->dim;
  mxArray * const kinds = mxCreateNumericArray( 1, &mxNrrdDim, mxINT32_CLASS, mxREAL );
  int * const     kinds_temp = (int *)mxGetData(kinds);
  for( unsigned int axIdx = 0; axIdx < nrrd->dim; axIdx++ )
    {
    kinds_temp[axIdx] = nrrd->axis[axIdx].kind;
    }
  mxSetFieldByNumber( structMx, 0, FIELDNAME_INDEX_kinds, kinds );
  myMexPrintf(  "after kind\n");

  /** spacedirections **/
  const mwSize    mxNrrdSpaceDim = nrrd->spaceDim;
  mxArray * const spacedirections = mxCreateNumericMatrix(mxNrrdSpaceDim, mxNrrdSpaceDim, mxDOUBLE_CLASS, mxREAL );
  double *const   spacedirections_temp = (double *)mxGetData(spacedirections);
    {
    unsigned int count = 0;
    for( unsigned int axIdx = 0; axIdx < mxNrrdDim; axIdx++ )
      {
      if( isGradientAxis( nrrd->axis[axIdx].kind ) )
        {
        // Do not fill out if it is a gradient direction.
        continue;
        }
      for( unsigned int sdIdx = 0; sdIdx < mxNrrdSpaceDim; sdIdx++ )
        {
        spacedirections_temp[count] = nrrd->axis[axIdx].spaceDirection[sdIdx];
        ++count;
        }
      }
    }
  mxSetFieldByNumber( structMx, 0, FIELDNAME_INDEX_spacedirections, spacedirections );
  myMexPrintf(  "after space directions\n");

  /** spaceunits **/
  mxArray * const spaceunits = mxCreateCellArray( 1, &mxNrrdSpaceDim );
  for( unsigned sdIdx = 0; sdIdx < mxNrrdSpaceDim; sdIdx++ )
    {
      mxArray *spaceunits_temp = mxCreateString( nrrd->spaceUnits[sdIdx] ? nrrd->spaceUnits[sdIdx] : "mm"); // if NULL default to "mm"
    mxSetCell(spaceunits, sdIdx, spaceunits_temp);
    }
  mxSetFieldByNumber( structMx, 0, FIELDNAME_INDEX_spaceunits, spaceunits );
  myMexPrintf(  "after space units\n");

  /** spacedefinition **/
  const char * const theSpaceString = airEnumStr(nrrdSpace, nrrd->space);
  mxArray *          spacedefinition = mxCreateString( theSpaceString );
  mxSetFieldByNumber( structMx, 0, FIELDNAME_INDEX_spacedefinition, spacedefinition );
  myMexPrintf(  "after space definition\n");

  /** spaceorigin **/
  const mwSize spaceorigin_size = mxNrrdSpaceDim;
  mxArray *    spaceorigin = mxCreateNumericArray( 1, &spaceorigin_size, mxDOUBLE_CLASS, mxREAL );
  double *     spaceorigin_temp = (double *)mxGetData(spaceorigin);
  for( unsigned int sdIdx = 0; sdIdx < mxNrrdSpaceDim; sdIdx++ )
    {
    spaceorigin_temp[sdIdx] = nrrd->spaceOrigin[sdIdx];
    }
  mxSetFieldByNumber( structMx, 0, FIELDNAME_INDEX_spaceorigin, spaceorigin );
  myMexPrintf(  "after space origin\n");

  /** measurementframe **/
  mxArray *measurementframe = mxCreateNumericMatrix(mxNrrdSpaceDim, mxNrrdSpaceDim, mxDOUBLE_CLASS, mxREAL );
  double * measurementframe_temp = (double *)mxGetData(measurementframe);
  for( unsigned int axIdx = 0; axIdx < mxNrrdSpaceDim; ++axIdx )
    {
    for( unsigned int sdIdx = 0; sdIdx < mxNrrdSpaceDim; ++sdIdx )
      {
        measurementframe_temp[axIdx * mxNrrdSpaceDim + sdIdx] = nrrd->measurementFrame[axIdx][sdIdx] != nrrd->measurementFrame[axIdx][sdIdx] ? 0.0 : nrrd->measurementFrame[axIdx][sdIdx];//check for nan
      myMexPrintf(  "nrrdLoad measurement frame: %lf\n", measurementframe_temp[axIdx * mxNrrdSpaceDim + sdIdx]);
      }
    }
  mxSetFieldByNumber( structMx, 0, FIELDNAME_INDEX_measurementframe, measurementframe );
  myMexPrintf(  "after measurement frame\n");

  /** modality, bvalue, & gradientdirections **/
  if( NULL != nrrd->kvp )
    {
    unsigned int *skip = NULL;
    unsigned int  skipNum = 0;
    /** modality **/
    mxArray * modality = mxCreateString( nrrd->kvp[1] );
    mxSetFieldByNumber( structMx, 0, FIELDNAME_INDEX_modality, modality );

    myMexPrintf(  "after set modality\n");

    /* use ten to parse the key/value pairs */
    Nrrd * ngradKVP = NULL, *nbmatKVP = NULL; double bKVP; double *info;
    tenDWMRIKeyValueParse(&ngradKVP, &nbmatKVP, &bKVP, &skip, &skipNum, nrrd);
    info = (double *)(ngradKVP->data);
    myMexPrintf(  "after tenDWMRIKeyValueParse");

    /** bvalue **/
    const mwSize    bvalue_size = 1;
    mxArray * const bvalue = mxCreateNumericArray( 1, &bvalue_size, mxDOUBLE_CLASS, mxREAL );
    double * const  bvalue_temp = (double *)mxGetData(bvalue);
    *bvalue_temp = bKVP;
    mxSetFieldByNumber( structMx, 0, FIELDNAME_INDEX_bvalue, bvalue );
    myMexPrintf(  "after BValue\n");

    /** gradientdirections **/
    /*First find the index that contains the graident directions! */
    unsigned int gradientSizeIndex = 0;
    for( unsigned int axIdx = 0; axIdx < nrrd->dim; axIdx++ )
      {
      if( isGradientAxis( nrrd->axis[axIdx].kind ) )
        {
        gradientSizeIndex = axIdx;
        break;
        }
      }
    myMexPrintf(  "Gradient size Index = %u\n", gradientSizeIndex);

    mxArray *gradientdirections = mxCreateNumericMatrix( sizeI[gradientSizeIndex], mxNrrdSpaceDim, mxDOUBLE_CLASS,
                                                         mxREAL );
    double *gradientdirections_temp = (double *)mxGetData(gradientdirections);
    for( unsigned int dwiIdx = 0; dwiIdx < sizeI[gradientSizeIndex]; dwiIdx++ )
      {
      gradientdirections_temp[dwiIdx] = info[dwiIdx * mxNrrdSpaceDim];
      gradientdirections_temp[dwiIdx + sizeI[gradientSizeIndex]] = info[dwiIdx * mxNrrdSpaceDim + 1];
      gradientdirections_temp[dwiIdx + sizeI[gradientSizeIndex] * 2] = info[dwiIdx * mxNrrdSpaceDim + 2];
      }
    mxSetFieldByNumber( structMx, 0, FIELDNAME_INDEX_gradientdirections, gradientdirections );
    myMexPrintf(  "after gradients\n");
    }

  /* read second time, now loading data */
  if( nrrdLoad(nrrd, filename, NULL) )
    {
    char * errPtr = biffGetDone(NRRD);
    airMopAdd(mop, errPtr, airFree, airMopAlways);
    snprintf(errBuff, NRRD_MAX_ERROR_MSG_SIZE, "%s: trouble reading NRRD:\n%s", me, errPtr);
    airMopError(mop);
    mexErrMsgTxt(errBuff);
    }
  myMexPrintf(  "after voxel load\n");

  airMopOkay(mop);
  plhs[0] = structMx;
  return;
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  try
    {
    nrrdLoadWithMetadata(nlhs, plhs, nrhs, prhs);
    }
  catch( ... )
    {
    printf("Exception in nrrdLoadWithMetaData\n");
    mexErrMsgTxt("Exception in nrrdLoadWithMetaData");
    }
}





