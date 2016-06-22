## NOTE: On mavericks you need to edit mexopts.sh to match required compiler options.
##  vim /Applications/MATLAB_R2013b.app/bin/mexopts.sh

BuildMex(MEXNAME nrrdLoadWithMetadata
  TARGETDIR "${${PRIMARY_PROJECT_NAME}_SOURCE_DIR}/MATLAB_SCRIPTS"
  SOURCE nrrdLoadWithMetadata.cxx nrrdCommon.cxx
  LIBRARIES ${Teem_LIBRARIES})
BuildMex(MEXNAME nrrdSaveWithMetadata
  TARGETDIR "${${PRIMARY_PROJECT_NAME}_SOURCE_DIR}/MATLAB_SCRIPTS"
  SOURCE nrrdSaveWithMetadata.cxx nrrdCommon.cxx
  LIBRARIES ${Teem_LIBRARIES})

BuildMex(MEXNAME itkLoadWithMetadata
  TARGETDIR "${${PRIMARY_PROJECT_NAME}_SOURCE_DIR}/MATLAB_SCRIPTS"
  SOURCE itkLoadWithMetadata.cxx nrrdCommon.cxx
  LIBRARIES ${ITK_LIBRARIES} ${Teem_LIBRARIES})

BuildMex(MEXNAME itkSaveWithMetadata
  TARGETDIR "${${PRIMARY_PROJECT_NAME}_SOURCE_DIR}/MATLAB_SCRIPTS"
  SOURCE itkSaveWithMetadata.cxx nrrdCommon.cxx
  LIBRARIES ${ITK_LIBRARIES} ${Teem_LIBRARIES})

BuildMex(MEXNAME vtkLoadPolyData
  TARGETDIR "${${PRIMARY_PROJECT_NAME}_SOURCE_DIR}/MATLAB_SCRIPTS"
  SOURCE vtkLoadPolyData.cxx
  LIBRARIES ${VTK_LIBRARIES} )

BuildMex(MEXNAME vtkSavePolyData
  TARGETDIR "${${PRIMARY_PROJECT_NAME}_SOURCE_DIR}/MATLAB_SCRIPTS"
  SOURCE vtkSavePolyData.cxx
  LIBRARIES ${VTK_LIBRARIES} )

