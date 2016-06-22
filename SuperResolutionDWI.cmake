include(${CMAKE_CURRENT_LIST_DIR}/Common.cmake)

#-----------------------------------------------------------------------------
enable_testing()
include(CTest)

find_package(ITK NO_MODULE REQUIRED)
include(${ITK_USE_FILE})

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Teem REQUIRED)
include(${Teem_USE_FILE})

link_directories(${CMAKE_CURRENT_BINARY_DIR}/../lib)

find_package(Matlab REQUIRED)
include_directories(${MATLAB_INCLUDE_DIR})

# Some test are failing due to inadequate test construction, but
# the code seems to do the correct thing on real data.
# This is also for tests that are huge, and can not be running
# a long time.
option(ENABLE_EXTENDED_TESTING "Enable tests that are long running, or where the test itself is in error." OFF)
mark_as_advanced(ENABLE_EXTENDED_TESTING)

#Set the global max TIMEOUT for CTest jobs.  This is very large for the moment
#and should be revisted to reduce based on "LONG/SHORT" test times, set to 1 hr for now
set(CTEST_TEST_TIMEOUT 1800 CACHE STRING "Maximum seconds allowed before CTest will kill the test." FORCE)
set(DART_TESTING_TIMEOUT ${CTEST_TEST_TIMEOUT} CACHE STRING "Maximum seconds allowed before CTest will kill the test." FORCE)

#-----------------------------------------------------------------------
# Setup locations to find externally maintained test data.
#-----------------------------------------------------------------------
include(SuperResolutionDWIExternalData)
set(TestData_DIR ${CMAKE_CURRENT_SOURCE_DIR}/TestData)

#-----------------------------------------------------------------------------
# Define list of module names
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
# Add module sub-directory if USE_<MODULENAME> is both defined and true
#-----------------------------------------------------------------------------
add_subdirectory(mexFiles)

ExternalData_Add_Target( ${PROJECT_NAME}FetchData )  # Name of data management target
