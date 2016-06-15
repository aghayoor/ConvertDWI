#! /usr/bin/env python

## \author Ali Ghayoor
##
## Main workflow for Human Connectome Project DWI data processing
## This workflow calls ? other WFs for:
## -Conversion to Nrrd and Direction Cosign Correction


"""
HCPWF.py
==============

The purpose of this pipeline is to complete all the pre-processing, super-resolution reconstruction and evaluation steps on Human Connectome Project DWI datasets. The main inputs to this pipeline are input DWI subject from HCP dataset and post processing structural MRIs and brain labelmap from BRAINSAutoWorkup (BAW).

Usage:
  HCPWF.py --inputDWIScan DWISCAN --inputT1Scan T1SCAN --inputT2Scan T2SCAN --inputBrainLabelsMapImage BLMImage --program_paths PROGRAM_PATHS --python_aux_paths PYTHON_AUX_PATHS --labelsConfigFile LABELS_CONFIG_FILE [--workflowCacheDir CACHEDIR] [--resultDir RESULTDIR]
  HCPWF.py -v | --version
  HCPWF.py -h | --help

Options:
  -h --help                                 Show this help and exit
  -v --version                              Print the version and exit
  --inputDWIScan DWISCAN                    Path to the input DWI scan for further processing
  --inputT1Scan T1SCAN                      Path to the input T1 scan from BAW
  --inputT2Scan T2SCAN                      Path to the input T2 scan from BAW
  --inputBrainLabelsMapImage BLMImage       Path to the input brain labels map image from BAW
  --program_paths PROGRAM_PATHS             Path to the directory where binary files are places
  --python_aux_paths PYTHON_AUX_PATHS       Path to the AutoWorkup directory
  --labelsConfigFile LABELS_CONFIG_FILE     Configuration file that defines region labels (.csv)
  --workflowCacheDir CACHEDIR               Base directory that cache outputs of workflow will be written to (default: ./)
  --resultDir RESULTDIR                     Outputs of dataSink will be written to a sub directory under the resultDir named by input scan sessionID (default: CACHEDIR)
"""
from __future__ import print_function

def runMainWorkflow(DWI_scan, T1_scan, T2_scan, labelMap_image, BASE_DIR, dataSink_DIR, PYTHON_AUX_PATHS, LABELS_CONFIG_FILE):
    print("Running the workflow ...")

    sessionID = os.path.basename(os.path.dirname(DWI_scan))
    subjectID = os.path.basename(os.path.dirname(os.path.dirname(DWI_scan)))
    siteID = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(DWI_scan))))

    #\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
    ####### Workflow ###################
    WFname = 'DWIWorkflow_CACHE_' + sessionID
    DWIWorkflow = pe.Workflow(name=WFname)
    DWIWorkflow.base_dir = BASE_DIR

    inputsSpec = pe.Node(interface=IdentityInterface(fields=['DWIVolume','T1Volume','T2Volume','LabelMapVolume']),
                         name='inputsSpec')

    inputsSpec.inputs.DWIVolume = DWI_scan
    inputsSpec.inputs.T1Volume = T1_scan
    inputsSpec.inputs.T2Volume = T2_scan
    inputsSpec.inputs.LabelMapVolume = labelMap_image

    ## DWI_baseline: original HCP DWI image that is converted to NRRD, corrected and aligned to physical space of input structral MR images.
    ## DWI_SR_NN: the high resolution DWI that is output of super-resolution reconstruction by Nearest Neighbor method.
    ## DWI_SR_IFFT: the high resolution DWI that is output of super-resolution reconstruction by zero-padded IFFT method.
    ## DWI_SR_TV: the high resolution DWI that is output of super-resolution reconstruction by Total Variation method.
    ## DWI_SR_WTV: the high resolution DWI that is output of super-resolution reconstruction by Weighted TV method.

    outputsSpec = pe.Node(interface=IdentityInterface(fields=['DWI_original_corrected','DWI_baseline','DWIBrainMask'
                                                              ,'DWI_SR_NN','DWI_SR_IFFT','DWI_SR_TV','DWI_SR_WTV'
                                                              ,'Baseline_ukfTracks','NN_ukfTracks','IFFT_ukfTracks','TV_ukfTracks','WTV_ukfTracks'
                                                              ]),
                          name='outputsSpec')
'''
    correctionWFname = 'CorrectionWorkflow_CACHE_' + sessionID
    myCorrectionWF = CreateCorrectionWorkflow(correctionWFname)

    measurementWFname = 'MeasurementWorkflow_CACHE_' + sessionID
    myMeasurementWF = CreateMeasurementWorkflow(measurementWFname, LABELS_CONFIG_FILE)

    # clone measurement WF to measure statistics from RISs estimated from non compressed sensing DWI scan
    measurementWithoutCSWFname = 'MeasurementWFWithoutCS_CACHE_' + sessionID
    MeasurementWFWithoutCS = myMeasurementWF.clone(name=measurementWithoutCSWFname)

    #Connect up the components into an integrated workflow.
    DWIWorkflow.connect([(inputsSpec,myCorrectionWF,[('T2Volume','inputsSpec.T2Volume'),
                                           ('DWIVolume','inputsSpec.DWIVolume'),
                                           ('LabelMapVolume','inputsSpec.LabelMapVolume'),
                                           ]),
                         (myCorrectionWF, myCSWF,[('outputsSpec.CorrectedDWI_in_T2Space','inputsSpec.DWI_Corrected_Aligned'),
                                                ('outputsSpec.DWIBrainMask','inputsSpec.DWIBrainMask'),
                                                ])
                       ])
'''
    ## Write all outputs with DataSink
    DWIDataSink = pe.Node(interface=nio.DataSink(), name='DWIDataSink')
    DWIDataSink.overwrite = True
    DWIDataSink.inputs.base_directory = dataSink_DIR
    #DWIDataSink.inputs.substitutions = []

    # Outputs (directory)
    DWIWorkflow.connect(outputsSpec, 'DWI_original_corrected', DWIDataSink, 'Outputs.@DWI_original_corrected')
    DWIWorkflow.connect(outputsSpec, 'DWI_baseline', DWIDataSink, 'Outputs.@DWI_baseline')
    DWIWorkflow.connect(outputsSpec, 'DWIBrainMask', DWIDataSink, 'Outputs.@DWIBrainMask')

    DWIWorkflow.write_graph()
    DWIWorkflow.run()


if __name__ == '__main__':
  import os
  import glob
  import sys

  from docopt import docopt
  argv = docopt(__doc__, version='1.0')
  print(argv)

  DWISCAN = argv['--inputDWIScan']
  assert os.path.exists(DWISCAN), "Input DWI scan is not found: %s" % DWISCAN

  T1SCAN = argv['--inputT1Scan']
  assert os.path.exists(T2SCAN), "Input T1 scan is not found: %s" % T1SCAN

  T2SCAN = argv['--inputT2Scan']
  assert os.path.exists(T2SCAN), "Input T2 scan is not found: %s" % T2SCAN

  LabelMapImage = argv['--inputBrainLabelsMapImage']
  assert os.path.exists(LabelMapImage), "Input Brain labels map image is not found: %s" % LabelMapImage

  PROGRAM_PATHS = argv['--program_paths']

  PYTHON_AUX_PATHS = argv['--python_aux_paths']

  LABELS_CONFIG_FILE = argv['--labelsConfigFile']

  if argv['--workflowCacheDir'] == None:
      print("*** workflow cache directory is set to current working directory.")
      CACHEDIR = os.getcwd()
  else:
      CACHEDIR = argv['--workflowCacheDir']
      assert os.path.exists(CACHEDIR), "Cache directory is not found: %s" % CACHEDIR

  if argv['--resultDir'] == None:
      print("*** data sink result directory is set to the cache directory.")
      RESULTDIR = CACHEDIR
  else:
      RESULTDIR = argv['--resultDir']
      assert os.path.exists(RESULTDIR), "Results directory is not found: %s" % RESULTDIR

  print('=' * 100)

  #\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
  #####################################################################################
  #     Prepend the shell environment search paths
  PROGRAM_PATHS = PROGRAM_PATHS.split(':')
  PROGRAM_PATHS.extend(os.environ['PATH'].split(':'))
  os.environ['PATH'] = ':'.join(PROGRAM_PATHS)

  CUSTOM_ENVIRONMENT=dict()

  # Platform specific information
  #     Prepend the python search paths
  PYTHON_AUX_PATHS = PYTHON_AUX_PATHS.split(':')
  PYTHON_AUX_PATHS.extend(sys.path)
  sys.path = PYTHON_AUX_PATHS

  import SimpleITK as sitk
  import nipype
  from nipype.interfaces import ants
  from nipype.interfaces.base import CommandLine, CommandLineInputSpec, TraitedSpec, File, Directory
  from nipype.interfaces.base import traits, isdefined, BaseInterface
  from nipype.interfaces.utility import Merge, Split, Function, Rename, IdentityInterface
  import nipype.interfaces.io as nio   # Data i/oS
  import nipype.pipeline.engine as pe  # pypeline engine
  import nipype.interfaces.matlab as matlab
  from nipype.interfaces.semtools import *
  #####################################################################################
'''
  from CorrectionWorkflow import CreateCorrectionWorkflow
  from CSWorkflow import CreateCSWorkflow
  from EstimationWorkflow import CreateEstimationWorkflow
  from TractographyWorkflow import CreateTractographyWorkflow
  from MeasurementWorkflow import CreateMeasurementWorkflow
'''
  exit = runMainWorkflow(DWISCAN, T1SCAN, T2SCAN, LabelMapImage, CACHEDIR, RESULTDIR, PYTHON_AUX_PATHS, LABELS_CONFIG_FILE)

  sys.exit(exit)
