#! /usr/bin/env python

## \author Ali Ghayoor
##
## Main workflow for Human Connectome Project DWI data processing
## This workflow calls ? other WFs for:
## -Conversion to Nrrd and Direction Cosign Correction


"""
HCPWF.py
========

The purpose of this pipeline is to complete all the pre-processing, super-resolution reconstruction and evaluation steps on Human Connectome Project DWI datasets. The main inputs to this pipeline are input DWI subject from HCP dataset and post processing structural MRIs and brain labelmap from BRAINSAutoWorkup (BAW).

Usage:
  HCPWF.py --inputDWIScan DWISCAN --inputT1Scan T1SCAN --inputT2Scan T2SCAN --inputStandardLabels FS_STANDARD_LABELS --inputLobeLabels LOBE_LABELS --program_paths PROGRAM_PATHS --python_aux_paths PYTHON_AUX_PATHS --labelsConfigFile LABELS_CONFIG_FILE [--workflowCacheDir CACHEDIR] [--resultDir RESULTDIR]
  HCPWF.py -v | --version
  HCPWF.py -h | --help

Options:
  -h --help                                 Show this help and exit
  -v --version                              Print the version and exit
  --inputDWIScan DWISCAN                    Path to the input DWI scan for further processing
  --inputT1Scan T1SCAN                      Path to the input T1 scan from BAW
  --inputT2Scan T2SCAN                      Path to the input T2 scan from BAW
  --inputStandardLabels FS_STANDARD_LABELS  Path to the standard freesurfer brain labels map image from BAW (fs_standard_label)
  --inputLobeLabels LOBE_LABELS             Path to the brain lobes label map image from BAW (lobe_label)
  --program_paths PROGRAM_PATHS             Path to the directory where binary files are places
  --python_aux_paths PYTHON_AUX_PATHS       Path to the AutoWorkup directory
  --labelsConfigFile LABELS_CONFIG_FILE     Configuration file that defines region labels (.csv)
  --workflowCacheDir CACHEDIR               Base directory that cache outputs of workflow will be written to (default: ./)
  --resultDir RESULTDIR                     Outputs of dataSink will be written to a sub directory under the resultDir named by input scan sessionID (default: CACHEDIR)
"""
from __future__ import print_function

def runMainWorkflow(DWI_scan, T1_scan, T2_scan, FS_standard_labelMap, lobes_labelMap, BASE_DIR, dataSink_DIR, PYTHON_AUX_PATHS, LABELS_CONFIG_FILE):
    print("Running the workflow ...")
    sessionID = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(DWI_scan))))

    #\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
    ####### Workflow ###################
    WFname = 'HCPWorkflow_CACHE_' + sessionID
    HCPWorkflow = pe.Workflow(name=WFname)
    HCPWorkflow.base_dir = BASE_DIR

    inputsSpec = pe.Node(interface=IdentityInterface(fields=['DWIVolume','T1Volume','T2Volume','FSLabelMapVolume','LobesLabelMapVolume']),
                         name='inputsSpec')

    inputsSpec.inputs.DWIVolume = DWI_scan
    inputsSpec.inputs.T1Volume = T1_scan
    inputsSpec.inputs.T2Volume = T2_scan
    inputsSpec.inputs.FSLabelMapVolume = FS_standard_labelMap # standard freesurfer labelmap
    inputsSpec.inputs.LobesLabelMapVolume = lobes_labelMap # brain lobes labelmap

    ## DWI_corrected_alignedSpace: input HCP DWI image that is converted to NRRD,
    #                              corrected and aligned to physical space of input structral MR images.
    ## DWI_Baseline: is the corrected aligned DWI image that is refactored to be volume interleaved,
    #                and the intensity content of each component is normalized between zero and one.
    ## DWI_SR_NN: the high resolution DWI that is output of super-resolution reconstruction by Nearest Neighbor method.
    ## DWI_SR_IFFT: the high resolution DWI that is output of super-resolution reconstruction by zero-padded IFFT method.
    ## DWI_SR_TV: the high resolution DWI that is output of super-resolution reconstruction by Total Variation method.
    ## DWI_SR_WTV: the high resolution DWI that is output of super-resolution reconstruction by Weighted TV method.

    outputsSpec = pe.Node(interface=IdentityInterface(fields=['DWI_corrected_originalSpace','DWI_corrected_alignedSpace','DWI_corrected_alignedSpace_B0',
                                                              'DWI_corrected_alignedSpace_masked','DWIBrainMask',
                                                              'StrippedT1_125','StrippedT2_125','MaximumGradientImage','EdgeMap',
                                                              'DWI_Baseline','DWI_SR_NN','DWI_SR_IFFT','DWI_SR_TV','DWI_SR_WTV',
                                                              'FA_distance','MD_distance','RD_distance','AD_distance',
                                                              'Frobenius_distance','Logeuclid_distance','Reimann_distance','Kullback_distance',
                                                              'Baseline_ukfTracts','NN_ukfTracts','IFFT_ukfTracts','TV_ukfTracts','WTV_ukfTracts',
                                                              'Baseline_cst_left','Baseline_cst_right',
                                                              'NN_cst_left','NN_cst_right',
                                                              'IFFT_cst_left','IFFT_cst_right',
                                                              'TV_cst_left','TV_cst_right',
                                                              'WTV_cst_left','WTV_cst_right',
                                                              'NN_overlap_coeficient','IFFT_overlap_coeficient',
                                                              'TV_overlap_coeficient','WTV_overlap_coeficient'
                                                              ]),
                          name='outputsSpec')

    ###
    PreProcWFname = 'PreprocessingWorkflow_CACHE_' + sessionID
    PreProcWF = CreatePreprocessingWorkFlow(PreProcWFname)

    SRWFname = 'SuperResolutionWorkflow_CACHE_' + sessionID
    SRWF = CreateSuperResolutionWorkflow(SRWFname, PYTHON_AUX_PATHS)

    DistWFname = 'DistanceImagesWorkflow_CACHE_' + sessionID
    DistWF = CreateDistanceImagesWorkflow(DistWFname)

    TractWFname = 'TractographyWorkflow_CACHE_' + sessionID
    TractWF = CreateTractographyWorkflow(TractWFname)
    ###

    #Connect up the components into an integrated workflow
    HCPWorkflow.connect([(inputsSpec,PreProcWF,[('DWIVolume','inputsSpec.DWIVolume'),
                                                ('T1Volume','inputsSpec.T1Volume'),
                                                ('T2Volume','inputsSpec.T2Volume'),
                                                ('FSLabelMapVolume','inputsSpec.LabelMapVolume'),
                                               ]),
                         (PreProcWF, outputsSpec, [('outputsSpec.DWI_corrected_originalSpace','DWI_corrected_originalSpace'),
                                                   ('outputsSpec.DWI_corrected_alignedSpace','DWI_corrected_alignedSpace'),
                                                   ('outputsSpec.DWI_corrected_alignedSpace_B0','DWI_corrected_alignedSpace_B0'),
                                                   ('outputsSpec.DWI_corrected_alignedSpace_masked','DWI_corrected_alignedSpace_masked'),
                                                   ('outputsSpec.DWIBrainMask','DWIBrainMask'),
                                                   ('outputsSpec.StrippedT1_125','StrippedT1_125'),
                                                   ('outputsSpec.StrippedT2_125','StrippedT2_125'),
                                                   ('outputsSpec.MaximumGradientImage','MaximumGradientImage'),
                                                   ('outputsSpec.EdgeMap','EdgeMap')
                                                  ]),
                         (PreProcWF, SRWF, [('outputsSpec.DWI_corrected_alignedSpace_masked','inputsSpec.DWIVolume'),
                                            ('outputsSpec.EdgeMap','inputsSpec.EdgeMap')
                                           ]),
                         (SRWF, outputsSpec, [('outputsSpec.DWI_Baseline','DWI_Baseline'),
                                              ('outputsSpec.DWI_SR_NN','DWI_SR_NN'),
                                              ('outputsSpec.DWI_SR_IFFT','DWI_SR_IFFT'),
                                              ('outputsSpec.DWI_SR_TV','DWI_SR_TV'),
                                              ('outputsSpec.DWI_SR_WTV','DWI_SR_WTV')
                                             ]),
                         (inputsSpec, DistWF, [('LobesLabelMapVolume','inputsSpec.LobesLabelMapVolume')]),
                         (PreProcWF, DistWF, [('outputsSpec.DWIBrainMask','inputsSpec.DWI_brainMask')]),
                         (SRWF, DistWF, [('outputsSpec.DWI_Baseline','inputsSpec.DWI_Baseline'),
                                         ('outputsSpec.DWI_SR_NN','inputsSpec.DWI_SR_NN'),
                                         ('outputsSpec.DWI_SR_IFFT','inputsSpec.DWI_SR_IFFT'),
                                         ('outputsSpec.DWI_SR_TV','inputsSpec.DWI_SR_TV'),
                                         ('outputsSpec.DWI_SR_WTV','inputsSpec.DWI_SR_WTV')
                                        ]),
                         (DistWF, outputsSpec, [('outputsSpec.FA_distance','FA_distance'),
                                                ('outputsSpec.MD_distance','MD_distance'),
                                                ('outputsSpec.RD_distance','RD_distance'),
                                                ('outputsSpec.AD_distance','AD_distance'),
                                                ('outputsSpec.Frobenius_distance','Frobenius_distance'),
                                                ('outputsSpec.Logeuclid_distance','Logeuclid_distance'),
                                                ('outputsSpec.Reimann_distance','Reimann_distance'),
                                                ('outputsSpec.Kullback_distance','Kullback_distance')
                                               ]),
                         (inputsSpec, TractWF, [('FSLabelMapVolume','inputsSpec.inputLabelMap')]),
                         (PreProcWF, TractWF, [('outputsSpec.DWIBrainMask','inputsSpec.DWI_brainMask')]),
                         (SRWF, TractWF, [('outputsSpec.DWI_Baseline','inputsSpec.DWI_Baseline'),
                                          ('outputsSpec.DWI_SR_NN','inputsSpec.DWI_SR_NN'),
                                          ('outputsSpec.DWI_SR_IFFT','inputsSpec.DWI_SR_IFFT'),
                                          ('outputsSpec.DWI_SR_TV','inputsSpec.DWI_SR_TV'),
                                          ('outputsSpec.DWI_SR_WTV','inputsSpec.DWI_SR_WTV')
                                         ]),
                         (TractWF, outputsSpec, [('outputsSpec.Baseline_ukfTracts','Baseline_ukfTracts'),
                                                 ('outputsSpec.NN_ukfTracts','NN_ukfTracts'),
                                                 ('outputsSpec.IFFT_ukfTracts','IFFT_ukfTracts'),
                                                 ('outputsSpec.TV_ukfTracts','TV_ukfTracts'),
                                                 ('outputsSpec.WTV_ukfTracts','WTV_ukfTracts'),
                                                 ('outputsSpec.Baseline_cst_left','Baseline_cst_left'),
                                                 ('outputsSpec.Baseline_cst_right','Baseline_cst_right'),
                                                 ('outputsSpec.NN_cst_left','NN_cst_left'),
                                                 ('outputsSpec.NN_cst_right','NN_cst_right'),
                                                 ('outputsSpec.IFFT_cst_left','IFFT_cst_left'),
                                                 ('outputsSpec.IFFT_cst_right','IFFT_cst_right'),
                                                 ('outputsSpec.TV_cst_left','TV_cst_left'),
                                                 ('outputsSpec.TV_cst_right','TV_cst_right'),
                                                 ('outputsSpec.WTV_cst_left','WTV_cst_left'),
                                                 ('outputsSpec.WTV_cst_right','WTV_cst_right'),
                                                 ('outputsSpec.NN_overlap_coeficient','NN_overlap_coeficient'),
                                                 ('outputsSpec.IFFT_overlap_coeficient','IFFT_overlap_coeficient'),
                                                 ('outputsSpec.TV_overlap_coeficient','TV_overlap_coeficient'),
                                                 ('outputsSpec.WTV_overlap_coeficient','WTV_overlap_coeficient')])
                         ])

    ## Write all outputs with DataSink
    DWIDataSink = pe.Node(interface=nio.DataSink(), name='DWIDataSink')
    DWIDataSink.overwrite = True
    DWIDataSink.inputs.base_directory = dataSink_DIR
    DWIDataSink.inputs.substitutions = [('Outputs/_ResampleToAlignedDWIResolution0/','Outputs/'),
                                        ('Outputs/_ResampleToAlignedDWIResolution1/','Outputs/'),
                                        ('Outputs/_ResampleToAlignedDWIResolution2/','Outputs/'),
                                        ('Outputs/DistanceImage/_ComputeDistanceImages0/','Outputs/DistanceImage/NN_Distances/'),
                                        ('Outputs/DistanceImage/_ComputeDistanceImages1/','Outputs/DistanceImage/IFFT_Distances/'),
                                        ('Outputs/DistanceImage/_ComputeDistanceImages2/','Outputs/DistanceImage/TV_Distances/'),
                                        ('Outputs/DistanceImage/_ComputeDistanceImages3/','Outputs/DistanceImage/WTV_Distances/'),
                                        ('Outputs/Tractography/_RunUKFt0/','Outputs/Tractography/'),
                                        ('Outputs/Tractography/_RunUKFt1/','Outputs/Tractography/'),
                                        ('Outputs/Tractography/_RunUKFt2/','Outputs/Tractography/'),
                                        ('Outputs/Tractography/_RunUKFt3/','Outputs/Tractography/'),
                                        ('Outputs/Tractography/_RunUKFt4/','Outputs/Tractography/'),
                                        ('Outputs/WMQL/_tract_querier0/','Outputs/WMQL/'),
                                        ('Outputs/WMQL/_tract_querier1/','Outputs/WMQL/'),
                                        ('Outputs/WMQL/_tract_querier2/','Outputs/WMQL/'),
                                        ('Outputs/WMQL/_tract_querier3/','Outputs/WMQL/'),
                                        ('Outputs/WMQL/_tract_querier4/','Outputs/WMQL/'),
                                        ('Outputs/Stats/_BhattacharyyaCoeficient0/','Outputs/Stats/'),
                                        ('Outputs/Stats/_BhattacharyyaCoeficient1/','Outputs/Stats/'),
                                        ('Outputs/Stats/_BhattacharyyaCoeficient2/','Outputs/Stats/'),
                                        ('Outputs/Stats/_BhattacharyyaCoeficient3/','Outputs/Stats/')
                                       ]

    # Outputs (directory)
    HCPWorkflow.connect(outputsSpec, 'DWI_corrected_originalSpace', DWIDataSink, 'Outputs.@DWI_corrected_originalSpace')
    HCPWorkflow.connect(outputsSpec, 'DWI_corrected_alignedSpace', DWIDataSink, 'Outputs.@DWI_corrected_alignedSpace')
    HCPWorkflow.connect(outputsSpec, 'DWI_corrected_alignedSpace_B0', DWIDataSink, 'Outputs.@DWI_corrected_alignedSpace_B0')
    HCPWorkflow.connect(outputsSpec, 'DWI_corrected_alignedSpace_masked', DWIDataSink, 'Outputs.@DWI_corrected_alignedSpace_masked')
    HCPWorkflow.connect(outputsSpec, 'DWIBrainMask', DWIDataSink, 'Outputs.@DWIBrainMask')
    HCPWorkflow.connect(outputsSpec, 'StrippedT1_125', DWIDataSink, 'Outputs.@StrippedT1_125')
    HCPWorkflow.connect(outputsSpec, 'StrippedT2_125', DWIDataSink, 'Outputs.@StrippedT2_125')
    HCPWorkflow.connect(outputsSpec, 'MaximumGradientImage', DWIDataSink, 'Outputs.@MaximumGradientImage')
    HCPWorkflow.connect(outputsSpec, 'EdgeMap', DWIDataSink, 'Outputs.@EdgeMap')
    # Outputs/SuperResolution
    HCPWorkflow.connect(outputsSpec, 'DWI_Baseline', DWIDataSink, 'Outputs.SuperResolution.@DWI_Baseline')
    HCPWorkflow.connect(outputsSpec, 'DWI_SR_NN', DWIDataSink, 'Outputs.SuperResolution.@DWI_SR_NN')
    HCPWorkflow.connect(outputsSpec, 'DWI_SR_IFFT', DWIDataSink, 'Outputs.SuperResolution.@DWI_SR_IFFT')
    HCPWorkflow.connect(outputsSpec, 'DWI_SR_TV', DWIDataSink, 'Outputs.SuperResolution.@DWI_SR_TV')
    HCPWorkflow.connect(outputsSpec, 'DWI_SR_WTV', DWIDataSink, 'Outputs.SuperResolution.@DWI_SR_WTV')
    # Outputs/DistanceImage
    HCPWorkflow.connect(outputsSpec, 'FA_distance', DWIDataSink, 'Outputs.DistanceImage.@FA_distance')
    HCPWorkflow.connect(outputsSpec, 'MD_distance', DWIDataSink, 'Outputs.DistanceImage.@MD_distance')
    HCPWorkflow.connect(outputsSpec, 'RD_distance', DWIDataSink, 'Outputs.DistanceImage.@RD_distance')
    HCPWorkflow.connect(outputsSpec, 'AD_distance', DWIDataSink, 'Outputs.DistanceImage.@AD_distance')
    HCPWorkflow.connect(outputsSpec, 'Frobenius_distance', DWIDataSink, 'Outputs.DistanceImage.@Frobenius_distance')
    HCPWorkflow.connect(outputsSpec, 'Logeuclid_distance', DWIDataSink, 'Outputs.DistanceImage.@Logeuclid_distance')
    HCPWorkflow.connect(outputsSpec, 'Reimann_distance', DWIDataSink, 'Outputs.DistanceImage.@Reimann_distance')
    HCPWorkflow.connect(outputsSpec, 'Kullback_distance', DWIDataSink, 'Outputs.DistanceImage.@Kullback_distance')
    # Outputs/Tractography
    HCPWorkflow.connect(outputsSpec, 'Baseline_ukfTracts', DWIDataSink, 'Outputs.Tractography.@Baseline_ukfTracts')
    HCPWorkflow.connect(outputsSpec, 'NN_ukfTracts', DWIDataSink, 'Outputs.Tractography.@NN_ukfTracts')
    HCPWorkflow.connect(outputsSpec, 'IFFT_ukfTracts', DWIDataSink, 'Outputs.Tractography.@IFFT_ukfTracts')
    HCPWorkflow.connect(outputsSpec, 'TV_ukfTracts', DWIDataSink, 'Outputs.Tractography.@TV_ukfTracts')
    HCPWorkflow.connect(outputsSpec, 'WTV_ukfTracts', DWIDataSink, 'Outputs.Tractography.@WTV_ukfTracts')
    # Outputs/WMQL
    HCPWorkflow.connect(outputsSpec, 'Baseline_cst_left', DWIDataSink, 'Outputs.WMQL.@Baseline_cst_left')
    HCPWorkflow.connect(outputsSpec, 'Baseline_cst_right', DWIDataSink, 'Outputs.WMQL.@Baseline_cst_right')
    HCPWorkflow.connect(outputsSpec, 'NN_cst_left', DWIDataSink, 'Outputs.WMQL.@NN_cst_left')
    HCPWorkflow.connect(outputsSpec, 'NN_cst_right', DWIDataSink, 'Outputs.WMQL.@NN_cst_right')
    HCPWorkflow.connect(outputsSpec, 'IFFT_cst_left', DWIDataSink, 'Outputs.WMQL.@IFFT_cst_left')
    HCPWorkflow.connect(outputsSpec, 'IFFT_cst_right', DWIDataSink, 'Outputs.WMQL.@IFFT_cst_right')
    HCPWorkflow.connect(outputsSpec, 'TV_cst_left', DWIDataSink, 'Outputs.WMQL.@TV_cst_left')
    HCPWorkflow.connect(outputsSpec, 'TV_cst_right', DWIDataSink, 'Outputs.WMQL.@TV_cst_right')
    HCPWorkflow.connect(outputsSpec, 'WTV_cst_left', DWIDataSink, 'Outputs.WMQL.@WTV_cst_left')
    HCPWorkflow.connect(outputsSpec, 'WTV_cst_right', DWIDataSink, 'Outputs.WMQL.@WTV_cst_right')
    # Outputs/Stats
    HCPWorkflow.connect(outputsSpec, 'NN_overlap_coeficient', DWIDataSink, 'Outputs.Stats.@NN_overlap_coeficient')
    HCPWorkflow.connect(outputsSpec, 'IFFT_overlap_coeficient', DWIDataSink, 'Outputs.Stats.@IFFT_overlap_coeficient')
    HCPWorkflow.connect(outputsSpec, 'TV_overlap_coeficient', DWIDataSink, 'Outputs.Stats.@TV_overlap_coeficient')
    HCPWorkflow.connect(outputsSpec, 'WTV_overlap_coeficient', DWIDataSink, 'Outputs.Stats.@WTV_overlap_coeficient')

    HCPWorkflow.write_graph()
    HCPWorkflow.run()


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
  assert os.path.exists(T1SCAN), "Input T1 scan is not found: %s" % T1SCAN

  T2SCAN = argv['--inputT2Scan']
  assert os.path.exists(T2SCAN), "Input T2 scan is not found: %s" % T2SCAN

  FSStandardLabelMap = argv['--inputStandardLabels']
  assert os.path.exists(FSStandardLabelMap), "Input standard freesurfer brain labelmap image is not found: %s" % FSStandardLabelMap

  LobesLabelMap = argv['--inputLobeLabels']
  assert os.path.exists(LobesLabelMap), "Input brain lobes labelmap image is not found: %s" % LobesLabelMap

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
  from PreprocessingWorkflow import CreatePreprocessingWorkFlow
  from SuperResolutionWorkflow import CreateSuperResolutionWorkflow
  from DistanceImagesWorkflow import CreateDistanceImagesWorkflow
  from TractographyWorkflow import CreateTractographyWorkflow

  exit = runMainWorkflow(DWISCAN, T1SCAN, T2SCAN, FSStandardLabelMap, LobesLabelMap, CACHEDIR, RESULTDIR, PYTHON_AUX_PATHS, LABELS_CONFIG_FILE)

  sys.exit(exit)
