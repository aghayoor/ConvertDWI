## \author Ali Ghayoor
##
## This workflow run UKFt on the high-resolution baseline image and each super-resolution reconstructed images
##

import os
import SimpleITK as sitk
import nipype
from nipype.interfaces import ants
from nipype.interfaces.base import CommandLine, CommandLineInputSpec, TraitedSpec, File, Directory
from nipype.interfaces.base import traits, isdefined, BaseInterface
from nipype.interfaces.utility import Merge, Split, Function, Rename, IdentityInterface
import nipype.interfaces.io as nio   # Data i/oS
import nipype.pipeline.engine as pe  # pypeline engine
from nipype.interfaces.semtools import *
from functools import reduce

def CreateTractographyWorkflow(WFname):
    ###### UTILITY FUNCTIONS #######
    #\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/#
    def MakeInputSRList(DWI_Baseline, DWI_SR_NN, DWI_SR_IFFT DWI_SR_TV, DWI_SR_WTV):
        imagesList = [DWI_Baseline, DWI_SR_NN, DWI_SR_IFFT, DWI_SR_TV, DWI_SR_WTV]
        return imagesList
    #################################
    #\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\#
    TractWF = pe.Workflow(name=WFname)

    inputsSpec = pe.Node(interface=IdentityInterface(fields=['DWI_brainMask',
                                                             'DWI_Baseline',
                                                             'DWI_SR_NN',
                                                             'DWI_SR_IFFT',
                                                             'DWI_SR_TV',
                                                             'DWI_SR_WTV',
                                                             ]),
                         name='inputsSpec')

    outputsSpec = pe.Node(interface=IdentityInterface(fields=['ukfTracks']),
                          name='outputsSpec')

    ##
    ## Step 1: Create input list
    ##
    MakeInputSRListNode = pe.Node(Function(function=MakeInputSRList,
                                           input_names=['DWI_Baseline','DWI_SR_NN','DWI_SR_IFFT','DWI_SR_TV','DWI_SR_WTV'],
                                           output_names=['imagesList']),
                                  name="MakeInputSRList")
    TractWF.connect([(inputsSpec,MakeInputSRListNode,[('DWI_Baseline','DWI_Baseline')
                                                      ,('DWI_SR_NN','DWI_SR_NN')
                                                      ,('DWI_SR_IFFT','DWI_SR_IFFT')
                                                      ,('DWI_SR_TV','DWI_SR_TV')
                                                      ,('DWI_SR_WTV','DWI_SR_WTV')
                                                     ])])
    ##
    ## Step 2: UKF Processing
    ##
    UKFNode = pe.MapNode(interface=UKFTractography(), name= "UKFRunRecordStates", iterfield=['dwiFile','tracts'])
    UKFNode.inputs.tracts = ['Baseline_ukfTracts.vtp','NN_ukfTracts.vtp','IFFT_ukfTracts.vtp','TV_ukfTracts.vtp','WTV_ukfTracts.vtp']
    UKFNode.inputs.numTensor = '2'
    UKFNode.inputs.freeWater = True ## default False
    UKFNode.inputs.minFA = 0.15
    UKFNode.inputs.minGA = 0.15
    UKFNode.inputs.seedFALimit = 0.06
    UKFNode.inputs.Ql = 70
    UKFNode.inputs.Rs = 0.025
    UKFNode.inputs.recordLength = 1.25
    UKFNode.inputs.recordTensors = True
    UKFNode.inputs.recordFreeWater = True
    UKFNode.inputs.recordFA = True
    UKFNode.inputs.recordTrace = True
    UKFNode.inputs.seedsPerVoxel = 1
    TractWF.connect(MakeInputSRListNode, 'imagesList', UKFNode, 'dwiFile')
    TractWF.connect(inputsSpec, 'DWI_brainMask', UKFNode, 'maskFile')
    TractWF.connect(UKFNode,'tracts',outputsSpec,'ukfTracks')

    return TractWF
