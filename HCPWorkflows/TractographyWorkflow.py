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
    def MakeInputSRList(DWI_Baseline, DWI_SR_NN, DWI_SR_IFFT, DWI_SR_TV, DWI_SR_WTV):
        imagesList = [DWI_Baseline, DWI_SR_NN, DWI_SR_IFFT, DWI_SR_TV, DWI_SR_WTV]
        return imagesList

    def runWMQL(input_tractography, input_atlas):
        import os
        from wmql import TractQuerier
        # prepare proper prefix
        sr_file_name = os.path.basename(input_tractography)
        sr_file_name_base = os.path.splitext(sr_file_name)[0]
        sr_name = sr_file_name_base.split('_',1)[0]
        out_prefix = sr_name + '_query'
        # run WMQL
        tract_querier = TractQuerier()
        tract_querier.inputs.input_atlas = input_atlas
        tract_querier.inputs.input_tractography = input_tractography
        tract_querier.inputs.out_prefix = out_prefix
        tract_querier.inputs.queries = ['cst.left' ,'cst.right']
        tract_querier.run()
        # check outputs
        output_cst_left_name = out_prefix + '_' + 'cst.left.vtp'
        output_cst_right_name = out_prefix + '_' + 'cst.right.vtp'
        output_cst_left = os.path.join(os.getcwd(), output_cst_left_name)
        output_cst_right = os.path.join(os.getcwd(), output_cst_right_name)
        assert os.path.isfile(output_cst_left), "Output cst tract file is not found: %s" % output_cst_left
        assert os.path.isfile(output_cst_right), "Output cst tract file is not found: %s" % output_cst_right
        return [output_cst_left,output_cst_right]
    #################################
    #\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\#
    TractWF = pe.Workflow(name=WFname)

    inputsSpec = pe.Node(interface=IdentityInterface(fields=['inputLabelMap',
                                                             'DWI_brainMask',
                                                             'DWI_Baseline',
                                                             'DWI_SR_NN',
                                                             'DWI_SR_IFFT',
                                                             'DWI_SR_TV',
                                                             'DWI_SR_WTV',
                                                             ]),
                         name='inputsSpec')

    outputsSpec = pe.Node(interface=IdentityInterface(fields=['ukfTracks','output_cst_left','output_cst_right']),
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

    ##
    ## Step 3: run WMQL
    ##
    tract_querier = pe.Node(interface=Function(function = runWMQL,
                                               input_names=['input_tractography','input_atlas'],
                                               output_names=['output_cst_left','output_cst_right']),
                            name="tract_querier")
    TractWF.connect(UKFNode,'tracts',tract_querier,'input_tractography')
    TractWF.connect(inputsSpec,'inputLabelMap',tract_querier,'input_atlas')
    TractWF.connect(tract_querier,'output_cst_left',outputsSpec,'output_cst_left')
    TractWF.connect(tract_querier,'output_cst_right',outputsSpec,'output_cst_right')

    return TractWF
