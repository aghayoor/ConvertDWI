## \author Ali Ghayoor
##
## This workflow runs the following preprocessing steps on input HCP DWI dataset:
## ...

import nipype
from nipype.interfaces import ants
from nipype.interfaces.base import CommandLine, CommandLineInputSpec, TraitedSpec, File, Directory
from nipype.interfaces.base import traits, isdefined, BaseInterface
from nipype.interfaces.utility import Merge, Split, Function, Rename, IdentityInterface
import nipype.interfaces.io as nio   # Data i/oS
import nipype.pipeline.engine as pe  # pypeline engine
from nipype.interfaces.semtools import *


def CreatePreprocessingWorkFlow(WFname):

    ###### UTILITY FUNCTIONS #######
    #\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/#
    def ExtractSingleShell(nrrd_data,nrrd_bvecs,nrrd_bvals,gradient_index):
        #mask = nrrd_bvals<1010 # corresponds to b=1000
        #mask = (nrrd_bvals==5) + (abs(nrrd_bvals/2000-1) < 1e-2) # corresponds to b=2000
        mask = (nrrd_bvals==5) + (abs(nrrd_bvals/3000-1) < 1e-2) # corresponds to b=3000
        nrrd_bvals = nrrd_bvals[mask]
        nrrd_bvecs = nrrd_bvecs[mask]
        remove_indices = np.where(np.invert(mask))
        nrrd_data = np.delete( nrrd_data, remove_indices, gradient_index)
        return nrrd_data, nrrd_bvecs, nrrd_bvals

    def CorrectDC(inNrrdDWI, outNrrdDWI, useSingleShell):
        from ReadWriteNrrDWI import ReadNAMICDWIFromNrrd, WriteNAMICDWIToNrrd
        nrrd_data,dwi_nrrd_header,nrrd_bvecs,nrrd_bvals,gradient_index = ReadNAMICDWIFromNrrd(DWI_nrrd_fn)
        # Correct direction cosign so the output data is loaded with correct alignment in Slicer
        dwi_nrrd_header[u'space directions'][2][1] = str(-float(dwi_nrrd_header[u'space directions'][2][1]))
        # Adjust gradient directions based on new correction_matrix
        correction_matrix = np.diag([1,-1,1])
        nrrd_bvecs = (np.asmatrix(nrrd_bvecs) * correction_matrix).view(type=np.ndarray)
        #
        if useSingleShell:
           nrrd_data,nrrd_bvecs,nrrd_bvals = ExtractSingleShell(nrrd_data,nrrd_bvecs,nrrd_bvals,gradient_index)
        #
        outNrrdDWIFilename = os.path.join(os.getcwd(), outNrrdDWI) # return output CS filename
        # write corrected nrrd file to disk
        WriteNAMICDWIToNrrd(outNrrdDWIFilename,nrrd_data,nrrd_bvecs,nrrd_bvals,dwi_nrrd_header)
        assert os.path.isfile(outNrrdDWIFilename), "Corrected Nrrd file is not found: %s" % outNrrdDWIFilename
        return outNrrdDWIFilename

    #################################
    #\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\#
    PreProcWF = pe.Workflow(name=WFname)

    inputsSpec = pe.Node(interface=IdentityInterface(fields=['DWIVolume',
                                                             'T1Volume','T2Volume',
                                                             'LabelMapVolume']),
                         name='inputsSpec')

    outputsSpec = pe.Node(interface=IdentityInterface(fields=['DWI_original_corrected',
                                                              'DWI_baseline',
                                                              'DWIBrainMask']),
                          name='outputsSpec')
    ##
    ## STEP 1: Convert input DWI file from Nifti to Nrrd
    ##
    DWIDIR = os.path.dirname(DWIVolume)
    bvecs_fn = os.path.join(DWIDIR,'bvecs')
    bvals_fn = os.path.join(DWIDIR,'bvals')

    dwiConvert = pe.Node(interface=DWIConvert(),name="DWIConvert")
    dwiConvert.inputs.conversionMode = 'FSLToNrrd'
    dwiConvert.inputs.inputBVectors = bvecs_fn
    dwiConvert.inputs.inputBValues = bvals_fn
    dwiConvert.inputs.allowLossyConversion = True # input data is float
    dwiConvert.inputs.transposeInputBVectors = True # bvecs are saved column-wise
    #dwiConvert.inputs.outputVolume = DWI_nrrd_fn

    PreProcWF.connect(inputsSpec,'DWIVolume',dwiConvert,'inputVolume')

    ##
    ## STEP 2: Correct direction cosign so the output data is loaded with correct alignment in Slicer
    ##
    CorrectDCNode = pe.Node(interface=Function(function = CorrectDC,
                                                          input_names=['inNrrdDWI','outNrrdDWI','useSingleShell'],
                                                          output_names=['outNrrdDWIFilename']),
                                       name="CorrectDC")
    CorrectDCNode.inputs.outNrrdDWI = 'DWI_original_corrected.nrrd'
    CorrectDCNode.inputs.useSingleShell = False
    PreProcWF.connect(dwiConvert, 'outputVolume', CorrectDCNode, 'inNrrdDWI')
    #PreProcWF.connect(CorrectDCNode, 'outNrrdDWIFilename', ?, '?')



