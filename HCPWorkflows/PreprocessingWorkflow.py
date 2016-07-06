## \author Ali Ghayoor
##
## This workflow runs the following preprocessing steps on input HCP DWI dataset:
## ...

import os
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
    def FindBVecBVals(DWIVolume):
        import os
        assert os.path.exists(DWIVolume), "File not found: %s" % DWIVolume
        DWIDIR = os.path.dirname(DWIVolume)
        bvecs_fn = os.path.join(DWIDIR,'bvecs')
        bvals_fn = os.path.join(DWIDIR,'bvals')
        return [bvecs_fn, bvals_fn]

    def CorrectDC(inNrrdDWI, outNrrdDWI, useSingleShell):
        import os
        import numpy as np
        from ReadWriteNrrdDWI import ReadNAMICDWIFromNrrd, WriteNAMICDWIToNrrd
        # utility function
        def ExtractSingleShell(nrrd_data,nrrd_bvecs,nrrd_bvals,gradient_index):
            #mask = nrrd_bvals<1010 # corresponds to b=1000
            mask = (nrrd_bvals==5) + (abs(nrrd_bvals/2000-1) < 1e-2) # corresponds to b=2000
            #mask = (nrrd_bvals==5) + (abs(nrrd_bvals/3000-1) < 1e-2) # corresponds to b=3000
            nrrd_bvals = nrrd_bvals[mask]
            nrrd_bvecs = nrrd_bvecs[mask]
            remove_indices = np.where(np.invert(mask))
            nrrd_data = np.delete( nrrd_data, remove_indices, gradient_index)
            return nrrd_data, nrrd_bvecs, nrrd_bvals
        #
        nrrd_data,dwi_nrrd_header,nrrd_bvecs,nrrd_bvals,gradient_index = ReadNAMICDWIFromNrrd(inNrrdDWI)
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

    def MaskDWI(inNrrdDWI, inputMask, outNrrdDWI):
        import os
        import numpy as np
        import SimpleITK as sitk
        from ReadWriteNrrdDWI import ReadNAMICDWIFromNrrd, WriteNAMICDWIToNrrd
        assert os.path.exists(inNrrdDWI), "File not found: %s" % inNrrdDWI
        assert os.path.exists(inputMask), "File not found: %s" % inputMask
        mask_img = sitk.ReadImage(inputMask)
        mask_data = sitk.GetArrayFromImage(mask_img)
        dwi_data,dwi_header,bvecs,bvals,gradient_index = ReadNAMICDWIFromNrrd(inNrrdDWI)
        if gradient_index != 0:
           raise ValueError('Program expects a gradient index of zero for HCP data, but gradient index is {0}'.format(gradient_index))
        numOfComponents = dwi_data.shape[gradient_index]
        for idx in range(numOfComponents):
            dwi_3d = dwi_data[idx,:,:,:]
            '''
            To convert the above numpy array to a b0 image, we need careful attention to the order of index and dimensions.
            ITK's Image class does not have a bracket operator. It has a GetPixel which takes an ITK Index object as an argument,
            which is an array ordered as (x,y,z). This is the convention that SimpleITK's Image class uses for the GetPixel method as well.
            While in numpy, an array is indexed in the opposite order (z,y,x).
            '''
            dwi_3d = np.transpose(dwi_3d,(2, 1, 0))
            dwi_3d = dwi_3d*mask_data
            dwi_3d = np.transpose(dwi_3d,(2, 1, 0))
            dwi_data[idx,:,:,:] = dwi_3d
        outMaskedDWIFilename = os.path.join(os.getcwd(), outNrrdDWI)
        # write corrected nrrd file to disk
        WriteNAMICDWIToNrrd(outMaskedDWIFilename,dwi_data,bvecs,bvals,dwi_header)
        assert os.path.isfile(outMaskedDWIFilename), "Masked DWI file is not found: %s" % outMaskedDWIFilename
        return outMaskedDWIFilename

    # remove the skull from T1/T2 volume
    def ExtractBRAINFromHead(inputVolume, brainLabels):
        import os
        import SimpleITK as sitk
        # utility function
        def LinearResampling(inputImage, refImage):
            resFilt = sitk.ResampleImageFilter()
            resFilt.SetReferenceImage(refImage)
            #resFilt.SetOutputPixelType(sitk.sitkFloat32)
            resFilt.SetInterpolator(sitk.sitkLinear)
            resampledImage = resFilt.Execute(inputImage)
            return resampledImage
        #
        # Remove skull from the head scan
        assert os.path.exists(inputVolume), "File not found: %s" % inputVolume
        assert os.path.exists(brainLabels), "File not found: %s" % brainLabels
        headImage = sitk.ReadImage(inputVolume)
        labelsMap = sitk.ReadImage(brainLabels)
        brain_mask = labelsMap>0
        ## dilate brain mask
        #dilateFilter = sitk.BinaryDilateImageFilter()
        #dilateFilter.SetKernelRadius(2)
        #brain_mask = dilateFilter.Execute( brain_mask )
        ##
        if (headImage.GetSpacing() != brain_mask.GetSpacing()):
            headImage = LinearResampling(headImage,brain_mask)
        brainImage = sitk.Cast(headImage,sitk.sitkFloat32) * sitk.Cast(brain_mask,sitk.sitkFloat32)
        outputVolume = os.path.realpath('Stripped_'+ os.path.basename(inputVolume))
        sitk.WriteImage(brainImage, outputVolume)
        return outputVolume

    def MakeBrainStripperInputFilesList(inputT1, inputT2):
        import os
        assert os.path.exists(inputT1), "File not found: %s" % inputT1
        assert os.path.exists(inputT2), "File not found: %s" % inputT2
        imagesList = [inputT1, inputT2]
        return imagesList

    def MakeResamplerInFileList(inputT1, inputT2, inputLabelMap):
        imagesList = [inputT1, inputT2, inputLabelMap]
        return imagesList

    def MakeGenerateEdgeMapInputList(inputT1, inputT2, inputBrainMask):
        import os
        import SimpleITK as sitk
        assert os.path.exists(inputT1), "File not found: %s" % inputT1
        assert os.path.exists(inputT2), "File not found: %s" % inputT2
        assert os.path.exists(inputBrainMask), "File not found: %s" % inputBrainMask
        brainMask = sitk.ReadImage(inputBrainMask)
        brainMask = brainMask > 0
        t1 = sitk.ReadImage(inputT1)
        t1_masked = sitk.Cast(t1,sitk.sitkInt16) * sitk.Cast(brainMask,sitk.sitkInt16)
        outputT1 = os.path.join(os.getcwd(), 'StrippedT1_125.nrrd')
        sitk.WriteImage(t1_masked,outputT1)
        t2 = sitk.ReadImage(inputT2)
        t2_masked = sitk.Cast(t2,sitk.sitkInt16) * sitk.Cast(brainMask,sitk.sitkInt16)
        outputT2 = os.path.join(os.getcwd(), 'StrippedT2_125.nrrd')
        sitk.WriteImage(t2_masked,outputT2)
        imagesList = [outputT1, outputT2]
        return imagesList

    # This function helps to pick desirable output from the output list
    def pickFromList(inlist,item):
        return inlist[item]

    #################################
    #\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\#

    PreProcWF = pe.Workflow(name=WFname)

    inputsSpec = pe.Node(interface=IdentityInterface(fields=['DWIVolume','T1Volume','T2Volume','LabelMapVolume']),
                         name='inputsSpec')

    outputsSpec = pe.Node(interface=IdentityInterface(fields=['DWI_corrected_originalSpace',
                                                              'DWI_corrected_alignedSpace',
                                                              'DWI_corrected_alignedSpace_masked',
                                                              'DWIBrainMask',
                                                              'StrippedT1_125',
                                                              'StrippedT2_125',
                                                              'MaximumGradientImage',
                                                              'EdgeMap']),
                          name='outputsSpec')

    ##
    ## STEP 1: Convert input DWI file from Nifti to Nrrd
    ##
    FindBVecBVals = pe.Node(interface=Function(function = FindBVecBVals,
                                               input_names=['DWIVolume'],
                                               output_names=['bvecs_fn','bvals_fn']),
                            name="FindBVecBVals")
    PreProcWF.connect(inputsSpec,'DWIVolume',FindBVecBVals,'DWIVolume')

    dwiConvert = pe.Node(interface=DWIConvert(),name="DWIConvert")
    dwiConvert.inputs.conversionMode = 'FSLToNrrd'
    dwiConvert.inputs.allowLossyConversion = True # input data is float
    dwiConvert.inputs.transposeInputBVectors = True # bvecs are saved column-wise
    dwiConvert.inputs.outputVolume = 'DWI_data.nrrd'
    PreProcWF.connect(inputsSpec,'DWIVolume',dwiConvert,'inputVolume')
    PreProcWF.connect(FindBVecBVals,'bvecs_fn',dwiConvert,'inputBVectors')
    PreProcWF.connect(FindBVecBVals,'bvals_fn',dwiConvert,'inputBValues')

    ##
    ## STEP 2: Correct direction cosign so the output data is loaded with correct alignment in Slicer
    ##
    CorrectDCNode = pe.Node(interface=Function(function = CorrectDC,
                                               input_names=['inNrrdDWI','outNrrdDWI','useSingleShell'],
                                               output_names=['outNrrdDWIFilename']),
                            name="CorrectDirectionCosign")
    CorrectDCNode.inputs.outNrrdDWI = 'DWI_corrected_originalSpace.nrrd'
    CorrectDCNode.inputs.useSingleShell = True
    PreProcWF.connect(dwiConvert, 'outputVolume', CorrectDCNode, 'inNrrdDWI')
    PreProcWF.connect(CorrectDCNode, 'outNrrdDWIFilename', outputsSpec, 'DWI_corrected_originalSpace')

    #
    ## STEP 3: align corrected DWI image to the space of T1/T2
    #
    #  DWI_corrected_originalSpace -> ExtractB0
    #                                     |
    #                                     ~
    #                             Register_B0_to_T2 -> gtractResampleInPlace -> DWI_corrected_alignedSpace
    #                                     ~
    #                                     |
    #                         T2 -> skullStrip
    #

    # Step3_1: remove the skull from the T2 volume
    MakeBrainStripperInputFilesListNode = pe.Node(Function(function=MakeBrainStripperInputFilesList,
                                                           input_names=['inputT1','inputT2'],
                                                           output_names=['imagesList']),
                                                  name="MakeBrainStripperInputFilesListNode")
    PreProcWF.connect([(inputsSpec,MakeBrainStripperInputFilesListNode,[('T1Volume','inputT1'),('T2Volume','inputT2')])])

    ExtractBRAINFromHeadNode = pe.MapNode(interface=Function(function = ExtractBRAINFromHead,
                                                             input_names=['inputVolume','brainLabels'],
                                                             output_names=['outputVolume']),
                                          name="ExtractBRAINFromHead",
                                          iterfield=['inputVolume'])

    PreProcWF.connect(inputsSpec, 'LabelMapVolume', ExtractBRAINFromHeadNode, 'brainLabels')
    PreProcWF.connect(MakeBrainStripperInputFilesListNode, 'imagesList', ExtractBRAINFromHeadNode, 'inputVolume')

    # Step3_2: extract B0 from DWI volume
    EXTRACT_B0 = pe.Node(interface=extractNrrdVectorIndex(),name="EXTRACT_B0")
    EXTRACT_B0.inputs.vectorIndex = 0
    EXTRACT_B0.inputs.outputVolume = 'B0_Image.nrrd'
    PreProcWF.connect(CorrectDCNode,'outNrrdDWIFilename',EXTRACT_B0,'inputVolume')

    # Step3_3: Register B0 to T2 space using BRAINSFit
    BFit_B0toT2 = pe.Node(interface=BRAINSFit(), name="BFit_B0toT2")
    BFit_B0toT2.inputs.costMetric = "MMI"
    BFit_B0toT2.inputs.numberOfSamples = 100000
    BFit_B0toT2.inputs.numberOfIterations = [1500]
    BFit_B0toT2.inputs.numberOfHistogramBins = 50
    BFit_B0toT2.inputs.maximumStepLength = 0.2
    BFit_B0toT2.inputs.minimumStepLength = [0.00005]
    BFit_B0toT2.inputs.useRigid = True
    BFit_B0toT2.inputs.useAffine = True
    BFit_B0toT2.inputs.maskInferiorCutOffFromCenter = 65
    BFit_B0toT2.inputs.maskProcessingMode = "ROIAUTO"
    BFit_B0toT2.inputs.ROIAutoDilateSize = 13
    BFit_B0toT2.inputs.backgroundFillValue = 0.0
    BFit_B0toT2.inputs.initializeTransformMode = 'useCenterOfHeadAlign'
    BFit_B0toT2.inputs.strippedOutputTransform = "B0ToT2_RigidTransform.h5"
    BFit_B0toT2.inputs.writeOutputTransformInFloat = True
    PreProcWF.connect(EXTRACT_B0, 'outputVolume', BFit_B0toT2, 'movingVolume')
    PreProcWF.connect(ExtractBRAINFromHeadNode, ('outputVolume', pickFromList, 1), BFit_B0toT2, 'fixedVolume')

    # Step3_4: ResampleInPlace the DWI image to T2 image space
    gtractResampleDWIInPlace_Trigid = pe.Node(interface=gtractResampleDWIInPlace(),
                                              name="gtractResampleDWIInPlace_Trigid")
    PreProcWF.connect(CorrectDCNode,'outNrrdDWIFilename',
                      gtractResampleDWIInPlace_Trigid,'inputVolume')
    PreProcWF.connect(BFit_B0toT2,'strippedOutputTransform',
                      gtractResampleDWIInPlace_Trigid,'inputTransform')
    gtractResampleDWIInPlace_Trigid.inputs.outputVolume = 'DWI_corrected_alignedSpace.nrrd'
    gtractResampleDWIInPlace_Trigid.inputs.outputResampledB0 = 'DWI_corrected_alignedSpace_B0.nrrd'
    PreProcWF.connect(gtractResampleDWIInPlace_Trigid, 'outputVolume', outputsSpec, 'DWI_corrected_alignedSpace')


    ##
    ## STEP 4: Resample T1/T2/Labelmap to the voxel space of aligned DWI
    ##
    MakeResamplerInFilesListNode = pe.Node(Function(function=MakeResamplerInFileList,
                                                    input_names=['inputT1','inputT2','inputLabelMap'],
                                                    output_names=['imagesList']),
                                           name="MakeResamplerInFilesListNode")
    PreProcWF.connect(ExtractBRAINFromHeadNode,('outputVolume', pickFromList, 0),
                      MakeResamplerInFilesListNode,'inputT1')
    PreProcWF.connect(ExtractBRAINFromHeadNode,('outputVolume', pickFromList, 1),
                      MakeResamplerInFilesListNode,'inputT2')
    PreProcWF.connect(inputsSpec, 'LabelMapVolume', MakeResamplerInFilesListNode, 'inputLabelMap')
    #

    ResampleToAlignedDWIResolution = pe.MapNode(interface=BRAINSResample(),
                                                name="ResampleToAlignedDWIResolution",
                                                iterfield=['inputVolume', 'pixelType', 'outputVolume'])
    ResampleToAlignedDWIResolution.inputs.interpolationMode = 'Linear'
    ResampleToAlignedDWIResolution.inputs.outputVolume = ['StrippedT1_125.nrrd','StrippedT2_125.nrrd','DWIBrainMask.nrrd']
    ResampleToAlignedDWIResolution.inputs.pixelType = ['ushort','ushort','binary']
    # warpTransform is identity
    PreProcWF.connect(gtractResampleDWIInPlace_Trigid,'outputResampledB0',ResampleToAlignedDWIResolution,'referenceVolume')
    PreProcWF.connect(MakeResamplerInFilesListNode,'imagesList',ResampleToAlignedDWIResolution,'inputVolume')
    PreProcWF.connect(ResampleToAlignedDWIResolution, ('outputVolume', pickFromList, 2),
                      outputsSpec, 'DWIBrainMask')

    ##
    ## STEP 5: Create main outputs to Super-Resolution Wrokflow (edgeMap and masked DWI)
    ##

    ## Step5_1: Make a list for input structral MR images
    MakeGenerateEdgeMapInputListNode = pe.Node(Function(function=MakeGenerateEdgeMapInputList,
                                                    input_names=['inputT1','inputT2','inputBrainMask'],
                                                    output_names=['imagesList']),
                                           name="MakeGenerateEdgeMapInputList")
    PreProcWF.connect(ResampleToAlignedDWIResolution, ('outputVolume', pickFromList, 0),
                      MakeGenerateEdgeMapInputListNode,'inputT1')
    PreProcWF.connect(ResampleToAlignedDWIResolution, ('outputVolume', pickFromList, 1),
                      MakeGenerateEdgeMapInputListNode,'inputT2')
    PreProcWF.connect(ResampleToAlignedDWIResolution, ('outputVolume', pickFromList, 2),
                      MakeGenerateEdgeMapInputListNode,'inputBrainMask')
    # pass output T1/T2 to outputSpecs
    PreProcWF.connect(MakeGenerateEdgeMapInputListNode, ('imagesList', pickFromList, 0),
                      outputsSpec, 'StrippedT1_125')
    PreProcWF.connect(MakeGenerateEdgeMapInputListNode, ('imagesList', pickFromList, 1),
                      outputsSpec, 'StrippedT2_125')

    ## Step5_2: Generate EdgeMap
    GEM = pe.Node(interface=GenerateEdgeMapImage(), name="EdgeMapImage")
    GEM.inputs.minimumOutputRange = 1
    GEM.inputs.maximumOutputRange = 255
    GEM.inputs.lowerPercentileMatching = 0.50
    GEM.inputs.upperPercentileMatching = 0.98 # a value between 92% to 98%
    GEM.inputs.outputMaximumGradientImage = 'MaximumGradientImage.nrrd'
    GEM.inputs.outputEdgeMap = 'EdgeMap.nrrd'
    PreProcWF.connect(MakeGenerateEdgeMapInputListNode, 'imagesList', GEM, 'inputMRVolumes')
    #PreProcWF.connect(ResampleToAlignedDWIResolution,('outputVolume', pickFromList, 2), GEM, 'inputMask') # input brain mask to calculate quantiles
    PreProcWF.connect(GEM, 'outputMaximumGradientImage', outputsSpec, 'MaximumGradientImage')
    PreProcWF.connect(GEM, 'outputEdgeMap', outputsSpec, 'EdgeMap')

    ##
    ## STEP5_3: Create masked DWI image
    ##
    MaskDWINode = pe.Node(interface=Function(function = MaskDWI,
                                               input_names=['inNrrdDWI','inputMask','outNrrdDWI'],
                                               output_names=['outMaskedDWIFilename']),
                            name="MaskDWINode")
    MaskDWINode.inputs.outNrrdDWI = 'DWI_corrected_alignedSpace_masked.nrrd'
    PreProcWF.connect(gtractResampleDWIInPlace_Trigid, 'outputVolume', MaskDWINode, 'inNrrdDWI')
    PreProcWF.connect(ResampleToAlignedDWIResolution, ('outputVolume', pickFromList, 2), MaskDWINode, 'inputMask')
    PreProcWF.connect(MaskDWINode, 'outMaskedDWIFilename', outputsSpec, 'DWI_corrected_alignedSpace_masked')

    return PreProcWF
