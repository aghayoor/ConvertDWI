## \author Ali Ghayoor
##
## This workflow computes several distance metrics for input high-resolution DWI and input super-resolution reconstructed images.
##

import os
import nipype
from nipype.interfaces.base import CommandLine, CommandLineInputSpec, TraitedSpec, File, Directory
from nipype.interfaces.base import traits, isdefined, BaseInterface
from nipype.interfaces.utility import Merge, Split, Function, Rename, IdentityInterface
import nipype.interfaces.io as nio   # Data i/oS
import nipype.pipeline.engine as pe  # pypeline engine
from nipype.interfaces.semtools import *

def CreateDistanceImagesWorkflow(WFname):
    ###### UTILITY FUNCTIONS #######
    #\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/#
    def ComputeDistanceImages(DWI_baseline,DWI_sr,DWI_brainMask):
        import os
        import numpy as np
        from scipy.linalg import logm
        from scipy.linalg import eigvalsh # eigvalsh(A,B) is joint eigenvalues of A and B
        import SimpleITK as sitk
        from ReadWriteNrrdDWI import ReadNAMICDWIFromNrrd, WriteNAMICDWIToNrrd
        #########
        ### distance functions ###
        # Frobenius distance
        def distance_euclid(tenfit_A, tenfit_B):
            return np.linalg.norm(tenfit_A.quadratic_form - tenfit_B.quadratic_form, ord='fro')
        # Log-Euclidian distance
        def distance_logeuclid(tenfit_A, tenfit_B):
            def fro_norm(A,B):
                return np.linalg.norm(A-B, ord='fro')
            return fro_norm(logm(tenfit_A.quadratic_form),logm(tenfit_B.quadratic_form))
        # Reimannian distance
        def distance_reimann(tenfit_A, tenfit_B):
            return np.sqrt((np.log(eigvalsh(tenfit_A.quadratic_form,tenfit_B.quadratic_form))**2).sum())
        # Kullback-Leibler distance
        def distance_kullback(tenfit_A, tenfit_B):
            A = tenfit_A.quadratic_form
            B = tenfit_B.quadratic_form
            dim = A.shape[0]
            kl = np.sqrt( np.trace( np.dot(np.linalg.inv(A),B)+np.dot(np.linalg.inv(B),A) ) - 2*dim )
            return 0.5*kl
        #########
        assert os.path.exists(DWI_baseline), "File not found: %s" % DWI_baseline
        assert os.path.exists(DWI_sr), "File not found: %s" % DWI_sr
        assert os.path.exists(DWI_brainMask), "File not found: %s" % DWI_brainMask
        mask = sitk.ReadImage(DWI_brainMask)
        # read DWI nrrd files
        data_base,nrrd_header_base,bvecs_base,bvals_base,grad_idx_base = ReadNAMICDWIFromNrrd(DWI_baseline)
        data_sr,nrrd_header_sr,bvecs_sr,bvals_sr,grad_idx_sr = ReadNAMICDWIFromNrrd(DWI_sr)
        # compute tensorfits
        from dipy.core.gradients import gradient_table
        from dipy.reconst.dti import TensorModel
        # base tensorfit
        gtab_base = gradient_table(bvals_base, bvecs_base)
        ten_base = TensorModel(gtab_base)
        tenfit_base = ten_base.fit(data_base)
        # super-res reconstructed tensorfit
        gtab_sr = gradient_table(bvals_sr, bvecs_sr)
        ten_sr = TensorModel(gtab_sr)
        tenfit_sr = ten_base.fit(data_sr)
        # create empty distance images
        size = [data_base.shape[0],data_base.shape[1],data_base.shape[2]] #data_base.shape[4] is gradient components
        fa_distance_image = sitk.Image(size, sitk.sitkFloat32)
        fa_distance_image.CopyInformation(mask)
        #
        md_distance_image = sitk.Image(size, sitk.sitkFloat32)
        md_distance_image.CopyInformation(mask)
        #
        rd_distance_image = sitk.Image(size, sitk.sitkFloat32)
        rd_distance_image.CopyInformation(mask)
        #
        ad_distance_image = sitk.Image(size, sitk.sitkFloat32)
        ad_distance_image.CopyInformation(mask)
        #
        frobenius_distance_image = sitk.Image(size, sitk.sitkFloat32)
        frobenius_distance_image.CopyInformation(mask)
        #
        logeuclid_distance_image = sitk.Image(size, sitk.sitkFloat32)
        logeuclid_distance_image.CopyInformation(mask)
        #
        reimann_distance_image = sitk.Image(size, sitk.sitkFloat32)
        reimann_distance_image.CopyInformation(mask)
        #
        kullback_distance_image = sitk.Image(size, sitk.sitkFloat32)
        kullback_distance_image.CopyInformation(mask)
        #
        # for loop to fill the distance images
        for i in xrange(size[0]):
            for j in xrange(size[1]):
                for k in xrange(size[2]):
                    fa_distance_image[i,j,k] = abs(tenfit_base[i,j,k].fa - tenfit_sr[i,j,k].fa)
                    md_distance_image[i,j,k] = abs(tenfit_base[i,j,k].md - tenfit_sr[i,j,k].md)
                    rd_distance_image[i,j,k] = abs(tenfit_base[i,j,k].rd - tenfit_sr[i,j,k].rd)
                    ad_distance_image[i,j,k] = abs(tenfit_base[i,j,k].ad - tenfit_sr[i,j,k].ad)
                    frobenius_distance_image[i,j,k] = distance_euclid(tenfit_base[i,j,k], tenfit_sr[i,j,k])
                    logeuclid_distance_image[i,j,k] = distance_logeuclid(tenfit_base[i,j,k], tenfit_sr[i,j,k])
                    reimann_distance_image[i,j,k] = distance_reimann(tenfit_base[i,j,k], tenfit_sr[i,j,k])
                    kullback_distance_image[i,j,k] = distance_kullback(tenfit_base[i,j,k], tenfit_sr[i,j,k])
        #
        maskf = sitk.Cast(mask, sitk.sitkFloat32)
        #
        ## create proper SR prefix for output file name
        sr_file_name = os.path.basename(DWI_sr)
        sr_file_name_base = os.path.splitext(sr_file_name)[0]
        sr_name = sr_file_name_base.split('_',2)[2]
        #
        fa_distance_image_fn = os.path.realpath(sr_name + '_FA_distance.nrrd')
        md_distance_image_fn = os.path.realpath(sr_name + '_MD_distance.nrrd')
        rd_distance_image_fn = os.path.realpath(sr_name + '_RD_distance.nrrd')
        ad_distance_image_fn = os.path.realpath(sr_name + '_AD_distance.nrrd')
        frobenius_distance_image_fn = os.path.realpath(sr_name + '_Frobenius_distance.nrrd')
        logeuclid_distance_image_fn = os.path.realpath(sr_name + '_Logeuclid_distance.nrrd')
        reimann_distance_image_fn = os.path.realpath(sr_name + '_Reimann_distance.nrrd')
        kullback_distance_image_fn = os.path.realpath(sr_name + '_Kullback_distance.nrrd')
        #
        ## mask out the background
        fa_distance_image = sitk.Multiply(fa_distance_image,maskf)
        md_distance_image = sitk.Multiply(md_distance_image,maskf)
        rd_distance_image = sitk.Multiply(rd_distance_image,maskf)
        ad_distance_image = sitk.Multiply(ad_distance_image,maskf)
        frobenius_distance_image = sitk.Multiply(frobenius_distance_image,maskf)
        logeuclid_distance_image = sitk.Multiply(logeuclid_distance_image,maskf)
        reimann_distance_image = sitk.Multiply(reimann_distance_image,maskf)
        kullback_distance_image = sitk.Multiply(kullback_distance_image,maskf)
        #
        ## clip outliers by computing 95 percentiles
        p = 95.0
        # FA
        fa_distance_arr = sitk.GetArrayFromImage(fa_distance_image)
        np.clip(fa_distance_arr, fa_distance_arr.min(), np.percentile(fa_distance_arr,p), fa_distance_arr)
        fa_distance_image = sitk.GetImageFromArray(fa_distance_arr)
        fa_distance_image.CopyInformation(mask)
        # MD
        md_distance_arr = sitk.GetArrayFromImage(md_distance_image)
        np.clip(md_distance_arr, md_distance_arr.min(), np.percentile(md_distance_arr,p), md_distance_arr)
        md_distance_image = sitk.GetImageFromArray(md_distance_arr)
        md_distance_image.CopyInformation(mask)
        # RD
        rd_distance_arr = sitk.GetArrayFromImage(rd_distance_image)
        np.clip(rd_distance_arr, rd_distance_arr.min(), np.percentile(rd_distance_arr,p), rd_distance_arr)
        rd_distance_image = sitk.GetImageFromArray(rd_distance_arr)
        rd_distance_image.CopyInformation(mask)
        # AD
        ad_distance_arr = sitk.GetArrayFromImage(ad_distance_image)
        np.clip(ad_distance_arr, ad_distance_arr.min(), np.percentile(ad_distance_arr,p), ad_distance_arr)
        ad_distance_image = sitk.GetImageFromArray(ad_distance_arr)
        ad_distance_image.CopyInformation(mask)
        # frobenius
        frobenius_distance_arr = sitk.GetArrayFromImage(frobenius_distance_image)
        np.clip(frobenius_distance_arr, frobenius_distance_arr.min(), np.percentile(frobenius_distance_arr,p), frobenius_distance_arr)
        frobenius_distance_image = sitk.GetImageFromArray(frobenius_distance_arr)
        frobenius_distance_image.CopyInformation(mask)
        # logeuclid
        logeuclid_distance_arr = sitk.GetArrayFromImage(logeuclid_distance_image)
        np.clip(logeuclid_distance_arr, logeuclid_distance_arr.min(), np.percentile(logeuclid_distance_arr,p), logeuclid_distance_arr)
        logeuclid_distance_image = sitk.GetImageFromArray(logeuclid_distance_arr)
        logeuclid_distance_image.CopyInformation(mask)
        # reimann
        reimann_distance_arr = sitk.GetArrayFromImage(reimann_distance_image)
        np.clip(reimann_distance_arr, reimann_distance_arr.min(), np.percentile(reimann_distance_arr,p), reimann_distance_arr)
        reimann_distance_image = sitk.GetImageFromArray(reimann_distance_arr)
        reimann_distance_image.CopyInformation(mask)
        # kullback
        kullback_distance_arr = sitk.GetArrayFromImage(kullback_distance_image)
        np.clip(kullback_distance_arr, kullback_distance_arr.min(), np.percentile(kullback_distance_arr,p), kullback_distance_arr)
        kullback_distance_image = sitk.GetImageFromArray(kullback_distance_arr)
        kullback_distance_image.CopyInformation(mask)
        #
        ## Write out the distance images
        sitk.WriteImage(fa_distance_image,fa_distance_image_fn)
        sitk.WriteImage(md_distance_image,md_distance_image_fn)
        sitk.WriteImage(rd_distance_image,rd_distance_image_fn)
        sitk.WriteImage(ad_distance_image,ad_distance_image_fn)
        sitk.WriteImage(frobenius_distance_image,frobenius_distance_image_fn)
        sitk.WriteImage(logeuclid_distance_image,logeuclid_distance_image_fn)
        sitk.WriteImage(reimann_distance_image,reimann_distance_image_fn)
        sitk.WriteImage(kullback_distance_image,kullback_distance_image_fn)
        #
        return [fa_distance_image_fn,
                md_distance_image_fn,
                rd_distance_image_fn,
                ad_distance_image_fn,
                frobenius_distance_image_fn,
                logeuclid_distance_image_fn,
                reimann_distance_image_fn,
                kullback_distance_image_fn]

    def MakeInputSRList(DWI_SR_NN, DWI_SR_IFFT, DWI_SR_TV, DWI_SR_WTV):
        imagesList = [DWI_SR_NN, DWI_SR_IFFT, DWI_SR_TV, DWI_SR_WTV]
        return imagesList

    def MakePurePlugsMaskInputList(inputT1, inputT2, inputIDWI):
        imagesList = [inputT1, inputT2, inputIDWI]
        return imagesList

    def CreateBrainPurePlugsMask(PurePlugsMask, DWI_brainMask):
        import os
        import SimpleITK as sitk
        assert os.path.exists(PurePlugsMask), "File not found: %s" % PurePlugsMask
        assert os.path.exists(DWI_brainMask), "File not found: %s" % DWI_brainMask
        ppmask = sitk.ReadImage(PurePlugsMask)
        brainmask = sitk.ReadImage(DWI_brainMask)
        brainPurePlugs = ppmask * sitk.Cast(brainmask,sitk.sitkUInt8)
        BrainPurePlugsMask = os.path.realpath('BrainPurePlugsMask.nrrd')
        sitk.WriteImage(brainPurePlugs,BrainPurePlugsMask)
        assert os.path.exists(BrainPurePlugsMask), "Output BrainPurePlugsMask file not found: %s" % BrainPurePlugsMask
        return BrainPurePlugsMask
    #################################
    #\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\#
    DistWF = pe.Workflow(name=WFname)

    inputsSpec = pe.Node(interface=IdentityInterface(fields=['inputT1',
                                                             'inputT2',
                                                             'LobesLabelMapVolume',
                                                             'DWI_brainMask',
                                                             'DWI_corrected_alignedSpace_masked',
                                                             'DWI_Baseline',
                                                             'DWI_SR_NN',
                                                             'DWI_SR_IFFT',
                                                             'DWI_SR_TV',
                                                             'DWI_SR_WTV'
                                                            ]),
                         name='inputsSpec')

    outputsSpec = pe.Node(interface=IdentityInterface(fields=['FA_distance',
                                                              'MD_distance',
                                                              'RD_distance',
                                                              'AD_distance',
                                                              'Frobenius_distance',
                                                              'Logeuclid_distance',
                                                              'Reimann_distance',
                                                              'Kullback_distance',
                                                              'idwi_image',
                                                              'PurePlugsMask'
                                                             ]),
                          name='outputsSpec')
    ##
    ## Step 1: Create input list
    ##
    MakeInputSRListNode = pe.Node(Function(function=MakeInputSRList,
                                           input_names=['DWI_SR_NN','DWI_SR_IFFT','DWI_SR_TV','DWI_SR_WTV'],
                                           output_names=['imagesList']),
                                           name="MakeInputSRList")
    DistWF.connect([(inputsSpec,MakeInputSRListNode,[('DWI_SR_NN','DWI_SR_NN'),
                                                     ('DWI_SR_IFFT','DWI_SR_IFFT'),
                                                     ('DWI_SR_TV','DWI_SR_TV'),
                                                     ('DWI_SR_WTV','DWI_SR_WTV')
                                                    ])])
    ##
    ## Step 2: Create distance images
    ##
    ComputeDistanceImages = pe.MapNode(interface=Function(function = ComputeDistanceImages,
                                       input_names=['DWI_baseline','DWI_sr','DWI_brainMask'],
                                       output_names=['fa_distance_image_fn','md_distance_image_fn',
                                                     'rd_distance_image_fn','ad_distance_image_fn',
                                                     'frobenius_distance_image_fn','logeuclid_distance_image_fn',
                                                     'reimann_distance_image_fn','kullback_distance_image_fn'
                                                    ]),
                                       name="ComputeDistanceImages",
                                       iterfield=['DWI_sr'])
    DistWF.connect(MakeInputSRListNode, 'imagesList', ComputeDistanceImages, 'DWI_sr')
    DistWF.connect([(inputsSpec,ComputeDistanceImages,[('DWI_Baseline','DWI_baseline'),
                                                       ('DWI_brainMask','DWI_brainMask')])])
    DistWF.connect([(ComputeDistanceImages,outputsSpec,[('fa_distance_image_fn','FA_distance'),
                                                        ('md_distance_image_fn','MD_distance'),
                                                        ('rd_distance_image_fn','RD_distance'),
                                                        ('ad_distance_image_fn','AD_distance'),
                                                        ('frobenius_distance_image_fn','Frobenius_distance'),
                                                        ('logeuclid_distance_image_fn','Logeuclid_distance'),
                                                        ('reimann_distance_image_fn','Reimann_distance'),
                                                        ('kullback_distance_image_fn','Kullback_distance')])])

    ##
    ## Step 3: Create IDWI image
    ##
    DTIEstim = pe.Node(interface=dtiestim(), name="DTIEstim")
    DTIEstim.inputs.method = 'wls'
    DTIEstim.inputs.threshold = 0
    #DTIEstim.inputs.correctionType = 'nearest'
    DTIEstim.inputs.tensor_output = 'DTI_Output.nrrd'
    DTIEstim.inputs.idwi = 'IDWI_Output.nrrd'
    DTIEstim.inputs.B0 = 'average_B0.nrrd'
    DistWF.connect(inputsSpec, 'DWI_corrected_alignedSpace_masked', DTIEstim, 'dwi_image')
    DistWF.connect(inputsSpec, 'DWI_brainMask', DTIEstim, 'brain_mask')
    DistWF.connect(DTIEstim, 'idwi', outputsSpec, 'idwi_image')

    ##
    ## Step 4: Generate Pure plugs mask
    ##

    ## Step 4_1: Make a list for input modality list
    MakePurePlugsMaskInputListNode = pe.Node(Function(function=MakePurePlugsMaskInputList,
                                                      input_names=['inputT1','inputT2','inputIDWI'],
                                                      output_names=['imagesList']),
                                             name="MakePurePlugsMaskInputList")
    DistWF.connect(inputsSpec, 'inputT1', MakePurePlugsMaskInputListNode, 'inputT1')
    DistWF.connect(inputsSpec, 'inputT2', MakePurePlugsMaskInputListNode, 'inputT2')
    DistWF.connect(DTIEstim, 'idwi', MakePurePlugsMaskInputListNode, 'inputIDWI')

    ## Step 4_2: Create the mask
    PurePlugsMaskNode = pe.Node(interface=GeneratePurePlugMask(), name="PurePlugsMask")
    PurePlugsMaskNode.inputs.threshold = 0.15
    PurePlugsMaskNode.inputs.outputMaskFile = 'PurePlugsMask.nrrd'
    DistWF.connect(MakePurePlugsMaskInputListNode, 'imagesList', PurePlugsMaskNode, 'inputImageModalities')

    ## Step 4_3:
    CreateBrainPurePlugsMaskNode = pe.Node(Function(function=CreateBrainPurePlugsMask,
                                                    input_names=['PurePlugsMask', 'DWI_brainMask'],
                                                    output_names=['BrainPurePlugsMask']),
                                           name="CreateBrainPurePlugsMask")
    DistWF.connect(PurePlugsMaskNode, 'outputMaskFile', CreateBrainPurePlugsMaskNode, 'PurePlugsMask')
    DistWF.connect(inputsSpec, 'DWI_brainMask', CreateBrainPurePlugsMaskNode, 'DWI_brainMask')
    DistWF.connect(CreateBrainPurePlugsMaskNode, 'BrainPurePlugsMask', outputsSpec, 'PurePlugsMask')

    ##
    ## Step 5: Generate different WM ROIs
    ##

    return DistWF
