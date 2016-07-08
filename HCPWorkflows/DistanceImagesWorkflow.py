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
        '''
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
        '''
        #
        maskf = sitk.Cast(mask, sitk.sitkFloat32)
        #
        # create proper SR prefix for output file name
        sr_file_name = os.path.basename(DWI_sr)
        sr_file_name_base = os.path.splitext(sr_file_name)[0]
        sr_name = sr_file_name_base.split('_',2)[2]
        #
        fa_distance_image = sitk.Multiply(fa_distance_image,maskf)
        fa_distance_image_fn = os.path.realpath(sr_name + '_FA_distance.nrrd')
        sitk.WriteImage(fa_distance_image,fa_distance_image_fn)
        #
        md_distance_image = sitk.Multiply(md_distance_image,maskf)
        md_distance_image_fn = os.path.realpath(sr_name + '_MD_distance.nrrd')
        sitk.WriteImage(md_distance_image,md_distance_image_fn)
        #
        rd_distance_image = sitk.Multiply(rd_distance_image,maskf)
        rd_distance_image_fn = os.path.realpath(sr_name + '_RD_distance.nrrd')
        sitk.WriteImage(rd_distance_image,rd_distance_image_fn)
        #
        ad_distance_image = sitk.Multiply(ad_distance_image,maskf)
        ad_distance_image_fn = os.path.realpath(sr_name + '_AD_distance.nrrd')
        sitk.WriteImage(ad_distance_image,ad_distance_image_fn)
        #
        frobenius_distance_image = sitk.Multiply(frobenius_distance_image,maskf)
        frobenius_distance_image_fn = os.path.realpath(sr_name + '_Frobenius_distance.nrrd')
        sitk.WriteImage(frobenius_distance_image,frobenius_distance_image_fn)
        #
        logeuclid_distance_image = sitk.Multiply(logeuclid_distance_image,maskf)
        logeuclid_distance_image_fn = os.path.realpath(sr_name + '_Logeuclid_distance.nrrd')
        sitk.WriteImage(logeuclid_distance_image,logeuclid_distance_image_fn)
        #
        reimann_distance_image = sitk.Multiply(reimann_distance_image,maskf)
        reimann_distance_image_fn = os.path.realpath(sr_name + '_Reimann_distance.nrrd')
        sitk.WriteImage(reimann_distance_image,reimann_distance_image_fn)
        #
        kullback_distance_image = sitk.Multiply(kullback_distance_image,maskf)
        kullback_distance_image_fn = os.path.realpath(sr_name + '_Kullback_distance.nrrd')
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

    def MakeInputSRList(DWI_SR_NN, DWI_SR_IFFT):#, DWI_SR_TV, DWI_SR_WTV):
        imagesList = [DWI_SR_NN, DWI_SR_IFFT]#, DWI_SR_TV, DWI_SR_WTV]
        return imagesList
    #################################
    #\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\#
    DistWF = pe.Workflow(name=WFname)

    inputsSpec = pe.Node(interface=IdentityInterface(fields=['DWI_brainMask'
                                                             ,'DWI_Baseline'
                                                             ,'DWI_SR_NN'
                                                             ,'DWI_SR_IFFT'
                                                             ,'DWI_SR_TV'
                                                             ,'DWI_SR_WTV'
                                                             ]),
                         name='inputsSpec')

    outputsSpec = pe.Node(interface=IdentityInterface(fields=['FA_distance',
                                                              'MD_distance',
                                                              'RD_distance',
                                                              'AD_distance',
                                                              'Frobenius_distance',
                                                              'Logeuclid_distance',
                                                              'Reimann_distance',
                                                              'Kullback_distance'
                                                              ]),
                          name='outputsSpec')
    ##
    ## Step 1: Create input list
    ##
    MakeInputSRListNode = pe.Node(Function(function=MakeInputSRList,
                                                    input_names=['DWI_SR_NN','DWI_SR_IFFT'
                                                                 #,'DWI_SR_TV','DWI_SR_WTV'
                                                                ],
                                                    output_names=['imagesList']),
                                           name="MakeInputSRList")
    DistWF.connect([(inputsSpec,MakeInputSRListNode,[('DWI_SR_NN','DWI_SR_NN')
                                                     ,('DWI_SR_IFFT','DWI_SR_IFFT')
                                                     #,('DWI_SR_TV','DWI_SR_TV')
                                                     #,('DWI_SR_WTV','DWI_SR_WTV')
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


    return DistWF
