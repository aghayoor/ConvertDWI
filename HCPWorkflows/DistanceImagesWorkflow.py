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
        from itertools import product
        import time
        from scipy.linalg import logm
        from scipy.linalg import eigvalsh # eigvalsh(A,B) is joint eigenvalues of A and B
        import SimpleITK as sitk
        from ReadWriteNrrdDWI import ReadNAMICDWIFromNrrd, WriteNAMICDWIToNrrd
        ######### distance functions ###
        # Frobenius distance
        def distance_euclid(A, B):
            return np.linalg.norm(A - B, ord='fro')
        # Log-Euclidian distance
        def distance_logeuclid(A, B):
            return distance_euclid(logm(A),logm(B))
        # Reimannian distance
        def distance_reimann(A, B):
            return np.sqrt((np.log(eigvalsh(A,B))**2).sum())
        # Kullback-Leibler distance
        def distance_kullback(A, B):
            dim = A.shape[0]
            kl = np.sqrt( np.trace( np.dot(np.linalg.inv(A),B)+np.dot(np.linalg.inv(B),A) ) - 2*dim )
            return 0.5*kl
        #########
        assert os.path.exists(DWI_baseline), "File not found: %s" % DWI_baseline
        assert os.path.exists(DWI_sr), "File not found: %s" % DWI_sr
        assert os.path.exists(DWI_brainMask), "File not found: %s" % DWI_brainMask
        # read brain mask file
        mask = sitk.ReadImage(DWI_brainMask)
        mask_data = sitk.GetArrayFromImage(mask)
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
        ## create RIS error arrays
        fa_distance_arr = abs(tenfit_base.fa-tenfit_sr.fa)
        md_distance_arr = abs(tenfit_base.md-tenfit_sr.md)
        rd_distance_arr = abs(tenfit_base.rd-tenfit_sr.rd)
        ad_distance_arr = abs(tenfit_base.ad-tenfit_sr.ad)
        ## mask out the background
        fa_distance_arr = np.transpose(fa_distance_arr,(2, 1, 0))
        md_distance_arr = np.transpose(md_distance_arr,(2, 1, 0))
        rd_distance_arr = np.transpose(rd_distance_arr,(2, 1, 0))
        ad_distance_arr = np.transpose(ad_distance_arr,(2, 1, 0))
        #
        fa_distance_arr = fa_distance_arr*mask_data
        md_distance_arr = md_distance_arr*mask_data
        rd_distance_arr = rd_distance_arr*mask_data
        ad_distance_arr = ad_distance_arr*mask_data
        ## clip outliers by computing 'p' percentiles
        p = 99.7
        np.clip(fa_distance_arr, fa_distance_arr.min(), np.percentile(fa_distance_arr,p), fa_distance_arr)
        np.clip(md_distance_arr, md_distance_arr.min(), np.percentile(md_distance_arr,p), md_distance_arr)
        np.clip(rd_distance_arr, rd_distance_arr.min(), np.percentile(rd_distance_arr,p), rd_distance_arr)
        np.clip(ad_distance_arr, ad_distance_arr.min(), np.percentile(ad_distance_arr,p), ad_distance_arr)
        ## create RIS distance images
        fa_distance_image = sitk.GetImageFromArray(fa_distance_arr)
        fa_distance_image.CopyInformation(mask)
        #
        md_distance_image = sitk.GetImageFromArray(md_distance_arr)
        md_distance_image.CopyInformation(mask)
        #
        rd_distance_image = sitk.GetImageFromArray(rd_distance_arr)
        rd_distance_image.CopyInformation(mask)
        #
        ad_distance_image = sitk.GetImageFromArray(ad_distance_arr)
        ad_distance_image.CopyInformation(mask)
        #
        size = [data_base.shape[0],data_base.shape[1],data_base.shape[2]] #data_base.shape[4] is gradient components
        frobenius_distance_arr = np.empty(size, dtype=float)
        logeuclid_distance_arr = np.empty(size, dtype=float)
        reimann_distance_arr = np.empty(size, dtype=float)
        kullback_distance_arr = np.empty(size, dtype=float)
        #
        tensors_base = tenfit_base.quadratic_form
        tensors_sr = tenfit_sr.quadratic_form
        # for loop to find other distance images
        tstart = time.time()
        for i, idx in enumerate(product(xrange(size[0]),xrange(size[1]),xrange(size[2]))):
            frobenius_distance_arr[idx] = distance_euclid(tensors_base[idx], tensors_sr[idx])
            logeuclid_distance_arr[idx] = distance_logeuclid(tensors_base[idx], tensors_sr[idx])
            reimann_distance_arr[idx] = distance_reimann(tensors_base[idx], tensors_sr[idx])
            kullback_distance_arr[idx] = distance_kullback(tensors_base[idx], tensors_sr[idx])
        elapsed = time.time() - tstart
        print('for loop took: ', elapsed)
        # mask out the background
        frobenius_distance_arr = np.transpose(frobenius_distance_arr,(2, 1, 0))
        logeuclid_distance_arr = np.transpose(logeuclid_distance_arr,(2, 1, 0))
        reimann_distance_arr = np.transpose(reimann_distance_arr,(2, 1, 0))
        kullback_distance_arr = np.transpose(kullback_distance_arr,(2, 1, 0))
        #
        frobenius_distance_arr = frobenius_distance_arr*mask_data
        logeuclid_distance_arr = logeuclid_distance_arr*mask_data
        reimann_distance_arr = reimann_distance_arr*mask_data
        kullback_distance_arr = kullback_distance_arr*mask_data
        ## clip outliers by computing 'p' percentiles
        p = 99.7
        np.clip(frobenius_distance_arr, frobenius_distance_arr.min(), np.percentile(frobenius_distance_arr,p), frobenius_distance_arr)
        np.clip(logeuclid_distance_arr, logeuclid_distance_arr.min(), np.percentile(logeuclid_distance_arr,p), logeuclid_distance_arr)
        np.clip(reimann_distance_arr, reimann_distance_arr.min(), np.percentile(reimann_distance_arr,p), reimann_distance_arr)
        np.clip(kullback_distance_arr, kullback_distance_arr.min(), np.percentile(kullback_distance_arr,p), kullback_distance_arr)
        ## create RIS distance images
        frobenius_distance_image = sitk.GetImageFromArray(frobenius_distance_arr)
        frobenius_distance_image.CopyInformation(mask)
        #
        logeuclid_distance_image = sitk.GetImageFromArray(logeuclid_distance_arr)
        logeuclid_distance_image.CopyInformation(mask)
        #
        reimann_distance_image = sitk.GetImageFromArray(reimann_distance_arr)
        reimann_distance_image.CopyInformation(mask)
        #
        kullback_distance_image = sitk.GetImageFromArray(kullback_distance_arr)
        kullback_distance_image.CopyInformation(mask)
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

    def CreateWhiteMatterROIMask(LobesLabelMapVolume,BrainPurePlugsMask):
        import os
        import SimpleITK as sitk
        ####
        def ResampleMask(in_mask, ref_mask):
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(ref_mask)
            resampler.SetInterpolator(sitk.sitkLabelGaussian) # Smoothly interpolate multi-label images
            res_mask = resampler.Execute(in_mask)
            return res_mask
        ####
        assert os.path.exists(LobesLabelMapVolume), "File not found: %s" % LobesLabelMapVolume
        assert os.path.exists(BrainPurePlugsMask), "File not found: %s" % BrainPurePlugsMask
        ## read in masks
        lobeLabels = sitk.ReadImage(LobesLabelMapVolume)
        ppmask = sitk.ReadImage(BrainPurePlugsMask)
        # resample input label map to lattice space of pure plug mask (DWI mask)
        lobeLabels = ResampleMask(lobeLabels, ppmask)
        ## define lobe labels
        temporal_wm = ((lobeLabels == 9703) + (lobeLabels == 9803))
        occipital_wm = ((lobeLabels == 9706) + (lobeLabels == 9806))
        frontal_wm = ((lobeLabels == 9707) + (lobeLabels == 9807))
        parietal_wm = ((lobeLabels == 9708) + (lobeLabels == 9808))
        ## define pure/ not pure labels
        temporal_wm_pure = temporal_wm * ppmask
        temporal_wm_NOTpure = temporal_wm * (1 - ppmask)
        #
        occipital_wm_pure = occipital_wm * ppmask
        occipital_wm_NOTpure = occipital_wm * (1 - ppmask)
        #
        frontal_wm_pure = frontal_wm * ppmask
        frontal_wm_NOTpure = frontal_wm * (1 - ppmask)
        #
        parietal_wm_pure = parietal_wm * ppmask
        parietal_wm_NOTpure = parietal_wm * (1 - ppmask)
        ## all above masks should have no overlap, so test it!
        wholemask = (temporal_wm_pure + temporal_wm_NOTpure
                     + occipital_wm_pure + occipital_wm_NOTpure
                     + frontal_wm_pure + frontal_wm_NOTpure
                     + parietal_wm_pure + parietal_wm_NOTpure)
        statFilter = sitk.StatisticsImageFilter()
        statFilter.Execute(wholemask)
        wholemask_max = statFilter.GetMaximum()
        if wholemask_max != 1:
            raise ValueError('White matter regions of interest must not overlap!')
        ## Now create a label map that only has our white matter regions of interest, such that:
        # frontal_wm_pure = 10
        # frontal_wm_NOTpure = 11
        # parietal_wm_pure = 20
        # parietal_wm_NOTpure = 21
        # occipital_wm_pure = 30
        # occipital_wm_NOTpure = 31
        # temporal_wm_pure = 40
        # temporal_wm_NOTpure = 41
        wm_roi_labels = ( (frontal_wm_pure*10) + (frontal_wm_NOTpure*11)
                        + (parietal_wm_pure*20) + (parietal_wm_NOTpure*21)
                        + (occipital_wm_pure*30) + (occipital_wm_NOTpure*31)
                        + (temporal_wm_pure*40) + (temporal_wm_NOTpure*41) )
        # write out all 8 ROIs to the disk
        wm_roi_labels_fn = os.path.realpath('wm_roi_labelmap.nrrd')
        sitk.WriteImage(wm_roi_labels, wm_roi_labels_fn)
        assert os.path.exists(wm_roi_labels_fn), "Output wm roi labelmap file not found: %s" % wm_roi_labels_fn
        return wm_roi_labels_fn

    def ComputeStatistics(wm_roi_labelmap,
                          FA_distance,MD_distance,RD_distance,AD_distance,
                          Frobenius_distance,Logeuclid_distance,
                          Reimann_distance,Kullback_distance):
        import os
        import SimpleITK as sitk
        ####
        def ReturnErrorImageStatsList(distImage_fn, roiMask_fn):
            distImage = sitk.ReadImage(distImage_fn)
            roi = sitk.ReadImage(roiMask_fn)
            statFilter = sitk.LabelStatisticsImageFilter()
            statFilter.Execute(distImage, roi)
            #
            frontal_pure_mean = statFilter.GetMean(10)
            frontal_NOTpure_mean = statFilter.GetMean(11)
            #
            parietal_pure_mean = statFilter.GetMean(20)
            parietal_NOTpure_mean = statFilter.GetMean(21)
            #
            occipital_pure_mean = statFilter.GetMean(30)
            occipital_NOTpure_mean = statFilter.GetMean(31)
            #
            temporal_pure_mean = statFilter.GetMean(40)
            temporal_NOTpure_mean = statFilter.GetMean(41)
            # Now create statistics list
            statsList = [format(frontal_pure_mean,'.8f'),
                         format(parietal_pure_mean,'.8f'),
                         format(occipital_pure_mean,'.8f'),
                         format(temporal_pure_mean,'.8f'),
                         format(frontal_NOTpure_mean,'.8f'),
                         format(parietal_NOTpure_mean,'.8f'),
                         format(occipital_NOTpure_mean,'.8f'),
                         format(temporal_NOTpure_mean,'.8f')]
            return statsList
        ####
        def writeLabelStatistics(filename,statisticsDictionary):
            import csv
            with open(filename, 'wb') as lf:
                headerdata = [['#Label', 'frontal_pure_mean', 'parietal_pure_mean',
                               'occipital_pure_mean', 'temporal_pure_mean',
                               'frontal_NOTpure_mean', 'parietal_NOTpure_mean',
                               'occipital_NOTpure_mean', 'temporal_NOTpure_mean']]
                wr = csv.writer(lf, delimiter=',')
                wr.writerows(headerdata)
                for key, value in sorted(statisticsDictionary.items()):
                    wr.writerows([[key] + value])
        ####
        assert os.path.exists(wm_roi_labelmap), "File not found: %s" % wm_roi_labelmap
        assert os.path.exists(FA_distance), "File not found: %s" % FA_distance
        assert os.path.exists(MD_distance), "File not found: %s" % MD_distance
        assert os.path.exists(RD_distance), "File not found: %s" % RD_distance
        assert os.path.exists(AD_distance), "File not found: %s" % AD_distance
        assert os.path.exists(Frobenius_distance), "File not found: %s" % Frobenius_distance
        assert os.path.exists(Logeuclid_distance), "File not found: %s" % Logeuclid_distance
        assert os.path.exists(Reimann_distance), "File not found: %s" % Reimann_distance
        assert os.path.exists(Kullback_distance), "File not found: %s" % Kullback_distance
        ## compute the stats
        statisticsDictionary={}
        statisticsDictionary['FA'] = ReturnErrorImageStatsList(FA_distance, wm_roi_labelmap)
        statisticsDictionary['MD'] = ReturnErrorImageStatsList(MD_distance, wm_roi_labelmap)
        statisticsDictionary['RD'] = ReturnErrorImageStatsList(RD_distance, wm_roi_labelmap)
        statisticsDictionary['AD'] = ReturnErrorImageStatsList(AD_distance, wm_roi_labelmap)
        statisticsDictionary['Frobenius'] = ReturnErrorImageStatsList(Frobenius_distance, wm_roi_labelmap)
        statisticsDictionary['Logeuclid'] = ReturnErrorImageStatsList(Logeuclid_distance, wm_roi_labelmap)
        statisticsDictionary['Reimann'] = ReturnErrorImageStatsList(Reimann_distance, wm_roi_labelmap)
        statisticsDictionary['Kullback'] = ReturnErrorImageStatsList(Kullback_distance, wm_roi_labelmap)
        ## create output filename
        fa_dist_fn = os.path.basename(FA_distance) # in a format of {SR}_FA_distance.nrrd
        base_name = os.path.splitext(fa_dist_fn)[0] # in a format of {SR}_FA_distance
        sr_name = base_name.split('_',2)[0] # SR name = NN,IFFT,TV,WTV
        statisticsFile = os.path.realpath(sr_name + '_errorImages_statistics.csv')
        # write statistics to a csv file
        writeLabelStatistics(statisticsFile,statisticsDictionary)
        return statisticsFile

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
                                                              'IDWI_image',
                                                              'PurePlugsMask',
                                                              'WM_ROI_Labelmap',
                                                              'errorImagesStatisticsFile'
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
    DistWF.connect(DTIEstim, 'idwi', outputsSpec, 'IDWI_image')

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

    ## Step 4_3: Create brain pure plugs mask
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
    CreateWhiteMatterROIMaskNode = pe.Node(Function(function=CreateWhiteMatterROIMask,
                                                    input_names=['LobesLabelMapVolume','BrainPurePlugsMask'],
                                                    output_names=['wm_roi_labels_fn']),
                                           name="CreateWhiteMatterROIMask")
    DistWF.connect(inputsSpec, 'LobesLabelMapVolume', CreateWhiteMatterROIMaskNode, 'LobesLabelMapVolume')
    DistWF.connect(CreateBrainPurePlugsMaskNode, 'BrainPurePlugsMask', CreateWhiteMatterROIMaskNode, 'BrainPurePlugsMask')
    DistWF.connect(CreateWhiteMatterROIMaskNode, 'wm_roi_labels_fn', outputsSpec, 'WM_ROI_Labelmap')

    ##
    ## Step 6: Compute Statistics: compute average of each distance image in each of ROIs.
    ##
    ComputeStats = pe.MapNode(interface=Function(function = ComputeStatistics,
                                                 input_names=['wm_roi_labelmap',
                                                              'FA_distance','MD_distance','RD_distance','AD_distance',
                                                              'Frobenius_distance','Logeuclid_distance',
                                                              'Reimann_distance','Kullback_distance'],
                                                 output_names=['statisticsFile']),
                              name="ComputeStatistics",
                              iterfield=['FA_distance','MD_distance','RD_distance','AD_distance',
                                         'Frobenius_distance','Logeuclid_distance',
                                         'Reimann_distance','Kullback_distance'])
    DistWF.connect(CreateWhiteMatterROIMaskNode, 'wm_roi_labels_fn', ComputeStats, 'wm_roi_labelmap')
    DistWF.connect([(ComputeDistanceImages,ComputeStats,[('fa_distance_image_fn','FA_distance'),
                                                         ('md_distance_image_fn','MD_distance'),
                                                         ('rd_distance_image_fn','RD_distance'),
                                                         ('ad_distance_image_fn','AD_distance'),
                                                         ('frobenius_distance_image_fn','Frobenius_distance'),
                                                         ('logeuclid_distance_image_fn','Logeuclid_distance'),
                                                         ('reimann_distance_image_fn','Reimann_distance'),
                                                         ('kullback_distance_image_fn','Kullback_distance')])])
    DistWF.connect(ComputeStats, 'statisticsFile', outputsSpec, 'errorImagesStatisticsFile')

    return DistWF
