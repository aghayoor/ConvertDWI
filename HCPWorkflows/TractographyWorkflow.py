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

    # This function helps to pick desirable output from the output list
    def pickFromList(inlist,item):
        return inlist[item]

    def MakeInputCSTList(NN_cst, IFFT_cst, TV_cst, WTV_cst):
        outputList = [NN_cst, IFFT_cst, TV_cst, WTV_cst]
        return outputList

    def CSTOverlap(gs_cst_left, gs_cst_right, sr_cst_left, sr_cst_right):
        # gs: gold standard, sr: super-resolution reconstructed
        import os
        #########
        def ComputeBhattacharyyaCoeficient(gs_bundle,sr_bundle):
            from tract_querier import tract_math, tractography
            import numpy as np
            ###
            def AlignFiber(tract):
                out_tract = tract
                vecs = np.diff(tract, axis=0)
                avg_vec = np.mean(vecs, axis=0)
                dots = np.array([ np.dot(avg_vec,[1,0,0]), np.dot(avg_vec,[0,1,0]), np.dot(avg_vec,[0,0,1]) ])
                idx = np.argmax(np.abs(dots))
                if (dots[idx] < 0):
                    out_tract = tract[::-1]
                return out_tract
            #
            def InterpolateFiber(tract):
                new_tract = tract
                # find point distances
                d = np.diff(tract, axis=0)
                pointdists = np.hypot(d[:,0], d[:,1], d[:,2])
                dist_threshold = min(pointdists.sum()/len(pointdists),1)
                offset = 0
                for i in xrange(len(pointdists)):
                    if pointdists[i] > dist_threshold:
                        new_p = (tract[i+1]+tract[i])/2
                        new_tract = np.insert(new_tract,i+1+offset,new_p,axis=0)
                        offset += 1
                return new_tract
            #
            def returnBundlePoints(bundle):
                in_tractography = tractography.tractography_from_files(bundle)
                tracts = in_tractography.tracts()
                ## sort all tracts direction to positive coordinate of their dominant orientation
                ## then interpolate tracts to make sure they are approximately equally placed
                for i in xrange(len(tracts)):
                    tracts[i] = AlignFiber(tracts[i])
                    tracts[i] = InterpolateFiber(tracts[i])
                #
                tracts_last_quarter = [tracts[i][4*len(tracts[i])/5:] for i in xrange(len(tracts))]
                #
                pts = np.vstack(tracts)
                pts_last_quarter = np.vstack(tracts_last_quarter)
                return pts, pts_last_quarter
            #
            def returnBhattCoef(gs_pts,sr_pts):
                from scipy import stats
                gs_xyz = np.array([np.linspace(gs_pts.min(0)[i],gs_pts.max(0)[i],100) for i in xrange(3)])
                sr_xyz = np.array([np.linspace(sr_pts.min(0)[i],sr_pts.max(0)[i],100) for i in xrange(3)])
                #
                gs_kde = np.array([ stats.gaussian_kde(gs_pts[:,i]) for i in xrange(3) ])
                gs_p = np.array([ gs_kde[i](gs_xyz[i]) for i in xrange(3) ])
                gs_p = np.array([gs_p[i]/gs_p.sum(1)[i] for i in xrange(3)])
                #
                sr_kde = np.array([ stats.gaussian_kde(sr_pts[:,i]) for i in xrange(3) ])
                sr_p = np.array([ sr_kde[i](sr_xyz[i]) for i in xrange(3) ])
                sr_p = np.array([sr_p[i]/sr_p.sum(1)[i] for i in xrange(3)])
                #
                coefs = np.array([ np.sqrt(gs_p[i]*sr_p[i]).sum() for i in xrange(3) ])
                return coefs.mean()
            ###
            [gs_pts, gs_pts_last_quarter] = returnBundlePoints(gs_bundle)
            [sr_pts, sr_pts_last_quarter] = returnBundlePoints(sr_bundle)
            #
            coef = returnBhattCoef(gs_pts,sr_pts)
            coef_last_quarter = returnBhattCoef(gs_pts_last_quarter,sr_pts_last_quarter)
            return [coef,coef_last_quarter]
        #########
        def writeLabelStatistics(filename,statsList):
            import csv
            label = os.path.splitext(os.path.basename(filename))[0].split('_',1)[0]
            with open(filename, 'wb') as lf:
                headerdata = [['#Label', 'cst_left', 'cst_right', 'cst', 'cst_left_top', 'cst_right_top', 'cst_top']]
                wr = csv.writer(lf, delimiter=',')
                wr.writerows(headerdata)
                wr.writerows([[label] + statsList])
        #########
        # compute Bhattacharyya Coeficient for cst.left/right/total
        bc_l,bc_l_q = ComputeBhattacharyyaCoeficient(gs_cst_left,sr_cst_left)
        bc_r,bc_r_q = ComputeBhattacharyyaCoeficient(gs_cst_right,sr_cst_right)
        bc_total = (bc_l + bc_r)/2.0
        bc_total_q = (bc_l_q + bc_r_q)/2.0
        statsList = [format(bc_l,'.4f'), format(bc_r,'.4f'), format(bc_total,'.4f'), format(bc_l_q,'.4f'), format(bc_r_q,'.4f'), format(bc_total_q,'.4f')]
        # create output file name
        srfn = os.path.basename(sr_cst_left)
        srfnbase = os.path.splitext(srfn)[0]
        srLabel = srfnbase.split('_',1)[0]
        fn = srLabel + '_BhattacharyyaCoeficient.csv'
        output_csv_file = os.path.join(os.getcwd(), fn)
        # write the stats list
        writeLabelStatistics(output_csv_file,statsList)
        assert os.path.isfile(output_csv_file), "Output Bhattacharyya coeficient file is not found: %s" % output_csv_file
        return output_csv_file

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

    outputsSpec = pe.Node(interface=IdentityInterface(fields=['Baseline_ukfTracts',
                                                              'NN_ukfTracts',
                                                              'IFFT_ukfTracts',
                                                              'TV_ukfTracts',
                                                              'WTV_ukfTracts',
                                                              'Baseline_cst_left','Baseline_cst_right',
                                                              'NN_cst_left','NN_cst_right',
                                                              'IFFT_cst_left','IFFT_cst_right',
                                                              'TV_cst_left','TV_cst_right',
                                                              'WTV_cst_left','WTV_cst_right',
                                                              'NN_overlap_coeficient',
                                                              'IFFT_overlap_coeficient',
                                                              'TV_overlap_coeficient',
                                                              'WTV_overlap_coeficient'
                                                             ]),
                          name='outputsSpec')

    ##
    ## Step 1: Create input list
    ##
    MakeInputSRListNode = pe.Node(Function(function=MakeInputSRList,
                                           input_names=['DWI_Baseline','DWI_SR_NN','DWI_SR_IFFT','DWI_SR_TV','DWI_SR_WTV'],
                                           output_names=['imagesList']),
                                  name="MakeInputSRList")
    TractWF.connect([(inputsSpec,MakeInputSRListNode,[('DWI_Baseline','DWI_Baseline'),
                                                      ('DWI_SR_NN','DWI_SR_NN'),
                                                      ('DWI_SR_IFFT','DWI_SR_IFFT'),
                                                      ('DWI_SR_TV','DWI_SR_TV'),
                                                      ('DWI_SR_WTV','DWI_SR_WTV')
                                                     ])])
    ##
    ## Step 2: UKF Processing
    ##
    UKFNode = pe.MapNode(interface=UKFTractography(), name= "RunUKFt", iterfield=['dwiFile','tracts'])
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
    TractWF.connect(UKFNode,('tracts', pickFromList, 0),outputsSpec,'Baseline_ukfTracts')
    TractWF.connect(UKFNode,('tracts', pickFromList, 1),outputsSpec,'NN_ukfTracts')
    TractWF.connect(UKFNode,('tracts', pickFromList, 2),outputsSpec,'IFFT_ukfTracts')
    TractWF.connect(UKFNode,('tracts', pickFromList, 3),outputsSpec,'TV_ukfTracts')
    TractWF.connect(UKFNode,('tracts', pickFromList, 4),outputsSpec,'WTV_ukfTracts')

    ##
    ## Step 3: run WMQL to extract cortico-spinal (CST) tract bundles
    ##
    tract_querier = pe.MapNode(interface=Function(function = runWMQL,
                                                  input_names=['input_tractography','input_atlas'],
                                                  output_names=['output_cst_left','output_cst_right']),
                               name="tract_querier", iterfield=['input_tractography'])
    TractWF.connect(UKFNode,'tracts',tract_querier,'input_tractography')
    TractWF.connect(inputsSpec,'inputLabelMap',tract_querier,'input_atlas')
    # baseline cst
    TractWF.connect(tract_querier,('output_cst_left', pickFromList, 0),outputsSpec,'Baseline_cst_left')
    TractWF.connect(tract_querier,('output_cst_right', pickFromList, 0),outputsSpec,'Baseline_cst_right')
    # NN cst
    TractWF.connect(tract_querier,('output_cst_left', pickFromList, 1),outputsSpec,'NN_cst_left')
    TractWF.connect(tract_querier,('output_cst_right', pickFromList, 1),outputsSpec,'NN_cst_right')
    # IFFT cst
    TractWF.connect(tract_querier,('output_cst_left', pickFromList, 2),outputsSpec,'IFFT_cst_left')
    TractWF.connect(tract_querier,('output_cst_right', pickFromList, 2),outputsSpec,'IFFT_cst_right')
    # TV cst
    TractWF.connect(tract_querier,('output_cst_left', pickFromList, 3),outputsSpec,'TV_cst_left')
    TractWF.connect(tract_querier,('output_cst_right', pickFromList, 3),outputsSpec,'TV_cst_right')
    # WTV cst
    TractWF.connect(tract_querier,('output_cst_left', pickFromList, 4),outputsSpec,'WTV_cst_left')
    TractWF.connect(tract_querier,('output_cst_right', pickFromList, 4),outputsSpec,'WTV_cst_right')

    ##
    ## Step 4: Make SR CST left/right lists
    ##

    # step 4_1: input cst.left
    cst_left_list = pe.Node(Function(function=MakeInputCSTList,
                                     input_names=['NN_cst', 'IFFT_cst', 'TV_cst', 'WTV_cst'],
                                     output_names=['outputList']),
                            name="cst_left_list")
    TractWF.connect([(tract_querier,cst_left_list,[(('output_cst_left', pickFromList, 1),'NN_cst'),
                                                   (('output_cst_left', pickFromList, 2),'IFFT_cst'),
                                                   (('output_cst_left', pickFromList, 3),'TV_cst'),
                                                   (('output_cst_left', pickFromList, 4),'WTV_cst')
                                                   ])])

    # step 4_2: input cst.right
    cst_right_list = pe.Node(Function(function=MakeInputCSTList,
                                      input_names=['NN_cst', 'IFFT_cst', 'TV_cst', 'WTV_cst'],
                                      output_names=['outputList']),
                             name="cst_right_list")
    TractWF.connect([(tract_querier,cst_right_list,[(('output_cst_right', pickFromList, 1),'NN_cst'),
                                                    (('output_cst_right', pickFromList, 2),'IFFT_cst'),
                                                    (('output_cst_right', pickFromList, 3),'TV_cst'),
                                                    (('output_cst_right', pickFromList, 4),'WTV_cst')
                                                    ])])

    ##
    ## Step 5: Compute Bhattacharyya coeficient to find overlap between CST tract bundles
    ##
    cst_overlap = pe.MapNode(interface=Function(function = CSTOverlap,
                                                input_names=['gs_cst_left', 'gs_cst_right', 'sr_cst_left', 'sr_cst_right'],
                                                output_names=['output_csv_file']),
                             name="BhattacharyyaCoeficient", iterfield=['sr_cst_left','sr_cst_right'])
    TractWF.connect(tract_querier,('output_cst_left', pickFromList, 0),cst_overlap,'gs_cst_left')
    TractWF.connect(tract_querier,('output_cst_right', pickFromList, 0),cst_overlap,'gs_cst_right')
    TractWF.connect(cst_left_list,'outputList',cst_overlap,'sr_cst_left')
    TractWF.connect(cst_right_list,'outputList',cst_overlap,'sr_cst_right')
    TractWF.connect(cst_overlap, ('output_csv_file', pickFromList, 0), outputsSpec, 'NN_overlap_coeficient')
    TractWF.connect(cst_overlap, ('output_csv_file', pickFromList, 1), outputsSpec, 'IFFT_overlap_coeficient')
    TractWF.connect(cst_overlap, ('output_csv_file', pickFromList, 2), outputsSpec, 'TV_overlap_coeficient')
    TractWF.connect(cst_overlap, ('output_csv_file', pickFromList, 3), outputsSpec, 'WTV_overlap_coeficient')

    return TractWF
