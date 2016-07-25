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
        def ComputeBhattacharyyaCoeficient(baseline_bundle, sr_bundle):
            import vtk
            import numpy as np
            ## read in each fiber bundle
            reader_gs = vtk.vtkXMLPolyDataReader()
            reader_gs.SetFileName(baseline_bundle)
            reader_gs.Update()
            gs_bundle = reader_gs.GetOutput()
            #
            reader_sr = vtk.vtkXMLPolyDataReader()
            reader_sr.SetFileName(sr_bundle)
            reader_sr.Update()
            sr_bundle = reader_sr.GetOutput()
            #
            gs_pts = np.array([gs_bundle.GetPoint(i) for i in xrange(gs_bundle.GetNumberOfPoints())])
            sr_pts = np.array([sr_bundle.GetPoint(i) for i in xrange(sr_bundle.GetNumberOfPoints())])
            #
            mn = np.minimum(gs_pts.min(0), sr_pts.min(0))
            mx = np.maximum(gs_pts.max(0), sr_pts.max(0))
            bins = np.ceil((mx - mn))
            #
            gs_hist = np.array([ np.histogram(gs_pts[:,i], bins=bins[i], density=True, range=(mn[i], mx[i]))[0] for i in xrange(3) ])
            sr_hist = np.array([ np.histogram(sr_pts[:,i], bins=bins[i], density=True, range=(mn[i], mx[i]))[0] for i in xrange(3) ])
            #
            coefs = np.array([ np.sqrt( (gs_hist[i]*sr_hist[i])/(gs_hist[i].sum()*sr_hist[i].sum()) ).sum() for i in xrange(3) ])
            #print(coefs)
            return coefs.mean()
            ##
        def writeLabelStatistics(filename,statsList):
            import csv
            label = os.path.splitext(os.path.basename(filename))[0].split('_',1)[0]
            with open(filename, 'wb') as lf:
                headerdata = [['#Label', 'left.cst', 'right.cst', 'cst']]
                wr = csv.writer(lf, delimiter=',')
                wr.writerows(headerdata)
                wr.writerows([[label] + statsList])
        #########
        # compute Bhattacharyya Coeficient for cst.left/right/total
        bc_l = ComputeBhattacharyyaCoeficient(gs_cst_left,sr_cst_left)
        if bc_l > 1: bc_l = 1
        bc_r = ComputeBhattacharyyaCoeficient(gs_cst_right,sr_cst_right)
        if bc_r > 1: bc_r = 1
        bc_total = (bc_l + bc_r)/2.0
        statsList = [format(bc_l,'.4f'), format(bc_r,'.4f'), format(bc_total,'.4f')]
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
