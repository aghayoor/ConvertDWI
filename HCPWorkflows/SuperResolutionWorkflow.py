## \author Ali Ghayoor
##
## This workflow runs "super-resolution reconstruction" in Matlab on an input DWI scan.
## The other image input to this workflow is the anatomical edgemap derived from structural MR images (T1/T2).
##

import os
import nipype
from nipype.interfaces import ants
from nipype.interfaces.base import CommandLine, CommandLineInputSpec, TraitedSpec, File, Directory
from nipype.interfaces.base import traits, isdefined, BaseInterface
from nipype.interfaces.utility import Merge, Split, Function, Rename, IdentityInterface
import nipype.interfaces.io as nio   # Data i/oS
import nipype.pipeline.engine as pe  # pypeline engine
from nipype.interfaces.semtools import *

def CreateSuperResolutionWorkflow(WFname, PYTHON_AUX_PATHS):
    if PYTHON_AUX_PATHS is not None:
        Path_to_Matlab_Func = PYTHON_AUX_PATHS[0]
        assert os.path.exists(Path_to_Matlab_Func), "Path to SR matlab function is not found: %s" % Path_to_Matlab_Func

    ###### UTILITY FUNCTIONS #######
    #\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/#
    # Run Super resolution reconstruction by Matlab
    def runSRbyMatlab(inputDWI,inputEdgeMap,Path_to_Matlab_Func):
        import os
        import nipype.interfaces.matlab as matlab
        script="runSR('"+inputDWI+"','"+inputEdgeMap+"','"+os.getcwd()+"')"
        mlab = matlab.MatlabCommand()
        mlab.set_default_matlab_cmd("matlab")
        mlab.inputs.single_comp_thread = False
        mlab.inputs.nodesktop = True
        mlab.inputs.nosplash = True
        mlab.inputs.paths = Path_to_Matlab_Func
        mlab.inputs.script = script
        mlab.run()
        baseline_fn = os.path.join(os.getcwd(), 'DWI_Baseline.nrrd')
        assert os.path.isfile(baseline_fn), "DWI baseline file is not found: %s" % baseline_fn
        NN_fn = os.path.join(os.getcwd(), 'DWI_SR_NN.nrrd')
        assert os.path.isfile(NN_fn), "DWI Nearest Neighbor file is not found: %s" % NN_fn
        IFFT_fn = os.path.join(os.getcwd(), 'DWI_SR_IFFT.nrrd')
        assert os.path.isfile(IFFT_fn), "DWI IFFT file is not found: %s" % IFFT_fn
        TV_fn = os.path.join(os.getcwd(), 'DWI_SR_TV.nrrd')
        assert os.path.isfile(TV_fn), "DWI TV file is not found: %s" % TV_fn
        WTV_fn = os.path.join(os.getcwd(), 'DWI_SR_WTV.nrrd')
        assert os.path.isfile(WTV_fn), "DWI WTV file is not found: %s" % WTV_fn
        return [baseline_fn,NN_fn,IFFT_fn,TV_fn,WTV_fn]
    #################################
    #\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\#

    SRWF = pe.Workflow(name=WFname)

    inputsSpec = pe.Node(interface=IdentityInterface(fields=['DWIVolume',
                                                             'EdgeMap']),
                         name='inputsSpec')

    outputsSpec = pe.Node(interface=IdentityInterface(fields=['DWI_Baseline',
                                                              'DWI_SR_NN',
                                                              'DWI_SR_IFFT',
                                                              'DWI_SR_TV',
                                                              'DWI_SR_WTV']),
                          name='outputsSpec')

    runSR=pe.Node(interface=Function(function = runSRbyMatlab,
                                     input_names=['inputDWI','inputEdgeMap','Path_to_Matlab_Func'],
                                     output_names=['baseline_fn','NN_fn','IFFT_fn','TV_fn','WTV_fn']),
                  name="runSR")
    runSR.inputs.Path_to_Matlab_Func = Path_to_Matlab_Func
    SRWF.connect([(inputsSpec,runSR,[('DWIVolume','inputDWI'),('EdgeMap','inputEdgeMap')])])
    SRWF.connect([(runSR,outputsSpec,[('baseline_fn','DWI_Baseline'),
                                      ('NN_fn','DWI_SR_NN'),
                                      ('IFFT_fn','DWI_SR_IFFT'),
                                      ('TV_fn','DWI_SR_TV'),
                                      ('WTV_fn','DWI_SR_WTV')])])

    return SRWF
