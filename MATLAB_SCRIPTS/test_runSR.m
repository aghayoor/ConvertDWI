clear all
close all

dwi_fn='/scratch/TESTS/IpythonNotebook/20160615_HCPWF/mainWF/Outputs_delme/DWI_corrected_alignedSpace_masked.nrrd';
edgemap_fn='/scratch/TESTS/IpythonNotebook/20160615_HCPWF/mainWF/Outputs_delme/EdgeMap.nrrd';
out_dir='/scratch/TESTS/IpythonNotebook/20160615_HCPWF/2_SRWF';

% Check that the file exists
assert(exist(dwi_fn, 'file') == 2, 'File does not exist');
assert(exist(edgemap_fn, 'file') == 2, 'File does not exist');

tic
    run_sr(dwi_fn,edgemap_fn,out_dir);
toc