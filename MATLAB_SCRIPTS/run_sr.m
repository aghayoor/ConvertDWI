%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This function run following super resolution methods on a dwi image
% - Nearest Neighbor
% - zero-padded IFFT
% - Total Variation
% - Edge guided Weighted Total Variation (WTV)
%
% Input: 'input_dwi_fn' filename in nhdr format
%      : 'input_mask_fn' name of the mask file
%      : 'output_dwi_dir' name of the output directory to store the output
%
% Output: the recovered diffusion signal data files in nrrd format.
%
% Ali Ghayoor, Hans J. Johnson, Greg Ongie, University of Iowa, June 2016
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function run_sr(input_dwi_fn, input_mask_fn, output_dwi_dir)
    
% Add main MATLAB_SCRIPTS code tree
thisFile = mfilename('fullpath');
addpath(fullfile(fileparts(thisFile),'support_functions'));

fprintf('Starting testCS:\n    input_dwi_fn: %s\n    input_mask_fn: %s\n    output_dwi_dir: %s\n',input_dwi_fn,input_mask_fn,output_dwi_dir


[ rawDWI ] = nrrdLoadWithMetadata(input_dwi_fn);
%[ reformattedDWI, voxelLatticeToAnatomicalSpace ] = nrrdReformatAndNormalize(rawDWI);
%[ dwi_struct, metric, counts ] = BalanceDWIReplications( reformattedDWI );

% read input mask file
in_mask = nrrdLoadWithMetadata(input_mask_fn);
%mask = in_mask.data ~= 0;

[estimatedSignal] = doSRestimate(dwi_struct, mask);

nrrdStrct = dwi_struct;
nrrdStrct.data = estimatedSignal;
%nrrdStrct.gradientdirections = estimatedGradients;

fprintf('Writing files to disk...\n');

nrrdSaveWithMetadata(output_dwi_fn,nrrdStrct)

end