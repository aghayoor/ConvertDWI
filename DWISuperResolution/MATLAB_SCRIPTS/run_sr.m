%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This function uses compressed sensing to recover
%the HARDI signal (single shell) from very vew gradient directions.
%A minimum of 16 directions are needed to recover crossing fiber signal.
%
% Input: 'input_dwi_fn' filename in nhdr format
%      : 'input_mask_fn' name of the mask file
%      : 'output_dwi_fn' name of the output filename to store the output
%      : 'denoising_factor' A parameter for adjusting how much de-noising 
%           is done at each iteration 0=no denoising, 0.025 for very little
%           denoising, and 0.05 for "typical scans" level of noise.
%      : 'new_gradients' optionally, supply the new gradients on which to
%           sample the recovered signal
%      : 'saveC' - optionally, save the coefficients for future use.
% Output: the recovered diffusion signal data file in nhdr format.
%
% Yogesh Rathi, Oleg Michailovich, June 20th 2011
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function run_cs(input_dwi_fn,input_mask_fn,output_dwi_fn,denoising_factor,new_gradients,saveC)
    % Add main MATLAB_SCRIPTS code tree
    thisFile = mfilename('fullpath');
    addpath(fullfile(fileparts(thisFile),'support_functions'));

fprintf('Starting testCS:\n    input_dwi_fn: %s\n    input_mask_fn: %s\n    output_dwi_fn: %s\n',input_dwi_fn,input_mask_fn,output_dwi_fn)


%
% KW QUESTION: How do the default gradients relate to the DWI magnitude images?
% Just choosing a bunch of icosahedron vertices as gradients without doing some
% of manipulation of the magnitudes means that you are generating a nonsense DWI output
% file.
if nargin >=5 && new_gradients
    nu = new_gradients;
else
    nu = SetDefaultGradients();
end

[ rawDWI ] = nrrdLoadWithMetadata(input_dwi_fn);
[ reformattedDWI, voxelLatticeToAnatomicalSpace ] = nrrdReformatAndNormalize(rawDWI);
[ dwi_struct, metric, counts ] = BalanceDWIReplications( reformattedDWI );
in_mask = nrrdLoadWithMetadata(input_mask_fn);
mask = in_mask.data ~= 0;

% change nu to be aligned in voxel lattice space
% This cause the anatomicalEstimatedGradients be the same for all output
% data after CS computations.
gradientsInVoxelLatticeSpace = (voxelLatticeToAnatomicalSpace\nu')';

%save('/tmp/before_doCSestimate.mat','-v7.3')
[estimatedSignal,estimatedGradients] = doCSestimate(dwi_struct, mask, gradientsInVoxelLatticeSpace, denoising_factor);

anatomicalEstimatedGradients = ( voxelLatticeToAnatomicalSpace*estimatedGradients' )';

nrrdStrct = dwi_struct;
nrrdStrct.data=single(estimatedSignal);
nrrdStrct.gradientdirections=anatomicalEstimatedGradients;

fprintf('Writing file to disk...\n');

nrrdSaveWithMetadata(output_dwi_fn,nrrdStrct)
if nargin == 5 && saveC
  save(output_fn,'c','-v7.3');
end
end

function [ nu ] =  SetDefaultGradients()
    % icosahedron produces a antipodal symmetric set of points,
    % so remove the redundant samples.

    % Add main MATLAB_SCRIPTS code tree
    thisFile = mfilename('fullpath');
    addpath(fullfile(fileparts(thisFile),'support_functions'));

    %ico1 = icosahedron(1); %42 gradient direction
    ico2 = icosahedron(2); %162  <-- Good default for most cases
    %ico3 = icosahedron(3); %642
    nu = ico2; %162 gradient directions

    n0 = size(nu,1)/2;
    nu = nu(1:n0,:); %81 gradient directions
end
