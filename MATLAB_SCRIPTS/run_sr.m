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
% Ali Ghayoor, Greg Ongie, Hans J. Johnson, University of Iowa, June 2016
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function run_sr(input_dwi_fn, input_edgemap_fn, output_dwi_dir)

% Add main MATLAB_SCRIPTS code tree
thisFile = mfilename('fullpath');
addpath(fullfile(fileparts(thisFile),'support_functions'));

fprintf('Starting testCS:\n    input_dwi_fn: %s\n    input_mask_fn: %s\n    output_dwi_dir: %s\n',input_dwi_fn,input_edgemap_fn,output_dwi_dir);

% read input DWI file
[ rawDWI ] = nrrdLoadWithMetadata(input_dwi_fn);
[ reformattedDWI ] = nrrdReformat(rawDWI);

% read input mask file
in_edgemap = nrrdLoadWithMetadata(input_edgemap_fn);
edgemap = in_edgemap.data;

%%
tic
[normalizedSignal,estimatedNNsignal,estimatedIFFTsignal,estimatedTVsignal,estimatedWTVsignal] = doSRestimate(reformattedDWI.data, edgemap, 2);
toc

%% Write output DWI_Baseline
% normalized reformatted DWI data
nrrdBaselineStrct = reformattedDWI;
nrrdBaselineStrct.data = normalizedSignal;
fprintf('Writing SRR DWI_Baseline file to disk...\n');
output_dwi_fn = strcat(output_dwi_dir,'/DWI_Baseline.nrrd');
nrrdSaveWithMetadata(output_dwi_fn,nrrdBaselineStrct);

%% Write output DWI_NN
nrrdNNStrct = reformattedDWI;
nrrdNNStrct.data = estimatedNNsignal;
fprintf('Writing SRR DWI_NN file to disk...\n');
output_dwi_fn = strcat(output_dwi_dir,'/DWI_SR_NN.nrrd');
nrrdSaveWithMetadata(output_dwi_fn,nrrdNNStrct);

%% Write output DWI_IFFT
nrrdIFFTStrct = reformattedDWI;
nrrdIFFTStrct.data = estimatedIFFTsignal;
fprintf('Writing SRR DWI_IFFT file to disk...\n');
output_dwi_fn = strcat(output_dwi_dir,'/DWI_SR_IFFT.nrrd');
nrrdSaveWithMetadata(output_dwi_fn,nrrdIFFTStrct);

%% Write output DWI_TV
nrrdTVStrct = reformattedDWI;
nrrdTVStrct.data = estimatedTVsignal;
fprintf('Writing SRR DWI_TV file to disk...\n');
output_dwi_fn = strcat(output_dwi_dir,'/DWI_SR_TV.nrrd');
nrrdSaveWithMetadata(output_dwi_fn,nrrdTVStrct);

%% Write output DWI_WTV
nrrdWTVStrct = reformattedDWI;
nrrdWTVStrct.data = estimatedWTVsignal;
fprintf('Writing SRR DWI_WTV file to disk...\n');
output_dwi_fn = strcat(output_dwi_dir,'/DWI_SR_WTV.nrrd');
nrrdSaveWithMetadata(output_dwi_fn,nrrdWTVStrct);

end
