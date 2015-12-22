% Test of FRI Edgemask
%% Solve for annihliating filters
load Data/dwi_testdata;

%Size of rectangular k-space sampling window used in algorithm
%can be any dimensions <= size(inputImage). 
%However, you should not need to take this larger than [256,256],
%even for images bigger than this. 
sample_siz = size(inputImage); 

%Size of annihilating filters. 
%increasing this will increase the detail of the edge
%mask, but algorithm  will be slower for large
%filter sizes. To reproduce the earlier result, use [71,71], but this is
%probably larger than necessary, and will take long to run.
filter_siz = [55,55];%[71,71];

%Regularization parameter. Set high to remove weak edges
%or to smooth edges in the case of noisy input.
%Algorithm is fairly insensitive to this parameter over a wide range.
%May need to take very high (e.g. 1e7) to see an effect.
edgelambda = 1e-5;                           

%solves for filters U and weights s 
%that define the edgemask in a resolution independent way.
%U and S are passed to the auxiliary function compute_fri_edgemask
%in order to generate the final edgemask.
[U,s] = compute_fri_filters(inputImage,sample_siz,filter_siz,edgelambda);
%% Combine annihlating filters to create edgemask
%output pixel dimensions of edgemask. Does not need to match the input size.
outres = [256,256];

%combines the filters U with weights s to create an edgemask at the
%specified resolution. This step can be modified later to alter 
%properties of the edgemask. For example, the parameter q below will effect
%the dynamic range of the mask. Compare q=0.5 vs q=1 vs q=2. But typically 
%q=1 works well for the weighted TV reconstructions.
q = 1;%
edgemask = compute_fri_edgemask(U,s,filter_siz,outres,q);
figure(1); imagesc(edgemask); colorbar;
