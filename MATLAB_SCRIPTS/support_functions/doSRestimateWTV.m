function [estimatedIFFTsignal,estimatedTVsignal,estimatedWTVsignal] = doSRestimateWTV( DWIIntensityData, edgemap )
% param: DWIIntensityData - is the dwi 4D data that is cleaned and normalized
% param: edgemap - a weight matrix derived from anatomical edge locations

[nx, ny, nz, numGrad]=size(DWIIntensityData);
[mnx,mny,mnz] =size(edgemap);

% Mask MUST be same size as image
assert( nx == mnx , 'Mask x size does not match DWI data size');
assert( ny == mny , 'Mask y size does not match DWI data size');
assert( nz == mnz , 'Mask z size does not match DWI data size');

parallel.defaultClusterProfile('local');
num_slots = getenv('NSLOTS');
if( num_slots )
  poolobj = parpool(STRTODOUBLE(num_slots));
else
  poolobj = parpool(4);
end

estimatedIFFTsignal = zeros(size(DWIIntensityData));
estimatedTVsignal = zeros(size(DWIIntensityData));
estimatedWTVsignal = zeros(size(DWIIntensityData));

%HACK
numGrad=1;

for c=1:numGrad
    X0 = DWIIntensityData(:,:,:,c);
    %%
    m = fftn(X0);       %m=fourier data
    inres = size(m);
    k = get_kspace_inds( inres ); %k=fourier indices
    %% Define Fourier Projection Operators
    res = inres; %output resolution
    lores = round(inres/2); %input lower resolution (use odd numbers)
    ind_samples = get_lowpass_inds(k,lores);
    [A,At] = defAAt_fourier(ind_samples, res); %Define function handles for fourier projection operators
    b = A(X0);       %low-resolution fourier samples
    Xlow = ifftn(reshape(b,lores));
    %% Zero-padded IFFT reconstruction
    X_IFFT = At(fftn(Xlow));
    estimatedIFFTsignal(:,:,:,c) = X_IFFT;
    %% Run Standard TV algorithm (no edgemap)
    lambda = 5e-4; %regularization parameter
    Niter = 25;  %number of iterations
    siz = size(edgemap);
    edgemap_allone = ones(siz);
    [X_TV, cost] = OpWeightedTV_PD_AHMOD(b,edgemap_allone,lambda,A,At,res,Niter);
    estimatedTVsignal(:,:,:,c) = X_TV;
    %% Run New Weighted TV algorithm
    lambda = 5e-4; %regularization parameter (typically in the range [1e-2,1], if original image scaled to [0,1])
    Niter = 25;  %number of iterations
    [X_WTV, cost] = OpWeightedTV_PD_AHMOD(b,edgemap,lambda,A,At,res,Niter); %see comments inside OpWeightedTV_PD_AHMOD.m
    estimatedWTVsignal(:,:,:,c) = X_WTV;
end

delete(poolobj)

end