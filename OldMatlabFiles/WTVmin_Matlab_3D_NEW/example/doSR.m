function [X_hr] = doSR(X_lr,edgemap)
%%
lowres = size(X_lr);
highres = size(edgemap);
k = get_kspace_inds( highres ); %k=fourier indices
ind_samples = get_lowpass_inds(k,lowres);
[A,At] = defAAt_fourier(ind_samples, highres); %Define function handles for fourier projection operators
%% Rescale data -- X_hr will have same scaling as X_lr_pad
FX_lr_pad = zeros(highres);
FX_lr_pad(ind_samples) = fftn(X_lr);
X_lr_pad = (prod(highres)/prod(lowres))*ifftn(FX_lr_pad); 
b = A(X_lr_pad);
%% Run Weighted L2 algorithm - ADMM version
lambda = 1e-3; %regularization parameter
Niter = 100;  %number of iterations
gam = 1;   %ADMM parameter, in range [0.1,10]
tol = 1e-8;  %convergence tolerance
tic
[X_hr, cost] = OpWeightedL2(b,edgemap,lambda,A,At,highres,Niter,tol,gam);
toc
%X_hr = NormalizeDataComponent(abs(X_hr));
end