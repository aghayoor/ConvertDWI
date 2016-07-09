clear all
close all

 % Add support functions directory
thisFile = mfilename('fullpath');
addpath(fileparts(fileparts(thisFile)));

load 'edgemask_t1t2_1ByGMI_3D.mat'
edgemap = edgemask;
edgemap = edgemap(:,:,70:80); %trim data to speed-up experiments

%% load high res data
load 'dwib0_testdata_3D.mat'
X_hr_base = NormalizeDataComponent(double(inputImage));
X_hr_base = X_hr_base(:,:,70:80); %trim data to speed-up experiments

%% create lowres image from highres image
m = fftn(edgemap);       %m=fourier data
highres = size(m);
k = get_kspace_inds( highres ); %k=fourier indices

lowres = round(highres/2); %input lower resolution (use odd numbers)

ind_samples = get_lowpass_inds(k,lowres);
[A,At] = defAAt_fourier(ind_samples, highres); %Define function handles for fourier projection operators
b = A(X_hr_base);       %low-resolution fourier samples -> 1st input to optimization algorithm
X_lr = real(ifftn(reshape(b,lowres)));

%% Run Super-Resolution reconstruction
X_sr1 = doSR(X_lr,edgemap);

SNR_WL2 = -20*log10(norm(X_sr1(:)-X_hr_base(:))/norm(X_hr_base(:)));
fprintf('WL2 output SNR (Weighted L2 ADMM) = %2.1f dB\n',SNR_WL2);

%% Assume lowres is a real input with dynamic range of [0 1]
X_lr_new = NormalizeDataComponent(X_lr);

%% Run Super-Resolution reconstruction
X_sr2 = doSR(X_lr_new,edgemap);

SNR_WL2 = -20*log10(norm(X_sr2(:)-X_hr_base(:))/norm(X_hr_base(:)));
fprintf('WL2 output SNR (Weighted L2 ADMM) = %2.1f dB\n',SNR_WL2);