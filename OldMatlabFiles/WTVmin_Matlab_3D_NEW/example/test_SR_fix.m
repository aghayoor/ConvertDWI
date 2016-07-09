clear all
close all

 % Add support functions directory
thisFile = mfilename('fullpath');
addpath(fileparts(fileparts(thisFile)));

load '/scratch/forHans/input_mat/edgemap.mat'
edgemap = edgemap(:,:,70:80); %trim data to speed-up experiments

%% load high res data
load '/scratch/forHans/input_mat/dwi_b0_hr.mat'
X_hr_base = NormalizeDataComponent(double(dwi_b0_hr));
X_hr_base = X_hr_base(:,:,70:80); %trim data to speed-up experiments

%% create lowres image from highres image
highres = size(edgemap);
k = get_kspace_inds( highres ); %k=fourier indices

lowres = round(highres/2); %input lower resolution

ind_samples = get_lowpass_inds(k,lowres);
FX_hr_base = fftn(X_hr_base);
X_lr = (prod(lowres)/prod(highres))*real(ifftn(reshape(FX_hr_base(ind_samples),lowres)));  % This puts X_lr in the range [0,1]
fprintf('max value of X_lr: %2.2f\n',max(abs(X_lr(:)))); %Note the max is not exactly 1--this is because you lose some intensity by low-pass filtering
%% Run Super-Resolution reconstruction
% Now X_sr1 will be scaled the same as X_lr
X_sr1 = doSR(X_lr,edgemap); 

SNR_WL2 = -20*log10(norm(X_sr1(:)-X_hr_base(:))/norm(X_hr_base(:)));
fprintf('WL2 output SNR (Weighted L2 ADMM) = %2.1f dB\n',SNR_WL2);