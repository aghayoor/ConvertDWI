clear all
close all

load '/scratch/TESTS/IpythonNotebook/20160615_HCPWF/2_SRWF/test_tune_parameters/matlabFiles/dwib0_testdata_3D.mat'
load '/scratch/TESTS/IpythonNotebook/20160615_HCPWF/2_SRWF/test_tune_parameters/matlabFiles/edgemask_t1t2_1ByGMI_3D.mat'

X0 = double(inputImage);
X0 = X0(:,:,70:80); %trim data to speed-up experiments
X0_size = size(X0);
X0_2d = X0(:,:,round(X0_size(3)/2)); % make input 2D %%%%%%%%%%%%%

edgemask = edgemask(:,:,70:80); %trim data to speed-up experiments
edgemask_size = size(edgemask);
edgemask_2d = edgemask(:,:,round(edgemask_size(3)/2)); % make edgemask 2D %%%%%%%%%%%%

figure(1); imagesc(abs(X0_2d),[0 1]); colorbar; title('ground truth');
figure(2); imagesc(edgemask_2d,[0 1]); colorbar; title('spatial weights image');

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
Xlow = real(ifftn(reshape(b,lores)));
% Show a 2D slice of low resolution image
Xlow_size = size(Xlow);
Xlow_2d = Xlow(:,:,round(Xlow_size(3)/2));
figure(3); imagesc(abs(Xlow_2d)); colorbar; title('low resolution input');

%% Zero-padded IFFT reconstruction
X_IFFT = At(b);
% Show a 2D slice of zero-padded IFFT image
X_IFFT_size = size(X_IFFT);
X_IFFT_2d = X_IFFT(:,:,round(X_IFFT_size(3)/2));
figure(4); imagesc(abs(X_IFFT_2d)); colorbar; title('hi resolution, zero-padded IFFT');

SNR_IFFT = -20*log10(norm(X_IFFT(:)-X0(:))/norm(X0(:)));
fprintf('Zero-padded IFFT output SNR = %2.1f dB\n',SNR_IFFT);

%% Run Standard TV algorithm (no edgemask) -- ADMM algorithm
lambda = 1e-4; %regularization parameter (typically in the range [1e-2,1], if original image scaled to [0,1])
Niter = 100;  %number of iterations (typically 20-100 for Fourier inversion)
beta = 10; %ADMM parameter
tic
[X_TV_AL, cost] = OpTV_AL_3D(b,lambda,A,At,res,Niter,beta);
toc
X_TV_AL_size = size(X_TV_AL);
X_TV_AL_2d = X_TV_AL(:,:,round(X_TV_AL_size(3)/2));
figure(5); imagesc(abs(X_TV_AL_2d),[0,1]); colorbar; title('hi resolution, TV_AL (no edgemask)');
figure(6); plot(cost); xlabel('iteration'); ylabel('cost');
figure(7); imagesc(abs(X_TV_AL_2d-X0_2d),[0,0.2]); colorbar; title('TV_AL error image');

SNR_TV_AL = -20*log10(norm(X_TV_AL(:)-X0(:))/norm(X0(:)));
fprintf('TV_AL (ADMM) output SNR = %2.1f dB\n',SNR_TV_AL);
fprintf('TV_AL (ADMM) final cost %6.4f\n',cost(end));

%% Run Standard TV algorithm (no edgemask) -- AHMOD from Chambolle & Pock
lambda = 1e-4; %regularization parameter (typically in the range [1e-2,1], if original image scaled to [0,1])
Niter = 200;  %number of iterations (typically 500-1000 for Fourier inversion)
tic
[X_TV_PD, cost] = OpWeightedTV_PD_AHMOD2(b,ones(res),lambda,A,At,res,Niter); %see comments inside OpWeightedTV_PD_AHMOD.m
toc
% Show a 2D slice of X_WTV image
X_TV_PD_size = size(X_TV_PD);
X_TV_PD_2d = X_TV_PD(:,:,round(X_TV_PD_size(3)/2));
figure(8); imagesc(abs(X_TV_PD_2d),[0,1]); colorbar; title('hi resolution, TV_PD');
figure(9); plot(cost); xlabel('iteration'); ylabel('cost');
figure(10); imagesc(abs(X_TV_PD_2d-X0_2d),[0,0.2]); colorbar; title('TV_PD error image');

SNR_TV_PD = -20*log10(norm(X_TV_PD(:)-X0(:))/norm(X0(:)));
fprintf('TV_PD (AHMOD) output SNR = %2.1f dB\n',SNR_TV_PD);
fprintf('TV_PD (AHMOD) final cost %6.4f\n',cost(end));

%% Run Weighted TV algorithm - Primal-Dual algorithm -- AHMOD from Chambolle & Pock
lambda = 1e-3; %regularization parameter (typically in the range [1e-2,1], if original image scaled to [0,1])
Niter = 200;  %number of iterations (typically 500-1000 for Fourier inversion)
tic
[X_WTV, cost] = OpWeightedTV_PD_AHMOD2(b,edgemask,lambda,A,At,res,Niter); %see comments inside OpWeightedTV_PD_AHMOD.m
toc
% Show a 2D slice of X_WTV image
X_WTV_size = size(X_WTV);
X_WTV_2d = X_WTV(:,:,round(X_WTV_size(3)/2));
figure(11); imagesc(abs(X_WTV_2d),[0,1]); colorbar; title('hi resolution, WTV');
figure(12); plot(cost); xlabel('iteration'); ylabel('cost');
figure(13); imagesc(abs(X_WTV_2d-X0_2d),[0,0.2]); colorbar; title('WTV error image');

SNR_WTV = -20*log10(norm(X_WTV(:)-X0(:))/norm(X0(:)));
fprintf('WTV (AHMOD) output SNR (Weighted TV algorithm) = %2.1f dB\n',SNR_WTV);
fprintf('WTV (AHMOD) final cost %6.4f\n',cost(end));

%% Run Weighted L2 algorithm - ADMM version
lambda = 1e-4; %regularization parameter
Niter = 100;  %number of iterations
gam = 0.5;   %ADMM parameter, in range [0.1,10]
tol = 1e-6;  %convergence tolerance
tic
[X_WL2, cost] = OpWeightedL2(b,edgemask,lambda,A,At,res,Niter,tol,gam);
toc
% Show a 2D slice
X_WL2_size = size(X_WL2);
X_WL2_2d = X_WL2(:,:,round(X_WL2_size(3)/2));
figure(14); imagesc(abs(X_WL2_2d),[0,1]); colorbar; title('hi resolution, WL2');
figure(15); plot(cost); xlabel('iteration'); ylabel('cost');
figure(16); imagesc(abs(X_WL2_2d-X0_2d),[0,0.2]); colorbar; title('WL2 error image');

SNR_WL2 = -20*log10(norm(X_WL2(:)-X0(:))/norm(X0(:)));
fprintf('WL2 output SNR (Weighted L2 algorithm) = %2.1f dB\n',SNR_WL2);
fprintf('WL2 final cost %2.4e\n',cost(end));

%% Comparison figure
% labelIFFT = sprintf('SNR=%6.1f',SNR_IFFT);
% labelTV   = sprintf('SNR=%6.1f',SNR_TV);
% labelWTV  = sprintf('SNR=%6.1f',SNR_WTV);
%
% figure(100);
% subplot(2,4,1);
% imshow(abs(X0_2d),[0 1]); title('baseline image'); axis image;
% subplot(2,4,2);
% imshow(abs(X_IFFT_2d),[0 1]); title('zero-padded IFFT'); xlabel(labelIFFT);
% subplot(2,4,3);
% imshow(abs(X_TV_2d),[0 1]); title('standard TV'); xlabel(labelTV);
% subplot(2,4,4);
% imshow(abs(X_WTV_2d),[0 1]); title('weighted TV'); xlabel(labelWTV);
% subplot(2,4,5);
% imagesc(abs(edgemask_2d),[0 1]); title('spatial weights image'); axis off; axis image;
% %imagesc(abs(Xlow_2d)); title('low-resolution input image'); axis off; axis image;
% subplot(2,4,6);
% imagesc(abs(X_IFFT_2d-X0_2d),[0 0.05]); title('IFFT error x 20');  axis off; axis image;
% subplot(2,4,7);
% imagesc(abs(X_TV_2d-X0_2d),[0 0.05]); title('TV error x 20');  axis off; axis image;
% subplot(2,4,8);
% imagesc(abs(X_WTV_2d-X0_2d),[0 0.05]); title('WTV error x 20');  axis off; axis image;
