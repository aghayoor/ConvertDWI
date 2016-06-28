clear all
close all

load '/scratch/TESTS/IpythonNotebook/20160615_HCPWF/2_SRWF/test_tune_parameters/matlabFiles/dwi_b0.mat'
%load '/scratch/TESTS/IpythonNotebook/20160615_HCPWF/2_SRWF/test_tune_parameters/matlabFiles/dwib0_traditionalway.mat'

load '/scratch/TESTS/IpythonNotebook/20160615_HCPWF/2_SRWF/test_tune_parameters/matlabFiles/edgemask_maskedQuantile.mat'
%load '/scratch/TESTS/IpythonNotebook/20160615_HCPWF/2_SRWF/test_tune_parameters/matlabFiles/edgemask_ipython.mat'

X0 = double(inputImage);
X0_size = size(X0);
X0_2d = X0(:,:,round(X0_size(3)/2)); % make input 2D %%%%%%%%%%%%%
%X0_2d = squeeze( X0(:,round(X0_size(2)/2),:) );

edgemask_size = size(edgemask);
edgemask_2d = edgemask(:,:,round(edgemask_size(3)/2)); % make edgemask 2D %%%%%%%%%%%%
%edgemask_2d = squeeze( edgemask(:,round(edgemask_size(2)/2),:) );

figure(1); imagesc(abs(X0_2d),[0,1]); colorbar; title('ground truth');
%figure(1); imshow(X0_2d,[0,1]); title('ground truth');
figure(2); imagesc(edgemask_2d,[0,1]); colorbar; title('spatial weights image');
%figure(2); imshow(edgemask_2d,[0 0.1]); title('spatial weights image');

%%
% Test algorithm on 2D images, then if verified, expand it to 3D
%
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
% Show a 2D slice of low resolution image
Xlow_size = size(Xlow);
Xlow_2d = Xlow(:,:,round(Xlow_size(3)/2));
figure(3); imagesc(abs(Xlow_2d)); colorbar; title('low resolution input');

%% Zero-padded IFFT reconstruction
X_IFFT = At(fftn(Xlow));
% Show a 2D slice of zero-padded IFFT image
X_IFFT_size = size(X_IFFT);
X_IFFT_2d = X_IFFT(:,:,round(X_IFFT_size(3)/2));
figure(4); imagesc(abs(X_IFFT_2d),[0,1]); colorbar; title('hi resolution, zero-padded IFFT');

SNR_IFFT = -20*log10(norm(X_IFFT(:)-X0(:))/norm(X0(:)));
fprintf('Zero-padded IFFT output SNR = %2.1f dB\n',SNR_IFFT);

%% Run Standard TV algorithm (no edgemask) -- new implementation
% lambda = 1e-4; %regularization parameter (typically in the range [1e-2,1], if original image scaled to [0,1])
% Niter = 100;  %number of iterations (typically 20-100 for Fourier inversion)
% [X_TV, cost] = OpTV_AL(b,lambda,A,At,res,Niter);
% figure(5); imagesc(abs(X_TV),[0,1]); colorbar; title('hi resolution, TV (no edgemask)');
% figure(6); plot(cost); xlabel('iteration'); ylabel('cost');
% SNR_TV = -20*log10(norm(X_TV(:)-X0(:))/norm(X0(:)));
% fprintf('TV output SNR = %2.1f dB\n',SNR_TV);

%% Run Standard TV algorithm (no edgemask)
lambda = 6e-4; %regularization parameter
Niter = 25;  %number of iterations (typically 500-1000 for Fourier inversion)
siz = size(edgemask);
edgemask_1 = ones(siz);
tic
[X_TV, cost] = OpWeightedTV_PD_AHMOD(b,edgemask_1,lambda,A,At,res,Niter);
toc
% Show a 2D slice of X_TV image
X_TV_size = size(X_TV);
X_TV_2d = X_TV(:,:,round(X_TV_size(3)/2));
figure(5); imagesc(abs(X_TV_2d),[0,1]); colorbar; title('hi resolution, TV (no edgemask)');
figure(6); plot(cost); xlabel('iteration'); ylabel('cost');

SNR_TV = -20*log10(norm(X_TV(:)-X0(:))/norm(X0(:)));
fprintf('TV output SNR = %2.1f dB\n',SNR_TV);

%% Run New Weighted TV algorithm
lambda = 6e-4; %regularization parameter (typically in the range [1e-2,1], if original image scaled to [0,1])
Niter = 25;  %number of iterations (typically 500-1000 for Fourier inversion)
tic
[X_WTV, cost] = OpWeightedTV_PD_AHMOD(b,edgemask,lambda,A,At,res,Niter); %see comments inside OpWeightedTV_PD_AHMOD.m
toc
% Show a 2D slice of X_WTV image
X_WTV_size = size(X_WTV);
X_WTV_2d = X_WTV(:,:,round(X_WTV_size(3)/2));
figure(7); imagesc(abs(X_WTV_2d),[0,1]); colorbar; title('hi resolution, WTV');
figure(8); plot(cost); xlabel('iteration'); ylabel('cost');

SNR_WTV = -20*log10(norm(X_WTV(:)-X0(:))/norm(X0(:)));
fprintf('WTV output SNR (Weighted TV algorithm) = %2.1f dB\n',SNR_WTV);

%% Comparison figure
labelIFFT = sprintf('SNR=%6.1f',SNR_IFFT);
labelTV   = sprintf('SNR=%6.1f',SNR_TV);
labelWTV  = sprintf('SNR=%6.1f',SNR_WTV);

figure(100);
subplot(2,4,1);
imshow(abs(X0_2d),[0 1]); title('baseline image'); axis image;
subplot(2,4,2);
imshow(abs(X_IFFT_2d),[0 1]); title('zero-padded IFFT'); xlabel(labelIFFT);
subplot(2,4,3);
imshow(abs(X_TV_2d),[0 1]); title('standard TV'); xlabel(labelTV);
subplot(2,4,4);
imshow(abs(X_WTV_2d),[0 1]); title('weighted TV'); xlabel(labelWTV);
subplot(2,4,5);
imagesc(abs(edgemask_2d),[0 1]); title('spatial weights image'); axis off; axis image;
%imagesc(abs(Xlow_2d)); title('low-resolution input image'); axis off; axis image;
subplot(2,4,6);
imagesc(abs(X_IFFT_2d-X0_2d),[0 0.05]); title('IFFT error x 20');  axis off; axis image;
subplot(2,4,7);
imagesc(abs(X_TV_2d-X0_2d),[0 0.05]); title('TV error x 20');  axis off; axis image;
subplot(2,4,8);
imagesc(abs(X_WTV_2d-X0_2d),[0 0.05]); title('WTV error x 20');  axis off; axis image;
