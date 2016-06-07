%
% Test of Weighted TV algorithm for super-resolution reconstruction of 2D MRI data.
%
%% Load Data
load Data/dwib0_testdata;
%load Data/dwig1_testdata;

%% Load Weights
%load Data/t2_edgemask_fri;   %edgemask estimated in stage one
load /scratch/TESTS/Aim2/MatlabFiles/edgemask_t1t2_1ByGMI.mat
%load /scratch/TESTS/Aim2/MatlabFiles/edgemask_t1t2_1ByGMI_square.mat

%%
X0 = double(inputImage);
m = fft2(X0);       %m=fourier data
inres = size(m);
indx = [0:((inres(2)/2)-1), -(inres(2)/2):-1];
indy = [0:((inres(1)/2)-1), -(inres(1)/2):-1];
[kx,ky] = meshgrid(indx,indy);
k(1,:) = kx(:);     %k=fourier indices
k(2,:) = ky(:);

figure(1); imagesc(abs(X0),[0,1]); colorbar; title('ground truth');
figure(2); imshow(edgemask,[0 1]); title('FRI spatial weights image');

%% Define Fourier Projection Operators
res = [256,256]; %output resolution
lores = [127,127]; %input resolution (use odd numbers)
%lores = [63,63];
ind_samples = find((abs(k(1,:)) <= (lores(2)-1)/2 & (abs(k(2,:)) <= (lores(1)-1)/2))); %Low-pass Fourier indices
[A,At] = defAAt_fourier(ind_samples, res); %Define function handles for fourier projection operators
b = A(X0);       %low-resolution fourier samples
Xlow = ifft2(reshape(b,lores));
figure(3); imagesc(abs(Xlow)); colorbar; title('low resolution input');

%% Zero-padded IFFT reconstruction
X_IFFT = At(fft2(Xlow));
figure(4); imagesc(abs(X_IFFT)); colorbar; title('hi resolution, zero-padded IFFT');
SNR_IFFT = -20*log10(norm(X_IFFT(:)-X0(:))/norm(X0(:)));
fprintf('Zero-padded IFFT output SNR = %2.1f dB\n',SNR_IFFT);

%% Run Standard TV algorithm (no edgemask)
% lambda = 1e-4; %regularization parameter (typically in the range [1e-2,1], if original image scaled to [0,1])
% Niter = 100;  %number of iterations (typically 20-100 for Fourier inversion)
% [X_TV, cost] = OpTV_AL(b,lambda,A,At,res,Niter); %see comments inside OpWeightedTV_PD.m
% figure(5); imagesc(abs(X_TV),[0,1]); colorbar; title('hi resolution, TV (no edgemask)');
% figure(6); plot(cost); xlabel('iteration'); ylabel('cost');
% SNR_TV = -20*log10(norm(X_TV(:)-X0(:))/norm(X0(:)));
% fprintf('TV output SNR = %2.1f dB\n',SNR_TV);

%% Run Standard TV algorithm (no edgemask)
lambda = 3e-2; %regularization parameter (typically in the range [1e-2,1], if original image scaled to [0,1])
Niter = 200;  %number of iterations (typically 20-100 for Fourier inversion)
siz = size(edgemask);
edgemask_1 = ones(siz);
[X_TV, cost] = OpWeightedTV_PD_AHMOD(b,edgemask_1,lambda,A,At,res,Niter);
figure(5); imagesc(abs(X_TV),[0,1]); colorbar; title('hi resolution, TV (no edgemask)');
figure(6); plot(cost); xlabel('iteration'); ylabel('cost');
SNR_TV = -20*log10(norm(X_TV(:)-X0(:))/norm(X0(:)));
fprintf('TV output SNR = %2.1f dB\n',SNR_TV);

%% Run New Weighted TV algorithm
lambda = 3e-2; %regularization parameter (typically in the range [1e-2,1], if original image scaled to [0,1])
Niter = 200;  %number of iterations (typically 500-1000 for Fourier inversion)
[X_WTV, cost] = OpWeightedTV_PD_AHMOD(b,edgemask,lambda,A,At,res,Niter); %see comments inside OpWeightedTV_PD_AHMOD.m
figure(7); imagesc(abs(X_WTV),[0,1]); colorbar; title('hi resolution, WTV new');
figure(8); plot(cost); xlabel('iteration'); ylabel('cost');
SNR_WTV = -20*log10(norm(X_WTV(:)-X0(:))/norm(X0(:)));
fprintf('WTV output SNR (New Weighted TV algorithm) = %2.1f dB\n',SNR_WTV);

%% Comparison figure
labelIFFT = sprintf('SNR=%6.1f',SNR_IFFT);
labelTV   = sprintf('SNR=%6.1f',SNR_TV);
labelWTV  = sprintf('SNR=%6.1f',SNR_WTV);

figure(100);
subplot(2,4,1);
imshow(abs(X0),[0 1]); title('baseline image'); axis image;
subplot(2,4,2);
imshow(abs(X_IFFT),[0 1]); title('zero-padded IFFT'); xlabel(labelIFFT);
subplot(2,4,3);
imshow(abs(X_TV),[0 1]); title('standard TV'); xlabel(labelTV);
subplot(2,4,4);
imshow(abs(X_WTV),[0 1]); title('weighted TV'); xlabel(labelWTV);
subplot(2,4,5);
%imagesc(abs(edgemask),[0 1]); title('spatial weights image'); axis off; axis image;
imagesc(abs(Xlow)); title('low-resolution input image'); axis off; axis image;
subplot(2,4,6);
imagesc(abs(X_IFFT-X0),[0 0.05]); title('IFFT error x 20');  axis off; axis image;
subplot(2,4,7);
imagesc(abs(X_TV-X0),[0 0.05]); title('TV error x 20');  axis off; axis image;
subplot(2,4,8);
imagesc(abs(X_WTV-X0),[0 0.05]); title('WTV error x 20');  axis off; axis image;

%colormap('default');
