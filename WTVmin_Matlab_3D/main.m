clear all
close all

load 'dwi_b0.mat'
load 'edgemask.mat'

X0 = double(inputImage);
X0_size = size(X0);
%X0 = X0(:,:,round(X0_size(3)/2)); % make input 2D %%%%%%%%%%%%%
X0 = squeeze( X0(:,round(X0_size(2)/2),:) ); 

edgemask_size = size(edgemask);
%edgemask = edgemask(:,:,round(edgemask_size(3)/2)); % make edgemask 2D %%%%%%%%%%%%
edgemask = squeeze( edgemask(:,round(edgemask_size(2)/2),:) );

figure(1); imagesc(abs(X0),[0,1]); colorbar; title('ground truth');
%figure(1); imshow(X0,[0,1]); title('ground truth');
figure(2); imagesc(edgemask,[0,0.1]); colorbar; title('spatial weights image');
%figure(2); imshow(edgemask,[0 0.1]); title('spatial weights image');

%%
% Test algorithm on 2D images, then if verified, expand it to 3D
%
%%
m = fft2(X0);       %m=fourier data
inres = size(m);
k = get_kspace_inds( inres ); %k=fourier indices

%% Define Fourier Projection Operators
res = inres; %output resolution
lores = round(inres/2); %input lower resolution (use odd numbers)

ind_samples = get_lowpass_inds(k,lores);
[A,At] = defAAt_fourier(ind_samples, res); %Define function handles for fourier projection operators                                                
b = A(X0);       %low-resolution fourier samples
Xlow = ifft2(reshape(b,lores));
figure(3); imagesc(abs(Xlow)); colorbar; title('low resolution input');

%% Zero-padded IFFT reconstruction
X_IFFT = At(fft2(Xlow));
figure(4); imagesc(abs(X_IFFT),[0,1]); colorbar; title('hi resolution, zero-padded IFFT');
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
lambda = 3e-2; %regularization parameter
Niter = 200;  %number of iterations (typically 500-1000 for Fourier inversion)
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
fprintf('WTV output SNR (Weighted TV algorithm) = %2.1f dB\n',SNR_WTV);
