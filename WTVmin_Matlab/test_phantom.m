% Test of Weighted TV algorithm for super-resolution reconstruction 
% of MRI data.
%% Load Data
load testdata;   %m=fourier data, k=fourier indices
load edgemask;   %edgemask estimated in stage one
X0 = ifft2(m);   %high-resolution ground truth in spatial domain
figure(1); imagesc(abs(X0),[0,1]); colorbar; title('ground truth');
figure(2); imagesc(edgemask,[0,1]); colorbar; title('edge mask');

%% Define Fourier Projection Operators
res = [256,256]; %output resolution
lores = [65,65]; %input resolution (use odd numbers)
ind_samples = find((abs(k(1,:)) <= (lores(2)-1)/2 & (abs(k(2,:)) <= (lores(2)-1)/2))); %Low-pass Fourier indices
[A,At] = defAAt_fourier(ind_samples, res); %Define function handles for fourier projection operators                                                
b = A(X0);       %low-resolution fourier samples
Xlow = ifft2(reshape(b,lores));
figure(3); imagesc(abs(Xlow)); colorbar; title('low resolution input');

%% Run New Weighted TV algorithm (works better with real data)
lambda = 5e-1; %regularization parameter (typically in the range [1e-2,1], if original image scaled to [0,1])
Niter = 1000;  %number of iterations (typically 500-1000 for Fourier inversion)
[X, cost] = OpWeightedTV_PD_AHMOD(b,edgemask,lambda,A,At,res,Niter); %see comments inside OpWeightedTV_PD_AHMOD.m
figure(4); imagesc(abs(X),[0,1]); colorbar; title('hi resolution output');
figure(5); plot(cost); xlabel('iteration'); ylabel('cost');
SNR = -20*log10(norm(X(:)-X0(:))/norm(X0(:)));
fprintf('WTV output SNR = %2.1f dB\n',SNR);
%% Run Old Weighted TV algorithm (works better with phantoms)
lambda = 5e-1; %regularization parameter (typically in the range [1e-2,1], if original image scaled to [0,1])
Niter = 1000;  %number of iterations (typically 500-1000 for Fourier inversion)
[X, cost] = OpWeightedTV_PD(b,edgemask,lambda,A,At,res,Niter); %see comments inside OpWeightedTV_PD.m
figure(6); imagesc(abs(X),[0,1]); colorbar; title('hi resolution output');
figure(7); plot(cost); xlabel('iteration'); ylabel('cost');
SNR = -20*log10(norm(X(:)-X0(:))/norm(X0(:)));
fprintf('WTV output SNR = %2.1f dB\n',SNR);
%% Run Standard TV algorithm (no edgemask)
lambda = 1e-4; %regularization parameter (typically in the range [1e-2,1], if original image scaled to [0,1])
Niter = 100;  %number of iterations (typically 20-100 for Fourier inversion)
[X2, cost] = OpTV_AL(b,lambda,A,At,res,Niter); %see comments inside OpWeightedTV_PD.m
figure(8); imagesc(abs(X2),[0,1]); colorbar; title('hi resolution output (no edgemask)');
figure(9); plot(cost); xlabel('iteration'); ylabel('cost');
SNR = -20*log10(norm(X(:)-X0(:))/norm(X0(:)));
fprintf('TV output SNR = %2.1f dB\n',SNR);