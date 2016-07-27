function [normalizedSignal,estimatedNNsignal,estimatedIFFTsignal,estimatedTVsignal,estimatedWTVsignal] = doSRestimate( dwi, edgemap, samplingFactor )
% param: DWIIntensityData - is the dwi 4D data that is cleaned and normalized
% param: edgemap - a weight matrix derived from anatomical edge locations

DWIIntensityData = dwi.data;
[nx, ny, nz, numComponents]=size(DWIIntensityData);
[mnx,mny,mnz] =size(edgemap);

% Mask MUST be same size as image
assert( nx == mnx , 'Mask x size does not match DWI data size');
assert( ny == mny , 'Mask y size does not match DWI data size');
assert( nz == mnz , 'Mask z size does not match DWI data size');

delete(gcp)
parallel.defaultClusterProfile('local');
num_slots = getenv('NSLOTS');
if( num_slots )
  poolobj = parpool(STRTODOUBLE(num_slots));
else
  poolobj = parpool(4);
end

normalizedSignal = NormalizeData(DWIIntensityData); % high-res image normalized between 0 and 1

estimatedNNsignal = single(zeros(size(DWIIntensityData))); % reconstructed by Nearest-Neighbor interpolation
estimatedIFFTsignal = single(zeros(size(DWIIntensityData))); % reconstructed by zero-padded IFFT
estimatedTVsignal = single(zeros(size(DWIIntensityData))); % reconstructed by Total Variation
estimatedWTVsignal = single(zeros(size(DWIIntensityData))); % reconstructed by Weighted Total Variation

%% Find lowpass indices to define Fourier Projection Operators [A,At]
highres = size(edgemap); %output resolution
lowres = round(highres/samplingFactor); %input lower resolution
k = get_kspace_inds( highres ); %k=fourier indices
lowpass_inds = get_lowpass_inds(k,lowres);
% Define function handles for fourier projection operators
[A,At] = defAAt_fourier(lowpass_inds, highres);

%% HACK for debugging
%numComponents=2; % processed compoenent
%fn = 0; %figure number

% parfor causes application out of memory!
% use for temporarly
for c=1:numComponents
    %% data component
    X0 = normalizedSignal(:,:,:,c);
    %% low-resolution fourier samples: input to SR algorithms
    b = A(X0);
    %% Nearest-Neighbor reconstruction
    % create lowres image from highres image
    FX0 = fftn(X0);
    Xlow = (prod(lowres)/prod(highres))*real(ifftn(reshape(FX0(lowpass_inds),lowres))); %This puts Xlow in the range [0,1]
    X_NN = nearestNeigborInterp(Xlow,highres);
    estimatedNNsignal(:,:,:,c) = X_NN;
    %% Zero-padded IFFT reconstruction
    Xlow = real(ifftn(reshape(b,lowres)));
    X_IFFT = At(fftn(Xlow));
    estimatedIFFTsignal(:,:,:,c) = X_IFFT;
    %% Run Standard TV algorithm (no edgemap)
    % lambda = 5e-4; %regularization parameter
    % Niter = 200;  %number of iterations
    % gam = 0.01; %set in the range [0.01,1].
    % Xinit = real(At(b)); %initialization
    % [X_TV, cost] = OpWeightedTV_PD_AHMOD(b,ones(highres),lambda,A,At,highres,Niter,gam,Xinit); %see comments inside OpWeightedTV_PD_AHMOD.m
    %
    lambda = 1e-3; %regularization parameter
    Niter = 100;   %number of iterations
    gam = 1;       %ADMM parameter, in range [0.1,10]
    tol = 1e-8;    %convergence tolerance
    [X_TV, cost] = OpWeightedL2(b,ones(highres),lambda,A,At,highres,Niter,tol,gam);
    estimatedTVsignal(:,:,:,c) = X_TV;
    %% Run New Weighted TV algorithm
    gradient_magnitude = norm(dwi.gradientdirections(c,:));
    if gradient_magnitude < 0.05 %% this component is a zero gradient %%
    %   lambda = 5e-4; %regularization parameter (typically in the range [1e-2,1], if original image scaled to [0,1])
    %   Niter = 200;  %number of iterations
    %   gam = 0.01; %set in the range [0.01,1].
    %   Xinit = real(At(b)); %initialization
    %   [X_WTV, cost] = OpWeightedTV_PD_AHMOD(b,edgemap,lambda,A,At,highres,Niter,gam,Xinit); %see comments inside OpWeightedTV_PD_AHMOD.m
    %   %%%%
        lambda = 1e-3; %regularization parameter
        Niter = 100;   %number of iterations
        gam = 1;       %ADMM parameter, in range [0.1,10]
        tol = 1e-8;    %convergence tolerance
        [X_WTV, cost] = OpWeightedL2(b,edgemap,lambda,A,At,highres,Niter,tol,gam);
        estimatedWTVsignal(:,:,:,c) = X_WTV;
    else %% this component is a non-zero gradient %%
        % diffusion component has lots of more high freqeuncy data, and
        % edges can be perturbed, so we need an edgemap that puts less
        % emphasis on edges
        emap = edgemap + std(edgemap(:));
        emap(emap>1) = 1; % the max value should always be one
        %
        lambda = 1e-3; %regularization parameter
        Niter = 100;   %number of iterations
        gam = 1;       %ADMM parameter, in range [0.1,10]
        tol = 1e-8;    %convergence tolerance
        [X_WTV, cost] = OpWeightedL2(b,emap,lambda,A,At,highres,Niter,tol,gam);
        estimatedWTVsignal(:,:,:,c) = X_WTV;
    end
    %% Just for sanity check and debugging
%     if(c<=2)
%         X0_size = size(X0);
%         X0_2d = X0(:,:,round(X0_size(3)/2));
%         fn=fn+1; figure(fn); imagesc(abs(X0_2d),[0 1]); colorbar; title('ground truth');
%
%         edgemap_size = size(edgemap);
%         edgemap_2d = edgemap(:,:,round(edgemap_size(3)/2));
%         fn=fn+1; figure(fn); imagesc(edgemap_2d,[0 1]); colorbar; title('spatial weights image');
%
%         Xlow_size = size(Xlow);
%         Xlow_2d = Xlow(:,:,round(Xlow_size(3)/2));
%         fn=fn+1; figure(fn); imagesc(abs(Xlow_2d)); colorbar; title('low resolution input');
%
%         X_NN_size = size(X_NN);
%         X_NN_2d = X_NN(:,:,round(X_NN_size(3)/2));
%         fn=fn+1; figure(fn); imagesc(abs(X_NN_2d),[0 1]); colorbar; title('hi resolution, nearest neigbour');
%
%         X_IFFT_size = size(X_IFFT);
%         X_IFFT_2d = X_IFFT(:,:,round(X_IFFT_size(3)/2));
%         fn=fn+1; figure(fn); imagesc(abs(X_IFFT_2d),[0 1]); colorbar; title('hi resolution, zero-padded IFFT');
%
%         X_L2_size = size(X_TV);
%         X_L2_2d = X_TV(:,:,round(X_L2_size(3)/2));
%         fn=fn+1; figure(fn); imagesc(abs(X_L2_2d),[0 1]); colorbar; title('hi resolution, TV');
%
%         X_WL2_size = size(X_WTV);
%         X_WL2_2d = X_WTV(:,:,round(X_WL2_size(3)/2));
%         fn=fn+1; figure(fn); imagesc(abs(X_WL2_2d),[0 1]); colorbar; title('hi resolution, WTV');
%
%         SNR_NN = -20*log10(norm(X_NN(:)-X0(:))/norm(X0(:)));
%         fprintf('Nearest-Neigbour output SNR = %2.1f dB\n',SNR_NN);
%
%         SNR_IFFT = -20*log10(norm(X_IFFT(:)-X0(:))/norm(X0(:)));
%         fprintf('Zero-padded IFFT output SNR = %2.1f dB\n',SNR_IFFT);
%
%         SNR_L2 = -20*log10(norm(X_TV(:)-X0(:))/norm(X0(:)));
%         fprintf('L2 output SNR (L2 algorithm) = %2.1f dB\n',SNR_L2);
%         fprintf('L2 final cost %2.4e\n',cost(end));
%
%         SNR_WL2 = -20*log10(norm(X_WTV(:)-X0(:))/norm(X0(:)));
%         fprintf('WL2 output SNR (Weighted L2 algorithm) = %2.1f dB\n',SNR_WL2);
%         fprintf('WL2 final cost %2.4e\n',cost(end));
%     end
end % end of for loop

delete(poolobj)

end

function [normArr] = NormalizeData(arr)
  % This function normalizes input matrix between zero and one.
  newMax = single(1);
  newMin = single(0);
  oldMax = single(max(arr(:)));
  oldMin = single(min(arr(:)));
  f = (newMax-newMin)/(oldMax-oldMin);
  normArr = (single(arr)-oldMin)*f+newMin;
end
