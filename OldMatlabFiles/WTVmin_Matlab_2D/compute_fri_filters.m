function [U,s] = compute_fri_filters(inputImage,sample_siz,filter_siz,lambda0)
%% IRLS Nuclear Norm Minimization
%Solves min_X (1/p)*||T(X)||_p^p + (lambda/2)*||AX-b||_2^2
%where A is a Fourier undersampling mask, and
%QX is the block Toeplitz matrix built from k-space data X
%Uses a two-stage IRLS MM approach, which alternates between updating
%an annihilating filter/edge mask and a least squares annihilation
%% Load data
m = fft2(double(inputImage));

%define k-space index sets
inres = size(m);
k0 = get_kspace_inds(inres);

res = 2*ceil(sample_siz/2)-1;  %sample window resolution (use odd nums)
filter_siz = 2*ceil(filter_siz/2)-1; %filter dimensions (use odd nums)
filter_siz2 = 2*filter_siz - [1,1];
overres = res + 2*filter_siz; %2*res-1;

%trim input data
ind_trim = (abs(k0(1,:)) <= (res(2)-1)/2) & (abs(k0(2,:)) <= (res(1)-1)/2);
m = reshape(m(ind_trim),res);
%redefine k-space indices for over-resolved grid
k = get_kspace_inds(overres);
ind_full = ( abs(k(1,:)) <= (res(2)-1)/2 ) & ( abs(k(2,:)) <= (res(1)-1)/2 );
ind_filter = ( abs(k(1,:)) <= (filter_siz(2)-1)/2 ) & ( abs(k(2,:)) <= (filter_siz(1)-1)/2 );
ind_filter2 = ( abs(k(1,:)) <= (filter_siz(2)-1) ) & ( abs(k(2,:)) <= (filter_siz(1)-1));

%% init variables and operators
%1st order derivatives
scaledz = max(res);
clear dz;
dz(:,:,1) = reshape((1i*2*pi*(k(1,:))).',overres)/(scaledz);
dz(:,:,2) = reshape((1i*2*pi*(k(2,:))).',overres)/(scaledz);

M = @(z) repmat(z,[1,1,size(dz,3)]).*dz;
Mt = @(Z) sum(Z.*conj(dz),3);
MtMmask = Mt(M(ones(overres)));

mask_pad = zeros(overres);
mask_pad(ind_full) = 1;

Atb_pad = zeros(overres);
Atb_pad(ind_full) = m;
x = m;
x_pad = zeros(overres);
x_pad(ind_full) = x;

%% run alg.
%lambda0 = 1e-5; %regularization parameter
lambda = 1/lambda0;
%parameters for epsilon update: eps = min(eps,gamma*s(r+1));
eps = 0;         %epsilson initialization
eta = 1.3;       %epsilon update is eps = eps/eta;
epsmin = 1e-5;
 
p = 0;               %p=1   is nuclear norm, p=0 is Shatten p=0 / log det penalty
q = 1-(p/2);         %q=1/2 is nuclear norm, q=1 is Shatten p=0 / log det penalty

y = lambda*Atb_pad;
y = y(:);

%cost = [];
%%
iter = 5;
for i=1:iter

gradx = M(x_pad);
gradx_ifft = ifft2(gradx);
sos = fft2(sum(conj(gradx_ifft).*gradx_ifft,3));
sos2 = fftshift(reshape(sos(ind_filter2),filter_siz2));
R = im2col(sos2,filter_siz);
R = rot90(R,-1);
[U,S] = eig(R+eps*eye(size(R)));
s = diag(S); %note: s = sing. values squared.
%figure(11); plot(0.5*log10(s)); title('sing vals, log scale'); drawnow;

%calculate cost
if p == 0
shatten = 0.5*sum(log(s-eps));
else
shatten = (1/p)*sum((s-eps).^(p/2));
end

%diff = x_pad-Atb_pad;
%thiscost = lambda0*shatten + 0.5*norm(diff(:)).^2; %objective function
%cost = [cost,thiscost];
%figure(12); plot(cost); title('cost'); drawnow;

%update epsilon
if i == 1
    eps = max(s)/1000; %inital epsilon
else    
    eps = max(eps/eta,epsmin);
end

%compute sos-polynomial
mu = zeros(overres);
for j=1:length(s)
    filter = zeros(overres);
    filter(ind_filter) = ifftshift(reshape(U(:,j),filter_siz));
    mu = mu + ((1/s(j))^q)*(abs(ifft2(filter)).^2);    
end

%skip ADMM problem in final iteration
if i == iter
    fprintf('Finished iteration %d of %d\n',i,iter);    
    break;
end
%figure(14); imagesc(sqrt(abs(mu))); colorbar; colormap jet; title('edgemask');    

%ADMM solution of WL2 problem
gam = max(mu(:))/10;  %tenth of sos-mask max value
L = zeros(size(dz)); %warm-start?
admm_relres = [];
for l = 1:200
    x_temp = x_pad;
    % Y subprob 
    Z = gam*(M(x_pad)-L);
    muinv = repmat((mu + gam).^(-1),[1,1,size(dz,3)]);
    Y = fft2(muinv.*ifft2(Z));

    % x subprob
    W = Atb_pad + (gam/lambda)*Mt(Y+L);
    x_pad = W./(mask_pad + (gam/lambda)*MtMmask);

    % L update
    L = L + Y - M(x_pad);

    admm_relres = [admm_relres, norm(x_pad(:)-x_temp(:))/norm(x_pad(:))];
    if (l >= 10) && (admm_relres(end) < 1e-5)
        break;
    end
end

fprintf('Finished iteration %d of %d\n',i,iter);
end

end