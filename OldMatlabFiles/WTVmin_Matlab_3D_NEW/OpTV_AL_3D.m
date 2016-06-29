function [X, cost] = OpTV_AL_3D(b,lambda,A,At,res,Niter,beta)
% OPTV_AL: Solves TV regularized inverse problems with an
% ADMM scheme. Minimizes the cost function
% X* = argmin_X ||A(X)-b||_2^2 + lambda || |D(X)| ||_1
% where     X* = recovered image
%           A  = linear measurement operator
%           b  = (noisy) measurements
%           |D(X)| = gradient magnitude at each pixel
%
% Inputs:  A = function handle representing the forward
%               model/measurement operator
%          At = function handle representing the backwards model/
%               the transpose of the measurment operator.
%               (e.g. if A is a downsampling, At is a upsampling)                    
%          b =  a vector of measurements; should match the
%               dimensions of A(X)
%          lambda = regularization parameter that balances data fidelity
%               and smoothness. set lambda high for more smoothing.
%          siz = output image size, e.g. siz = [512,512]
%          Niter = is the number of iterations; should be ~500-1000
%          beta = ADMM parameter (beta=10 should work well for images scaled between [0,1])
%         
% Output:  X = high-resolution output image
%          cost = array of cost function value vs. iteration

%Define AtA fourier mask
p_image = zeros(res); p_image(1,1,1) = 1;
AtA = fftn(At(A(p_image)));

%Define derivative operators D, Dt, and DtD
[D,Dt] = defDDt;
DtD = fftn(Dt(D(p_image)));

%Initialize X
Atb = At(b);
X = Atb;
DX = D(X);
G = zeros(size(DX));

% Begin alternating minimization alg.
cost = zeros(1,Niter);
for i=1:Niter                   
    %Shrinkage step
    Z = DX + G;
    Z1 = Z(:,:,:,1);
    Z2 = Z(:,:,:,2);
    Z3 = Z(:,:,:,3);
    AZ = sqrt(abs(Z1).^2 + abs(Z2).^2 + abs(Z3).^2);
    shrinkZ = shrink(AZ,1/beta); %shrinkage of gradient mag.
    Z(:,:,:,1) = shrinkZ.*Z1; 
    Z(:,:,:,2) = shrinkZ.*Z2;
    Z(:,:,:,3) = shrinkZ.*Z3;    

    %Inversion step 
    F1 = fftn(2*Atb + lambda*beta*Dt(Z-G));
    F2 = lambda*beta*DtD + 2*AtA;
    X = ifftn(F1./F2);

    %Calculate error        
    DX = D(X);
    NDX = sqrt(abs(DX(:,:,:,1)).^2 + abs(DX(:,:,:,2)).^2 + abs(DX(:,:,:,3)).^2);        
    diff = A(X)-b;
    cost(i) = norm(diff(:)).^2 + lambda*sum(NDX(:)); %objective function
        
    %Lagrange multiplier update
    G = G + DX - Z;        
end
