function [X, cost, resvec] = OpWeightedL2(b,edgemask,lambda,A,At,res,iter,tol,gam)
% OPWEIGHTEDL2: Solves weighted L2 regularized inverse problems.
% Minimizes the cost function
% X* = argmin_X ||A(X)-b||_2^2 + lambda ||W |D(X)| ||_2^2
% where     X* = recovered image
%           A  = linear measurement operator
%           b  = (noisy) measurements
%           W  = diagonal weight matrix built from the edge mask
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
%         
% Output:  X = high-resolution output image
%          cost = array of cost function value vs. iteration
%Define AtA fourier mask
p_image = zeros(res); p_image(1,1,1) = 1;
AtAhat = fftn(At(A(p_image)));

%Define derivative operators D, Dt, and DtD
[D,Dt] = defDDt;
DtDhat = fftn(Dt(D(p_image)));

mu = edgemask;
Atb = At(b);
X = Atb;
DX = D(X);
L = zeros(size(DX));
resvec = zeros(1,iter);
cost = zeros(1,iter);
for i = 1:iter
    % Y subprob 
    Z = gam*(DX+L);
    muinv = repmat((2*mu + gam).^(-1),[1,1,1,3]);
    Y = muinv.*Z;

    % X subprob
    X = ifftn(fftn(2*Atb + lambda*gam*Dt(Y-L))./(2*AtAhat + lambda*gam*DtDhat));   

    % L update
    DX = D(X);
    residue = DX-Y;
    L = L + residue;
    
    resvec(i) = norm(residue(:))/norm(Y(:));
    if (iter > 10) && (resvec(i) < tol)
        return;
    end
    
    %Calculate cost function      
    WDX = repmat(sqrt(mu),[1,1,1,3]).*DX;
    diff = A(X)-b;
    cost(i) = norm(diff(:)).^2 + lambda*norm(WDX(:)).^2; %cost function
end