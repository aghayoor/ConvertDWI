function [X, cost] = OpWeightedTV_PD_AHMOD(b,edgemask,lambda,A,At,res,Niter)
% OPWEIGHTEDTV_PD_AHMOD: Solves weighted TV regularized inverse problems with a
% primal-dual scheme. Minimizes the cost function
% X* = argmin_X ||A(X)-b||_2^2 + lambda ||W |D(X)| ||_1
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
%  
% Note this implementation is an adaptation of the AHMOD algorithm from:
% Chambolle, A., & Pock, T. (2011). A first-order primal-dual algorithm for 
%           convex problems with applications to imaging. 
%           Journal of Mathematical Imaging and Vision, 40(1), 120-145.

%Define AtA fourier mask
p_image = zeros(res,'double'); p_image(1,1) = 1;
AtAhat = fft2(At(A(p_image)));
Atbhat = fft2(At(b));

%Define derivative operator
[D,Dt] = defDDt;
%Defined weighted derivative operators
Wbig = repmat(edgemask,[1,1,2]);
WD = @(x) Wbig.*D(x);
WDt = @(x) Dt(Wbig.*x);

%Initialize variables
Atb = At(b);
X = Atb;
Xhat = X;
WDX = WD(X);
P = zeros(size(WDX));

lambda2 = lambda/2;
gamma = 0.35*(1/lambda2);

L = sqrt(12);
tau = 1/L;
sigma = 1/((L^2)*tau); 
theta = 1/sqrt(1+2*gamma*tau);

prox = @(x,lambda,tau) ... 
    ifft2((tau*Atbhat + lambda*fft2(x))./(lambda + tau*AtAhat));

% Begin primal-dual AHMOD algorithm
cost = zeros(1,Niter);
for i=1:Niter    
    %Dual Step
    P = projInfty(P + sigma*WD(Xhat));        

    %Primal Step
    Xold = X;
    X = prox(X - tau*WDt(P),lambda2,tau);    
    
    %Update Step-sizes with AHMOD rules
    theta = 1/sqrt(1+2*gamma*tau);          
    tau = theta*tau;
    sigma = sigma/theta;
        
    Xhat = X+theta*(X-Xold);         
    
    %Calculate cost function      
    WDX = WD(X);
    NWDX = sqrt(abs(WDX(:,:,1)).^2 + abs(WDX(:,:,2)).^2);
    diff = A(X)-b;
    cost(i) = norm(diff(:)).^2 + lambda*sum(NWDX(:)); %cost function
end
