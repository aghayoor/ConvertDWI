function [X, cost] = OpWeightedTV_AL(b,edgemask,lambda,A,At,siz,Niter,beta)
% OPTV: Solves TV regularized inverse problems with an alternating 
% minimization algorithm. Returns X, the recovered image, and earray,
% the values of the cost function for all iterations.
%
% Based on the fTVd implementation: 
% http://www.caam.rice.edu/~optimization/L1/ftvd/

%Define AtA
p_image = zeros(siz,'double'); p_image(1,1,1) = 1;
AtA = fftn(At(A(p_image)));

%Define derivative operator
[D,Dt] = defDDt;
%Defined weighted derivative 
Wbig = repmat(edgemask,[1,1,1,3]);
WD = @(x) Wbig.*D(x);
WDt = @(x) Dt(Wbig.*x); %assuming weights are real
%Define Fourier symbol for WDtWD
%WDtWD = fftn(WDt(WD(p_image)));
%DtD = fftn(Dt(D(p_image)));

%Initialize X
Atb = At(b);
X = At(b);
DX = WD(X);
G = zeros(size(DX)); %lagrange multiplier
Z = zeros(size(DX)); %aux variable

pcg_iter = 100;
pcg_tol = 1e-12;

% Begin alternating minimization alg.
cost = [];
for i=1:Niter     
        %Inversion step         
        B = @(x) reshape(lambda*beta*WDt(WD(reshape(x,siz))) + 2*At(A(reshape(x,siz))),[prod(siz),1]);
        c = reshape(2*Atb + lambda*beta*WDt(Z-G),[prod(siz),1]);
        [X,flag,relres,iter,resvec] = pcg(B,c,pcg_tol,pcg_iter,[],[],X(:));
        relres
        X = reshape(X,siz);
%         F1 = fft2(2*At(b) + lambda*beta*Dt(Z-G));
%         F2 = lambda*beta*DtD + 2*AtA;
%         X = ifft2(F1./F2);

        %Calculate error        
        DX = WD(X);
        DX1 = DX(:,:,:,1); 
        DX2 = DX(:,:,:,2);
        DX3 = DX(:,:,:,3);        
        ADX = sqrt(abs(DX1).^2 + abs(DX2).^2 +  abs(DX3).^2); %isotropic TV           
        diff = A(X)-b;
        thiscost = norm(diff(:)).^2 + lambda*sum(ADX(:)); %objective function
        cost = [cost,thiscost];

        %Shrinkage step
        %H = delta*DX + (1-delta)*Z; %over-relaxation
        %Z = H + G;
        Z = DX + G;
        Z1 = Z(:,:,:,1);
        Z2 = Z(:,:,:,2);
        Z3 = Z(:,:,:,3);
        AZ = sqrt(abs(Z1).^2 + abs(Z2).^2+ abs(Z3).^2);
        shrinkZ = shrink(AZ,1/beta); %shrinkage of gradient mag.
        Z(:,:,:,1) = shrinkZ.*Z1; 
        Z(:,:,:,2) = shrinkZ.*Z2;
        Z(:,:,:,3) = shrinkZ.*Z3; 
        

%         %Convergence test
%         if ii > 2
%             if abs((cost(end)-cost(end-1)))<tol
%                 break;
%             end
%         end  
    %Lagrange multiplier update
    %G = G + H - Z;
    G + DX - Z; 
    %Increase continuation parameter
    %beta = beta*bfactor;
        
end
