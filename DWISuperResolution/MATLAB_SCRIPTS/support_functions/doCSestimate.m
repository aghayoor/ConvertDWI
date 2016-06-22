
function [estimatedSignal, estimatedGradients] = doCSestimate( dwi_struct, mask, new_gradients, denoising_factor )
% param: dwi_struct - is the structure of dwi data that is cleaned and normalized,
% but with full intensity values
% param: mask - locations inside the brain where calculations should be
% done
% param: new_gradients - these should be the set of gradients used for
% computing the output gradient directions (THEY SHOULD NOT have anti-podal
% replicants.
% denoising_factor -- A parameter for adjusting how much de-noising is done at each iteration 0=no denoising, 0.025 for very little denoising, and 0.05 for "typical scans" level of noise.
%output
% estimatedSignal - The estimated value in the new gradient space
% estimatedGradients - the gradients for the ouput (same as new_gradients)

[DWIIntensityData, gradientDirections, bValue, spaceDirectionMatrix, spaceOrigin, averagedB0, measFrame] = AverageB0AndExtractIntensity( dwi_struct );

% Estimate Signal (must be done on data scaled by averagedB0)
numGradientDirections = size(DWIIntensityData,4);
% Scale down gradient direction data by averagedB0 data
DWIIntensityData = single(DWIIntensityData) ./ averagedB0(:,:,:,ones(1,numGradientDirections));
%remove negative values
DWIIntensityData(DWIIntensityData<0)=eps;

% Attempt to give both positive and negative gradient directions to cover
% the same space as the output directions (antipodal)
% test_results_quality
% HACK:  No need in doubling
if 1 == 0 %% Mathematically this should not be necessary, but it appears to produce more "visually appealing" results
    DWIIntensityData = cat(4,DWIIntensityData,DWIIntensityData);
    gradientDirections = [gradientDirections;-gradientDirections];
    % HACK -Removed -- why? halve the number of gradients numGradientDirections = size(gradientDirections,1)/2;
    numGradientDirections = size(gradientDirections,1);
    gradientDirections=gradientDirections(1:numGradientDirections,:);
else
    estimatedGradients = new_gradients;

%ofn = sprintf('%s_%s.nhdr',input_dwi(1:end-5),output_fn);
n0 = size(new_gradients,1);
new_gradients=new_gradients(1:n0,:);

[nx, ny, nz, d]=size(DWIIntensityData);
[mnx,mny,mnz] =size(mask);

% Mask MUST be same size as image
assert( nx == mnx , 'Mask x size does not match DWI data size');
assert( ny == mny , 'Mask y size does not match DWI data size');
assert( nz == mnz , 'Mask z size does not match DWI data size');

numGradientVoxels = nx*ny*nz;

%setup for CompressedSensing
J=2;
D0=diag(1e-6*[300; 300; 1700]);
[rho,p]=optimband(bValue,D0,icosahedron(3));
rho=rho*2^(J*p);

psi=buildridges(J,rho,1,p,0);
[v,M]=multisample(J);
A=buildsensor(gradientDirections,v,psi);
A0=buildsensor(new_gradients,v,psi);

%parameter setup
lmd=0.06;                   % Lagrangian L1
myu=denoising_factor;       % Lagrangian TV
% set myu=0; for no denoising.
% set it to 0.025 for very little denoising and to the current value of 0.05 if there is good amount
% of noise in the data (which typical scans have).
gama=0.5;                   % Bregman parameter
NIT=2000;                   % Max number of FISTA iterations
tol=1e-3;                   % Relative change in Chambolle

id=find(mask~=0);

parallel.defaultClusterProfile('local');
num_slots = getenv('NSLOTS');
if( num_slots )
  poolobj = parpool(str2num(num_slots));
else
  poolobj = parpool(4);
end

u=step2(DWIIntensityData,myu,tol);
c=step1(DWIIntensityData,A,lmd,NIT,id);   % initialization of ridgelet coefficients

Ac=reshape(reshape(c,[numGradientVoxels M])*A',[nx ny nz numGradientDirections]);
p=Ac-u;

TNIT=3;                     % number of outer iterations
for itr=1:TNIT,
    fprintf(1,'Iteration %d of %d\t',[itr TNIT]);

    t=u-p;
    c=step1(t,A,lmd/gama,NIT,id);
    Ac=reshape(reshape(c,[numGradientVoxels M])*A',[nx ny nz numGradientDirections]);

    t=(1/(1+gama))*(DWIIntensityData+gama*(Ac+p));
    u=step2(t,myu/(1+gama),tol);

    p=p+(Ac-u);

    tv=sum(sum(sum(sqrt((Ac(:,[2:ny,ny],:,:)-Ac).^2+(Ac([2:nx,nx],:,:,:)-Ac).^2+(Ac(:,:,[2:nz,nz],:)-Ac).^2))));
    f=0.5*sum((Ac(:)-DWIIntensityData(:)).^2)+lmd*sum(abs(c(:)))+myu*sum(tv);

    fprintf(1,'Cost = %f\n',f);
end

delete(poolobj)

estimatedSignal=reshape(reshape(c,[numGradientVoxels M])*A0',[nx ny nz n0]);   %estimated signal

% Scale up  gradient direction data by averagedB0 data
estimatedSignal = estimatedSignal.* averagedB0(:,:,:,ones(1,n0)); %multiply by the B0 image
%% Insert the B0 back in
estimatedSignal = cat(4,averagedB0,estimatedSignal); %add B0 to the data
estimatedGradients=[0 0 0;new_gradients];

end
