function [estimatedSignal, estimatedGradients] = doSRestimateWTV( dwi_struct, mask )
% param: dwi_struct - is the structure of dwi data that is cleaned and normalized,
% but with full intensity values
% param: mask - locations inside the brain where calculations should be done

[DWIIntensityData, gradientDirections, bValue, spaceDirectionMatrix, spaceOrigin, averagedB0, measFrame] = AverageB0AndExtractIntensity( dwi_struct );

%remove negative values
DWIIntensityData(DWIIntensityData<0)=eps;

[nx, ny, nz, numGrad]=size(DWIIntensityData);
[mnx,mny,mnz] =size(mask);

% Mask MUST be same size as image
assert( nx == mnx , 'Mask x size does not match DWI data size');
assert( ny == mny , 'Mask y size does not match DWI data size');
assert( nz == mnz , 'Mask z size does not match DWI data size');

parallel.defaultClusterProfile('local');
num_slots = getenv('NSLOTS');
if( num_slots )
  poolobj = parpool(STRTODOUBLE(num_slots));
else
  poolobj = parpool(4);
end

for itr=1:numGrad,
    %ToDo
end

delete(poolobj)

%% Insert the B0 back in
estimatedSignal = cat(4,averagedB0,estimatedSignal); %add B0 to the data
estimatedGradients=[0 0 0;new_gradients];

end