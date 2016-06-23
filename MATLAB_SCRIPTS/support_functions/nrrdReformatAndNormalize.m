function [reformattedDWI] = nrrdReformatAndNormalize(rawDWI)
 %  This function will review parameters from a dwi struct to ensure
 %  that all necessary information is provided, inject "resonable defaults" for missing
 %  values, and permute the data into a canonical organization for subsequent
 %  data processing.
 %
 % --The measurement frame is reset to voxel lattice orientation (and then
 %   set to identity)
 % --The order is permuted to consistent space vs. gradient memory layout
 %
 % Author Hans J. Johnson, Ali Ghayoor

  reformattedDWI = rawDWI;

  %XXXXXXXXXXXXX
  % Make an identity measurement frame by rotating gradients to be
  % interpreted in voxel lattice orientation.
  if ~isfield(reformattedDWI, 'measurementframe')
    reformattedDWI.measurementframe = eye(3);
  end

  % Remove the measurement frame from the gradient direcitons
  reformattedDWI.gradientdirections = ( reformattedDWI.measurementframe*reformattedDWI.gradientdirections' )';
  % Force measurement from to Identity matrix as it is already applied to the gradient directions.
  reformattedDWI.measurementframe = eye(3);
  %XXXXXXXXXXXXX

  %XXXXXXXXXXXXX
  % Permute the order of data to be in a cononical format
  order = [1 2 3 4];
  %find the storage format
  switch size(reformattedDWI.gradientdirections,1)
      case size(reformattedDWI.data,1)
         order = [2 3 4 1];
      case size(reformattedDWI.data,2)
         order = [1 4 3 2];
      case size(reformattedDWI.data,3)
         order = [1 2 4 3];
  end
  reformattedDWI.data = permute(reformattedDWI.data,order);
  reformattedDWI.centerings = reformattedDWI.centerings(order);
  reformattedDWI.kinds = reformattedDWI.kinds(order);
  %XXXXXXXXXXXXX

  %XXXXXXXXXXXXX
  % Normalize DWI components between zero and one
  DWIIntensityData = single(reformattedDWI.data);
  numGradientDirs = size(reformattedDWI.gradientdirections,1);
  for c=1:numGradientDirs
      data_component_3D = DWIIntensityData(:,:,:,c);
      data_component_3D = NormalizeDataComponent(data_component_3D);
      DWIIntensityData(:,:,:,c) = data_component_3D;
  end
  %remove negative values
  DWIIntensityData(DWIIntensityData<0)=eps;
  %
  reformattedDWI.data = DWIIntensityData;
  %XXXXXXXXXXXXX
end

function [normArr] = NormalizeDataComponent(arr)
  % This function normalizes a 3D matrix between zero and one.
  newMax = single(1);
  newMin = single(0);
  oldMax = single(max(arr(:)));
  oldMin = single(min(arr(:)));
  f = (newMax-newMin)/(oldMax-oldMin);
  normArr = (arr-oldMin)*f+newMin;
end
