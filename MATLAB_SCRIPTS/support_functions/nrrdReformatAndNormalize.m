function [rotatedGradientDWI, voxelLatticeToAnatomicalSpace ] = nrrdReformatAndNormalize(rawDWI)
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

  consistentDWI = rawDWI;

  %XXXXXXXXXXXXX
  % Make an identity measurement frame by rotating gradients to be
  % interpreted in voxel lattice orientation.
  if ~isfield(consistentDWI, 'measurementframe')
    consistentDWI.measurementframe = eye(3);
  end

 %make adjustment of the spacedirections and measurement frame
 spaceDirectionMatrix = rawDWI.spacedirections;
 voxel = [norm(spaceDirectionMatrix(:,1));norm(spaceDirectionMatrix(:,2));norm(spaceDirectionMatrix(:,3))]';
 directionCosines = spaceDirectionMatrix./repmat(voxel,[3 1]);

  % Remove the measurement frame from the gradient direcitons.  This makes
  % gradients relative to the voxel lattice.
  rotatedGradientDWI = consistentDWI;
  anatomicalToVoxelLatticeSpace = directionCosines\consistentDWI.measurementframe; % inv(DC)*measFrame
  voxelLatticeToAnatomicalSpace = inv(anatomicalToVoxelLatticeSpace); % This is return by this function to be used after CS computations
  rotatedGradientDWI.gradientdirections = ( anatomicalToVoxelLatticeSpace*consistentDWI.gradientdirections' )';
  % Force measurement from to Identity matrix as it is already applied to the gradient directions.
  rotatedGradientDWI.measurementframe = eye(3);
  %XXXXXXXXXXXXX

  %XXXXXXXXXXXXX
  % Permute the order of data to be in a cononical format
  order = [1 2 3 4];
  %find the storage format
  switch size(rotatedGradientDWI.gradientdirections,1)
      case size(rotatedGradientDWI.data,1)
         order = [2 3 4 1];
      case size(rotatedGradientDWI.data,2)
         order = [1 4 3 2];
      case size(rotatedGradientDWI.data,3)
         order = [1 2 4 3];
  end
  rotatedGradientDWI.data = permute(rotatedGradientDWI.data,order);
  rotatedGradientDWI.centerings = rotatedGradientDWI.centerings(order);
  rotatedGradientDWI.kinds = rotatedGradientDWI.kinds(order);
  %XXXXXXXXXXXXX

  % Normalize DWI components between zero and one
  numGradientDirs = size(rotatedGradientDWI.gradientdirections,1);
  for c=1:numGradientDirs
      data_component_3D = rotatedGradientDWI.data(:,:,:,c);
      data_component_3D = NormalizeDataComponent(data_component_3D);
      rotatedGradientDWI.data(:,:,:,c) = data_component_3D;
  end

end

function [normArr] = NormalizeDataComponent(arr)
  % This function normalizes a 3D matrix between zero and one.
  newMax = 1.0;
  newMin = 0.0;
  oldMax = double(max(arr(:)));
  oldMin = double(min(arr(:)));
  f = (newMax-newMin)/(oldMax-oldMin);
  normArr = (arr-oldMin)*f+newMin;
end
