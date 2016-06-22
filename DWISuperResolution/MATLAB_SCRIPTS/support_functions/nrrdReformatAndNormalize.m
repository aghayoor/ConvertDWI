function [rotatedGradientDWI, voxelLatticeToAnatomicalSpace ] = nrrdReformatAndNormalize(rawDWI)
 %  This function will review parameters from a dwi struct to ensure
 %  that all necessary information is provided, inject "resonable defaults" for missing
 %  values, and permute the data into a canonical organization for subsequent
 %  data processing.
 %
 % --The measurement frame is reset to voxel lattice orientation (and then
 %   set to identity)
 % --The gradient directions are normailzed
 % --The order is permuted to consistent space vs. gradient memory layout
 %
 % Author Hans J. Johnson

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
  rotatedGradientDWI = consistentDWI
  anatomicalToVoxelLatticeSpace = inv(directionCosines)*consistentDWI.measurementframe;
  voxelLatticeToAnatomicalSpace = inv(anatomicalToVoxelLatticeSpace); % This is return by this function to be used after CS computations
  rotatedGradientDWI.gradientdirections = ( anatomicalToVoxelLatticeSpace*consistentDWI.gradientdirections' )';
  %XXXXXXXXXXXXX

  % Renormalize to ensure unit length gradients (force all bValues to be
  % the same!!
  for j = 1:size(rotatedGradientDWI.gradientdirections,1)
      gnorm = norm(rotatedGradientDWI.gradientdirections(j,:));
      if gnorm > eps
       rotatedGradientDWI.gradientdirections(j,:) = rotatedGradientDWI.gradientdirections(j,:)/gnorm;
       % Should issue a warning here, because it indicates new b0 values
      end
  end

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
end
