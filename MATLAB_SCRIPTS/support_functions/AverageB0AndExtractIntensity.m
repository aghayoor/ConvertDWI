function [intensityData, voxelLatticeAlignedGradientDirections, bValue, spaceDirectionMatrix, spaceOrigin, averagedB0, measFrame] = AverageB0AndExtractIntensity( dwi )
%AverageB0AndExtractIntensity Extract components of dwi data
%   This function extracts the important 
%   processing units from a dwi data structure
%   to make it suitable for algorithmic processing
 
  %XXXXXXXXXXXXX
  % Average all B0's into a single B0 and separate gradient volumes
  % first find the number of baseline data
  b0_indicies=[];
  gradient_indicies=[];
  numGradientDirs = size(dwi.gradientdirections,1);
  for j =1:numGradientDirs
      %the baseline images
      if norm(dwi.gradientdirections(j,:)) < eps
          b0_indicies = [b0_indicies;j]; %#ok<AGROW>
      else
          gradient_indicies = [gradient_indicies;j]; %#ok<AGROW>
      end
  end

  if isempty(b0_indicies)
      fprintf('Data does not have a B0 image. \n');
      return;
  end

  % separate out baselines and average
  averagedB0 = dwi.data(:,:,:,b0_indicies);
  averagedB0 = mean(averagedB0, 4);
  averagedB0(averagedB0==0) = 1;  % If the average is exactly zero, then set to 1 to avoid division by zero.  Could this be done better?

  % separate out gradient intensity volumes
  intensityData  = dwi.data(:,:,:,gradient_indicies); % drop null-gradient slices

  bValue = dwi.bvalue;
  spaceOrigin = dwi.spaceorigin;
  measFrame = dwi.measurementframe;
 
  spaceDirectionMatrix = dwi.spacedirections;
  %voxel = [norm(spaceDirectionMatrix(:,1));norm(spaceDirectionMatrix(:,2));norm(spaceDirectionMatrix(:,3))]';

  % actual gradient directions used
  voxelLatticeAlignedGradientDirections = dwi.gradientdirections(gradient_indicies,:);

end

