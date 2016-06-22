function [ newDWI, metric, counts ] = BalanceDWIReplications( oldDWI )
% BalanceDWIReplications tried to make a
% heuristic for averaging dwi data together.
% gd is the input gradient directions
% output - metric is an array of degrees to the nearest gradient
  gd = oldDWI.gradientdirections;
  n = size(gd,1);
  
  metric=ones(n,1)*90;
  isGradient = ones(n,1);
  isLess5Degrees = zeros(n,1);
  counts = ones(n,1);
  metric(1) = 0;
  
  
  for ii=1:n
    if norm(gd(ii,:)) < 1e-4
        isGradient(ii) = 0;
        isLess5Degrees(ii) = 0;
        continue;
    end
    for jj=1:n
      if ii ~= jj
        angleBetween = acos(abs(dot(gd(ii,:),gd(jj,:))))*180/pi;
        metric(ii) = min( metric(ii), angleBetween );
        metric(jj) = min( metric(jj), angleBetween );
        if angleBetween < 5
            isLess5Degrees(ii) = 1;
            isLess5Degrees(jj) = 1;
            counts(ii) = counts(ii) + 1;
            counts(jj) = counts(jj) + 1;
        end
      end
    end
  end
  
  GradientsNeedDuplications = ( (counts < max(counts) ).*max(counts) - counts .* (counts < max(counts) ) ) .* isGradient;
  newDWI=oldDWI;
  for ii=1:n
      if GradientsNeedDuplications(ii) > 0
        newDWI = replicateGradient(newDWI, ii, GradientsNeedDuplications(ii) );
      end
  end
end


function [newDWI] = replicateGradient(oldDWI,gdindex, count)
   % [ errorStatus, outDWI, refDWI ] = OriginalTestCS( )
   % dwigt=nrrdLoadWithMetadata('/Users/johnsonhj/src/CompressedSensingDWI/TestSuite/Synthetic-Ground-truth_02.nhdr')
   % size(dwigt.data)
   % new_data=dwigt(:,:,:,4)
   % new_data=dwigt.data(:,:,:,4)
   % new_data=dwigt.data(:,:,:,4);
   % size(new_data)
   % bigger=cat(4,dwigt.data,new_data);
   % size(bigger)
   % new_data = 0.001.*new_data.*randn(size(new_data));
   newDWI=oldDWI;
   gradientDataToDuplicate = newDWI.data(:,:,:,gdindex);
   gradientDirectionToDuplicate = newDWI.gradientdirections(gdindex,:);
   for c=1:count
       newDWI.data = cat(4,newDWI.data,gradientDataToDuplicate);
       newDWI.gradientdirections = cat(1,newDWI.gradientdirections,gradientDirectionToDuplicate);
   end
  if 0 == 1
    %testFn = fullfile(fileparts(pwd),'TestSuite','Synthetic-Test-data_02.nhdr');
    testFn = '/Users/johnsonhj/Dropbox/DATA/0657_57309_DWI-79_concat_QCed.nrrd';
    rawDWI = nrrdLoadWithMetadata(testFn);
    [ DWI ] = nrrdReformatAndNormalize(rawDWI);
    %gd = DWI.gradientdirections;
  end
end

