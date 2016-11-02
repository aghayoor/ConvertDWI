%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This function reads the DWI Dicom file and the gradients table csv file,
% and converts the input raw DWI data to the NRRD format with proper tags.
% Input: 'input_dwi_dicom_dir' input DICOM directory
%      : 'input_dwi_header_fn' name of the gradients csv file
%      : 'output_nrrd_dwi_dir' name of the output directory to write out dwi in nrrd format
%
% Output: DWI data in nrrd format.
%
% Ali Ghayoor
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function convert_dwi(input_dwi_dicom_dir, input_dwi_header_fn, output_dwi_nrrd_dir,file_with_ref_matadata_fn)

% Add main MATLAB_SCRIPTS code tree
%thisFile = mfilename('fullpath');
%addpath(fullfile(fileparts(thisFile),'support_functions'));

fprintf('Starting conversion:\n    input_dwi_dicom_dir: %s\n    input_dwi_header_fn: %s\n    output_nrrd_dwi_dir: %s\n',input_dwi_dicom_dir,input_dwi_header_fn,output_dwi_nrrd_dir);

% Number of volumes
Nvol = 28;

% Find the DICOM files in the data location
d = dir(fullfile(input_dwi_dicom_dir,'*.dcm'));

for i = 1:length(d),

  % Filename
  fname = fullfile(input_dwi_dicom_dir,d(i).name);

  % DICOM Data
  x = dicomread(fname);

  % DICOM Header
  y = dicominfo(fname);

  % Allocate for our 4D NRRD file
  if i == 1,
    Nslice = length(d)/Nvol;
    sz = [size(x) Nslice Nvol];
    img = zeros(sz);
  end

  % Populate
  inum = y.InstanceNumber;

  % Coordinates
  v = ceil(inum/Nslice);
  z = mod(inum-1,Nslice)+1;
  img(:,:,z,v) = x';

  % Progress
  perccount(i,length(d));

end

%% Now we need to create the proper image header using
% the infomration from the DICOM header and from the gradient table.
%% space directions
% 1)
% direction cosine
% "ImageOrientationPatient" tag that gives the coordinate frame in L-P by default using 6 elements
dc = reshape(y.ImageOrientationPatient,3,2);
% Then, to find the LPS orientation matrix, we should use the cross product to get the S-axis.
s_axis = [dc(2,1)*dc(3,2)-dc(3,1)*dc(2,2); dc(3,1)*dc(1,2)-dc(1,1)*dc(3,2); dc(1,1)*dc(2,2)-dc(2,1)*dc(1,2)];
dc = [dc s_axis]; % each column is corresponding to one axis (L-P-S)

% 2)
% spacing
sp = diag([y.PixelSpacing' y.SliceThickness]);

% 3)
% then, space directions is computed as:
sd = dc*sp*[1 0 0; 0 1 0; 0 0 -1];

%% gradient directions
grads = csvread(input_dwi_header_fn);

%% there is one bvalue for single shell data
bval = 750;

%% space units
% units = {'mm','mm','mm'}';

%% create nrrdStrct

% nrrdStrct=struct('data',img);
% nrrdStrct.bvalue = bval;
% nrrdStrct.modality = 'DWMRI';
% nrrdStrct.spacedefinition = 'left-posterior-superior';
% nrrdStrct.spaceunits = units;
% nrrdStrct.measurementframe = eye(3);
% nrrdStrct.gradientdirections = grads;
% nrrdStrct.spacedirections = sd;
% nrrdStrct.spaceorigin = y.ImagePositionPatient;
% nrrdStrct.space = 5;
% nrrdStrct.kinds = [1;1;1;6];
% nrrdStrct.centerings = [2;2;2;2];

% Using above struct cause an exception in nrrdSaveWithMetaData
% A HACK solution:
% First load a reference nrrd struct
[ nrrdStrct ] = nrrdLoadWithMetadata(file_with_ref_matadata_fn);

% Then change the corresponding fields
nrrdStrct.data = img;
nrrdStrct.bvalue = bval;
nrrdStrct.measurementframe = eye(3);
nrrdStrct.gradientdirections = grads;
nrrdStrct.spacedirections = sd;
nrrdStrct.spaceorigin = y.ImagePositionPatient;

%% Write out
oname = fullfile(output_dwi_nrrd_dir,[y.PatientID '.nrrd']);
fprintf('Writing output DWI nrrd file to disk...\n');
nrrdSaveWithMetadata(oname,nrrdStrct);

end
