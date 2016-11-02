close all

testbase='/Users/invicro/Desktop/Ali/DWIConvertDicomToNrrd/TestDWI';

input_dwi_dicom_dir=fullfile(testbase,'dwi_dicom');
input_dwi_header_fn=fullfile(testbase,'dwi_header','b-vectors-GE-4b0-24b750.csv');
output_dwi_nrrd_dir=fullfile(testbase,'dwi_nrrd');

% Check that the file exists
assert(exist(input_dwi_dicom_dir, 'dir') == 7, 'Input Dicom directory does not exist');
assert(exist(input_dwi_header_fn, 'file') == 2, 'Input gradient table does not exist');

if exist(output_dwi_nrrd_dir, 'dir') ~= 7
    mkdir output_dwi_nrrd_dir;
end

% hack:
file_with_ref_matadata_fn=fullfile(testbase,'dwi_nrrd_onetestcase','ANONM80FVB14I.nrrd');
assert(exist(file_with_ref_matadata_fn, 'file') == 2, 'File with reference metadata does not exist');

tic
    convert_dwi(input_dwi_dicom_dir,input_dwi_header_fn,output_dwi_nrrd_dir,file_with_ref_matadata_fn);
toc
