function runSR(inputDWIFilename,inputEdgeMapFileName,outputDir)

    % Add main MATLAB_SCRIPTS code tree
    thisFile = mfilename('fullpath');
    addpath(fullfile(fileparts(fileparts(thisFile)),'MATLAB_SCRIPTS'));

    % Check that the file exists
    assert(exist(inputDWIFilename, 'file') == 2, 'File does not exist');
    assert(exist(inputEdgeMapFileName, 'file') == 2, 'File does not exist');

    tic
        run_sr(inputDWIFilename,inputEdgeMapFileName,outputDir);
    toc

end
