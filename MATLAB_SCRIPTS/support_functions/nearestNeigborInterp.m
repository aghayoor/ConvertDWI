function [im_NN] = nearestNeigborInterp(im,outres)
% nearest neighbor interpolation of a 3D matrix
inres = size(im);
[r,c,z]=ndgrid(linspace(1,inres(1),outres(1)),...
               linspace(1,inres(2),outres(2)),...
               linspace(1,inres(3),outres(3)));
im_NN = interp3(im,c,r,z,'nearest'); % c=x, r=y
end