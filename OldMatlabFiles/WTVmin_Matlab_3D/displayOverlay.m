close all

[M,N] = size(X0_2d); 
block_size = 25; 
P = ceil(M / block_size); 
Q = ceil(N / block_size); 
alpha = checkerboard(block_size, P, Q) > 0; 

alpha1 = alpha(1:M, 1:N);
alpha2 = 1-alpha1;
alpha = .25;

figure;
out = X0_2d + edgemask_2d.*alpha; % overlay : choose alpha or alpha1 or alpha2
imagesc(out);
%imshow(out);