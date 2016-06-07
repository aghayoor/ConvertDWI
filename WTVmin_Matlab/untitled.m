load Data/dwi_edgemask_fri;
figure(1); imagesc(edgemask,[0,1]); colorbar; title('dwi fri edge mask');

load Data/dwi_lowresx2_edgemask_fri;
figure(2); imagesc(edgemask,[0,1]); colorbar; title('dwi lowres x2 fri edge mask');

load Data/t1_edgemask_fri;
figure(3); imagesc(edgemask,[0,1]); colorbar; title('t1 fri edge mask');

load Data/t2_edgemask_fri;
figure(4); imagesc(edgemask,[0,1]); colorbar; title('t2 fri edge mask');