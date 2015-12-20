function edgemask = compute_fri_edgemask(U,s,filter_siz,outres,q)
%% Build edgemask from annihilating filters U and weights s
kout = get_kspace_inds(outres);
ind_filter_out = (abs(kout(1,:)) <= (filter_siz(2)-1)/2 ) & ( abs(kout(2,:)) <= (filter_siz(1)-1)/2);
mu2 = zeros(outres);
for j=1:size(U,2)
    filter = zeros(outres);
    filter(ind_filter_out) = ifftshift(reshape(U(:,j),filter_siz));
    mu2 = mu2 + ((1/s(j))^q)*(abs(ifft2(filter)).^2);    
end
mu2 = mu2/max(abs(mu2(:)));
edgemask = sqrt(abs(mu2));
end