load('EnhancedNet-DIV2K-train800.mat','scores');
% load EDSR-DIV2K_train0-99

scores_per = struct([]);
sum_per = 0;

for ii = 1: 800
    image = scores(ii)
    per = 1/2 * ((10 -image.Ma) + image.NIQE)
    scores_per(ii).name = scores(ii).name;
    scores_per(ii).MSE = scores(ii).MSE;
    scores_per(ii).Ma = scores(ii).Ma;
    scores_per(ii).NIQE = scores(ii).NIQE;
    scores_per(ii).Perceptual = per;
    sum_per = sum_per + per;
end

save('EnhancedNet-DIV2K800-per.mat','scores_per');
ave_per = sum_per/800;