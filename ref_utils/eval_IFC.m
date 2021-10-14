addpath ./ifcvec_release/;
addpath matlabPyrTools/;
warning('off');
root_dir = '/fileserver/tanyang/projects/ref_sr_ytan/mutiscale_warping_train/sr_results_on_LightField_data/';
exp_list = dir(root_dir);

for i=3:length(exp_list)
    cur_dir = fullfile(root_dir,exp_list(i).name); 
    cur_dir_list = dir(cur_dir);
    sum_ifc = 0.0;
    sum_psnr = 0.0;
    for j=3:length(cur_dir_list)
        path_gt = fullfile(cur_dir,cur_dir_list(j).name,'gt.png');
        path_sr = fullfile(cur_dir,cur_dir_list(j).name,'sr.png');
        img_gt = imread(path_gt);
        img_sr = imread(path_sr);
        ifc_ = ifcvec(rgb2gray(img_gt), rgb2gray(img_sr));
        sum_ifc = sum_ifc + ifc_;
        psnr_ = compute_psnr(im2single(img_gt),im2single(img_sr));
        sum_psnr = sum_psnr + psnr_;

    end 
    exp_list(i).name
    sum_ifc/(length(cur_dir_list) - 2)
    sum_psnr/(length(cur_dir_list) - 2)
end
        
