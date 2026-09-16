function [wfunc_file,wmeanfunc]=normalise(vfunc_list,deformation,cmeanfunc_file)

%cnt=1;
%for i=1:length(vfunc_file)
%     for j=1:length(vfunc_file{i})
%         vfunc_list{cnt,:}=vfunc_file{i}{j};
%         cnt=cnt+1;
%     end
%end

            
%normalise all the functional images to MNI space
    clear matlabbatch
    matlabbatch{1}.spm.spatial.normalise.write.subj.def = {deformation};
    matlabbatch{1}.spm.spatial.normalise.write.subj.resample = vfunc_list;
    matlabbatch{1}.spm.spatial.normalise.write.woptions.bb = [-78 -112 -70
        78 76 85];
    matlabbatch{1}.spm.spatial.normalise.write.woptions.vox = [2 2 2];
    matlabbatch{1}.spm.spatial.normalise.write.woptions.interp = 4;
    
    %run the job
    spm_jobman('run',matlabbatch)
   
%normalise the mean functional images to MNI space (for later coreg check)
    clear matlabbatch
    matlabbatch{1}.spm.spatial.normalise.write.subj.def = {deformation};
    matlabbatch{1}.spm.spatial.normalise.write.subj.resample = {cmeanfunc_file};
    matlabbatch{1}.spm.spatial.normalise.write.woptions.bb = [-78 -112 -70
        78 76 85];
    matlabbatch{1}.spm.spatial.normalise.write.woptions.vox = [2 2 2];
    matlabbatch{1}.spm.spatial.normalise.write.woptions.interp = 4;
    
    %run the job
    spm_jobman('run',matlabbatch)

%save the path of the normalised functional data
wfunc_file=cell(length(vfunc_file),1);
for i=1:length(vfunc_file)
    for j=1:length(vfunc_file{i})
        [wfunc_p,wfunc_n, wfunc_e]=fileparts(vfunc_file{i}{j});
        wfunc_file{i}{j,:}=[wfunc_p '/' insertBefore(wfunc_n,1,'w'), wfunc_e];
    end
end
%save the path of normalised mean functional data
[mean_p,mean_n,mean_e]=fileparts(cmeanfunc_file);
wmeanfunc=[mean_p '/' insertBefore(mean_n,1,'w') mean_e];

