function s_file = smoothing2(file,fwhm)

cnt=1;
for i=1:length(file)
    for j=1:length(file{i})
        wfunc_list{cnt,1}=file{i}{j};
        cnt=cnt+1;
    end
end 

clear matlabbatch
matlabbatch{1}.spm.spatial.smooth.data = wfunc_list;
matlabbatch{1}.spm.spatial.smooth.fwhm = [fwhm fwhm fwhm];
matlabbatch{1}.spm.spatial.smooth.dtype = spm_type('float32'); %what is this? The default is 0
matlabbatch{1}.spm.spatial.smooth.im = 0;
matlabbatch{1}.spm.spatial.smooth.prefix = sprintf('s%d_',fwhm);
spm_jobman('run',matlabbatch)

% [p,n,e] = fileparts(file);
% s_file = fullfile(p,[matlabbatch{1}.spm.spatial.smooth.prefix n e]);

%save the path of the normalised functional data 
s_file=cell(length(file),1);
for i=1:length(file)
    for j=1:length(file{i})
        [sfunc_p,sfunc_n,sfunc_e]=fileparts(file{i}{j});
    s_file{i}{j,:}=[sfunc_p '/' insertBefore(sfunc_n,1,matlabbatch{1}.spm.spatial.smooth.prefix) sfunc_e];
    end
end
