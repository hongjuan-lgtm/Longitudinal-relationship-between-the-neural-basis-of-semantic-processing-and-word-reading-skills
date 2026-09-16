{\rtf1\ansi\ansicpg1252\cocoartf2870
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 function contrast_f(mat,contrasts,weights)\
\
matlabbatch=[];\
matlabbatch\{1\}.spm.stats.con.spmmat = \{mat\};\
for ii=1:length(contrasts)\
matlabbatch\{1\}.spm.stats.con.consess\{ii\}.tcon.name = contrasts\{ii\};\
matlabbatch\{1\}.spm.stats.con.consess\{ii\}.tcon.weights = weights\{ii\};\
matlabbatch\{1\}.spm.stats.con.consess\{ii\}.tcon.sessrep = 'none';\
end\
matlabbatch\{1\}.spm.stats.con.delete = 1;\
%run the job\
spm_jobman('run', matlabbatch);\
\
end}