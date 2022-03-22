function model_selection(filename)

if ispc
    addpath('X:\Toolbox\SPM12')
elseif isunix
    addpath('/Data/Toolbox/SPM12')
end

results_folder = './model_results/';

%AIC
data = csvread([results_folder filename],1,1);
data = -1 * data;
[alpha,exp_r,xp,pxp,bor] = spm_BMS(data);

%wirte results
fid = fopen([results_folder filename],'r');
header = textscan(fid,'%s', 1, 'Delimiter','\n', 'headerlines', 0);
header{1}{1}(1:7) = [];
fclose(fid);

fid = fopen([results_folder 'model_selection_' filename], 'w');
fprintf(fid, '%s', header{1}{1});
fprintf(fid, '\n%s,','alpha');fprintf(fid, '%f,', alpha);
fprintf(fid, '\n%s,','exp_r');fprintf(fid, '%f,', exp_r);
fprintf(fid, '\n%s,','xp');fprintf(fid, '%f,', xp);
fprintf(fid, '\n%s,','pxp');fprintf(fid, '%f,', pxp);
fclose(fid);
