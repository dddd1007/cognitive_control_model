%function model_selection(filename)

results_folder = "/Users/dddd1007/project2git/cognitive_control_model/data/output/summary/"
filename = "2nd_model_selection_aic.csv"

%AIC
data = csvread(append(results_folder,filename),1,1);
data = -1 * data;
[alpha,exp_r,xp,pxp,bor] = spm_BMS(data);

%wirte results
fid = fopen(append(results_folder,filename),'r');
header = textscan(fid,'%s', 1, 'Delimiter','\n', 'headerlines', 0);
header{1}{1}(1:7) = [];
fclose(fid);

fid = fopen(append(results_folder,'model_selection_',filename), 'w');
fprintf(fid, '%s', header{1}{1});
fprintf(fid, '\n%s,','alpha');fprintf(fid, '%f,', alpha);
fprintf(fid, '\n%s,','exp_r');fprintf(fid, '%f,', exp_r);
fprintf(fid, '\n%s,','xp');fprintf(fid, '%f,', xp);
fprintf(fid, '\n%s,','pxp');fprintf(fid, '%f,', pxp);
fclose(fid);
