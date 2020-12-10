%AIC
data = csvread(append("/Users/dddd1007/project2git/cognitive_control_model/data/output/RLModels/model_selection/model_selection_final.csv"),1,1);
data = -1 * data;
[alpha,exp_r,xp,pxp,bor] = spm_BMS(data);

%wirte results
fid = fopen(append("/Users/dddd1007/project2git/cognitive_control_model/data/output/RLModels/model_selection/model_selection_final.csv"),'r');
header = textscan(fid,'%s', 1, 'Delimiter','\n', 'headerlines', 0);
header{1}{1}(1:7) = [];
fclose(fid);

fid = fopen(append("/Users/dddd1007/project2git/cognitive_control_model/data/output/RLModels/model_selection/model_selection_final_result.csv"), 'w');
fprintf(fid, '%s', header{1}{1});
fprintf(fid, '\n%s,','alpha');fprintf(fid, '%f,', alpha);
fprintf(fid, '\n%s,','exp_r');fprintf(fid, '%f,', exp_r);
fprintf(fid, '\n%s,','xp');fprintf(fid, '%f,', xp);
fprintf(fid, '\n%s,','pxp');fprintf(fid, '%f,', pxp);
fclose(fid);
