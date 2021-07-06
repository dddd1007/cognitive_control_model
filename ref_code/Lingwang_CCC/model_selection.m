csv_file_path = "/Users/dddd1007/Downloads/model_selection_miniblock.csv"

%AIC
data = csvread(append(csv_file_path),1,1);
data = -1 * data;
[alpha,exp_r,xp,pxp,bor] = spm_BMS(data);

%wirte results
fid = fopen(append(csv_file_path),'r');
header = textscan(fid,'%s', 1, 'Delimiter','\n', 'headerlines', 0);
header{1}{1}(1:7) = [];
fclose(fid);

fid = fopen(append("/Users/dddd1007/Downloads/model_selection_miniblock.csv"), 'w');
fprintf(fid, '%s', header{1}{1});
fprintf(fid, '\n%s,','alpha');fprintf(fid, '%f,', alpha);
fprintf(fid, '\n%s,','exp_r');fprintf(fid, '%f,', exp_r);
fprintf(fid, '\n%s,','xp');fprintf(fid, '%f,', xp);
fprintf(fid, '\n%s,','pxp');fprintf(fid, '%f,', pxp);
fclose(fid);
