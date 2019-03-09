clc;
clear all;

byte_order = 'be';

speech_path = 'noisyN203_SNR20_be/';
speech_path_list = dir(strcat(speech_path, '*.lsp'));
file_num = length(speech_path_list);

for i = 1:file_num
    filename = speech_path_list(i).name;
    fid = fopen(strcat(speech_path, filename), 'r',sprintf('ieee-%s',byte_order))
    nframes = fread(fid,1,'int32');
    sampPeriod = fread(fid,1,'int32');
    sampSize = fread(fid,1,'int16');
    paramKind = fread(fid,1,'int16');
    htkdata = fread(fid,(sampSize/4)*  nframes,'float32');
    htkdata = reshape(htkdata,sampSize/4, nframes);
    save(strcat('noisy_speech/',int2str(i),'.mat'), 'htkdata');
    fclose(fid);
end

speech_path = 'CleanN203_SNR20_be/';
speech_path_list = dir(strcat(speech_path, '*.lsp'));
file_num = length(speech_path_list);

for i = 1:file_num
    filename = speech_path_list(i).name;
    fid = fopen(strcat(speech_path, filename), 'r',sprintf('ieee-%s',byte_order))
    nframes = fread(fid,1,'int32');
    sampPeriod = fread(fid,1,'int32');
    sampSize = fread(fid,1,'int16');
    paramKind = fread(fid,1,'int16');
    htkdata = fread(fid,(sampSize/4)*  nframes,'float32');
    htkdata = reshape(htkdata,sampSize/4, nframes);
    save(strcat('clean_speech/',int2str(i),'.mat'), 'htkdata');
    fclose(fid);
end
    
