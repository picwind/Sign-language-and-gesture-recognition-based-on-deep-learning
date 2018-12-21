clear all;
close all;
h=424;
w=512;
CVAL    = 1;    % The defaul C value for SVR
WIN     = 20;   % The window size of the poooling layers
STRIDE  = 1;    % The stride of the pooling layer
% define the network architecture
net = defineNetwork(CVAL,WIN,STRIDE); 
nFrames=32;
Depths = zeros(h,w,nFrames);            
DMM = zeros(h,w);
p = genpath('D:\大创数据集\new32jpg');% 获得文件夹data下所有子文件的路径，这些路径存在字符串p中，以';'分割
length_p = size(p,2);%字符串p的长度
path = {};%建立一个单元数组，数组的每个单元中包含一个目录
temp = [];
for i = 1:length_p %寻找分割符';'，一旦找到，则将路径temp写入path数组中
    if p(i) ~= ';'
        temp = [temp p(i)];
    else 
        temp = [temp '\']; %在路径的最后加入 '\'
        path = [path ; temp];
        temp = [];
    end
end  
clear p length_p temp;
%至此获得data文件夹及其所有子文件夹（及子文件夹的子文件夹）的路径，存于数组path中。
%下面是逐一文件夹中读取图像
file_num = size(path,1);% 子文件夹的个数
for i = 1:file_num
    file_path =  path{i}; % 图像文件夹路径
    img_path_list = dir(strcat(file_path,'*.jpg'));
    img_num = length(img_path_list); %该文件夹中图像数量
    if img_num > 0
        for j = 1:img_num
            image_name = img_path_list(j).name;% 图像名
            image =  imread(strcat(file_path,image_name));
            fprintf('%d %d %s\n',i,j,strcat(file_path,image_name));% 显示正在处理的路径和图像名
            depthmap = rgb2gray(image);
            depthmap = im2double(depthmap);
            if j== 1
                [counts,binLocations]=imhist(depthmap);
                maxv = max(counts(128:end));
                myloc = find(counts == maxv);
                posi = binLocations(myloc);
                threshould = posi - 0.10;
            end
            for kk = 1:w*h
                if depthmap(kk)>threshould
                    depthmap(kk) = 0;
                end
            end
            Depths(:,:,j) = depthmap;
        end
        dx = zeros(h,w);
dy = zeros(h,w);
dt = zeros(h,w);
imagenormed_dx = zeros(nFrames,h*w);
imagenormed_dy = imagenormed_dx; imagenormed_dt = imagenormed_dx;
for f = 1:nFrames-1
    % smooth
    frame1 = medfilt2(Depths(:, :, f), [5, 5]);
    frame2 = medfilt2(Depths(:, :, f + 1), [5, 5]);
  
    % derivatives along x/y/t
    [dx(:, :), dy(:, :)] = gradient(frame1);
    dt(:, :) = frame2 - frame1;
    imagenormed_dx(f,:) =  reshape(dx(:, :),1,h*w);
    imagenormed_dy(f,:) =  reshape(dy(:, :),1,h*w);
    imagenormed_dt(f,:) =  reshape(dt(:, :),1,h*w);
end
                
% get the encoding of the sequenc
[im_WFF_r,im_WFR_r,im_WRF_r,im_WRR_r] = passNetwork(imagenormed_dx,h,w,net);
[im_WFF_g,im_WFR_g,im_WRF_g,im_WRR_g] = passNetwork(imagenormed_dy,h,w,net);
[im_WFF_b,im_WFR_b,im_WRF_b,im_WRR_b] = passNetwork(imagenormed_dt,h,w,net);
% ff  = getBackRGBImage(im_WFF_r,im_WFF_g,im_WFF_b);
fr  = getBackRGBImage(im_WFR_r,im_WFR_g,im_WFR_b);
rf  = getBackRGBImage(im_WRF_r,im_WRF_g,im_WRF_b);
% rr  = getBackRGBImage(im_WRR_r,im_WRR_g,im_WRR_b);
    filename1=['D:\workspace\matlab\FR_RF\FR\',num2str(i),'.jpg'];
    filename2=['D:\workspace\matlab\FR_RF\RF\',num2str(i),'.jpg'];
    imwrite(fr,filename1,'jpg');
    imwrite(rf,filename2,'jpg');
    end
end
