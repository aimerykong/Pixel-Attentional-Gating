%% add path and setup configuration
% clc; clear; close all;

addpath(genpath('../libs'));
path_to_matconvnet = '../matconvnet-1.0-beta23_modifiedDagnn';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));

mean_r = 123.68; %means to be subtracted and the given values are used in our training stage
mean_g = 116.779;
mean_b = 103.939;
mean_bgr = reshape([mean_b, mean_g, mean_r], [1,1,3]);
mean_rgb = reshape([mean_r, mean_g, mean_b], [1,1,3]);
%% read matconvnet model
load('imdb.mat');
flagSaveFig = false; % {true false} whether to store the resul
%% read data
for imgIdx = 1:length(imdb.train_image)
    %% read the image and annotation
    cur_path_to_image = imdb.train_image{imgIdx};
    cur_path_to_annot = imdb.train_annot{imgIdx};
    cur_path_to_depth = imdb.train_depth{imgIdx};
    cur_path_to_normal = imdb.train_normal{imgIdx};
    
    fprintf('%d/%d %s\n', imgIdx, length(imdb.train_image), cur_path_to_image);
    %%
    new_path_to_normal = strrep(cur_path_to_normal,'/normal/','/localNormal/');
    [folderName, fileName, ext]= fileparts(new_path_to_normal);
    if ~isdir(folderName)
        mkdir(folderName);
    end
    cur_image = imread(cur_path_to_image);
    %cur_annot = imread(cur_path_to_annot);
    %cur_depth = imread(cur_path_to_depth);
    cur_normal = imread(cur_path_to_normal);
    
    cur_image = single(cur_image);
    %cur_annot = single(cur_annot);
    %cur_depthLog10 = log10(single(cur_depth));
    cur_normal = single(cur_normal);
    cur_normal = (single(cur_normal)-127.5)/127.5;
    a = sqrt(sum(cur_normal.^2,3));
    normalMask = (a<1.1 & a>0.8);
    cur_normal = bsxfun(@times, cur_normal, normalMask);
    %% analysis
    nclick = 1;
    figure(1);
    normalMapShow = cur_normal/2+0.5;
    [H,W,C] = size(normalMapShow);
    
    figure(1);
    imshow(normalMapShow); title(sprintf('gtNormal')); axis off image;
    
    [x,y] = ginput(nclick);
    x = round(x);
    y = round(y);
    fprintf('y=%d,x=%d\n',y,x)
        
    normalMapShow(:,x,1) = 1;
    normalMapShow(:,x,2) = 0;
    normalMapShow(:,x,3) = 0;
    imagesc(normalMapShow); title(sprintf('gtNormal')); axis off image;
    
    XYmatrix = cat(3, normalMapShow(:,:,1), normalMapShow(:,:,3));
    XYmatrix = reshape(XYmatrix, [H*W, 2]);
    
    landmarkNormal = squeeze(cur_normal(y, x, :));
    landmarkNormalXY = [landmarkNormal(1); landmarkNormal(3)];
    landmarkNormalXY = landmarkNormalXY./norm(landmarkNormalXY);
    
    canonical_theta = acos(landmarkNormalXY'*[0;1]);
    rotR = [cos(canonical_theta), -sin(canonical_theta);
        sin(canonical_theta), cos(canonical_theta)];
    
    if norm( rotR*landmarkNormalXY - [0;1]) > 0.01
        canonical_theta = -1*canonical_theta;
    end
    
    new_normal = cur_normal;
    for curx = 1:W
        theta = canonical_theta + (curx-x)*2*pi/W;
        rotR = [cos(theta), -sin(theta);
            sin(theta), cos(theta)]; % counterclockwise
        cur_xList = cur_normal(:,curx,1);
        cur_yList = cur_normal(:,curx,3);
        cur_xy = [cur_xList(:)';cur_yList(:)'];
        new_xy = rotR*cur_xy;
        
        new_normal(:,curx,1) = new_xy(1,:);
        new_normal(:,curx,3) = new_xy(2,:);
    end
    
%     for curx = 1:x-1
%         theta = canonical_theta + (curx-x)*2*pi/W;
%         rotR = [cos(theta), -sin(theta);
%             sin(theta), cos(theta)]; % counterclockwise
%         cur_xList = cur_normal(:,curx,1);
%         cur_yList = cur_normal(:,curx,3);
%         cur_xy = [cur_xList(:)';cur_yList(:)'];
%         new_xy = rotR*cur_xy;
%         
%         new_normal(:,curx,1) = new_xy(1,:);
%         new_normal(:,curx,3) = new_xy(2,:);
%     end
    
    figure(1);
    new_normal = bsxfun(@times, new_normal, normalMask);
    imshow(new_normal/2+0.5); title(sprintf('gtNormal')); axis off image;
    % cur_normal = (single(cur_normal)-127.5)/127.5;
    new_normal = new_normal*127.5+127.5;
    %% save
%     imwrite(uint8(new_normal), new_path_to_normal);
end
%% leaving blank
a = sum(A.^2,3);
a(a<0.1) = 1;
A = bsxfun(@rdivide, A, a);

b = sum(B.^2,3);
b(a<0.1) = 1;
B = bsxfun(@rdivide, B, b);


T=sum(A.*B,3);

