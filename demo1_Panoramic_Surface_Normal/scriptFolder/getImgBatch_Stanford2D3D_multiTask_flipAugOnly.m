function [imBatch, segBatch, depthBatch, normalBatch, maskBatch] = getImgBatch_Stanford2D3D_multiTask_flipAugOnly(images, mode, varargin)
% CNN_IMAGENET_GET_BATCH  Load, preprocess, and pack images for CNN evaluation
opts.imageSize = [1024, 2048] ;
opts.border = [32, 32] ;
opts.cropSize = [1, 1] ;
opts.stepSize = [32, 32] ;
opts.lambda = 1 ;
opts.keepAspect = true ;
opts.numAugments = 0 ; % flip?
opts.transformation = 'none' ;  % 'stretch' 'none'
opts.averageImage = reshape( [123.6800 116.7800 103.9400], [1,1,3]) ;
% opts.rgbVariance = 1*ones(1,1,'single') ; % default: zeros(0,3,'single') ;
opts.rgbVariance = zeros(0,3,'single') ;
opts.classNum = 11;

opts.interpolation = 'bilinear' ;
opts.numThreads = 1 ;
opts.prefetch = false ;
opts.imdb = [];
opts.trainAugmentation = false;

opts = vl_argparse(opts, varargin);
imdb = opts.imdb;

if opts.trainAugmentation 
    rotation_prob = 0.5;
    scaling_prob = 0.5;
else
    rotation_prob = 0.;
    scaling_prob = 0.;
end

gammaRange = [-0.03, 0.03];
%% read mat file for image and label
imBatch = zeros(opts.imageSize(1)-opts.border(1), opts.imageSize(2)-opts.border(2), 3, numel(images), 'single') ;
segBatch = zeros((opts.imageSize(1)-opts.border(1)), (opts.imageSize(2)-opts.border(2)), 1, numel(images), 'single') ;
depthBatch = ones((opts.imageSize(1)-opts.border(1)), (opts.imageSize(2)-opts.border(2)), 1, numel(images), 'single') ;
normalBatch = ones((opts.imageSize(1)-opts.border(1)), (opts.imageSize(2)-opts.border(2)), 3, numel(images), 'single') ;
maskBatch = zeros((opts.imageSize(1)-opts.border(1)), (opts.imageSize(2)-opts.border(2)), 1, numel(images), 'single') ;

for img_i = 1:numel(images)
    %% read the image and annotation
    if  strcmpi(mode, 'val')
        path4image = imdb.train_image{images(img_i)};
        path4seg = imdb.train_annot{images(img_i)};
        path4normal = imdb.train_normal{images(img_i)};
        path4depth = imdb.train_depth{images(img_i)};        
        flag_flip = 0;
    else        
        path4image = imdb.train_image{images(img_i)};
        path4seg = imdb.train_annot{images(img_i)};
        path4normal = imdb.train_normal{images(img_i)};
        path4depth = imdb.train_depth{images(img_i)};                
        flag_flip = rand(1)>0.5;
    end
        
    cur_image = imread(path4image);    
    cur_seg = imread(path4seg);
    cur_normal = imread(path4normal);
    cur_depth = imread(path4depth);
        
    imageSize = size(cur_image);
    imageSize = imageSize(1:2);
    border = opts.border;
    %% augmentation
    if strcmpi(mode, 'train')
        xstart = randperm(opts.border(2) / opts.stepSize(2) + 1,1)*opts.stepSize(2) - opts.stepSize(2) + 1;
        ystart = randperm(opts.border(1) / opts.stepSize(1) + 1,1)*opts.stepSize(1) - opts.stepSize(1) + 1;
        xend = imageSize(2) - (border(2) - xstart+1);
        yend = imageSize(1) - (border(1) - ystart+1);
        %% crop augmentation
        cur_image = cur_image(ystart:yend, xstart:xend,:);
        cur_seg = cur_seg(ystart:yend, xstart:xend,:);
        cur_depth = cur_depth(ystart:yend, xstart:xend,:);  
        cur_normal = cur_normal(ystart:yend, xstart:xend,:);        
        %% flip augmentation 
        if flag_flip
            cur_image = fliplr(cur_image);
            cur_seg = fliplr(cur_seg);
            cur_depth = fliplr(cur_depth);
            cur_normal = fliplr(cur_normal); cur_normal(:,:,1)=-cur_normal(:,:,1);            
        end
        %% gamma augmentation
        if rand(1)>1 % 0.3
            cur_image = cur_image / 255;
            Z = gammaRange(1) + (gammaRange(2)-gammaRange(1)).*rand(1);
            gamma = log(0.5 + 1/sqrt(2)*Z) / log(0.5 - 1/sqrt(2)*Z);
            cur_image = cur_image.^gamma * 255;
        end
        %% RGB jittering
        %if rand(1)>1 % 0.3
        %    jitterRGB = rand(1,1,3)*0.4+0.8;
        %    cur_image = bsxfun(@times, cur_image, jitterRGB);
        %end        
        %% random rotation
        %{
        if rand(1)>1  % rand(1)>(1-rotation_prob) && min(imageSize)>200
            rangeDegree = -15:1:15;
            angle = randsample(rangeDegree, 1);
            if angle~=0
                W = size(cur_image,2);
                H = size(cur_image,1);
                Hst = ceil(W*abs(sin(angle/180*pi)));
                Wst = ceil(H*abs(sin(angle/180*pi)));
                
                cur_image = imrotate(cur_image, angle, 'bicubic');
                cur_seg = imrotate(cur_seg, angle, 'nearest');
                cur_inst = imrotate(cur_inst, angle, 'nearest');                
                cur_boundary = imrotate(cur_boundary, angle, 'nearest');
                cur_depth = imrotate(cur_depth, angle);
                cur_normal = imrotate(cur_normal, angle);
                cur_normalMask = imrotate(cur_normalMask, angle, 'nearest');
                                
                % take care of surface normal 
                R = [cos(angle/180*pi), -sin(angle/180*pi); sin(angle/180*pi), cos(angle/180*pi)]; % counterclockwise
                rot_surface_x = imrotate(cur_normal(:,:,1), angle, 'bicubic');
                rot_surface_y = imrotate(cur_normal(:,:,2), angle, 'bicubic');
                rot_surface_z = imrotate(cur_normal(:,:,3), angle, 'bicubic');
                rot_surface_x = rot_surface_x(max(1,Hst):end-Hst, max(1,Wst):end-Wst, :);
                rot_surface_y = rot_surface_y(max(1,Hst):end-Hst, max(1,Wst):end-Wst, :);
                rot_surface_z = rot_surface_z(max(1,Hst):end-Hst, max(1,Wst):end-Wst, :);
                
                A = [rot_surface_x(:)'; rot_surface_y(:)'];
                A = R*A;
                rot_surface_x = reshape(A(1,:), [size(rot_surface_x,1), size(rot_surface_x,2)]);
                rot_surface_y = reshape(A(2,:), [size(rot_surface_y,1), size(rot_surface_y,2)]);                
                cur_normal = cat(3, rot_surface_x, rot_surface_y, rot_surface_z);
                
                % get the valid area
                cur_image = cur_image(max(1,Hst):end-Hst, max(1,Wst):end-Wst, :);
                cur_seg = cur_seg(max(1,Hst):end-Hst, max(1,Wst):end-Wst, :);
                cur_inst = cur_inst(max(1,Hst):end-Hst, max(1,Wst):end-Wst, :);
                cur_boundary = cur_boundary(max(1,Hst):end-Hst, max(1,Wst):end-Wst, :);
                cur_depth = cur_depth(max(1,Hst):end-Hst, max(1,Wst):end-Wst, :);
                cur_normal = cur_normal(max(1,Hst):end-Hst, max(1,Wst):end-Wst, :);
                cur_normalMask = cur_normalMask(max(1,Hst):end-Hst, max(1,Wst):end-Wst, :);
    
                cur_image(cur_image<0) = 0;
                cur_image(cur_image>255) = 255;
            end
        end
        %% random scaling
        sz = size(cur_image); sz = sz(1:2);
        scaleFactorList = 1;
        if rand(1) > 1 % (1-scaling_prob)  && min(sz)>200
            scaleFactorList = 0.6:0.01:1.5;
            scaleFactorList = randsample(scaleFactorList, 1);
        end
        
        if max(scaleFactorList*sz)>800 % if too large to fit in memory
            tmp = 800 / max(scaleFactorList*sz);
            scaleFactorList = scaleFactorList*tmp;
            curRandScaleFactor = scaleFactorList*sz;
        else
            curRandScaleFactor = scaleFactorList*sz;
        end
        cur_depth = cur_depth ./ scaleFactorList;
        curRandScaleFactor = round(curRandScaleFactor/8)*8;
        if curRandScaleFactor~=0
            cur_image = imresize(cur_image, curRandScaleFactor);
            cur_seg = imresize(cur_seg, curRandScaleFactor, 'nearest');
            cur_inst = imresize(cur_inst, curRandScaleFactor, 'nearest');
            cur_boundary = imresize(cur_boundary, curRandScaleFactor, 'nearest');
            cur_depth = imresize(cur_depth, curRandScaleFactor);
            cur_normal = imresize(cur_normal, curRandScaleFactor);
            cur_normalMask = imresize(cur_normalMask, curRandScaleFactor, 'nearest');
        end
        %}
    elseif strcmpi(mode, 'val')        
        xstart = 1;
        ystart = 1;
        xend = imageSize(2)-opts.border(2);
        yend = imageSize(1)-opts.border(1);
        %% crop 
        cur_image = cur_image(ystart:yend, xstart:xend,:);
        cur_seg = cur_seg(ystart:yend, xstart:xend,:);
        cur_depth = cur_depth(ystart:yend, xstart:xend,:);
        cur_normal = cur_normal(ystart:yend, xstart:xend,:);
    end
    %% return
    cur_segMask = single(cur_seg)==0;
    
    cur_normal = (single(cur_normal)-127.5)/127.5;        
    a = sqrt(sum(cur_normal.^2,3));
    cur_normalMask = (a<1.1 & a>0.8);
    a(cur_normalMask==0) = 1;
    cur_normal = cur_normal ./ repmat(a, [1,1,3]);    
    cur_normal = bsxfun(@times, cur_normal, cur_normalMask);
    
    cur_depthMask = (cur_depth~=65535 & cur_depth>1);
    cur_depth = log10(single(cur_depth));    
    
    completeMask = cur_depthMask & cur_normalMask & cur_segMask;    
    
    if ~opts.trainAugmentation
        imBatch(:,:,:,img_i) = bsxfun(@minus, single(cur_image), opts.averageImage) ;
        segBatch(:,:,:,img_i) = single(cur_seg);        
        depthBatch(:,:,:,img_i) = bsxfun(@times, cur_depth, cur_depthMask);
        normalBatch(:,:,:,img_i) = bsxfun(@times, cur_normal, cur_normalMask);
        maskBatch(:,:,:,img_i) = completeMask;
    else
        imBatch = bsxfun(@minus, single(cur_image), opts.averageImage) ;
        segBatch = single(cur_seg);
        depthBatch = bsxfun(@times, cur_depth, cur_depthMask);
        normalBatch = bsxfun(@times, cur_normal, cur_normalMask);
        maskBatch = completeMask;
    end
end
% finishFlag = true;
%% leaving blank











