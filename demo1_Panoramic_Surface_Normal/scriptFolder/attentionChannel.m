classdef attentionChannel < dagnn.ElementWise
    properties
        numChannels
        numInputs
        SIZE_
        globalMean
        globalVariance 
    end
    methods
        function outputs = forward(obj, inputs, params)
            obj.numInputs = numel(inputs);
            X = inputs{1};
            gpuMode = isa(X, 'gpuArray');           
            % [height width channels batchsize]
            [h, w, ch, bs] = size(X);
            obj.SIZE_ = [h, w, ch, bs];             
                        
            bn_mult = params{1};
            bn_bias = params{2};            
            
            EPSILON = 0.00001;
            Y = bsxfun(@minus, inputs{1}, reshape(obj.globalMean, [1, 1, numel(obj.globalMean)]));            
            Y = bsxfun(@rdivide, Y, reshape(EPSILON+obj.globalVariance,[1, 1, numel(obj.globalVariance)]));
            
            Y = bsxfun(@times, Y, reshape(bn_mult,[1, 1, numel(bn_mult)]));           
            Y = bsxfun(@plus, Y, reshape(bn_bias,[1, 1, numel(bn_bias)]));            
                                                
            % official version of batch normalization
%             Y = vl_nnbnorm(inputs{1}, bn_mult,bn_bias, 'moments', bn_moments, 'epsilon', EPSILON) ;
%             figure(1); subplot(1,3,1);imagesc(mean(A,3)), colorbar, axis off image;
%             subplot(1,3,2);imagesc(mean(Y,3)), colorbar, axis off image;
%             subplot(1,3,3);imagesc(mean(A-Y,3)), colorbar, axis off image;
            
            outputs{1} = Y;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs = cell(1, numel(inputs));
            dzdy = derOutputs{1};
            EPSILON = 0.00001;
            %gpuMode = isa(inputs{1}, 'gpuArray');
            
            dzdx = bsxfun(@rdivide, dzdy, reshape(EPSILON+obj.globalVariance,[1, 1, numel(obj.globalVariance)]));
            dzdx = bsxfun(@times, dzdx, reshape(params{1},[1, 1, numel(params{1})]));
            
            Y = bsxfun(@minus, inputs{1}, reshape(obj.globalMean, [1, 1, numel(obj.globalMean)]));
            Y = bsxfun(@rdivide, Y, reshape(EPSILON+obj.globalVariance,[1, 1, numel(obj.globalVariance)]));
            Y = Y .* dzdy;
            dzd_multiplier = sum(sum(sum(Y,1),2),4);            
            dzd_bias = sum(sum(sum(dzdy,1),2),4);
            
            
            derInputs{1} = dzdx;
            derParams{1} = dzd_multiplier(:);
            derParams{2} = dzd_bias(:);    
        end
        
        function params = initParams(obj)
            params{1} = ones(obj.numChannels,1,'single') ;
            params{2} = zeros(obj.numChannels,1,'single') ;
        end
        
        
        function obj = attentionChannel(varargin)
            obj.load(varargin) ;
        end
    end
end
