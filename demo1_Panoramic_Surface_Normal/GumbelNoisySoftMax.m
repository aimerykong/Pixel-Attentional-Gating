classdef GumbelNoisySoftMax < dagnn.ElementWise
    properties
        t=0.1
    end
    
    properties (Transient)
        noiseMat
        noisyActivation
        cleanActiviation
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            if strcmpi(obj.net.mode, 'train') || strcmpi(obj.net.mode, 'normal') 
                obj.noiseMat = rand(size(inputs{1}));
                obj.noiseMat = (obj.noiseMat + 0.00001) / 1.001;
                obj.noiseMat = -log(-log(obj.noiseMat));
                
                %obj.noiseMat = ones(size(inputs{1}));
                
                obj.cleanActiviation = vl_nnsoftmax(inputs{1}) ;
                obj.cleanActiviation = (obj.cleanActiviation+0.001) / 1.001;
                obj.noisyActivation = (log(obj.cleanActiviation)+obj.noiseMat) / obj.t;
                outputs{1} = vl_nnsoftmax(obj.noisyActivation);
            elseif strcmpi(obj.net.mode, 'val') || strcmpi(obj.net.mode, 'test') 
%                 obj.noiseMat = ones(size(inputs{1}));
                obj.cleanActiviation = vl_nnsoftmax(inputs{1}) ;
%                 obj.cleanActiviation = (obj.cleanActiviation+0.001) / 1.001;
%                 obj.noisyActivation = (log(obj.cleanActiviation)+obj.noiseMat) / obj.t;
%                 outputs{1} = vl_nnsoftmax(obj.noisyActivation);
                
                [~, A] = max(obj.cleanActiviation, [], 3);
                B = gpuArray(zeros(size(obj.cleanActiviation), 'single'));
                for i = 1:size(B, 3)
                    B(:,:,i) = (A==i);
                end
                outputs{1} = B;
            else                
                error('Invalid running mode!');
            end
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            der_noisyActivation = vl_nnsoftmax(obj.noisyActivation, derOutputs{1});
            der_cleanActivation = (0.999/obj.t) .* der_noisyActivation ./ obj.cleanActiviation;
            derInputs{1} = vl_nnsoftmax(inputs{1}, der_cleanActivation);
            
            %derInputs{1} = vl_nnsoftmax(inputs{1}, derOutputs{1}) ;
            derParams = {} ;
        end
        
        function obj = GumbelNoisySoftMax(varargin)
            obj.load(varargin) ;
        end
    end
end
