# Pixel-wise Attentional Gating for Parsimonious Pixel Labeling

For paper, slides and poster, please refer to our [project page](http://www.ics.uci.edu/~skong2/PAG.html "pixel-attentional-gating")


![alt text](http://www.ics.uci.edu/~skong2/image/PAG_splashFigure.png "visualization")


TBA

**keywords**: Spatial Attention, Dynamic Computation, Per-Pixel Labeling, Semantic Segmentation, Monocular Depth, Surface Normal, Boundary Detection.



MatConvNet is used in our project, and some functions are changed/added. Please compile accordingly by adjusting the path --

```python
LD_LIBRARY_PATH=/usr/local/cuda/lib64:local matlab 

path_to_matconvnet = './libs/matconvnet-1.0-beta23_modifiedDagnn/';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(fullfile(path_to_matconvnet, 'matlab'));
vl_compilenn('enableGpu', true, ...
               'cudaRoot', '/usr/local/cuda', ...
               'cudaMethod', 'nvcc', ...
               'enableCudnn', true, ...
               'cudnnRoot', '/usr/local/cuda/cudnn/lib64') ;

```



last update: 03/13/2018

Shu Kong

aimerykong At g-m-a-i-l dot com

