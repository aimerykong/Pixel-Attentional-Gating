# Pixel-wise Attentional Gating for Parsimonious Pixel Labeling

For paper, slides and poster, please refer to our [project page](http://www.ics.uci.edu/~skong2/PAG.html "pixel-attentional-gating")


![alt text](https://github.com/aimerykong/Pixel-Attentional-Gating/blob/master/demo1_Panoramic_Surface_Normal/model/demo1_splashFig.png?raw=true "visualization")

Tracking the following scripts provides a good way to understand how to run/train/test models.

1. 'main000_calibrateNormal_train.m' and 'main000_calibrateNormal_val.m' show how to calibrate the panoramic surface normal in an interactive manner. Please refer to appendix of our arxiv paper for details, with script 'main000_understanding.m' that shows why it works.

2. 'main001_demo_visualize.m' reads a demo model and pass through a few demo images for visualization. The model is able to allocate dynamic computation to all pixels of an image. Please download the model through the goolge link listed below.

3. 'main002_demo_train_Res5ScaleAttention_pG2345_p07.m' demonstrates how to train a model. If training, you need to prepare the 'imdb' structure that indicates the path of each image&annotation. 



Please download the demo model from the [google drive](https://drive.google.com/open?id=1VmwFLzxlstsoAaq0UrOsZZuLLVUJaCiN), and put it to path './model/main006normal_Res5ScaleAttention_pG2345_p07/softmax_net-epoch-140.mat'




MatConvNet is used in our project, and some functions are changed/added. Please compile accordingly by adjusting the path --

```python
LD_LIBRARY_PATH=/usr/local/cuda/lib64:local matlab 

path_to_matconvnet = '../matconvnet-1.0-beta23_modifiedDagnn/';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(fullfile(path_to_matconvnet, 'matlab'));
vl_compilenn('enableGpu', true, ...
               'cudaRoot', '/usr/local/cuda', ...
               'cudaMethod', 'nvcc', ...
               'enableCudnn', true, ...
               'cudnnRoot', '/usr/local/cuda/cudnn/lib64') ;

```


If you find our model/method/dataset useful, please cite our work ([draft at arxiv](https://arxiv.org/abs/XXXXXX)):

    @inproceedings{kong2018pag,
      title={Pixel-wise Attentional Gating for Parsimonious Pixel Labeling},
      author={Kong, Shu and Fowlkes, Charless},
      booktitle={arxiv},
      year={2018}
    }



last update: 06/28/2018

Shu Kong

aimerykong At g-m-a-i-l dot com

