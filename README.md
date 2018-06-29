# Pixel-wise Attentional Gating for Parsimonious Pixel Labeling

For [paper](https://arxiv.org/abs/1805.01556) and [slides](https://www.ics.uci.edu/~skong2/slides/20180514_AIML_UCI.pdf), please refer to our [project page](http://www.ics.uci.edu/~skong2/PAG.html "pixel-attentional-gating")

Our entry to Robust Vision Challenge can be found [here](http://www.robustvision.net/leaderboard.php?benchmark=depth).


![alt text](http://www.ics.uci.edu/~skong2/image/PAG_splashFigure.png "visualization")


To achieve parsimonious inference in per-pixel labeling tasks with a limited
computational budget, we propose a **Pixel-wise Attentional Gating** unit
(**PAG**) that learns to selectively process a subset of spatial
locations at each layer of a deep convolutional network.  PAG is a generic,
architecture-independent, problem-agnostic mechanism that can be readily
``plugged in'' to an existing model with fine-tuning.  We utilize PAG in two
ways: 1) learning spatially varying pooling fields that improve model
performance without the extra computation cost associated with multi-scale
pooling, and 2) learning a dynamic computation policy for each pixel to
decrease total computation while maintaining accuracy.


We extensively evaluate PAG on a variety of per-pixel labeling tasks, including
semantic segmentation, boundary detection, monocular depth and surface normal
estimation.  We demonstrate that PAG allows competitive or state-of-the-art
performance on these tasks.  Our experiments show that PAG learns dynamic
spatial allocation of computation over the input image which provides better
performance trade-offs compared to related approaches (e.g., truncating deep
models or dynamically skipping whole layers).  Generally, we observe PAG can
reduce computation by 10% without noticeable loss in accuracy and
performance degrades gracefully when imposing stronger computational constraints.

**Keywords** Spatial Attention, Dynamic Computation, Per-Pixel Labeling, Semantic
Segmentation, Monocular Depth, Surface Normal, Boundary Detection.


Several demos are included as below.
As for details on the training, demo and code, please go into each demo folder.

1. [demo1: Panoramic Surface Normal Estimation](https://github.com/aimerykong/Pixel-Attentional-Gating/tree/master/demo1_Panoramic_Surface_Normal)  [**Ready**]

2. demo2: Boundary Detection [[!!!TOOD!!!]]

3. demo3: Semantic Segmentation  [[!!!TOOD!!!]]

4. demo4: Monocular Depth Estimation  [[!!!TOOD!!!]]



Please download those models from the [google drive](https://drive.google.com/open?id=1dIjVTL5Q4s4Lviol7kzwkCda1p04GoK3).





MatConvNet is used in our project, and some functions are changed/added. Please compile accordingly by adjusting the path --

```python
LD_LIBRARY_PATH=/usr/local/cuda/lib64:local matlab 

path_to_matconvnet = './matconvnet-1.0-beta23_modifiedDagnn/';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(fullfile(path_to_matconvnet, 'matlab'));
vl_compilenn('enableGpu', true, ...
               'cudaRoot', '/usr/local/cuda', ...
               'cudaMethod', 'nvcc', ...
               'enableCudnn', true, ...
               'cudnnRoot', '/usr/local/cuda/cudnn/lib64') ;

```

See also [Recurrent Scene Parsing with Perspective Understanding in-the Loop](https://github.com/aimerykong/Recurrent-Scene-Parsing-with-Perspective-Understanding-in-the-loop) which adapts depth map for pooling field selection.

If you find our model/method/dataset useful, please cite our work ([draft at arxiv](https://arxiv.org/abs/1805.01556)):

    @inproceedings{kong2018pag,
      title={Pixel-wise Attentional Gating for Parsimonious Pixel Labeling},
      author={Kong, Shu and Fowlkes, Charless},
      booktitle={arxiv 1805.01556},
      year={2018}
    }



last update: 06/27/2018

Shu Kong

aimerykong At g-m-a-i-l dot com

pag4ppl@gmail.com



