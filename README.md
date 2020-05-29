
# Deep RL based Image Captioning with Embedding Reward
## Introduction
Image captioning is the process of generating syntactically and semantically correct sentence of an image. It is one of the most difficult problems in computer vision due to its complexity in understanding the visual content of the image and depicting it in a natural language sentence. Recent advances in deep learning based technologies  helped to handle the difficulties present in the image captioning process. Most of the state-of-the-art approaches follow an encoder decoder mechanism which sequentially predicts the words of a sentence  but  we use a decision making  framework in the image captioning process which uses policy and value network to collaboratively form a sentence. In this code we discuss the approach of-" Deep Reinforcement Learning based Image Captioning with Embedding Reward"
## Dataset
We use MSCOCO for image captioning. Wedirectly imported files from  "http://cs231n.stanford.edu/coco_captioning.zip"
When we download and unzip from link we get following files
'''
- coco2014_captions.h5
- train2014_images.txt
- train2014_vgg16_fc7.h5
- val2014_images.txt
- val2014_vgg16_fc7.h5 
- coco2014_vocab.json
- train2014_urls.txt
- train2014_vgg16_fc7_pca.h5
- val2014_urls.txt
- val2014_vgg16_fc7_pca.h5
''''
 coco2014_vocab.json contains word mapping to index. In .h5 files there will be a numpy vector which are represents captions as a vector.
## COCO API
we utilize the functions  from cocoAPI 
- CODE FOR IMPORTING
- !git clone https://github.com/waleedka/coco
- !pip install -U setuptools
- !pip install -U wheel
- !make install -C coco/PythonAPI
- !pip install git+https://github.com/salaniz/pycocoevalcap
## Dependencies
- numpy==1.16.2 
- torch==1.2.0
- nltk==3.4
- pycocoevalcap==1.0
- h5py==2.9.0
- matplotlib==3.0.3
- future==0.17.1
- imageio==2.6.1
- torchsummary==1.5.1
## Model architecture
                  CNN      RNN
- Policy Network  VGG-16   LSTM
- Value Network   VGG-16   LSTM
- Reward Network  VGG-16   GRU
## for coco evaluations we import from following code.
https://github.com/kelvinxu/arctic-captions/blob/master/metrics.py After downloading from link we paste into metrics.py for which we directly import for evaluation of captions.
## Starting
First we should run PolicyNetwork.py ,ValueNetwork.py, RewardNetwork.py before training them as they initialize networks.
## Results

![](https://github.com/bpavankalyan/ImageCaptioningreinforce/blob/master/Screenshot%20from%202020-05-27%2018-53-52.png?raw=true
)

![](https://github.com/bpavankalyan/ImageCaptioningreinforce/blob/master/Screenshot%20from%202020-05-27%2018-54-03.png?raw=true
)
![](https://github.com/bpavankalyan/ImageCaptioningreinforce/blob/master/Screenshot%20from%202020-05-28%2013-02-37.png?raw=true
)
![](https://github.com/bpavankalyan/ImageCaptioningreinforce/blob/master/Screenshot%20from%202020-05-28%2013-02-57.png?raw=true
)

## Reference
@misc{ren2017deep,
    title={Deep Reinforcement Learning-based Image Captioning with Embedding Reward},
    author={Zhou Ren and Xiaoyu Wang and Ning Zhang and Xutao Lv and Li-Jia Li},
    year={2017},
    eprint={1704.03899},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
