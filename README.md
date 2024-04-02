# 3D Rotation and Translation for Hyperbolic Knowledge Graph Embedding

This paper has been accepted to appear at the conference [EACL2024](https://2024.eacl.org) and be published in the main proceedings [3H-TH](https://aclanthology.org/2024.eacl-long.90/) (Oral Presentation).

This is the PyTorch implementation of the [3H-TH](http://arxiv.org/abs/2305.13015) [6] model for knowledge graph embedding (KGE). 
This project is based on [AttH](https://github.com/HazyResearch/KGEmb) [5]. Thanks for their contributions.

## Citation

If you want to cite this paper or want to use this code, please cite the following paper:

```
@inproceedings{zhu-shimodaira-2024-3d,
    title = "3{D} Rotation and Translation for Hyperbolic Knowledge Graph Embedding",
    author = "Zhu, Yihua  and
      Shimodaira, Hidetoshi",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-long.90",
    pages = "1497--1515",
    abstract = "The main objective of Knowledge Graph (KG) embeddings is to learn low-dimensional representations of entities and relations, enabling the prediction of missing facts. A significant challenge in achieving better KG embeddings lies in capturing relation patterns, including symmetry, antisymmetry, inversion, commutative composition, non-commutative composition, hierarchy, and multiplicity. This study introduces a novel model called 3H-TH (3D Rotation and Translation in Hyperbolic space) that captures these relation patterns simultaneously. In contrast, previous attempts have not achieved satisfactory performance across all the mentioned properties at the same time. The experimental results demonstrate that the new model outperforms existing state-of-the-art models in terms of accuracy, hierarchy property, and other relation patterns in low-dimensional space, meanwhile performing similarly in high-dimensional space.",
}

```

or:


```
Yihua Zhu and Hidetoshi Shimodaira. 2024. 3D Rotation and Translation for Hyperbolic Knowledge Graph Embedding. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1497–1515, St. Julian’s, Malta. Association for Computational Linguistics.
```

## Models

* TransE (TE) [1]
* RotatE (2E) [2]
* QuatE  (3E) [3]
* MuRP   (TH) [4]
* RotH   (2H) [5]
* 2E-TE       [6]
* 3E-TE       [6]
* 3H          [6]
* **3H-TH**   [6]
* 2E-TE-2H-TH [6]
* 3E-TE-3H-TH [6]

## Initialization

1. environment (we need torch, numpy, tqdm):

```bash
conda create --name ThreeH_TH_env
source activate ThreeH_TH_env
pip install -r requirements.txt
```

2. set environment variables.

We use three files to do experiments for three datasets. Thus, we should set envirment variables for each files. For example, we should open 3H-TH_WN18RR file first then set environment. (we can open 3H-TH_FB237 (cd 3H-TH_FB237) and 3H-TH_FB15K (cd 3H-TH_FB15K)when we want to do experiments for datasets FB15K-237 and FB15K.)

```bash
cd 3H-TH_WN18RR


KGHOME=$(pwd)
export PYTHONPATH="$KGHOME:$PYTHONPATH"
export LOG_DIR="$KGHOME/logs"
export DATA_PATH="$KGHOME/data"
```
Then we can activate our environment:

```bash
source activate ThreeH_TH_env
```

## Data

I have uploaded all the data that we need to use in three files.
But we need to unzip the big dataset as following:
```bash
cd 3H-TH_FB15K/data/FB15K
unzip to_skip.pickle.zip

cd 3H-TH_FB237/data/FB237
unzip to_skip.pickle.zip
```

## usage

To train and evaluate a KG embedding model for the link prediction task, use the run.py script. And we can use the file "examples", "train_3E_TE.sh" and "train_3H_TH.sh" means the examples in Euclidean and hyperbolic space, respectively.

```bash
usage: run.py [-h] [--dataset {FB15K,WN,WN18RR,FB237,YAGO3-10}]
              [--model {TransE,RotatE,QuatE,RotH,ThreeE_TE,TwoE_TE,TH,ThreeH,ThreeH_TH,ThreeE_TE_ThreeH_TH, TwoE_TE_TwoH_TH}]
              [--regularizer {N3,N2}] [--reg REG]
              [--optimizer {Adagrad,Adam,SGD,SparseAdam,RSGD,RAdam}]
              [--max_epochs MAX_EPOCHS] [--patience PATIENCE] [--valid VALID]
              [--rank RANK] [--batch_size BATCH_SIZE]
              [--neg_sample_size NEG_SAMPLE_SIZE] [--dropout DROPOUT]
              [--init_size INIT_SIZE] [--learning_rate LEARNING_RATE]
              [--gamma GAMMA] [--bias {constant,learn,none}]
              [--dtype {single,double}] [--double_neg] [--debug] [--multi_c]

Knowledge Graph Embedding

optional arguments:
  -h, --help            show this help message and exit
  --dataset {FB15K,WN18RR,FB237}
                        Knowledge Graph dataset
  --model {TransE,RotatE,QuatE,RotH,ThreeE_TE,TwoE_TE,TH,ThreeH,ThreeH_TH,ThreeE_TE_ThreeH_TH, TwoE_TE_TwoH_TH}
                        Knowledge Graph embedding model
  --regularizer {N3,N2}
                        Regularizer
  --reg REG             Regularization weight
  --optimizer {Adagrad,Adam,SparseAdam}
                        Optimizer
  --max_epochs MAX_EPOCHS
                        Maximum number of epochs to train for
  --patience PATIENCE   Number of epochs before early stopping
  --valid VALID         Number of epochs before validation
  --rank RANK           Embedding dimension
  --batch_size BATCH_SIZE
                        Batch size
  --neg_sample_size NEG_SAMPLE_SIZE
                        Negative sample size, -1 to not use negative sampling
  --dropout DROPOUT     Dropout rate
  --init_size INIT_SIZE
                        Initial embeddings' scale
  --learning_rate LEARNING_RATE
                        Learning rate
  --gamma GAMMA         Margin for distance-based losses
  --bias {constant,learn,none}
                        Bias type (none for no bias)
  --dtype {single,double}
                        Machine precision
  --double_neg          Whether to negative sample both head and tail entities
  --debug               Only use 1000 examples for debugging
  --multi_c             Multiple curvatures per relation
```
For example:

```bash
python run.py \
            --dataset WN18RR \
            --model ThreeH_TH \
            --rank 32 \
            --regularizer N3 \
            --reg 0.0 \
            --optimizer Adam \
            --max_epochs 500 \
            --patience 15 \
            --valid 5 \
            --batch_size 500 \
            --neg_sample_size 100 \
            --init_size 0.001 \
            --learning_rate 0.001 \
            --gamma 0.0 \
            --bias learn \
            --dtype double \
            --double_neg \
            --multi_c 
```


## Reference

[1] Bordes, Antoine, et al. "Translating embeddings for modeling multi-relational data." Advances in neural information processing systems. 2013.

[2] Sun, Zhiqing, et al. "Rotate: Knowledge graph embedding by relational rotation in complex space." International Conference on Learning Representations. 2019.

[3] Zhang S, Tay Y, Yao L, et al. Quaternion knowledge graph embeddings. Advances in neural information processing systems, 2019, 32.

[4] Balazevic I, Allen C, Hospedales T. Multi-relational poincaré graph embeddings. Advances in Neural Information Processing Systems, 2019, 32.

[5] Chami I, Wolf A, Juan D C, et al. Low-dimensional hyperbolic knowledge graph embeddings. arXiv preprint arXiv:2005.00545, 2020.

[6] Zhu Y, Shimodaira H. 3D Rotation and Translation for Hyperbolic Knowledge Graph Embedding. arXiv preprint arXiv:2305.13015, 2023.
