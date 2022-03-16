# TASTER: Temporal Knowledge Graph Embedding via Sparse Transfer Matrix

## Introduction

This toolbox is exploited to support the work in "Domain Knowledge-enhanced Variable Selection for Biomedical Data Analysis", submitted to Information Sciences Journal. We provide a PyTorch implementation of the TASTER model for temporal knowledge graph embedding (TKGE). 

## **Implemented details**

- Evaluation Metrics: MRR, MR, HITS@1, HITS@3, HITS@10 (filtered)
- Loss Function: Uniform Negative Sampling
- Used Datasets: ICEWS14, ICEWS05-15, Wikidata12k

- Knowledge Graph Data:
  - *entities.dict*: a dictionary map entities to unique ids
  - *relations.dict*: a dictionary map relations to unique ids
  - *train.txt*: the KGE model is trained to fit this data set
  - *valid.txt*: create a blank file if no validation data is available
  - *test.txt*: the KGE model is evaluated on this data set

## Train & Test

Please create a python project in your local workstation, and import these files contained in the project. Run run_icews.py and run_wiki.py to train and test on the  ICEWS and Wikidata12k datasets respectively. The former is for data with timestamps and the latter is for data with time intervals.

For example, this command train a TASTER model on Wikidata12k dataset with GPU 0.

```sh
CUDA_VISIBLE_DEVICES=0 python run_wiki.py \
-dropout 0.4  -lr=0.005 -dim 200 -model TASTER -dig -epochs 200 --local \
-reg1 1e-6 -bs 2 --block -eval_step 5 -step 5
```

Check argparse configuration at src/run_wiki.py (run_icews.py ) for more arguments and more details.