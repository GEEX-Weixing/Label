# Locally-Biased Graph Contrastive Learning Impels GNNs Against Label Scarcity
## Framework:
 ![Image text](https://github.com/GEEX-Weixing/Label/blob/main/data/LABEL_0.pdf)
[LABEL_0.pdf](https://github.com/user-attachments/files/19901207/LABEL_0.pdf)

## Run:
```
sh train.sh
```
### Dataset: Cora, CiteSeer, PubMed, Amazon Conputers, Coauthor CS, Ogbn-Arxiv
```
python run.py
```
### Dataset: Texas, Wisconsin, Amazon ratings, Chameleon, Squirrel
```
python run_h.py
```
### Mask-Label
```
python run_m.py
```
## Acknowledgements:
Thanks to the following projects, in no particular order

* [M3S](https://github.com/Junseok0207/M3S_Pytorch)
* [Graph-MLP](https://github.com/yanghu819/Graph-MLP)
