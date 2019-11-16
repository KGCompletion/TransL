# TransL

The source code of paper "Translating Embedding with Local Connection for Knowledge Graph Completion".

Link prediction on FB15k-237:

||Raw MRR|Filter MRR|Hits@1|Hits@3|Hits@10|
|:---|:---|:---|:---|:---|:---|
|unif|0.227|0.342|0.244|0.379|0.535|
|bern|0.248|0.355|0.260|0.389|0.551|

Triplet classification on WN11 and FB13:

||WN11|FB13|
|:---|:---|:---|
|unif|0.861|0.838|
|bern|0.866|0.856|

## Data
We provide FB15k-237, FB13 and WN11 datasets used for the tasks of link prediction and triplet classification.<br>
Each dataset in the following format, containing five files:<br>
* entity2id.txt: all entities and corresponding ids, format (entity, id)  
* relation2id.txt: all relations and corresponding ids, format (relation, id)  
* test.txt: testing file, format (head_entity, relation, tail_entity, label)
* train.txt: training file, format (head_entity, relation, tail_entity)
* valid.txt: validation file, format (head_entity, relation, tail_entity, label)

## Training
Usage: 
```python
python code/train.py
```
You can change the hyper-parameters.    
-dim: entity and relation sharing embedding dimension  
-margin_pos: margin of positive triplets  
-margin_neg: margin of negative triplets  
-rate: learning rate  
-batch: batch size  
-epoch: number of training epoch  
-method: stratege of constructing negative triplets, options: unif, bern  
-data: dataset of the model, options: FB15k-237, WN11, FB13  


## Testing
Usage:
Link prediction:
```python
python code/test-lp.py
```
Triplet classification:
```python
python code/test-tc.py
```

You can change the hyper-parameters.    
-dim: entity and relation sharing embedding dimension  
-margin_pos: margin of positive triplets  
-margin_neg: margin of negative triplets  
-rate: learning rate  
-batch: batch size  
-epoch: number of training epoch  
-method: stratege of constructing negative triplets, options: unif, bern  
-data: dataset of the model, options: FB15k-237, WN11, FB13  

It will evaluate on test.txt and report the results.
