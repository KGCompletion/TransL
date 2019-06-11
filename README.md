# TransL

The source code of paper "Translating Embedding with Local Connection for Knowledge Graph Completion".

## Data
We provide FB13 and WN11 datasets used for the task of triplet classification.<br>
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
-data: dataset of the model, options: WN11, FB13  

## Validation
Usage: 
```python
python code/valid.py
```
You can change the hyper-parameters.    
-dim: entity and relation sharing embedding dimension  
-margin_pos: margin of positive triplets  
-margin_neg: margin of negative triplets  
-rate: learning rate  
-batch: batch size  
-epoch: number of training epoch  
-method: stratege of constructing negative triplets, options: unif, bern  
-data: dataset of the model, options: WN11, FB13  
-start: begining of the threshold  
-end: end of the threshold  

## Testing
Usage: 
```python
python code/test.py
```
You can change the hyper-parameters.    
-dim: entity and relation sharing embedding dimension  
-margin_pos: margin of positive triplets  
-margin_neg: margin of negative triplets  
-rate: learning rate  
-batch: batch size  
-epoch: number of training epoch  
-method: stratege of constructing negative triplets, options: unif, bern  
-data: dataset of the model, options: WN11, FB13  

It will evaluate on test.txt and report the accucacy of triplet classification.
