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

## Validation
Usage: 
```python
python code/valid.py
```
## Testing
Usage: 
```python
python code/test.py
```
It will evaluate on test.txt and report the accucacy of triplet classification.
