# Capstone Project

## Getting Started
### Software Requirement
- Python 3.7
- Numpy
- Gensim 4.1.2
- sklearn 0.21.3+

### Usage
Please put this file under the directory ```measuring-founding-strategy``` 

Then run the file for the complete pipeline
```
python helper.py
```
The final results will be stored in folder boolean_clustering_optK

## Pipeline
The Pipeline is made of several parts.

1. Reading all the html files from target folder and extract timestamps and texts
2. Train Doc2Vec model using the extracted info
3. Compute similarity scores and write results to csv files

