# Capstone Project

## Getting Started
### Software Requirement
- Python 3.7
- Numpy
- Gensim 4.1.2
- sklearn 0.21.3+

### Usage
Please put this file under the directory ```measuring-founding-strategy``` 

Company html files should be stored in 
```
../out2/company_name/timestamp/*.html
```
Then run the file for the complete pipeline
```
python helper.py
```
The final results will be stored in folder combined_pivot_si

## Pipeline
The Pipeline is made of several parts.

1. Reading all the html files from target folder and extract timestamps and texts
2. Train Doc2Vec model using the extracted info
3. Compute similarity scores and write results to csv files
4. Use mAP to evaluate how the model works on document retrieval task.
5. Find the best number of clusters using Silhouette score
6. Train cluters using Dbscan instead of k-means
7. Determine threshold using the mapping function and user sensitivity

