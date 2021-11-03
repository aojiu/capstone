import sys
import os
sys.path.append(os.path.abspath('crawler'))
sys.path.append(os.path.abspath('download'))
sys.path.append(os.path.abspath('text_analysis'))
from text_analysis.website_text import website_text
from text_analysis.website_text_dataset import website_text_dataset
from text_analysis.similarity_estimator import similarity_estimator

import pandas as pd
import numpy as np
import glob
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string, preprocess_documents
from gensim.models.doc2vec import Doc2Vec , TaggedDocument
import nltk
from sklearn.cluster import KMeans

# get the training documents and associated timestamps
def get_time_and_webtext(company_folder):
    '''
    Arg: 
    company_folder: the folder named after the company name. There are subfolders with timestamp.

    Return:
    train_documents: documents to train doc2vec model
    timestamps: timestamps of documents we use to train the model
    ''' 
    train_documents = []
    timestamps = []
    for time_folder in sorted(glob.glob(company_folder+ '/*')):
        print(time_folder)
        text = ''
        time = time_folder.split("/")[-1]
        time_truncate=time[:(len(time)-1)]
        company_name = time_folder.split("/")[-2]
        for html_file in sorted(glob.glob(time_folder+ '/*.html')):
#             # prepare train documents and times 
            single_web = website_text(path = html_file, domain = company_name, year = time_truncate[:4])
            encoded_string = single_web.load_page(html_file)[:400000].encode("ascii", "ignore")
            decode_string = encoded_string.decode()
            text+=" "
            text+=decode_string
#         company_name = html_file.split("/")[-3]
        train_documents.append(text)
        timestamps.append(time_truncate)
    return train_documents, timestamps, company_name


def get_all_data(all_data_path):
    '''
    Arg: 
    all_data_path: the folder contains all company folders

    Return:
    all_train_documents: documents of all companies to train one doc2vec model
    all_timestamps: timestamps of documents of all companies used to train the doc2vec model
    index_dict: a dictionary of indices for companies. 
                    The first index is the start index of company in the all_train_documents. 
                    The second index is the end position.
    ''' 
    index_dict = {}
    curr_ind = 0
    all_train_documents = []
    all_timestamps=[]
    for folder in sorted(glob.glob(all_data_path+'/*')):
        
        train_documents, timestamps, company_name = get_time_and_webtext(folder)
        
        index_dict[company_name] = [curr_ind, curr_ind+len(timestamps)]
        curr_ind+=len(timestamps)
        all_train_documents+=train_documents
        all_timestamps+=timestamps
    return all_train_documents, all_timestamps, index_dict


def train_doc2vec_model(documents, save_path):    
    '''
    Arg: 
    documents: training documents for the company.
    save_path: where the model is saved
    
    Return:
    model: trained doc2vec model
    ''' 
    all_docs = []
    # preprocessing
    counter = 0
    for train_doc in documents:
        doc = train_doc[:150000] if len(train_doc) > 150000 else train_doc
        if (counter%100) == 0:
            print("{0} .. len: {1}".format(counter,len(doc)))
        counter += 1
        doc = remove_stopwords(doc)
#            doc = re.sub(r'[^\w\s]','',doc)
        doc_tokens =nltk.word_tokenize(doc.lower())
        all_docs.append(doc_tokens)
    # bfollow gensim instructions
    final_documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(all_docs)]
    # train the model
    # hyperparameter tuning
#     model = Doc2Vec(documents = final_documents,
#                         vector_size=700,
#                         window=7,
#                         min_count =3)
    
    model = Doc2Vec(documents = final_documents,
                        vector_size=256,
                        window=7, min_count = 3)
    
    
    model.save(save_path + ".word2vec.model")
    return model
    
def load_model(path):
    model = Word2Vec.load(path + ".word2vec.model")
    return model


# this function extract the embedding from the doc2vec
# the timestamp of the documents is specified in the input
def get_timestamp_embeddings(target_times, training_times, model):
    '''
    Arg: 
    target_times: list of timestamps we are interested in evaluating
    training_times: list of timestamps of documents we use to train the model
    model: the trained doc2vec model we use to extract the embeddings

    Return:
    embedding_lst: a numpy array of embeddings we are interested in
    '''
    # get the embedding size of the model
    vector_size = model.docvecs.get_normed_vectors()[0].shape[0]
    # to store all the embeddings we want
    embedding_lst = np.zeros((len(target_times), vector_size))
    # get the index of embedding for each timestamp we are interested in
    for i in range(len(target_times)):
        try:
            index = training_times.index(target_times[i])
            embedding = model.docvecs.get_normed_vectors()[index]
            # store this target embedding in the numpy array
            embedding_lst[i, :] = embedding
        except: 
            print("passed")
            print(target_times[i])
            pass
    return embedding_lst

# doc embedding for each training corpus the input is embedding and times
def get_sim_scores_embedding(embeddings, times):
    '''
    Arg: 
    model: trained doc2vec model
    times: timestamps of documents we want

    Return:
    df: dataframe that has timestamp and similarity of company texts
    '''
    sim_dict = {}
    sim_scores = []
    titles = []
    for i in range(len(times)):
        if i ==0: pass
        else:
            similarity = np.dot(embeddings[i,:], embeddings[i-1,:])
            title = times[i-1] + " vs " + times[i]
            sim_scores.append(similarity)
            titles.append(title)
    sim_dict["similarity"] = sim_scores
    sim_dict["timestamp"] = titles
    df = pd.DataFrame(sim_dict)
    return df


def sim_scores_all_companies(model, all_timestamps, company_index_dict):
    '''
    Arg: 
    model: trained doc2vec model
    all_timestamps: timestamps of documents we want
    company_index_dict: dictionary with company first index and last last index in the model

    Return:
    Save dataframe
    '''
    vector_size = model.docvecs.get_normed_vectors()[0].shape[0]
    for key, value in company_index_dict.items():
        company_timestamps = all_timestamps[value[0]:value[1]]
        
        # each company will have an embedding list
        embedding_lst = np.zeros((len(company_timestamps), vector_size))
        # get all embedding for this company
        for i in range(value[0], value[1]):
            # get the document embedding using the index
            embedding = model.docvecs.get_normed_vectors()[i]
            # store this target embedding in the numpy array
            embedding_lst[i-value[0], :] = embedding
        df = get_sim_scores_embedding(embedding_lst, company_timestamps)
        file_name = key+".csv"
        df.to_csv(file_name)


# doc embedding for each training corpus
def get_sim_scores(model, times):
    '''
    Arg: 
    model: trained doc2vec model
    times: timestamps of documents we want

    Return:
    df: dataframe that has timestamp and similarity of company texts
    '''
    sim_dict = {}
    sim_scores = []
    titles = []
    for i in range(len(times)):
        if i ==0: pass
        else:
            similarity = np.dot(model.docvecs.get_normed_vectors()[i], model.docvecs.get_normed_vectors()[i-1])
            title = times[i-1] + " vs " + times[i]
            sim_scores.append(similarity)
            titles.append(title)
    sim_dict["similarity"] = sim_scores
    sim_dict["timestamp"] = titles
    df = pd.DataFrame(sim_dict)
    return df

def clustering(embeddings, n):
    '''
    Arg: 
    embeddings: nunpy array of embeddings we use to do clusterings
    n: number of clusters

    Return:
    clustering_labels: labels for which ckuster does the timestamp belong to for different n
    '''
    clustering_labels = []
    for i in range(3, n+1):
        kmeans = KMeans(n_clusters=i, random_state=0).fit(embeddings)
#         labels =kmeans.labels_
        clustering_labels.append(kmeans.labels_)
    return clustering_labels

def expand_df(times, clustering_labels):
    '''
    Arg: 
    df: dataframe contains sim score and timestamp
    clustering_labels: clustering labels for different n

    Return:
    df: expanded df with clustering info
    '''
    df = pd.DataFrame({"timestamp":times})
    for i in range(len(clustering_labels)):
        num_cluster = i+3
        col_name = "cluster n="+str(num_cluster)
        df[col_name] = clustering_labels[i]
        
    return df


if __name__ == '__main__':
    # get all the info we need for the model and further analysis
    all_documents, all_time, company_index_dict = get_all_data('../out2')
    # train model 
    model_airbnb = train_doc2vec_model(all_documents, "saved_model_all_companies")
    # save all similarity score to csv files
    # sim_scores_all_companies(model_airbnb, all_time, company_index_dict)