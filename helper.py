import sys
import os

import nltk
nltk.download('punkt')

import math
from scipy.interpolate import interp1d

sys.path.append(os.path.abspath('crawler'))
sys.path.append(os.path.abspath('download'))
sys.path.append(os.path.abspath('text_analysis'))
from text_analysis.website_text import website_text
from text_analysis.website_text_dataset import website_text_dataset
from text_analysis.similarity_estimator import similarity_estimator
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN

import pandas as pd
import numpy as np
import glob
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string, preprocess_documents
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from sklearn.cluster import KMeans
import os


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
    for time_folder in sorted(glob.glob(company_folder + '/*')):
        print(time_folder)
        text = ''
        time = time_folder.split("/")[-1]
        time_truncate = time[:(len(time))]
        company_name = time_folder.split("/")[-2]
        for html_file in sorted(glob.glob(time_folder + '/*.html')):
            #             # prepare train documents and times
            single_web = website_text(path=html_file, domain=company_name, year=time_truncate[:4])
            encoded_string = single_web.load_page(html_file)[:400000].encode("ascii", "ignore")
            decode_string = encoded_string.decode()
            text += " "
            text += decode_string
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
    all_timestamps = []
    for folder in sorted(glob.glob(all_data_path + '/*')):
        train_documents, timestamps, company_name = get_time_and_webtext(folder)

        index_dict[company_name] = [curr_ind, curr_ind + len(timestamps)]
        curr_ind += len(timestamps)
        all_train_documents += train_documents
        all_timestamps += timestamps
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
        if (counter % 100) == 0:
            print("{0} .. len: {1}".format(counter, len(doc)))
        counter += 1
        doc = remove_stopwords(doc)
        #            doc = re.sub(r'[^\w\s]','',doc)
        doc_tokens = nltk.word_tokenize(doc.lower())
        all_docs.append(doc_tokens)
    # bfollow gensim instructions
    final_documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(all_docs)]
    # train the model
    # hyperparameter tuning
    #     model = Doc2Vec(documents = final_documents,
    #                         vector_size=700,
    #                         window=7,
    #                         min_count =3)

    model = Doc2Vec(documents=final_documents,
                    vector_size=256,
                    window=7, min_count=3)

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
        if i == 0:
            pass
        else:
            similarity = np.dot(embeddings[i, :], embeddings[i - 1, :])
            title = times[i - 1] + " vs " + times[i]
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
    sim_scores_dict: a dictionary {company name: corresponding sim score DataFrame} 
    '''
    sim_scores_dict = {}
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
            embedding_lst[i - value[0], :] = embedding
        df = get_sim_scores_embedding(embedding_lst, company_timestamps)
        file_name = key + ".csv"
        df.to_csv('sim_result/'+file_name)
        sim_scores_dict[key] = df

    return sim_scores_dict


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
        if i == 0:
            pass
        else:
            similarity = np.dot(model.docvecs.get_normed_vectors()[i], model.docvecs.get_normed_vectors()[i - 1])
            title = times[i - 1] + " vs " + times[i]
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
    for i in range(3, n + 1):
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
    df = pd.DataFrame({"timestamp": times})
    for i in range(len(clustering_labels)):
        num_cluster = i + 3
        col_name = "cluster n=" + str(num_cluster)
        df[col_name] = clustering_labels[i]

    return df


def clustering_silhouette(embeddings, n):
    '''
    Arg:
    embeddings: nunpy array of embeddings we use to do clusterings
    n: number of clusters

    Return:
    clustering_labels: labels for which ckuster does the timestamp belong to for different n
    K: optimal K clusters determined by the silhouette method
    '''
    clustering_labels = {}
    sil = {}
    for i in range(3, n + 1):
        kmeans = KMeans(n_clusters=i, random_state=0).fit(embeddings)
        #         labels =kmeans.labels_
        clustering_labels[i] = kmeans.labels_
        #         clustering_labels.append(kmeans.labels_)
        sil[i] = silhouette_score(embeddings, kmeans.labels_, metric='cosine')
    #         sil.append(silhouette_score(embeddings, kmeans.labels_, metric = 'euclidean'))

    if sil:
        opt_K = max(sil, key=lambda x: sil[x])
        return opt_K, sil[opt_K], clustering_labels[opt_K]
    else:
        return 0, 0, []


def expand_df_optK(times, opt_clustering_labels, opt_K):
    '''
    Arg:
    df: dataframe contains sim score and timestamp
    clustering_labels: clustering labels for different n

    Return:
    df: expanded df with clustering info
    '''
    df = pd.DataFrame({"timestamp": times})
    col_name = "cluster k=" + str(opt_K)
    df[col_name] = opt_clustering_labels

    return df


def get_ave_dist(data):
  dist_list = []
  for i in range(len(data)-1):
  # dist = np.linalg.norm(data[i] - data[i+1])
    dist_list.append(np.linalg.norm(data[i] - data[i+1]))
  ave_dist = np.array(dist_list).mean()
  return ave_dist 


def get_dist_std(data):
  std_list = []
  for i in range(len(data)-1):
  # dist = np.linalg.norm(data[i] - data[i+1])
    std_list.append(np.linalg.norm(data[i] - data[i+1]))
  dist_std = np.array(dist_list).std()
  return dist_std, std_list

def dbscan_model(model,company_index_dict):
  dbscan_dict = {}
  sil_score_db_dict = {}
  for comp_name in company_index_dict.keys():

    # add time dimension to vectors
    data = model.dv.vectors[company_index_dict[comp_name][0]:company_index_dict[comp_name][1]]
    data = np.append(data, np.array([[i] for i in range(data.shape[0])]), axis=1)

    if len(data) != 1:
      # initiate DBscan clustering, using average distance as eps
      ave_dist = get_ave_dist(data)
      # data_std = get_std_dist(data)
      dbs = DBSCAN(eps=ave_dist , min_samples=2).fit(data)
      labels = dbs.labels_
      # Number of clusters in labels, ignoring noise if present.
      n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

      # get the silhouette score from the dbscan labels
      # sil_score_db_dict: key is company name, value is a tuple of sil score and the corresponding # of clusters
      if len(set(labels)) > 1:
        sil_score_db = silhouette_score(data, labels, metric='cosine')
        sil_score_db_dict[comp_name] = (sil_score_db, n_clusters)
      else:
        sil_score_db_dict[comp_name] = (0, 1)

      #clean output as pivots = 1
      labels_clean = [0]
      for i in range(1, len(labels)):
        if labels[i-1] == labels[i]:
          labels_clean.append(0)
        else:
          labels_clean.append(1)
    else:
      labels_clean = [1]
    dbscan_dict[comp_name] = labels_clean

  return dbscan_dict, sil_score_db_dict


def boolean_cluster_all_companies(model, company_index_dict):
    '''
        Arg:
        model: trained doc2vec model
        company_index_dict: dictionary with company first index and last last index in the model

        Return:
        A list of all the company names for further score column join
        Save cluster and boolean cluster dataframes for each company
        boolean_cluster_dict: a dictionary {company name: corresponding clustering boolean result DataFrame}

    '''
    company_name_list = []
    boolean_cluster_dict = {}
    opt_num_of_clusters_dict = {}

    dbscan_dict, sil_score_db_dict = dbscan_model(model, company_index_dict)

    for k, v in company_index_dict.items():
        company_name = k
        embeddings = model.dv.vectors[v[0]:v[1]]
        embeddings = np.append(embeddings, np.array([[i] for i in range(embeddings.shape[0])]), axis=1)
        target_times = all_time[v[0]:v[1]]

        if len(embeddings) >= 12:
            n = 10
        else:
            n = len(embeddings) - 2

        opt_K, sil_score_optK, opt_clusters = clustering_silhouette(embeddings, n)

        if opt_K > 0:
            company_name_list.append(company_name)

            # get the optimal # of clusters between Kmeans and dbscan
            for db_cn, db_tuple in sil_score_db_dict.items():
                if db_cn == company_name:
                    dn_n_clusters = db_tuple[1]
                    db_sil_score = db_tuple[0]
                    if db_sil_score >= sil_score_optK:
                        opt_num_of_clusters_dict[company_name] = (dn_n_clusters, "dbscan")
                    else:
                        opt_num_of_clusters_dict[company_name] = (opt_K, "Kmeans")

            df_cluster_optK = expand_df_optK(target_times, opt_clusters, opt_K)
            # save dataframe in a new dir called clustering_optK/
            outdir1 = "./clustering_optK"
            if not os.path.exists(outdir1):
                os.mkdir(outdir1)

            df_cluster_optK.to_csv("clustering_optK/" + company_name + ".csv")

            ts = df_cluster_optK['timestamp']
            ts_list = []
            for i in range(len(ts) - 1):
                ts_list.append(str(ts[i]) + ' vs ' + str(ts[i + 1]))

            cluster_start_index = list(df_cluster_optK.columns).index('timestamp') + 1
            cluster_matrix = df_cluster_optK.to_numpy()[:, cluster_start_index:]
            boolean_matrix = np.zeros((cluster_matrix.shape[0] - 1, cluster_matrix.shape[1]))

            for i in range(boolean_matrix.shape[0]):
                for j in range(boolean_matrix.shape[1]):
                    boolean_matrix[i][j] = (cluster_matrix[i][j] != cluster_matrix[i + 1][j])

            df_cluster_boolean = pd.DataFrame(boolean_matrix.astype(int),
                                              columns=df_cluster_optK.columns[cluster_start_index:])
            df_cluster_boolean.insert(0, 'timestamp', ts_list)
            # save dataframe in a new dir called boolean_clustering_optK/
            outdir2 = "./boolean_clustering_optK"
            if not os.path.exists(outdir2):
                os.mkdir(outdir2)

            df_cluster_boolean.to_csv("boolean_clustering_optK/" + company_name + ".csv")
            boolean_cluster_dict[company_name] = df_cluster_boolean

    return company_name_list, boolean_cluster_dict, opt_num_of_clusters_dict


def combine_dfs(company_name_list, sim_scores_dict, boolean_cluster_dict, dbscan_dict):
    """
    Save combined similarity score + boolean_cluster result + dbscan result for each company
    """
    for company_name in company_name_list:
        df_sim_score = sim_scores_dict[company_name].set_index('timestamp')
        df_boolean_cluster = boolean_cluster_dict[company_name].set_index('timestamp')

        combined_df = df_sim_score.join(df_boolean_cluster)
        combined_df['label_db'] = dbscan_dict[company_name][1:]

        outdir3 = "./combined"
        if not os.path.exists(outdir3):
            os.mkdir(outdir3)
        combined_df.to_csv("combined/" + company_name + ".csv")

# Notes for Ningxin
# For the input K for mapping(K, sensitivity), you might iterate throughthe opt_num_of_clusters_dict as follows:
# K = opt_num_of_clusters_dict[company_name][0] for company_name in company_name_list

def sigmoid(x):
  return(1 / (1 + np.exp(-x)))

def pivot_detect(company_name_list, sensitivity_index, opt_num_of_clusters_dict, sim_scores_dict):
    '''
    Arg:
    company_name_list: a list of company names
    sensitivity_index: the sensitivity index 1-10
    opt_num_of_clusters_dict: optimal # of clusters among Kmeans and sbscan models
    sim_scores_dict: a dictionary {company name: corresponding cosine similarity score DataFrame}
    
    Return:
    sim_scores_dict2: a dictionary {company name: corresponding sim score DataFrame with pivot classification result}
    '''
    company_threshold = {}
    if sensitivity_index < 1 or sensitivity_index > 10:
        raise ValueError('Sensitivity index must be with 1 to 10')
        
    if not os.path.exists('combined_pivot_si' + str(sensitivity_index)):
        os.makedirs('combined_pivot_si' + str(sensitivity_index))
        print('combined_pivot_si' + str(sensitivity_index) + " is created")
    
    sim_scores_dict2 = sim_scores_dict.copy()
    
    mapping = interp1d([0.1, 10],[-1, 10])
    
    for company in company_name_list:
        K = opt_num_of_clusters_dict[company][0]
        SI = sensitivity_index
        x = float(mapping(SI/K))

        threshold = sigmoid(x)
        # store threshold for each company
        company_threshold[company] = threshold

        company_df = sim_scores_dict2[company]
        
        company_df['pivot'] = company_df['similarity'] < threshold
        company_df['pivot'] = company_df['pivot'].astype(int)
        company_df.to_csv('combined_pivot_si' + str(sensitivity_index) + "/" + company + ".csv")
        
    return sim_scores_dict2


if __name__ == '__main__':
    # create directories
    if not os.path.exists('sim_result'):
        os.makedirs('sim_result')
        print("sim_result is created")
    if not os.path.exists('boolean_clustering_optK'):
        os.makedirs('boolean_clustering_optK')
        print("boolean_clustering_optK is created")
    if not os.path.exists('clustering_optK'):
        os.makedirs('clustering_optK')
        print("clustering_optK is created")
    if not os.path.exists('combined'):
        os.makedirs('combined')
        print("combined is created")
    # get all the info we need for the model and further analysis
    all_documents, all_time, company_index_dict = get_all_data('../out2')
    # train model 
    model = train_doc2vec_model(all_documents, "saved_model_all_companies")
    # # save all similarity score to csv files, and to our sim scores dictionary
    sim_scores_dict = sim_scores_all_companies(model, all_time, company_index_dict)
    # # save clusering result & all boolean cluster with the optimal K (highest silhouette score) to csv files
    # and save the clustering boolean
    # get the optimal # of clusters among Kmeans and sbscan models
    company_name_list, boolean_cluster_dict, opt_num_of_clusters_dict = boolean_cluster_all_companies(model, company_index_dict)
    # print out the optimal # of clusters for each company and indicate whether is Kmeans or dbscan model
    #print(opt_num_of_clusters_dict)
    
    dbscan_dict, sil_score_db_dict = dbscan_model(model,company_index_dict)
    # # save combined result to csv files
    combine_dfs(company_name_list, sim_scores_dict, boolean_cluster_dict, dbscan_dict)

    si = 5
    # adds pivot detection columns on each similarity score csv files
    # and save under new directory specifying the sensitivity index
    similarity_pivot_df = pivot_detect(company_name_list, si, opt_num_of_clusters_dict, sim_scores_dict)