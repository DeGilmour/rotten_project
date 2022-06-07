from sklearn.svm import LinearSVC
import nlpaug.augmenter.word as naw
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import nlpaug
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
import string
import matplotlib.pyplot as plt
from sklearn import metrics
import matplotlib.pyplot as mplt
import numpy as np
import joblib
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import nltk
import re
import seaborn as sns
from nltk.corpus import stopwords

# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
stop_words = set(stopwords.words('english'))
nltk.download('omw-1.4')

class PreProcessing:

    def __init__(self, dataset_file=None):
        self.rotten_ds = pd.read_csv(
            'rotten_tomatoes_movies.csv', sep=',', parse_dates=True)
        print(self.rotten_ds.shape)

    def to_single_genre(self, dataset):
        return dataset[0]

    def split_genres(self, genres):
        list_genre = []
        for i in genres:
            if ' & ' in i:
                i = i.split(' & ')
                if isinstance(i, list):
                    i = ", ".join(i)
            list_genre.append(i)
        list_genre = list(set(list_genre))
        new_list_genre = []
        for genre in list_genre:
            x = genre
            if ', ' in genre:
                x, y = genre.split(', ')
                new_list_genre.append(x)
                new_list_genre.append(y)
            else:
                new_list_genre.append(x)
        return new_list_genre

    def get_selected_columns(self):
        self.rotten_ds = self.rotten_ds.filter(
            ['movie_title', 'movie_info', 'genre_list'])

    def drop_na_rows(self):
        self.rotten_ds = self.rotten_ds.dropna()

    def get_dataset(self):
        return self.rotten_ds

    def clean_movie_info(self, text):
        text = str(text)
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        text = re.sub("\'", "", text)
        text = re.sub("[^a-zA-Z]", " ", text)
        text = re.sub(r'\w*\d\w*', '', text)
        text = ' '.join(text.split())
        text = text.lower()
        no_stopword_text = [w for w in text.split() if not w in stop_words]
        return ' '.join(no_stopword_text)

    def augment_dataset(self, n_aug, augment=True):
        if not augment:
            return 
        print('Started augmenting data, will augment it {} times'.format(n_aug))
        aug = naw.SynonymAug(aug_src='wordnet', aug_max=5)
        df1 = self.rotten_ds[['movie_title', 'movie_info', 'genre_list']]
        df2 = pd.DataFrame(columns=['movie_title', 'movie_info', 'genre_list'])
        dict_df = {}
        df_list = []
        df_to_copy = df1[~df1['genre_list'].isin(
            ['Action & Adventure', 'Comedy', 'Drama'])]
        df_list.append(df_to_copy)
        for x in range(n_aug):
            x_df = pd.DataFrame(
                columns=['movie_title', 'movie_info', 'genre_list'])
            for i in df1:
                if i in ['movie_info']:
                    x_df[i] = df_list[-1][i].apply(aug.augment)
                else:
                    x_df[i] = df_list[-1][i]
            df_list.append(x_df)

        df2 = pd.concat(df_list, ignore_index=True)
        print(df2)
        df = df1.append(df2, ignore_index=True)
        self.rotten_ds = df

    def create_training_dataset(self, augment=True):
        self.rotten_ds['genre_list'] = self.rotten_ds['genres'].str.split(', ')
        self.drop_na_rows()
        self.get_selected_columns()
        self.rotten_ds['genre_list'] = self.rotten_ds['genre_list'].apply(
            self.split_genres)
        # Change augmentation value here
        self.augment_dataset(3, augment=augment)
        print("Cleaning the movie_info column")
        self.rotten_ds['movie_info'] = self.rotten_ds['movie_info'].apply(
            lambda x: self.clean_movie_info(x))
        print("Size of dataset: ", self.rotten_ds.shape)


class TrainModel:

    def __init__(self, df_sampl, dataset_split):
        self.rotten_ds_new = df_sampl
        print('Inside Training')
        print(self.rotten_ds_new.shape)
        self.dataset_split = dataset_split
        self.split_vars = {}

    def genre_to_label(self):
        multilabel_binarizer = MultiLabelBinarizer()
        multilabel_binarizer.fit(self.rotten_ds_new['genre_list'])
        y = multilabel_binarizer.transform(self.rotten_ds_new['genre_list'])
        joblib.dump(multilabel_binarizer, 'multilabel_binarizer.pkl')
        print(multilabel_binarizer.classes_)
        return y

    def text_to_labels(self):
        self.rotten_ds_new['genre_list_enc'] = self.rotten_ds_new['genre_list'].apply(
            lambda x: x.split(',')[0])
        multilabel_binarizer = MultiLabelBinarizer()
        y = multilabel_binarizer.fit_transform(
            self.movies_new['genre_list_enc'])
        joblib.dump(multilabel_binarizer, 'multilabel_binarizer.pkl')
        print(multilabel_binarizer.classes_)
        return y

    def to_get_dummies(self):
        genre = pd.get_dummies(self.movies_new['genre_list'], drop_first=False)
        var_answer = pd.concat([self.movies_new, genre], axis=1)
        return var_answer

    def set_split_vars(self, xt, xv, yt, yv):
        self.split_vars['xtrain'] = xt
        self.split_vars['xval'] = xv
        self.split_vars['ytrain'] = yt
        self.split_vars['yval'] = yv

    def train_model(self):
        print('Initiated Training Just now.....')
        y = self.genre_to_label()
        x = self.movies_new[['movie_info']].values.ravel()
        xtrain, xval, ytrain, yval = train_test_split(
            x, y, test_size=self.dataset_split, random_state=9)
        tfidf_vectorizer = TfidfVectorizer()
        print('setting the split vars')
        self.set_split_vars(xtrain, xval, ytrain, yval)
        xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
        xval_tfidf = tfidf_vectorizer.transform(xval)
        joblib.dump(tfidf_vectorizer, 'tf_idf_vectorizer_model.pkl')
        lr = LinearSVC()
        clf = OneVsRestClassifier(lr)
        clf.fit(xtrain_tfidf, ytrain)
        joblib.dump(clf, 'genre_predictor_model.pkl')
        print('Training Completed')
        y_pred = clf.predict(xval_tfidf)
        print(f1_score(yval, y_pred, average="micro"))
        f1 = f1_score(yval, y_pred, average="micro")
        print("F1-score: ", f1)
        print("Classification Report: ")
        print(metrics.classification_report(yval, y_pred))
        return f1


class TestModel(PreProcessing):
    # 0.9
    def __init__(self, title, description):
        print('Testing new data')
        self.title = title
        self.description = description
        data = [[title, description]]
        self.df = pd.DataFrame(data, columns=['movie_title', 'movie_info'])
        # Load the model from the file
        self.genre_prediction = joblib.load('model/genre_predictor_model.pkl')

    def predict_new_movie(self):
        self.df['movie_info'] = self.df['movie_info'].apply(
            lambda x: self.clean_movie_info(x))
        self.tfidf_vectorizer = joblib.load(
            'model/tf_idf_vectorizer_model.pkl')
        xval_tfidf = self.tfidf_vectorizer.transform(self.df['movie_info'])
        y_pred = self.genre_prediction.predict(xval_tfidf)
        self.multilabel_binarizer = joblib.load(
            'model/multilabel_binarizer.pkl')
        # Inverse transforming the code labels to the string genres
        op = self.multilabel_binarizer.inverse_transform(y_pred)
        return op

