import joblib
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string
import joblib


# df = df.dropna()
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
stop_words = set(stopwords.words('english'))


class PreProcessing:
    def __init__(self):
        self.movies_df = pd.read_csv(
            'model/rotten_tomatoes_movies.csv', sep=',', parse_dates=True)

    def clean_dataset(self):
        self.movies_df['movie_info'] = self.movies_df['movie_info'].fillna('')
        self.movies_df['genres'] = self.movies_df['genres'].fillna('')
        self.movies_df['actors'] = self.movies_df['actors'].fillna('')
        self.movies_df['genres'] = self.movies_df['genres'].apply(
            self.split_genres)
        self.movies_df['directors'] = self.movies_df['directors'].fillna('')
        self.movies_df['tomatometer_rating'] = self.movies_df['tomatometer_rating'].fillna(
            0)
        self.movies_df["movie_info"] = self.movies_df['movie_info'].apply(
            self.clean_movie_info)
        self.movies_df["combined_features"] = self.movies_df.apply(
            self.combined_features, axis=1)
        return self.movies_df

    def combined_features(self, row):
        return row['genres'] + " " + row['movie_info'] + " " + row['directors']

    def split_genres(self, genres):
        genres = [genres]
        list_genre = []
        for i in genres:
            if ' & ' in i:
                i = i.split(' & ')
                if isinstance(i, list):
                    i = ", ".join(i)
            list_genre.append(i)
        list_genre = list(set(list_genre))
        return list_genre[0]

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


class MovieRecomender(PreProcessing):
    
    def __init__(self):
        self.movies_df = pd.read_csv('model/cleaned_rotten_ds.csv')

    def search_indices(self, indices, movie_name):
        fist_word_pattern = '\d+'
        match = ""
        for i in indices.iteritems():
            real_movie_name = i[0]
            movie_name_ = movie_name.strip().lower()
            movie_name_to_be_found = i[0].strip().lower()
            if movie_name_ in movie_name_to_be_found:
                return real_movie_name
            elif movie_name_ == movie_name_to_be_found:
                return real_movie_name
        return real_movie_name

    def get_similar_movies(self, movie_title, indices, cos_sim):
        try:
            curr_index = indices[movie_title]
        except KeyError:
            searched = self.search_indices(indices, movie_title)
            curr_index = indices[searched]
        # More then 1 movie
        if isinstance(curr_index, pd.core.series.Series):
            list_index = []
            curr_index_ = curr_index[0]
            for i in curr_index:
                list_index.append(i)
            new_df = self.movies_df.loc[self.movies_df.index.isin(list_index)]
            new_df = new_df.nlargest(1, ['tomatometer_count'])
            curr_index = pd.Series(new_df.index, index=new_df['movie_title']).drop_duplicates()[
                new_df['movie_title']]
            curr_index = curr_index[0]
        sim_scores = list(enumerate(cos_sim[curr_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Spliting to take the 10 most similars
        sim_scores = sim_scores[1:20]

        movie_indices = [i[0] for i in sim_scores]
        return self.movies_df[['movie_title', 'genres', 'actors', 'directors']].iloc[movie_indices]

    def find_movies_calculate(self, movie_title):

        tf_idf_vectorizer_object = TfidfVectorizer(
            stop_words='english', analyzer=self.clean_movie_info)

        tf_idf_matrix = tf_idf_vectorizer_object.fit_transform(
            self.movies_df['movie_info'])

        cos_sim = cosine_similarity(tf_idf_matrix, tf_idf_matrix)
        joblib.dump(cos_sim, 'cos_sim.pkl')
        title_to_index = pd.Series(
            self.movies_df.index, index=self.movies_df['movie_title']).drop_duplicates()

        tf_idf_matrix_genres = tf_idf_vectorizer_object.fit_transform(
            self.movies_df['genres'])

        cos_sim_genres = cosine_similarity(
            tf_idf_matrix_genres, tf_idf_matrix_genres)
        joblib.dump(cos_sim_genres, 'cos_sim_genres.pkl')
        title_to_index_genres = pd.Series(
            self.movies_df.index, index=self.movies_df['genres']).drop_duplicates()

        # self.movies_df['directors'] = self.movies_df['directors'].fillna('')

        # tf_idf_matrix_directors= tf_idf_vectorizer_object.fit_transform(self.movies_df['directors'])

        # cos_sim_directors= cosine_similarity(tf_idf_matrix_directors, tf_idf_matrix_directors)

        # title_to_index_directors = pd.Series(self.movies_df.index, index=self.movies_df['directors']).drop_duplicates()

        cos_sim = cos_sim + cos_sim_genres

        res = pd.concat([title_to_index, title_to_index_genres])

        print('Movies similar to ' + movie_title + ' are:')
        similar_movies = self.get_similar_movies(
            movie_title=movie_title, indices=res, cos_sim=cos_sim)
        similar_movies = pd.DataFrame(similar_movies)
        return similar_movies

    def find_movies(self, movie_title):
        cos_sim = joblib.load('cos_sim.pkl')
        title_to_index = pd.Series(
            self.movies_df.index, index=self.movies_df['movie_title']).drop_duplicates()
        cos_sim_genres = joblib.load('cos_sim_genres.pkl')
        # title_to_index_genres = pd.Series(self.movies_df.index, index=self.movies_df['genres']).drop_duplicates()
        cos_sim = cos_sim + cos_sim_genres
        # res = pd.concat([title_to_index, title_to_index_genres])
        print('Movies similar to ' + movie_title + ' are:')
        similar_movies = self.get_similar_movies(
            movie_title=movie_title, indices=title_to_index, cos_sim=cos_sim)
        similar_movies = pd.DataFrame(similar_movies)
        return similar_movies.to_html()
