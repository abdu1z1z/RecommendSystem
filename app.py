# app.py

from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
from surprise import SVD, Dataset, Reader
from joblib import load, dump
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from functools import lru_cache

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# تحميل البيانات
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

def process_movie_features():
    global movies_with_features, features_matrix, similarity_matrix, mlb
    movies['genres_list'] = movies['genres'].apply(lambda x: x.split('|'))
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(movies['genres_list'])
    genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)
    movies_with_features = pd.concat([movies[['movieId']], genres_df], axis=1)
    features_matrix = movies_with_features.drop('movieId', axis=1)
    similarity_matrix = cosine_similarity(features_matrix)

process_movie_features()

# تحميل النموذج أو تدريبه إذا لم يكن موجودًا
try:
    model = load('recommendation_model.joblib')
except:
    def retrain_model():
        reader = Reader(rating_scale=(0.5, 5))
        data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
        trainset = data.build_full_trainset()
        global model
        model = SVD()
        model.fit(trainset)
        dump(model, 'recommendation_model.joblib')
    retrain_model()

# إعدادات TMDb API
TMDB_API_KEY = '4d531df5e61cb4c59e24b2c797bce1d1'  # استبدل YOUR_TMDB_API_KEY بمفتاح TMDb API الخاص بك
TMDB_BASE_URL = 'https://api.themoviedb.org/3'
IMAGE_BASE_URL = 'https://image.tmdb.org/t/p/w500'

# دالة لجلب تفاصيل الفيلم من TMDb API
@lru_cache(maxsize=1000)
def get_movie_details(movie_title):
    try:
        # البحث عن الفيلم باستخدام العنوان
        search_url = f"{TMDB_BASE_URL}/search/movie"
        params = {
            'api_key': TMDB_API_KEY,
            'query': movie_title
        }
        response = requests.get(search_url, params=params)
        data = response.json()

        if data['results']:
            movie_data = data['results'][0]
            movie_id = movie_data['id']

            # الحصول على تفاصيل الفيلم باستخدام movie_id
            details_url = f"{TMDB_BASE_URL}/movie/{movie_id}"
            params = {
                'api_key': TMDB_API_KEY,
                'append_to_response': 'credits'
            }
            response = requests.get(details_url, params=params)
            details = response.json()

            return {
                'poster_path': details.get('poster_path'),
                'overview': details.get('overview'),
                'release_date': details.get('release_date'),
                'rating': details.get('vote_average'),
                'genres': [genre['name'] for genre in details.get('genres', [])],
                'cast': [cast['name'] for cast in details.get('credits', {}).get('cast', [])[:5]],
                'director': next((crew['name'] for crew in details.get('credits', {}).get('crew', []) if crew['job'] == 'Director'), None)
            }
        else:
            return None
    except Exception as e:
        print(f"Error fetching details for movie '{movie_title}': {e}")
        return None

# دالة لجلب الأفلام الشعبية من TMDb API
@lru_cache(maxsize=1)
def get_popular_movies():
    try:
        url = f"{TMDB_BASE_URL}/movie/popular"
        params = {
            'api_key': TMDB_API_KEY,
            'language': 'en-US',
            'page': 1
        }
        response = requests.get(url, params=params)
        data = response.json()
        movies = data['results'][:10]  # الحصول على أول 10 أفلام
        return movies
    except Exception as e:
        print(f"Error fetching popular movies: {e}")
        return []

def get_recent_watched_movies(user_id, n=10):
    watched_movies = ratings[ratings['userId'] == int(user_id)][['movieId', 'rating', 'timestamp']]
    watched_movies = watched_movies.merge(movies[['movieId', 'title', 'genres']], on='movieId', how='left')
    watched_movies = watched_movies.sort_values(by='timestamp', ascending=False).head(n)
    return watched_movies.to_dict('records')

def recommend_based_on_recent(user_id, n_recommendations=5):
    user_id = int(user_id)
    # الحصول على آخر 10 أفلام شاهدها المستخدم
    recent_movies = get_recent_watched_movies(user_id)
    recent_movie_ids = [movie['movieId'] for movie in recent_movies]
    movie_indices = movies_with_features[movies_with_features['movieId'].isin(recent_movie_ids)].index
    if len(movie_indices) == 0:
        # إذا لم يكن لدى المستخدم أفلام حديثة، نوصي بأفلام شائعة
        popular_movies = ratings.groupby('movieId')['rating'].count().sort_values(ascending=False)
        top_movie_ids = popular_movies.head(n_recommendations).index
        recommended_movies = movies[movies['movieId'].isin(top_movie_ids)]
        return recommended_movies[['title', 'genres']].to_dict('records')
    else:
        # جمع درجات التشابه للأفلام المشاهدة حديثًا
        similarity_scores = similarity_matrix[movie_indices].mean(axis=0)
        watched_movie_ids = ratings[ratings['userId'] == user_id]['movieId'].unique()
        unseen_movie_indices = movies_with_features[~movies_with_features['movieId'].isin(watched_movie_ids)].index
        unseen_similarity_scores = similarity_scores[unseen_movie_indices]
        # الحصول على أعلى n توصية
        top_indices = unseen_similarity_scores.argsort()[-n_recommendations:][::-1]
        recommended_movie_ids = movies_with_features.iloc[unseen_movie_indices[top_indices]]['movieId'].values
        recommended_movies = movies[movies['movieId'].isin(recommended_movie_ids)]
        return recommended_movies[['title', 'genres']].to_dict('records')

def retrain_model():
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    global model
    model = SVD()
    model.fit(trainset)
    dump(model, 'recommendation_model.joblib')

# صفحات التطبيق
# ... الكود السابق ...

@lru_cache(maxsize=1)
def get_genre_mapping():
    url = f"{TMDB_BASE_URL}/genre/movie/list"
    params = {
        'api_key': TMDB_API_KEY,
        'language': 'en-US'
    }
    response = requests.get(url, params=params)
    data = response.json()
    genres = data.get('genres', [])
    genre_mapping = {genre['id']: genre['name'] for genre in genres}
    return genre_mapping


genre_mapping = get_genre_mapping()

# تعديل مسار index لتمرير genre_mapping
@app.route('/')
def index():
    popular_movies = get_popular_movies()
    return render_template('index.html', popular_movies=popular_movies, IMAGE_BASE_URL=IMAGE_BASE_URL, genre_mapping=genre_mapping)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # الحصول على معرف فريد جديد
        if ratings['userId'].max() is not None:
            user_id = int(ratings['userId'].max()) + 1
        else:
            user_id = 1  # إذا لم يكن هناك مستخدمون في البيانات
        session['user_id'] = user_id
        return redirect(url_for('select_movies'))
    return render_template('register.html')

@app.route('/select_movies', methods=['GET', 'POST'])
def select_movies():
    if 'user_id' in session:
        user_id = int(session['user_id'])
        if request.method == 'POST':
            action = request.form.get('action')

            # الحصول على الأفلام المختارة من الجلسة أو تهيئتها كقائمة فارغة
            selected_movie_ids = session.get('selected_movie_ids', [])

            # تحديث قائمة الأفلام المختارة بناءً على الاختيارات الحالية
            current_selections = request.form.getlist('movies')
            current_selections = [int(movie_id) for movie_id in current_selections]

            if action == 'submit':
                # عند الإرسال النهائي، حفظ التقييمات
                ratings_data = []
                for movie_id in selected_movie_ids + current_selections:
                    ratings_data.append({'userId': user_id, 'movieId': int(movie_id), 'rating': 5.0, 'timestamp': pd.Timestamp.now().timestamp()})
                new_ratings = pd.DataFrame(ratings_data)
                global ratings
                ratings = pd.concat([ratings, new_ratings], ignore_index=True)
                ratings.to_csv('ratings.csv', index=False)
                retrain_model()
                # مسح قائمة الأفلام المختارة من الجلسة
                session.pop('selected_movie_ids', None)
                return redirect(url_for('user_profile', user_id=user_id))
            else:
                # تحديث قائمة الأفلام المختارة في الجلسة
                # إذا كانت الصفحة الأولى، استبدل القائمة؛ إذا كانت صفحات أخرى، أضف إليها
                if action == 'search':
                    selected_movie_ids = current_selections
                else:
                    # إضافة الاختيارات الحالية إلى القائمة السابقة
                    selected_movie_ids = list(set(selected_movie_ids + current_selections))

                # حفظ القائمة في الجلسة
                session['selected_movie_ids'] = selected_movie_ids

                # تحديث الصفحة بناءً على البحث أو تغيير الصفحة
                search_query = request.form.get('search', '')
                page = int(request.form.get('page', 1))
        else:
            # الطريقة GET: تهيئة الاختيارات
            selected_movie_ids = session.get('selected_movie_ids', [])
            search_query = request.args.get('search', '')
            page = request.args.get('page', 1, type=int)

        per_page = 20

        if search_query:
            filtered_movies = movies[movies['title'].str.contains(search_query, case=False, na=False)]
        else:
            filtered_movies = movies

        total_movies = len(filtered_movies)
        pages = (total_movies - 1) // per_page + 1
        start = (page - 1) * per_page
        end = start + per_page
        movies_paginated = filtered_movies.iloc[start:end]

        # جلب تفاصيل الأفلام وإضافة poster_path
        movies_list = movies_paginated.to_dict('records')
        for movie in movies_list:
            details = get_movie_details(movie['title'])
            if details:
                movie['poster_path'] = details.get('poster_path')
                movie['genres'] = details.get('genres')
            else:
                movie['poster_path'] = None
                movie['genres'] = []

        return render_template('select_movies.html',
                               movies=movies_list,
                               search_query=search_query,
                               page=page,
                               pages=pages,
                               selected_movie_ids=selected_movie_ids,
                               IMAGE_BASE_URL=IMAGE_BASE_URL)
    else:
        return redirect(url_for('register'))

@app.route('/user/<int:user_id>')
def user_profile(user_id):
    recent_watched_movies = get_recent_watched_movies(user_id)
    recommended_movies = recommend_based_on_recent(user_id)

    # جلب تفاصيل الأفلام للمشاهدة والتوصيات
    for movie in recent_watched_movies:
        details = get_movie_details(movie['title'])
        if details:
            movie.update(details)

    for movie in recommended_movies:
        details = get_movie_details(movie['title'])
        if details:
            movie.update(details)

    return render_template(
        'user_profile.html',
        watched_movies=recent_watched_movies,
        recommended_movies=recommended_movies,
        user_id=user_id,
        IMAGE_BASE_URL=IMAGE_BASE_URL
    )


@app.route('/search_user', methods=['GET', 'POST'])
def search_user():
    if request.method == 'POST':
        user_id = request.form['user_id']
        if int(user_id) in ratings['userId'].unique():
            return redirect(url_for('user_profile', user_id=user_id))
        else:
            return render_template('search_user.html', error='User ID not found.')
    return render_template('search_user.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
