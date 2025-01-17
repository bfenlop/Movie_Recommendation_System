from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import streamlit as st

# Cargar datos
items_final = pd.read_csv('items_final_poster.csv')

# Preprocesar las películas
def preprocess_movies(df):
    def extract_genres(row):
        return " ".join([col for col in df.columns[2:15] if row[col] == 1])

    df['combined_features'] = (
        df['title'] + " " +
        df['OMDb_Plot'].fillna('') + " " +
        df['OMDb_Year'].astype(str) + " " +
        df.apply(extract_genres, axis=1)
    )
    return df

# Vectorización
def vectorize_features(movie_data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movie_data['combined_features'])
    return tfidf, tfidf_matrix

# Filtrar y paginar películas iniciales con balance entre géneros
def get_paginated_movies(movie_data, num_movies_per_genre=15, movies_per_page=9):
    genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Drama', 
              'Fantasy', 'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller']
    
    # Crear un DataFrame vacío para almacenar las películas seleccionadas
    selected_movies = pd.DataFrame()

    # Seleccionar un número limitado de películas por género, aleatorias para mayor diversidad
    for genre in genres:
        genre_movies = movie_data[movie_data[genre] == 1].sample(n=num_movies_per_genre, random_state=42)
        selected_movies = pd.concat([selected_movies, genre_movies])
    
    # Dividir las películas en páginas
    paginated_movies = [
        selected_movies.iloc[i:i + movies_per_page]
        for i in range(0, len(selected_movies), movies_per_page)
    ]
    return paginated_movies

# Recomendaciones
def recommend_movies(selected_ids, movie_data, tfidf_matrix, num_recommendations=10, filters=None):
    selected_indices = movie_data[movie_data['id'].isin(selected_ids)].index
    if selected_indices.empty:
        return pd.DataFrame()  # Si no hay películas seleccionadas, retornar un DataFrame vacío
    
    selected_tfidf = tfidf_matrix[selected_indices]
    similarity_scores = cosine_similarity(selected_tfidf, tfidf_matrix).mean(axis=0)
    
    similar_indices = np.argsort(similarity_scores)[::-1]
    recommended = movie_data.iloc[similar_indices]
    
    if filters:
        for key, condition in filters.items():
            if key == 'genre':
                recommended = recommended[recommended.apply(condition, axis=1)]
            elif key in recommended.columns:
                if callable(condition):
                    recommended = recommended[recommended[key].apply(condition)]
                elif isinstance(condition, (list, tuple)):
                    recommended = recommended[recommended[key].isin(condition)]
                else:
                    recommended = recommended[recommended[key] == condition]
    
    recommended = recommended[~recommended['id'].isin(selected_ids)]
    
    return recommended.head(num_recommendations)

# Cargar y procesar datos
movies_preprocessed = preprocess_movies(items_final)
tfidf, tfidf_matrix = vectorize_features(movies_preprocessed)

# Crear páginas de películas iniciales equilibradas y aleatorias
paginated_movies = get_paginated_movies(movies_preprocessed)

# Configuración del estado de sesión
if 'show_portada' not in st.session_state:
    st.session_state.show_portada = True
if 'page_number' not in st.session_state:
    st.session_state.page_number = 0
if 'selected_movies' not in st.session_state:
    st.session_state.selected_movies = set()

MOVIES_PER_PAGE = 9

# Función de actualización de selección
def update_selection(movie_key):
    if movie_key in st.session_state.selected_movies:
        st.session_state.selected_movies.discard(movie_key)
    else:
        st.session_state.selected_movies.add(movie_key)

# Mostrar portada
if st.session_state.show_portada:
    st.title("Movie Recommendation System")
    st.write("Find the perfect movie based on your preferences and let our system guide you!")
    portada_url = "portada.jpg"
    st.image(portada_url, use_container_width=True)

    if st.button("Start the Adventure!", key="start_button"):
        st.session_state.show_portada = False

else:
    st.write("### Movie Recommendation System")
    st.write("Select movies you like and apply filters to get personalized recommendations!")

    # Barra lateral para filtros
    with st.sidebar:
        st.write("#### Filters")
        genre_filter = st.multiselect(
            "Filter by genre:",
            options=['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller']
        )
        year_filter = st.slider("Filter by release year (films after):", min_value=1900, max_value=2025, step=1)
        score_filter = st.slider("Minimum IMDb score:", min_value=0.0, max_value=10.0, step=0.1)

    # Barra lateral para historial de películas seleccionadas
    with st.sidebar:
        st.write("#### Your selected movies:")
        if st.session_state.selected_movies:
            for movie_key in st.session_state.selected_movies:
                movie_row = movies_preprocessed[movies_preprocessed['id'] == int(movie_key.split("_")[1])]
                if not movie_row.empty:
                    title = movie_row.iloc[0]['title']
                    poster = movie_row.iloc[0]['OMDb_Poster']
                    st.image(poster if isinstance(poster, str) else "placeholder.jpg", width=100)
                    st.caption(title)

    # Mostrar películas de la página actual
    current_page = st.session_state.page_number
    current_movies = paginated_movies[current_page]

    cols = st.columns(3)
    for i, (_, row) in enumerate(current_movies.iterrows()):
        col = cols[i % 3]
        with col:
            st.image(row['OMDb_Poster'] if isinstance(row['OMDb_Poster'], str) else "placeholder.png", width=120)
            movie_key = f"{row['title']}_{row['id']}"
            is_selected = movie_key in st.session_state.selected_movies

            checkbox = st.checkbox(
                row['title'],
                key=movie_key,
                value=is_selected,
                on_change=update_selection,
                args=(movie_key,)
            )

    # *** Controles de navegación (botones Previous y Next) ***
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Previous", key="prev_button", disabled=current_page == 0):
            st.session_state.page_number = max(current_page - 1, 0)
    with col2:
        st.write(f"Page {current_page + 1} of {len(paginated_movies)}")
    with col3:
        if st.button("Next", key="next_button", disabled=current_page + 1 >= len(paginated_movies)):
            st.session_state.page_number = min(current_page + 1, len(paginated_movies) - 1)

    # Generar filtros
    filters = {}

    # Filtro por género
    if genre_filter:
        filters['genre'] = lambda x: any(x[genre] == 1 for genre in genre_filter)

    # Filtro por año (películas posteriores a year_filter)
    if year_filter:
        filters['OMDb_Year'] = lambda x: x >= year_filter

    # Filtro por puntuación
    if score_filter:
        filters['IMDB_Score'] = lambda x: x >= score_filter

    # Obtener recomendaciones en base a los filtros y selecciones
    selected_ids = [int(movie_key.split("_")[1]) for movie_key in st.session_state.selected_movies]
    
    if selected_ids:
        recommendations = recommend_movies(
            selected_ids, movies_preprocessed, tfidf_matrix, filters=filters)
    else:
        recommendations = pd.DataFrame()

    if recommendations.empty:
        st.write("No recommendations found based on your filters.")
    else:
        st.write("### Your Recommendations:")

        # Mostrar las recomendaciones de manera detallada
        for _, movie in recommendations.iterrows():
            cols = st.columns([1, 2])
            with cols[0]:
                st.image(movie['OMDb_Poster'] if isinstance(movie['OMDb_Poster'], str) else "placeholder.jpg", width=200)
            with cols[1]:
                st.write(f"**Title**: {movie['title']}")
                st.write(f"**Year**: {movie['OMDb_Year']}")
                st.write(f"**Plot**: {movie['OMDb_Plot']}")
                st.write(f"**Runtime**: {movie['OMDb_Runtime']} minutes")
                st.write(f"**IMDb Score**: {movie['IMDB_Score']}")


