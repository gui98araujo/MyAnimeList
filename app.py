import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from PIL import Image
import requests

# Baixe os recursos necessários do NLTK
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Carregue o conjunto de dados
animes_df = pd.read_csv("animes.csv")

# Pré-processamento de texto
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(str(text).lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_tokens)

# Aplicação do pré-processamento à sinopse dos animes
animes_df['preprocessed_synopsis'] = animes_df['synopsis'].apply(preprocess_text)

# Vetorização usando TF-IDF
tfidf_vectorizer = TfidfVectorizer()
anime_tfidf_matrix = tfidf_vectorizer.fit_transform(animes_df['preprocessed_synopsis'])

# Função para obter os 5 animes mais prováveis para uma dada sinopse
def get_top_animes_for_synopsis(synopsis):
    preprocessed_synopsis = preprocess_text(synopsis)
    synopsis_tfidf_vector = tfidf_vectorizer.transform([preprocessed_synopsis])
    cosine_similarities_with_synopsis = linear_kernel(synopsis_tfidf_vector, anime_tfidf_matrix)
    similar_anime_indices = cosine_similarities_with_synopsis.argsort()[0][::-1][:5]
    return animes_df.iloc[similar_anime_indices][['title', 'score', 'synopsis', 'ranked', 'link', 'img_url']]

# Configuração do título do aplicativo Streamlit e remoção da barra lateral
st.set_page_config(page_title="Recomendação de Animes", page_icon="🎬", layout="wide")

# Widget para inserir a sinopse
example_synopsis = st.text_area("Insira a sinopse:", "In a world of magic and mystery, a young hero embarks on an epic journey.")

# Obtém os animes recomendados
recommended_animes = get_top_animes_for_synopsis(example_synopsis)

# Exibindo os resultados
st.subheader("Animes Recomendados para a Sinopse:")
for index, row in recommended_animes.iterrows():
    st.write(f"**Title:** {row['title']}")
    st.write(f"**Score:** {row['score']}")
    st.write(f"**Synopsis:** {row['synopsis']}")
    st.write(f"**Ranked:** {row['ranked']}")
    st.write(f"**Link:** {row['link']}")
    
    # Exibindo a imagem redimensionada
    image = Image.open(requests.get(row['img_url'], stream=True).raw)
    image.thumbnail((150, 150))  # Ajuste o tamanho desejado aqui
    st.image(image, caption='Anime Image', use_column_width=False)
    st.write("---")
