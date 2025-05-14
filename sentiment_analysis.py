from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# 1. Text Preprocessing
##################################################

df_= pd.read_csv("dataset/womensClothingReviews.csv", sep=",")
df = df_.copy()
df.head()


df.isna().sum()

#Title verisinde 3810 tane boş veri bulunuyor. Yani kullanıcılar genellikle title kısmını girmemişler veya yorum yapmamışlar.
# Bu sebeple verimizden title kısmını atabiliriz.

df = df.drop(['Title', "Unnamed: 0"], axis=1)
df.isna().sum()

df = df.dropna(subset=["Review Text", "Division Name", "Department Name", "Class Name"])
df.isna().sum()
df.head()

df = df.reset_index(drop=True)

# Müşteri yaş aralığına bakalım

fig = px.histogram(df['Age'], marginal='box',
                   labels={'value': 'Age'})

fig.update_traces(marker=dict(line=dict(color='#87ceff', width=2)))
fig.update_layout(title_text='Distribution of the Age of the Customers',
                  title_x=0.5, title_font=dict(size=20))
fig.show()

# Yaşa göre ratinglere bakalım

fig = px.histogram(df['Age'], color=df['Rating'],
                   labels={'value': 'Age',
                           'color': 'Rating'}, marginal='box')
fig.update_traces(marker=dict(line=dict(color='#87ceff', width=2)))
fig.update_layout(title_text='Distribution of the Age and Rating',
                  title_x=0.5, title_font=dict(size=20))
fig.update_layout(barmode='overlay')
fig.show()

# Rating recommend arasındaki ilişkiye bakalım.

fig = px.histogram(df['Rating'], color=df['Recommended IND'],
                   labels={'value': 'Rating',
                           'color': 'Recommended?'})
fig.update_layout(bargap=0.2)
fig.update_traces(marker=dict(line=dict(color='#87ceff', width=2)))
fig.update_layout(title_text='Relationship between Ratings and Recommendation',
                  title_x=0.5, title_font=dict(size =20))
fig.update_layout(barmode='group')
fig.show()

df.head()

# REGULAR EXPRESSION
# 1.Metinlerdeki büyük-küçük harfleri standart formata getir.
# 2. Noktalama işaretlerini kaldır
# 3. Sayıları kaldır(işimize yaramayacaksa)
# 4. Sosyal medya verisinde emojiler,verilen linkler kaldırılabilir.( işe yaramayacaksa)

# Normalizing Case Folding

df['Review Text'] = df['Review Text'].str.lower()
df['Review Text'].head()

# Punctuations - noktalama işaretlerini kaldırıyoruz.

df['Review Text'] = df['Review Text'].str.replace('[^\w\s]', '')

# Numbers
df['Review Text'] = df['Review Text'].str.replace('\d+', '') # burada d+ ifadesi ile sayıları siliyoruz.

# Stopwords - Bu kavram dilde kullanılan ancak herhangi bir anlamı olmayan yaygın ifadelerdir. is-of-the vb
#verimiz ing olduğu için eng stopwordleri almamız lazım


# nltk.download('stopwords')

sw = stopwords.words('english')

df['Review Text'] = df['Review Text'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

# Rarewords- Bu ise metindeki nadir kelimeleri bulmamıza yarar.Bunları veriden çıkararak daha verimli çalışma sağlayabiliriz.
#Önce kelimenin frekansını buluyoruz. Bu veriyi geçici bir dataframee atıyoruz. Az geçenleri drop ediyoruz.

temp_df = pd.Series(' '.join(df['Review Text']).split()).value_counts()

drops = temp_df[temp_df <= 1]

df['Review Text'] = df['Review Text'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

# Tokenization

#nltk.download("punkt")

df["Review Text"].apply(lambda x: TextBlob(x).words).head()

# Lemmatization - Kelimeleri köklerine ayırarak -s, -ies takısı gibi takıları kaldırıyoruz.

# import nltk
# nltk.download('wordnet')

df['Review Text'] = df['Review Text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
df['Review Text'].head()

# 2. Text Visualization - Kelimeleri sayısal anlamları olabilecek hale getirebiliriz.

# Terim Frekanslarının Hesaplanması

tf = df["Review Text"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

tf.columns = ["words", "tf"]

tf.sort_values("tf", ascending=False)

# Barplot
tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()

df.head()

# Wordcloud

text = " ".join(i for i in df["Review Text"])

wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud.to_file("wordcloud.png")

# 3. Sentiment Analysis

df["Review Text"].head()

# nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

df["Review Text"][0:10].apply(lambda x: sia.polarity_scores(x))

df["Review Text"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])

df["polarity_score"] = df["Review Text"].apply(lambda x: sia.polarity_scores(x)["compound"])

# 4. Feature Engineering

df["Review Text"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df["sentiment_label"] = df["Review Text"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df["sentiment_label"].value_counts()

df.head()

df.groupby("sentiment_label")["Rating"].mean()

df["sentiment_label"] = LabelEncoder().fit_transform(df["sentiment_label"])

y = df["sentiment_label"]
X = df["Review Text"]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


vectorizer = TfidfVectorizer()
X_vect = vectorizer.fit_transform(X)

# TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf_word_vectorizer = TfidfVectorizer()
X_tf_idf_word = tf_idf_word_vectorizer.fit_transform(X)


tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range=(2, 3))
X_tf_idf_ngram = tf_idf_ngram_vectorizer.fit_transform(X)

# 5. Sentiment Modeling

# Logistic Regression
###############################

log_model = LogisticRegression().fit(X_tf_idf_word, y)

cross_val_score(log_model,
                X_tf_idf_word,
                y,
                scoring="accuracy",
                cv=5).mean()


new_review = pd.Series("this product is great")
new_review = pd.Series("look at that shit very bad")
new_review = pd.Series("it was good but I am sure that it fits me")

new_review = TfidfVectorizer().fit(X).transform(new_review)

log_model.predict(new_review)

random_review = pd.Series(df["Review Text"].sample(1).values)

new_review = TfidfVectorizer().fit(X).transform(random_review)

log_model.predict(new_review)

df.tail()


# Random Forests

# TF-IDF Word-Level
rf_model = RandomForestClassifier().fit(X_tf_idf_word, y)
cross_val_score(rf_model, X_tf_idf_word, y, cv=5, n_jobs=-1).mean()

# TF-IDF N-GRAM
rf_model = RandomForestClassifier().fit(X_tf_idf_ngram, y)
cross_val_score(rf_model, X_tf_idf_ngram, y, cv=5, n_jobs=-1).mean()

# Hiperparametre Optimizasyonu
###############################

rf_model = RandomForestClassifier(random_state=17)

rf_params = {"max_depth": [8, None],
             "max_features": [7, "sqrt"],
             "min_samples_split": [2, 5, 8],
             "n_estimators": [100, 200]}

rf_best_grid = GridSearchCV(rf_model,
                            rf_params,
                            cv=5,
                            n_jobs=-1,
                            verbose=1).fit(X_vect, y)


rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X_vect, y)


cross_val_score(rf_final, X_vect, y, cv=5, n_jobs=-1).mean()