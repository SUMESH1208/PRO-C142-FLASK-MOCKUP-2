import pandas as pd

df = pd.read_csv('articles1.csv')
df = df[df['title'].notna()]

from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
countMatrix = count.fit_transform(df['title'])

from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(countMatrix, countMatrix)

df = df.reset_index()
indices = pd.Series(df.index, index=df['contentId'])

def get_recommendations(contentId):
    idx = indices[int(contentId)]
    sim_scores = list(enumerate(cosine_sim2[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    article_indices = [i[0] for i in sim_scores]
    return df[["url", "title", "text", "lang", "total_events"]].iloc[article_indices].values.tolist()