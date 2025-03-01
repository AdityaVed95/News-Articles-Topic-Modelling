from fastapi import FastAPI, Request, Form, Depends, File, UploadFile, Cookie
from starlette.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
from schemas import Article
import spacy
import pandas as pd
from fastapi.responses import RedirectResponse, FileResponse

with open('input_all_articles.txt', 'r', encoding='utf8') as f:
    articles = f.read().split('@delimiter')

df = pd.read_csv("output_labelled_dataset.csv")

from gensim import models, corpora
from gensim import similarities

lda_model = models.LdaModel.load("./trained-model/lda_model.gensim")
dictionary = corpora.Dictionary.load("./trained-model/lda_dictionary.gensim")
corpus_bow = corpora.MmCorpus("./trained-model/lda_corpus.mm")
lda_index = similarities.MatrixSimilarity.load("./trained-model/lda_index.sim")

# topic names extracted from LLM :
topic_names = {
    0: "Family & Relationships",
    1: "Entertainment & Media",
    2: "Environmental & Natural Disasters",
    3: "International Affairs & Military",
    4: "Crime & Law Enforcement",
    5: "Thoughts & Opinions",
    6: "Health & Drugs Research",
    7: "Education & School Safety",
    8: "War & Terrorism",
    9: "Elections & Politics",
    10: "Economy & Business",
    11: "Transportation & Aviation",
    12: "Technology & Computing",
    13: "Art & Aesthetics   ",
    14: "Law & Legal Issues",
    15: "Urban Life & Infrastructure",
    16: "Healthcare & Medicine",
    17: "Social Movements & Protests",
    18: "Gender & Identity",
    19: "Sports & Competitions"
}

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

nlp = spacy.load('en_core_web_lg',disable=['parser','ner'])

def token_filter(tokenized_doc):
    filtered_tokens = []

    for token in tokenized_doc:
        if(token.is_alpha and token.pos_ in ['NOUN','VERB','ADJ'] and token.is_punct == False and token.is_space == False and token.is_stop == False):
            filtered_tokens.append(token.lemma_)
    
    # returns filtered_tokens of a particular doc object
    return filtered_tokens


def get_topics_of_article(article_id : str,min_topic_prob=0.1):
    topics = sorted(lda_model.get_document_topics(corpus_bow[article_id],minimum_probability=min_topic_prob), key=lambda tup: tup[1])[::-1]
    final_topics = []
    for topic in topics:
        final_topics.append([topic_names[topic[0]],round(topic[1]*100)])
    return final_topics

def get_top_words_of_topic(topic_id):
    result = lda_model.show_topic(topic_id)
    top_words = []
    for pair in result:
        top_words.append(pair[0])
    return top_words

def get_topic_id_from_topic_name(topic_name):
    for key,value in topic_names.items():
        if(value == topic_name):
            return key

def get_top_words_of_article(topics_of_article):
    top_words_of_article = []
    for item in topics_of_article:
        topic_name = item[0]
        topic_id = get_topic_id_from_topic_name(topic_name)
        top_words_of_topic = get_top_words_of_topic(topic_id)
        top_words_of_article.extend(top_words_of_topic)
    
    return top_words_of_article

def get_similar_articles(article_bow, first_m_words, top_n=5):
    similar_docs = lda_index[lda_model[article_bow]]
    top_n_docs = sorted(enumerate(similar_docs), key=lambda item: -item[1])[1:top_n+1]
    # Return a list of tuples with each tuple: (article id, similarity score, first m words of article)
    return list(map(lambda entry: (entry[0], round(entry[1]*100), articles[entry[0]][:first_m_words]), top_n_docs))

@app.get("/get_article_by_id/{article_id}")
def get_article(article_id : int, request:Request):
    article = articles[article_id]
    topics_of_article = get_topics_of_article(article_id)
    top_words = get_top_words_of_article(topics_of_article)

    article_bow = corpus_bow[article_id]
    similar_articles = get_similar_articles(article_bow,250,5)


    return templates.TemplateResponse("get_article_by_id.html",{"request":request,"article":article,"topics_of_article":topics_of_article,"top_words":top_words[:10],"similar_articles":similar_articles})

@app.get("/home")
def get_article(request:Request):
    return templates.TemplateResponse("home.html",{"request":request,"topic_names":topic_names})

@app.get("/get_articles_by_topic/{topic_id}")
def get_articles_by_topic(topic_id : int, request : Request):
    topic_name = topic_names[topic_id]
    # show first 100 articles on this topic
    limit = 100
    articles_ids = df[df["topic_name"] == topic_name]["article_id"].head(limit).tolist()
    articles_dict = {}
    # show first 250 words of each article
    for article_id in articles_ids:
        articles_dict[article_id] = articles[article_id][:250]
    
    return templates.TemplateResponse("get_articles_by_topic.html",{"request":request,"topic_name":topic_name,"articles_dict":articles_dict})

@app.get("/get_topic_of_new_article")
def get_topic_of_new_article(request:Request):
    # return FileResponse("vuejs-topic.html", media_type="text/html")
    return templates.TemplateResponse("get_topic_of_new_article.html",{"request":request})

@app.post("/api_get_topic_of_new_article")
def api_get_topic_of_new_article(article_object : Article, request : Request):
    article_dict = article_object.model_dump()
    article_content = article_dict["article_content"]
    article_tokens = list(map(token_filter, [nlp(article_content)]))[0]
    article_bow = dictionary.doc2bow(article_tokens)
    article_topics = sorted(lda_model.get_document_topics(article_bow), key=lambda tup: tup[1])[::-1]
    
    final_article_topics = []
    for topic_id, match_ratio in article_topics:
        if(match_ratio>0.1):
            final_article_topics.append([topic_id, topic_names[topic_id], round(match_ratio*100)])
        else:
            break
    
    # final_article_topics contains topic_id, topic_name, match%
    result = {"final_article_topics":final_article_topics}
    return result
    

@app.post("/api_get_similar_articles")
def api_get_similar_articles(article_object : Article, request : Request):
    article_dict = article_object.model_dump()
    article_content = article_dict["article_content"]
    article_tokens = list(map(token_filter, [nlp(article_content)]))[0]
    article_bow = dictionary.doc2bow(article_tokens)
    similar_articles = get_similar_articles(article_bow,250,5)
    result = {"similar_articles":similar_articles}
    return result



if __name__ == "__main__":
    uvicorn.run("main_app:app",reload=True,port=8001)
