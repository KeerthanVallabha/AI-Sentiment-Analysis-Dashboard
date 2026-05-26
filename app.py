import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from transformers import pipeline
from wordcloud import WordCloud

################################################
# PAGE SETTINGS
################################################

st.set_page_config(
page_title="Sentiment Dashboard",
layout="wide"
)

st.title(
"📊 Social Media Sentiment Analysis Dashboard"
)

st.write(
"AI-powered sentiment analysis using NLP + BERT"
)

################################################
# LOAD BERT MODEL
################################################

@st.cache_resource
def load_model():

    return pipeline(
        "sentiment-analysis"
    )

model=load_model()

################################################
# UPLOAD DATA
################################################

uploaded=st.file_uploader(
"Upload CSV",
type=["csv"]
)

if uploaded:

    df=pd.read_csv(
        uploaded,
        encoding="latin-1",
        header=None
    )

    df.columns=[
    "target",
    "id",
    "date",
    "flag",
    "user",
    "text"
    ]

    df=df.head(1000)

################################################
# SENTIMENT PREDICTION
################################################

    sentiments=[]

    scores=[]

    for txt in df["text"]:

        try:

            result=model(
            str(txt)[:512]
            )[0]

            sentiments.append(
            result["label"]
            )

            scores.append(
            result["score"]
            )

        except:

            sentiments.append(
            "Neutral"
            )

            scores.append(0)

    df["Sentiment"]=sentiments

    df["Confidence"]=scores

################################################
# METRICS
################################################

    c1,c2,c3=st.columns(3)

    c1.metric(
    "Rows",
    len(df)
    )

    c2.metric(
    "Positive",
    sum(
    df["Sentiment"]=="POSITIVE"
    )
    )

    c3.metric(
    "Negative",
    sum(
    df["Sentiment"]=="NEGATIVE"
    )
    )

################################################
# DATA
################################################

    st.subheader(
    "Analyzed Data"
    )

    st.dataframe(
    df[[
    "text",
    "Sentiment",
    "Confidence"
    ]]
    )

################################################
# BAR GRAPH
################################################

    fig=px.histogram(
        df,
        x="Sentiment",
        color="Sentiment",
        title="Sentiment Distribution"
    )

    st.plotly_chart(
    fig,
    use_container_width=True
    )

################################################
# PIE CHART
################################################

    pie=px.pie(
    df,
    names="Sentiment",
    title="Percentage"
    )

    st.plotly_chart(
    pie
    )

################################################
# CONFIDENCE GRAPH
################################################

    fig=px.box(
    df,
    x="Sentiment",
    y="Confidence"
    )

    st.plotly_chart(
    fig
    )

################################################
# WORD CLOUD
################################################

    st.subheader(
    "Word Cloud"
    )

    text=" ".join(
    df["text"]
    )

    wc=WordCloud(
    width=900,
    height=400,
    background_color="white"
    ).generate(text)

    fig,ax=plt.subplots()

    ax.imshow(wc)

    ax.axis("off")

    st.pyplot(fig)

################################################
# SEARCH
################################################

    keyword=st.text_input(
    "Search Tweets"
    )

    if keyword:

        filtered=df[
        df["text"]
        .str.contains(
        keyword,
        case=False,
        na=False
        )
        ]

        st.write(filtered)

################################################
# DOWNLOAD
################################################

    csv=df.to_csv(
    index=False
    )

    st.download_button(
    "Download Results",
    csv,
    "analysis.csv"
    )