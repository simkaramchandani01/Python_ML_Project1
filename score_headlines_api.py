#!/usr/bin/env python
# coding: utf-8

# In[18]:


import logging


# In[11]:


from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import pipeline


# #### Logging

# In[19]:


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# #### Transformer Model

# In[20]:


logger.info("Loading sentiment analysis model...")
sentiment_analyzer = pipeline("sentiment-analysis")
logger.info("Model loaded successfully.")


# #### Create FastAPI

# In[13]:


app = FastAPI()


# #### Defining Model

# In[14]:


class HeadlinesRequest(BaseModel):
    headlines: List[str]


# #### Classification

# In[21]:


def classify_headline(headline: str) -> str:
    try:
        result = sentiment_analyzer(headline)[0]
        logger.debug(f"Headline: {headline} | Sentiment: {result}")
        if result["label"] == "POSITIVE":
            return "Optimistic"
        elif result["label"] == "NEGATIVE":
            return "Pessimistic"
        else:
            return "Neutral"
    except Exception as e:
        logger.error(f"Error processing headline: {headline}. Exception: {e}")
        return "Neutral"


# #### Status

# In[22]:


@app.get("/status")
def status():
    logger.info("Status check request received.")
    return {"status": "OK"}


# #### Score Headlines

# In[23]:


@app.post("/score_headlines")
def score_headlines(request: HeadlinesRequest):
    logger.info(f"Scoring request received with {len(request.headlines)} headlines.")
    labels = [classify_headline(headline) for headline in request.headlines]
    return {"labels": labels}

