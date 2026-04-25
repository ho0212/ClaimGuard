import os
import base64
from typing import TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langgraph.graph import StateGraph, START, END