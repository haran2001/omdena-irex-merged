import json
import pickle
import re
from collections import Counter
from typing import Any, List, Optional, Type, Union

import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
# from google.colab import drive
from serpapi import GoogleSearch


from langchain import LLMChain, OpenAI, PromptTemplate, SerpAPIWrapper
from langchain.agents import (AgentExecutor, AgentType, LLMSingleActionAgent,
                              Tool, initialize_agent, load_tools)
from langchain.callbacks.manager import (AsyncCallbackManagerForToolRun,
                                          CallbackManagerForToolRun)
from langchain.chains import LLMChain, SequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate, StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import BaseTool
from langchain.utilities import (GoogleSearchAPIWrapper, GoogleSerperAPIWrapper,
                                  WikipediaAPIWrapper)
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch import nn
from transformers import BertModel

from faknow.model.layers.layer import TextCNNLayer
from faknow.model.model import AbstractModel
from faknow.data.dataset.text import TextDataset
from faknow.data.process.text_process import TokenizerFromPreTrained
from faknow.evaluate.evaluator import Evaluator

import torch
from torch.utils.data import DataLoader
import gdown
from pathlib import Path
import streamlit as st

import constants
import os


openai_api_key=constants.OPENAI_API_KEY
serper_ai_key=constants.SERPER_API_KEY

os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY
os.environ["SERPER_API_KEY"] = constants.SERPER_API_KEY


def verify_checkpoint(model_name, f_checkpoint, gID):
    if not f_checkpoint.exists():
        load_model_from_gd(model_name, gID)
    return f_checkpoint.exists()


def load_model_from_gd(model_name, gID):
    # save_dest = Path("models")
    save_dest = Path("assets/models")
    save_dest.mkdir(exist_ok=True)
    output = f"assets/models/{model_name}"
    # f_checkpoint = Path(f"models//{model_name}")
    with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
        gdown.download(id=gID, output=output, quiet=False)
        # gdown.download(f"https://drive.google.com/uc?id=1klOgwmAUsjkVtTwMi9Cqyheednf_U18n", output)


#model structure
class _MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 embed_dims: List[int],
                 dropout_rate: float,
                 output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(nn.Linear(input_dim, embed_dim))
            layers.append(nn.BatchNorm1d(embed_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """

        Args:
            x (Tensor): shared feature from domain and text, shape=(batch_size, embed_dim)

        """
        return self.mlp(x)


class _MaskAttentionLayer(torch.nn.Module):
    """
    Compute attention layer
    """
    def __init__(self, input_size: int):
        super(_MaskAttentionLayer, self).__init__()
        self.attention_layer = torch.nn.Linear(input_size, 1)

    def forward(self,
                inputs: Tensor,
                mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        weights = self.attention_layer(inputs).view(-1, inputs.size(1))
        if mask is not None:
            weights = weights.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(weights, dim=-1).unsqueeze(1)
        outputs = torch.matmul(weights, inputs).squeeze(1)
        return outputs, weights


class MDFEND(AbstractModel):
    r"""
    MDFEND: Multi-domain Fake News Detection, CIKM 2021
    paper: https://dl.acm.org/doi/10.1145/3459637.3482139
    code: https://github.com/kennqiang/MDFEND-Weibo21
    """
    def __init__(self,
                 pre_trained_bert_name: str,
                 domain_num: int,
                 mlp_dims: Optional[List[int]] = None,
                 dropout_rate=0.2,
                 expert_num=5):
        """

        Args:
            pre_trained_bert_name (str): the name or local path of pre-trained bert model
            domain_num (int): total number of all domains
            mlp_dims (List[int]): a list of the dimensions in MLP layer, if None, [384] will be taken as default, default=384
            dropout_rate (float): rate of Dropout layer, default=0.2
            expert_num (int): number of experts also called TextCNNLayer, default=5
        """
        super(MDFEND, self).__init__()
        self.domain_num = domain_num
        self.expert_num = expert_num
        self.bert = BertModel.from_pretrained(
            pre_trained_bert_name)
        self.embedding_size = self.bert.config.hidden_size
        self.loss_func = nn.BCELoss()
        if mlp_dims is None:
            mlp_dims = [384]

        filter_num = 64
        filter_sizes = [1, 2, 3, 5, 10]
        experts = [
            TextCNNLayer(self.embedding_size, filter_num, filter_sizes)
            for _ in range(self.expert_num)
        ]
        self.experts = nn.ModuleList(experts)

        self.gate = nn.Sequential(
            nn.Linear(self.embedding_size * 2, mlp_dims[-1]), nn.ReLU(),
            nn.Linear(mlp_dims[-1], self.expert_num), nn.Softmax(dim=1))

        self.attention = _MaskAttentionLayer(self.embedding_size)

        self.domain_embedder = nn.Embedding(num_embeddings=self.domain_num,
                                            embedding_dim=self.embedding_size)
        self.classifier = _MLP(320, mlp_dims, dropout_rate)

    def forward(self, token_id: Tensor, mask: Tensor,
                domain: Tensor) -> Tensor:
        """

        Args:
            token_id (Tensor): token ids from bert tokenizer, shape=(batch_size, max_len)
            mask (Tensor): mask from bert tokenizer, shape=(batch_size, max_len)
            domain (Tensor): domain id, shape=(batch_size,)

        Returns:
            FloatTensor: the prediction of being fake, shape=(batch_size,)
        """
        text_embedding = self.bert(token_id,
                                   attention_mask=mask).last_hidden_state
        attention_feature, _ = self.attention(text_embedding, mask)

        domain_embedding = self.domain_embedder(domain.view(-1, 1)).squeeze(1)

        gate_input = torch.cat([domain_embedding, attention_feature], dim=-1)
        gate_output = self.gate(gate_input)

        shared_feature = 0
        for i in range(self.expert_num):
            expert_feature = self.experts[i](text_embedding)
            shared_feature += (expert_feature * gate_output[:, i].unsqueeze(1))

        label_pred = self.classifier(shared_feature)

        return torch.sigmoid(label_pred.squeeze(1))

    def calculate_loss(self, data) -> Tensor:
        """
        calculate loss via BCELoss

        Args:
            data (dict): batch data dict

        Returns:
            loss (Tensor): loss value
        """

        token_ids = data['text']['token_id']
        masks = data['text']['mask']
        domains = data['domain']
        labels = data['label']
        output = self.forward(token_ids, masks, domains)
        return self.loss_func(output, labels.float())

    def predict(self, data_without_label) -> Tensor:
        """
        predict the probability of being fake news

        Args:
            data_without_label (Dict[str, Any]): batch data dict

        Returns:
            Tensor: one-hot probability, shape=(batch_size, 2)
        """

        token_ids = data_without_label['text']['token_id']
        masks = data_without_label['text']['mask']
        domains = data_without_label['domain']


        output_prob = self.forward(token_ids, masks,domains)

        return output_prob


#loading model weights
max_len, bert = 178 , 'dccuchile/bert-base-spanish-wwm-uncased'
tokenizer = TokenizerFromPreTrained(max_len, bert)

# dataset
batch_size = 64
domain_num = 12



# Info Extraction function
def info_extraction(subject,event, topic, length=50, api_key=serper_ai_key, min_search=20):
    # Define the mapping of topics to their prioritized sources
    topic_priority_map = {
        "Politics": {
            "POLITICAL PARTIES": {"name": "ACCIÓN CIUDADANA", "link": "https://accion-ciudadana.org/"},
            "TRANSPARENCY": {"name": "TRACODA", "link": "https://tracoda.info/"},
            "ELECTIONS": {"name": "VOTANTE", "link": "https://twitter.com/somosvotante"},
            "CORRUPTION": {"name": "ALAC", "link": "https://twitter.com/ALAC_SV"},
        },
        "Social": {
            "GENDER": {"name": "ORMUSA", "link": "https://ormusa.org/"},
            "VIOLENCE": {"name": "ASDEUH", "link": "https://asdehu.com/"},
            "ENVIRONMENT": {"name": "ACUA", "link": "https://www.acua.org.sv/"},
            "MIGRATION": {"name": "GMIES", "link": "https://gmies.org/"},
        },
        "Economy": {
            "BUDGET": {"name": "FUNDE", "link": "https://funde.org/"},
            "MACROECONOMY": {"name": "ICEFI", "link": "https://mail.icefi.org/etiquetas/el-salvador"},
        }
    }

    # Fetch summary from the internet using the info_extraction function
    params = {
        "engine": "google",
        "q": subject + event,
        "api_key": api_key
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results["organic_results"]

    # Flatten all priority sources for secondary ranking
    all_priority_sources = {info['link'] for _, approaches in topic_priority_map.items() for _, info in approaches.items()}
    topic_linked_sources = [info['link'] for approach, info in topic_priority_map.get(topic, {}).items()]

    # Initialize summary list with rank
    summary = []
    for result in organic_results[:min_search]:
        snippet = result['snippet']
        source_url = result['link']
        words = snippet.split()[:length]  # Select the first 'length' words
        truncated_snippet = ' '.join(words)

        # Rank determination
        if source_url in topic_linked_sources:
            rank = 1  # Highest priority: Directly linked to the topic
        elif source_url in all_priority_sources:
            rank = 2  # Secondary priority: In the priority map but not directly linked to the topic
        else:
            rank = 3  # Lowest priority: Not in the priority map

        # Append the summary with metadata including rank
        summary.append({
            "snippet": truncated_snippet,
            "source": source_url,
            "rank": rank
        })

    # Sort the summary list by 'rank'
    sorted_summary = sorted(summary, key=lambda x: x['rank'])

    return sorted_summary
# Build Agent Wrapper
class FakeDetectionWrapper:
    def __init__(self, client):
        self.client = client

class FilterAgent(FakeDetectionWrapper):
    def __init__(self, client):
        super().__init__(client)

        self.filter_agent = """ You are an agent that must label the subjetivity of news {news}, if the news is a personal opinion, impossible to verify (1), if
        the news is an objective statement (0) if the statement is about an event or fact that can potentially be verified with evidence,
        even if the evidence is not currently available in the news {news} provided.it expresses a personal opinion or cannot be verified objectively.

        present the label  in a JSON structure
        (
          "label": label,
        )
        """
        self.prompt_filter_agent = PromptTemplate(template=self.filter_agent, input_variables=["news"])

        # Corrected the LLMChain instantiation
        self.llm_chain_filter_agent = LLMChain(prompt=self.prompt_filter_agent, llm=self.client)

    def _run_filter_layer(self,news):
        self.news = news

        try:
            filter_layer_output = self.llm_chain_filter_agent.run({'news': self.news})
            return filter_layer_output
        except Exception as e:
            print(e)
            return "Error filter layer"

class ClassAgent(FakeDetectionWrapper):
    def __init__(self, client):
        super().__init__(client)

        self.class_agent = """ You are an agent with the task of identifying the elements subject of a news {news} .
        you will identify the subject, the event and the field the news belongs to Politics, Economics and Social.
        you will provide a Json Structure :
          (
          "subject":  subject of the news,
          "event": event described,
         "topic": field the news belongs to (Politics, Economics or Social
        )
        """
        self.prompt_class_agent = PromptTemplate(template=self.class_agent, input_variables=["news"])

        # Corrected the LLMChain instantiation
        self.llm_chain_class_agent = LLMChain(prompt=self.prompt_class_agent, llm=self.client)

    def _run_class_branch(self,news):
        self.news = news

        try:
            class_agent_output = self.llm_chain_class_agent.run({'news': self.news})
            return class_agent_output
        except Exception as e:
            print(e)
            return "Error class layer"

class SummaryAgent(FakeDetectionWrapper):
    def __init__(self, client):
        super().__init__(client)

        self.summary_agent = """Make a summary of the input {RAG_input} in 200 words, discard contradicting info
         and narrow it down to stable and verifiable data, avoid contradictory information in the text

        """
        self.prompt_summary_agent = PromptTemplate(template=self.summary_agent, input_variables=["RAG_input"],output_key="summary")

        # Corrected the LLMChain instantiation
        self.llm_chain_summary_agent = LLMChain(prompt=self.prompt_summary_agent, llm=self.client)

    def _run_summary_branch(self,RAG_input):
        self.RAG_input = RAG_input

        try:
            summary_agent_output = self.llm_chain_summary_agent.run({'RAG_input': self.RAG_input})
            return summary_agent_output
        except Exception as e:
            print(e)
            return "Error summary layer"


class DecisionAgent(FakeDetectionWrapper):
    def __init__(self, client):
        super().__init__(client)

        self.decision_agent = """You will be presented with a piece of news {news} and contextual information gathered from the internet {context}.
        Your task is to evaluate whether the {news} is genuine or not, based solely on:

        - Correlation with information from external sources {context},
        - The likelihood of it being false as determined by an ML algorithm {probability}, if it is higher than 0.6 it is highly likely to be fake.

        Based on these criteria, you must decide if the evidence supports the authenticity of the news. Your conclusion should include:

        "category": A label indicating whether the news is fake or real,
        "reasoning": A detailed explanation supporting your classification.

        """
        self.prompt_decision_agent = PromptTemplate(template=self.decision_agent, input_variables=["news","context","probability"])

        # Corrected the LLMChain instantiation
        self.llm_chain_decision_agent = LLMChain(prompt=self.prompt_decision_agent, llm=self.client)

    def _run_decision_branch(self,news,context, probability):
        self.news=news
        self.context=context
        self.probability=probability

        try:
            decision_agent_output = self.llm_chain_decision_agent.run({'news':self.news,'context':self.context,'probability':self.probability})
            return decision_agent_output
        except Exception as e:
            print(e)
            return "Error decision layer"

class ProcessPipeline:
    def __init__(self, client, news , path , tokenizer : TokenizerFromPreTrained ):
        self.filter_agent = FilterAgent(client)
        self.class_agent = ClassAgent(client)
        self.summary_agent = SummaryAgent(client)
        self.decision_agent = DecisionAgent(client)
        self.news = news
        self.client = client
        self.path = path
        self.tokenizer = tokenizer

    def process_news(self, use_filter=True):
        if use_filter:
            # Attempt to run the filter layer
            try:
                filter_result = self.filter_agent._run_filter_layer(self.news)
                print("Filter Result:", filter_result)
                # Assuming filter_result is a JSON string
                filter_result = json.loads(filter_result)
                if filter_result['label'] == 1:
                    return {"message": "News is subjective, process stopped."}
            except Exception as e:
                print(f"Error during filtering: {e}")
                # Optional: Decide if you want to halt processing or continue despite the error
                return {"error": "Failed to process the news at the filtering stage."}

        # Proceed with classification and summary
        try:
            class_result = self.class_agent._run_class_branch(self.news)
            print("Classification Result:", class_result)
            data = json.loads(class_result)
            subject = data['subject']
            event = data['event']
            topic = data['topic']
            print(f"Subject: {subject}, Event: {event}, Topic: {topic}")
        except Exception as e:
            print(f"Error during classification: {e}")
            return {"error": "Failed to classify the news."}

        try:
            summary = info_extraction(subject, event, topic)
            print('Search Results:', summary)
            context = self.summary_agent._run_summary_branch(summary)
            print('Summary:', context)
        except Exception as e:
            print(f"Error during summarization: {e}")
            return {"error": "Failed to summarize the news."}

        # Integrate ML model predictions (Placeholder for actual model integration)
        try:
            domains = {"Politics" : 8 ,
                       "Social" : 11,
                        "Economy" : 4
                       }
            data = {"text" : ["hello word"],
                    "domain" : [0] ,
                    "label": [0]
                    }
            labels = {0: "real",
                      1 :"Fake"}


            data["text"].append(self.news)
            data["domain"].append(domains[topic])
            data["label"].append(0)
            df = pd.DataFrame(data)
            df.to_json(self.path, orient='records')
            data = TextDataset(self.path, ['text'], self.tokenizer)
            batch_size = 3
            data_loader = DataLoader(data,batch_size , shuffle=False)
            outputs = []
            for batch_data in data_loader:
                outputs.append(MDFEND_MODEL.predict(batch_data))
            predictions = outputs[-1]
            output_prob = predictions[-1]
            predicted_probability = output_prob.item()
            predicted_label = round(predicted_probability)
            predicted_category  = labels[predicted_label]


            print(f"Predicted Category: {predicted_category}, Probability: {predicted_probability}")
        except Exception as e:
            print(f"Error during prediction: {e}")
            return {"error": "Failed to predict the news category."}

        # Make the final decision
        try:
            decision_result = self.decision_agent._run_decision_branch(self.news, context, predicted_probability)
            return {"decision": decision_result}
        except Exception as e:
            print(f"Error during decision-making: {e}")
            return {"error": "Failed to make a final decision on the news."}
        

##### INIT MODEL ###########
model5 = "last-epoch-model-2024-02-27-15_22_42_6.pth"
f_checkpoint = Path(f"assets/models//{model5}")
if verify_checkpoint(model5, f_checkpoint, "17KR1gHm85PfNJOxqTwdX3R5CaQbMR58c"):
    MODEL_SAVE_PATH = f"assets/models/last-epoch-model-2024-02-27-15_22_42_6.pth"
    MDFEND_MODEL = MDFEND(bert, domain_num, expert_num=15, mlp_dims=[2024, 1012, 606])
    MDFEND_MODEL.load_state_dict(
        torch.load(f=MODEL_SAVE_PATH, map_location=torch.device("cpu"))
    )

domain_num = 12

MDFEND_MODEL = MDFEND(bert, domain_num , expert_num=15 , mlp_dims = [2024 ,1012 ,606])


MDFEND_MODEL.load_state_dict(torch.load(f=MODEL_SAVE_PATH , map_location=torch.device('cpu')))

#####################