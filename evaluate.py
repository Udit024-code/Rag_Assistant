# add 10 questions to test the retrieval system

import pickle

import os
from rag_app import rag_pipeline
from sentence_transformers import SentenceTransformer

from retrieval import retrieve
import faiss

TEST_SET = [
    {
        'question': 'What are the three structural blind spots that leave the AWE question unanswered in India?',
        'expected_keywords': ['aggregate', 'long-run', 'heterogeneity', 'persistence'],
        'expected_pages': [4]
    },
    {
        'question': 'What dataset is used in this study, and how many households does it track?',
        'expected_keywords': ['cphs', 'cmie', '174,000', '80,000'],
        'expected_pages': [6, 7]
    },
    {
        'question': 'What are the three core research questions (RQs) defined in the study framework?',
        'expected_keywords': ['exist', 'vary', 'persist', 're-employment'],
        'expected_pages': [4, 5]
    },
    {
        'question': 'How does the study use the COVID-19 pandemic as a natural experiment to eliminate selection bias?',
        'expected_keywords': ['lockdown', 'exogenous', 'unemployment', 'bias'],
        'expected_pages': [1, 7]
    },
    {
        'question': 'What is the specific percentage point gap between India\'s FLFPR and Vietnam\'s rate according to the global benchmarking data?',
        'expected_keywords': ['48.1', '72.1%', 'third'],
        'expected_pages': [2]
    },
    {
        'question': 'According to the data from Deshpande (2020), what percentage of women worked at least once in 4 years?',
        'expected_keywords': ['44%', '44', 'deshpande', 'once'],
        'expected_pages': [8]
    },
    {
        'question': 'What econometric model is proposed to test the persistence of the Added Worker Effect after male employment recovers?',
        'expected_keywords': ['difference-in-differences', 'did', 'three periods'],
        'expected_pages': [4, 6, 7]
    },
    {
        'question': 'According to the McKinsey Global Institute framework, how much could India add to its GDP through gender parity?',
        'expected_keywords': ['0.7t', '16%', 'mckinsey', 'parity'],
        'expected_pages': [10]
    },
    {
        'question': 'How is the existing Indian FLFP literature thematic focus distributed compared to this study\'s design?',
        'expected_keywords': ['skewed', 'structural', '45%', 'equal-weight'],
        'expected_pages': [9]
    },
    {
        'question': 'What is the critical policy implication if an increase in FLFP during a crisis is confirmed to be driven by the Added Worker Effect?',
        'expected_keywords': ['distress-entry', 'vulnerability', 'cyclical', 'multipliers'],
        'expected_pages': [10]
    }
]

def evaluate_retrieval(test_set, index, chunks, model):
    hits = 0
    for test in test_set:
        # Run your existing retrieval logic
        results = retrieve(test['question'], index, chunks, model, top_k=3)
        retrieved_pages = [r['page'] for r in results]
        
        # Check if ANY expected page was retrieved
        if any(p in retrieved_pages for p in test['expected_pages']):
            hits += 1
            
    return hits / len(test_set)  # recall score 0.0 to 1.0

index = faiss.read_index('chunks.index')
with open('chunks_metadata.pkl', 'rb') as f:
    chunks = pickle.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')
evaluation_score = evaluate_retrieval(TEST_SET, index, chunks, model)
print(f"Retrieval Evaluation Score: {evaluation_score:.2f}")

def evaluate_answers(test_set, index, chunks):
    hits = 0
    for test in test_set:
        # Run the full end-to-end pipeline (returns answer + retrieved chunks)
        answer, _sources = rag_pipeline(test['question'], index, chunks)
        answer = answer.lower()
        
        # Check if ANY of the expected keywords appear in the generated answer
        if any(kw.lower() in answer for kw in test['expected_keywords']):
            hits += 1
            
    return hits / len(test_set)  # answer quality score 0.0 to 1.0


# Run the end-to-end answer quality evaluation
generation_score = evaluate_answers(TEST_SET, index, chunks)
print(f"End-to-End Generation Score: {generation_score * 100:.1f}%")