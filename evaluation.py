import pandas as pd
import os
import utils
import argparse
from sentence_transformers import SentenceTransformer
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_save_path", type=str)
    args = parser.parse_args()

    # 1. Load data
    path = "" 
    ds = pd.read_csv(path) 
    
    # 2. BM25 Part
    # 2.1. Load FAQ Question
    faq_db = pd.read_csv("")
    faq_db.ffill(axis=0, inplace=True)
    faq_question = faq_db[['Question']] #.tolist()

    # 2.2. Make BM25 Model
    tokenizer_type = 'bert'
    bm25 = utils.BM25(faq_question, tokenizer_type=tokenizer_type)
    bm25.apply_bm25()
    
    # 2.3. Load embedding model
    faq_answer = faq_db[['Answer']]#.tolist()
    embedding = utils.Embedding(model='sbert', name=args.model_save_path)
    vector_db = embedding.make_vector_DB(faq_answer)

    # 3. Evaluation
    total = 0

    bm25_top1_score = 0
    bm25_top1 = []
    bm25_top2 = []
    bm25_top3 = []
    
    vector_top1_score = 0
    vector_top2_score = 0
    vector_top3_score = 0
    
    vector_top1 = []
    vector_top2 = []
    vector_top3 = []
    
    for idx, row in ds.iterrows():
        user_question = row['user_query']

        total += 1

        # get BM25 score
        if tokenizer_type == 'okt':
            query_length = len(bm25.tokenizer.morphs(user_question)) 
            answers_scores = bm25.bm25.get_scores(bm25.tokenizer.morphs(user_question))
        elif tokenizer_type == 'bert':
            query_length = len(bm25.tokenizer.tokenize(user_question))
            answers_scores = bm25.bm25.get_scores(bm25.tokenizer.tokenize(user_question))
            
        sorted_indices = np.argsort(answers_scores)[::-1] # score 값에 대한 sorting 된 index 결과
        sorted_values = answers_scores[sorted_indices] # score 값에 대한 sorting 된 결과
        
        
        # bm25_top_3_idx = bm25.bm25.get_top_n(user_question, bm25.df['Question'].tolist(), n=3)
        '''
        if sorted_values[0]/query_length < 1: # user query가 길 수록 높은 점수 받기 때문에 norm 진행.
            bm25_top1.append('')
            bm25_top2.append('')
            bm25_top3.append('')           
        else:
            bm25_top1.append(faq_db.loc[sorted_indices[0]]['Answer'])
            bm25_top2.append(faq_db.loc[sorted_indices[1]]['Answer'])
            bm25_top3.append(faq_db.loc[sorted_indices[2]]['Answer'])
        ''' 
        bm25_top1.append(faq_db.loc[sorted_indices[0]]['Answer'])
        bm25_top2.append(faq_db.loc[sorted_indices[1]]['Answer'])
        bm25_top3.append(faq_db.loc[sorted_indices[2]]['Answer'])
        
        try:
            if bm25_top1[-1].strip() == row['Answer'].strip():
                bm25_top1_score += 1
        except AttributeError:
            pass
        
        # get embedding score
        vector_top_3_df = embedding.retrieval(vector_db, user_question, top_n=3)
        
        vector_top1.append(vector_top_3_df.iloc[0]['Answer'])
        vector_top2.append(vector_top_3_df.iloc[1]['Answer'])
        vector_top3.append(vector_top_3_df.iloc[2]['Answer'])
        
        vector_top3_list = [i.strip() for i in vector_top_3_df.Answer]
        
        try:
            if vector_top1[-1].strip() == row['Answer'].strip():
                vector_top1_score += 1
        except AttributeError:
            pass
            
        try:
            if vector_top2[-1].strip() == row['Answer'].strip() or vector_top1[-1].strip() == row['Answer'].strip():
                vector_top2_score += 1
        except AttributeError:
            pass

        try:
            if vector_top3[-1].strip() == row['Answer'].strip() or vector_top2[-1].strip() == row['Answer'].strip() or vector_top1[-1].strip() == row['Answer'].strip():
                vector_top3_score += 1
        except AttributeError:
            pass
        
        
        
    ds['vector_top1'] = vector_top1
    ds['vector_top2'] = vector_top2
    ds['vector_top3'] = vector_top3

    ds['bm25_top1'] = bm25_top1
    ds['bm25_top2'] = bm25_top2
    ds['bm25_top3'] = bm25_top3
    
    ds.to_csv('', index=False)

    print(f"BM25 Top1 Score: {bm25_top1_score/total}")
    print(f"Vector Top1 Score: {vector_top1_score/total}")
    print(f"Vector Top2 Score: {vector_top2_score/total}")
    print(f"Vector Top3 Score: {vector_top3_score/total}")
    