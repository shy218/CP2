import numpy as np
import pandas as pd
import math
import re
import gensim
import matplotlib.pyplot as plt
import json

def dotest():
    test_config = json.load(open('config/test-params.json'))
    #Sample from the dataset DBLP
    df=pd.read_csv(test_config['auto_phrase'], sep='\t',error_bad_lines=False, header=None, names=["quality score", "phrase"])
    df=df[df['quality score']>0.5]

    np.random.seed(42)
    samples=np.random.choice(len(df), 100, replace=False)-1
    sample_df=df.iloc[samples].sort_values(by=['quality score'],ascending=False)
    sample_df=sample_df.reset_index()
    sample_df=sample_df.drop(columns=['index'])
    #Label the data manually
    result=pd.read_csv(test_config['result'])
    window=np.arange(0.5,1,0.01)

    #Calculate the Precision and Recall
    x=[]
    y=[]
    for i in window:
        TP=result[result['quality score']>i]['manually check'].sum()
        TP_FP=result[result['quality score']>i].shape[0]
        TP_FN=result['manually check'].sum()
        if math.isnan(TP/TP_FP):
            break
        y.append(TP/TP_FP)
        x.append(TP/TP_FN)
    x=x[::-1]
    #Plot the curve
    plt.plot(x,y)
    plt.axis('equal')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(test_config['pr_curve'])

    print('The Precision-Recall Curve is saved in PR.png')

    df_for_embed=df.copy()

    df_for_embed.phrase=df_for_embed.phrase.apply(lambda x: re.sub(r'([^\s])\s([^\s])', r'\1_\2',x))
    phrase_for_embed=df_for_embed.phrase.values
    phrase_for_embed=[[i] for i in phrase_for_embed]

    model = gensim.models.Word2Vec(
            phrase_for_embed,
            size=100,
            window=5,
            min_count=1,
            workers=4,
            iter=5)

    phrase=['parallel_programs','pedestrian_detection','lr_parsing']
    results=[]
    for i in phrase:
        results.append(model.wv.most_similar(positive=i,topn=5))

    print(result.head(20))
    print('Sample result of high-quality phrases')
