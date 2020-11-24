import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

def process(DBLP):
    # Read from text file
    with open(DBLP, 'rb') as f:
        lines = f.readlines()
    token_count = 0
    sentence_count = 0
    for line in lines:
        # We ignore the last char, because it is newline
        token_count += len(line.decode()[:-1].replace('.', ' ').replace(',',' ').split())
        # We ignore the last two chars, because it is newline and maybe a dot.
        sentence_count += len(line.decode()[:-2].split('.'))
    print('The number of documents: %s  The number of sentences: %s  The number of words: %s' % (len(lines), sentence_count, token_count))
    word_list = []
    for line in lines:
        for word in line.decode()[:-1].replace('.', ' ').replace(',',' ').split():
            word_list.append(word)
    print('The number of unique words: '+ str(len(set(word_list))))
    document_length = []
    for line in lines:
        document_length.append(len(line.decode()[:-1].replace('.', '').replace(',','').split()))
    filter_length = []
    for l in document_length:
        if l < 400:
            filter_length.append(l)
    plt.hist(filter_length, bins=int(200/2))
    plt.savefig('dochist.png')
    plt.show()
    print("The number of outliers document: %i" % (len(document_length) - len(filter_length)))
    sentence_length = []
    for line in lines:
        for sentence in line.decode()[:-1].replace(',', ' ').split('.'):
            if len(sentence) != 0:
                sentence_length.append(len(sentence))
    filter_sentence_length = []
    for l in sentence_length:
        if l < 400:
            filter_sentence_length.append(l)
    plt.hist(sentence_length, bins=int(200/2))
    plt.savefig('sentencehist.png')
    plt.show()
    print("The number of outliers sentence: %i" % (len(sentence_length) - len(filter_sentence_length)))
    from collections import Counter
    c = Counter(list(set(word_list)))
    word_count = list(dict(c).values())
    filter_word_count = []
    for l in word_count:
        if l < 50:
            filter_word_count.append(l)

    plt.hist(filter_word_count, bins=int(50/2))
    plt.savefig('wordhist.png')
    plt.show()
    rare = np.count_nonzero(np.array(word_count) < 5)

    print("The number of rare tokens is: %i" % rare)

def create_df(single,multi,All):
    dblp_qmulti=pd.read_fwf(multi,header=None,names=['quality score','phrase'])
    dblp_qmulti['phrase length']=dblp_qmulti['phrase'].apply(lambda x : len(x.split()))
    dblp_qmulti=dblp_qmulti[dblp_qmulti['phrase length']>1]
    dblp_qsingle=pd.read_fwf(single,header=None,names=['quality score','phrase'])
    dblp_all=pd.read_fwf(All,header=None,names=['quality score','phrase'])
    dblp_all['phrase length']=dblp_all['phrase'].apply(lambda x : len(str(x).split()))
    return dblp_qsingle,dblp_qmulti,dblp_all

def histplot(outdir,dblp_qsingle,dblp_qmulti):
    f, axes = plt.subplots(1, 2,figsize=(15,5))
    sns.histplot(dblp_qmulti, x="quality score",stat="probability",bins=50,kde=True,ax=axes[1]).set_title('distribution multi-words phrase quality score')
    sns.histplot(dblp_qsingle, x="quality score",stat="probability",bins=50,kde=True,ax=axes[0]).set_title('distribution single-words phrase quality score')
    f.savefig(os.path.join(outdir, 'histogram.png'))

def boxplot(outdir,dblp_all):
    f, ax= plt.subplots(1, 1,figsize=(7.5,5))
    sns.boxplot(y="quality score", x="phrase length", data=dblp_all,ax=ax).set_title('boxplot phrase quality score vs phrase length')
    f.savefig(os.path.join(outdir, 'boxplot.png'))

def kdeplot(outdir,dblp_all):
    kde=sns.displot(dblp_all, x="quality score",hue='phrase length',kind='kde',palette="tab10")
    kde.savefig(os.path.join(outdir, 'kde.png'))

def edcfplot(outdir,dblp_all):
    ecdf=sns.displot(dblp_all, x="quality score",hue='phrase length',palette="tab10", kind="ecdf")
    ecdf.savefig(os.path.join(outdir,'ecdf.png'))

def generate_stats(outdir, **kwargs):
    os.makedirs(outdir, exist_ok=True)
    parent_dir=os.getcwd()
    DBLP=parent_dir+kwargs['DBLP']
    single=parent_dir+kwargs['single']
    multi=parent_dir+kwargs['multi']
    All=parent_dir+kwargs['all']
    process(DBLP)
    dblp_qsingle,dblp_qmulti,dblp_all=create_df(single,multi,All)
    histplot(outdir,dblp_qsingle,dblp_qmulti)
    boxplot(outdir,dblp_all)
    kdeplot(outdir,dblp_all)
    edcfplot(outdir,dblp_all)
    print("The charts are generated in " + outdir)    
