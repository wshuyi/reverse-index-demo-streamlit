#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import jieba
import gensim
import jieba.analyse


def get_sample_docs_from_files():
    docs = []
    for i in range(3):
        filename = f"doc{i+1}.txt"
        with open(filename) as f:
            data = f.read()
            docs.append(data)
    return docs



def get_stopwords_list():
    filename = "stopwords.txt"
    with open(filename) as f:
        data = f.read()
    stopwords = data.split('\n')
    return stopwords


def preprocessing(docs, stopwords):
    texts = []

    for doc in docs:
        texts.append(list(jieba.cut(doc)))

    mydict = gensim.corpora.Dictionary(texts)

    myword_frec_list = build_myword_frec_list(mydict,texts)
        
    position_list = build_word_pos_dict(docs)
    
    reversed_index = build_reverse_index(mydict, myword_frec_list, stopwords)
        
    return mydict, myword_frec_list, position_list, reversed_index



def build_myword_frec_list(mydict, texts):
    
    corpus = [mydict.doc2bow(doc) for doc in texts]
    
    myword_frec_list = [] 

    for sentence in corpus:
        my_word_frec = {}
        for item in sentence:
            k = item[0]
            v = item[1]
            my_word_frec[k] = v
        myword_frec_list.append(my_word_frec)
    return myword_frec_list


def get_word_doc_freq_dict(word, mydict, myword_frec_list):
    myid = mydict.token2id[word]
    doc_freq_dict = {}
    for i in range(len(myword_frec_list)):
        current_doc = myword_frec_list[i]
        try:
            freq = current_doc[myid]
            doc_freq_dict[i] = freq
        except:
            pass
    return doc_freq_dict


def build_word_pos_dict(docs):
    position_list = []

    for i in range(len(docs)):

        tokenize_result = list(jieba.tokenize(docs[i]))

        position_dict = {}
        for item in tokenize_result:
            word = item[0]
            start = item[1]
            end = item[2]
            if word in position_dict:
                position_dict[word].append((start, end))
            else:
                position_dict[word] = [(start, end)]
        position_list.append(position_dict)
        i = i + 1 # increase index
        
    return position_list


# build reverse index
def build_reverse_index(mydict, myword_frec_list, stopwords):
    reversed_index = {}
    word_list = list(mydict.values())
    for word in word_list:
        if word not in stopwords:
            reversed_index[word] = list(get_word_doc_freq_dict(word, mydict, myword_frec_list).keys())
    return reversed_index


def search_word_position_in_docs(docs, word, position_list):
    position_result = {}
    for i in range(len(docs)):
        pos_dict = position_list[i]
        if word in pos_dict:
            position_result[i] = pos_dict[word]
    return position_result


def get_reversed_index(word, reversed_index):
    return reversed_index[word]


def display_reversed_index_pretty(reversed_index, words=None):
    if not words:
        words = list(reversed_index.keys())

    for word in words:
        st.write(f"{word} shows in document {get_reversed_index(word, reversed_index)}")



def pretty_print_single_word(word, docs, mydict, myword_frec_list, position_list, reversed_index):
    if word in reversed_index:
        reversed_index_list = get_reversed_index(word, reversed_index)
        
        word_position_dict = search_word_position_in_docs(docs, word, position_list)
        word_doc_freq_dict = get_word_doc_freq_dict(word, mydict, myword_frec_list)
        
        st.write(f"{word} shows in document {reversed_index_list}")
        for idx in reversed_index_list:
            st.write(f"{word} shows in document {idx}: {word_doc_freq_dict[idx]} times")
            st.write(f"The positions are: {word_position_dict[idx]}")
    else:
        st.write(f"Error! {word} does not show in any document!")
        


def get_keywords_tfidf(docs, topK=10):
    whole_keywords = []
    for i in range(len(docs)):
        keywords =  jieba.analyse.extract_tags(docs[i], topK)
        for keyword in keywords:
            if keyword not in whole_keywords:
                whole_keywords.append(keyword)
    return whole_keywords





docs = get_sample_docs_from_files()

docs[0] = st.text_area('Input Doc Text 0', docs[0])
docs[1] = st.text_area('Input Doc Text 1', docs[1])
docs[2] = st.text_area('Input Doc Text 2', docs[2])

# st.write(docs[1])



stopwords = get_stopwords_list()

stopwords = st.text_area('stop words list', '\n'.join(stopwords)).split('\n')

mydict, myword_frec_list, position_list, reversed_index = preprocessing(docs, stopwords) 

# if st.button('analyze'):
#     mydict, myword_frec_list, position_list, reversed_index = preprocessing(docs, stopwords)
#     st.write('analyze ... done!')


only_keywords = st.sidebar.checkbox('only top keywords?')

if st.sidebar.button('show reversed index:'):
    if only_keywords:
        keywords = get_keywords_tfidf(docs)
        # display_reversed_index_pretty(reversed_index, keywords)
    else:
        keywords = None
    display_reversed_index_pretty(reversed_index, keywords)
    # st.write('analyze ... done!')
    
    

word = st.sidebar.text_input('Input the word you are interested')
if st.sidebar.button('Analyze the specific word:'):
    # mydict, myword_frec_list, position_list, reversed_index = preprocessing(docs, stopwords)
    pretty_print_single_word(word, docs, mydict, myword_frec_list, position_list, reversed_index)


# word = "我们"
# pretty_print_single_word(word, docs, mydict, myword_frec_list, position_list, reversed_index)





# word = "。"
# pretty_print_single_word(word, docs, mydict, myword_frec_list, position_list, reversed_index)





# word = "的"
# pretty_print_single_word(word, docs, mydict, myword_frec_list, position_list, reversed_index)





# 





# 






