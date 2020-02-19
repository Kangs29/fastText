
# coding: utf-8

# In[ ]:

import os
import numpy as np
import time


os.chdir(r"C:\Users\조강\Desktop\Word2Vec\A. Data\OneBillionDataSet")
data_path = os.listdir()

def Ngram(word,n):
    ngrams=[]
    for i in range(len(word)-(n-1)):
        ngrams.append(word[i:i+n])
    return ngrams


def Ngram2_6(word):
    Ngrams=[]
    word = "#%s#" % word
    
    if len(word) == 3:
        for n in [2]:
            Ngrams+=Ngram(word,n)
    elif len(word) == 4:
        for n in [2,3]:
            Ngrams+=Ngram(word,n)
    elif len(word) == 5:
        for n in [2,3,4]:
            Ngrams+=Ngram(word,n)
    else:
        for n in [2,3,4,5]:
            Ngrams+=Ngram(word,n)

    Ngrams+=[word]
    
    return Ngrams



''' 오래 걸려서 봉인
print(" - N-gram Dictionary")
for num in range(len(data_path)):
    before=time.time()
    edit = open(data_path[num],'r',encoding='utf-8')
    for sentence in edit.readlines():
        
        for word in sentence.split():
            word = word.lower()
            
            ngrams=Ngram2_6(word)
            for ngram in ngrams:
                if not ngram in ngram_dict:
                    ngram_dict[ngram] = len(ngram_dict)
                    index_dict[len(index_dict)] = ngram 
                    freq_dict[ngram] = 1
                else:
                    freq_dict[ngram] += 1
    print(num,'/',len(data_path),' - Time :',int(time.time()-before),'s')
'''

print(" - N-gram Dictionary")
Edit_freq_dict = {}

before=time.time()
for num in range(len(data_path)):
    edit = open(data_path[num],'r',encoding='utf-8')
    for sentence in edit.readlines():
        
        for word in sentence.split():
            word = word.lower()
            
            if not word in Edit_freq_dict:
                Edit_freq_dict[word] = 1
            else:
                Edit_freq_dict[word] += 1
                
    if num % 32 == 0:
        print(num,'/',len(data_path),' - Time :',int(time.time()-before),'s')
        before=time.time()
        
min_count=5
print(" - Removing the word under frequency 5")

word_dict = {}
freq_dict = {}
word_index_dict = {}
ngram_dict = {}
index_dict = {}


for word in Edit_freq_dict:
    if Edit_freq_dict[word] < min_count:
        continue
    else:
        word_dict[word] = len(word_dict)
        word_index_dict[len(word_index_dict)] = word 

        freq_dict[word] = Edit_freq_dict[word]
        
        ngrams=Ngram2_6(word)
        for ngram in ngrams:
            if not ngram in ngram_dict:
                ngram_dict[ngram] = len(ngram_dict)
                index_dict[len(index_dict)] = ngram
                
print("The number of word (lower) :",len(word_dict))
print("The number of ngram (lower) :",len(ngram_dict))


# In[ ]:

import numpy as np
# Making Sigmoid Lookup Table
EXP_TABLE_SIZE = 1000
MAX_EXP = 6
SigmoidTable=[]
for i in range(EXP_TABLE_SIZE):
    exp_=np.exp(i/EXP_TABLE_SIZE*(MAX_EXP*2)-MAX_EXP)
    SigmoidTable.append(exp_/(exp_+1))
    
def SIGMOID(logit): # 시그모이드 테이블을 만드는 함수(효율적 코딩을 위해)
    global EXP_TABLE_SIZE
    global MAX_EXP
    global SigmoidTable
    
    index = int(((logit)+MAX_EXP)/(MAX_EXP*2)*EXP_TABLE_SIZE)
    if index >= EXP_TABLE_SIZE:
        index=EXP_TABLE_SIZE-1
    
    if index < 0:
        index=0
    
    return SigmoidTable[index]

def SubSamplingProb(t=1e-4):
    global freq_dict
    
    freq_sum=0
    for i in freq_dict:
        freq_sum+=freq_dict[i]

    Subsampling_prob=dict()
    for i in freq_dict:
        if 1-np.sqrt(t/(freq_dict[i]/freq_sum))>0:
            Subsampling_prob[i]=1-np.sqrt(t/(freq_dict[i]/freq_sum))
        else:
            Subsampling_prob[i]=0.0
            
    return Subsampling_prob

subsampling_prob = SubSamplingProb()

def UnigramTable():
    global freq_dict
    global word_index_dict
    
    power = 0.75
    sum_of_pows = 0
    for word in freq_dict:
        sum_of_pows += pow(freq_dict[word],power)


    table_size=int(1e8)
    unigram_table = [0]*table_size

    num=0
    word_idx=0
    cum_probability = pow(freq_dict[word_index_dict[word_idx]],power)/sum_of_pows


    for num in range(table_size):
        word = word_index_dict[word_idx]
        unigram_table[num]=word_dict[word]
        if (num/table_size)>cum_probability:
            word_idx+=1
            cum_probability+=pow(freq_dict[word_index_dict[word_idx]],power)/sum_of_pows
            
    return unigram_table

unigram_table = UnigramTable()


# In[ ]:

from ctypes import *
next_random=1
def RandomProbability():
    global next_random
    
    next_random=next_random*25214903917+11
    next_random=c_ulonglong(next_random).value # unsigned long long
    return int(bin(next_random & 0xFFFF),2)/0x10000 # (next_random & 0xFFFF)/ (float) 0x10000

def PairWord(sentence,position_):
    global R_window
    
    if len(sentence)>R_window:
        target=sentence[position_]
        if position_ < R_window:
            contexts=sentence[:position_+(R_window+1)]
            del contexts[-(R_window+1)]
        elif position_ > len(sentence)-R_window:
            contexts=sentence[position_-(R_window):]
            del contexts[R_window]
        else:
            contexts=sentence[position_-R_window:position_+R_window+1]
            del contexts[R_window]
    else:
        target=sentence[position_]
        contexts=sentence[:]
        del contexts[position_]

    return target, contexts

def SentenceSubsampling(sentence):
    global subsampling_prob
    global word_dict
    
    SubSentence=[]
    for word in sentence.split():
        word = word.lower()
        if word in word_dict:
            if subsampling_prob[word] >= RandomProbability():
                continue
            else:
                SubSentence.append(word_dict[word])
    return SubSentence

def RandomWindow():
    global window_size
    global next_random

    window_size=5
    next_random=next_random*25214903917+11
    next_random=c_ulonglong(next_random).value # unsigned long long

    window_sampled = next_random % window_size +1
    
    return window_sampled


# In[ ]:

word_to_ngram={}
for word in word_dict:
    sett=[]
    ns = Ngram2_6(word)
    for ns_ in ns:
        if ns_ in ngram_dict:
            sett.append(ngram_dict[ns_])
    word_to_ngram[word_dict[word]]=sett


# In[ ]:

# Spearson rank-order correlation
def Similarity(word1,word2):
    word1=word1.lower()
    word2=word2.lower()
    word1 = np.sum(W_in[NgramOOV(word1)],axis=0)
    word1_norm = np.sqrt(np.sum(np.square(word1)))

    word2 = np.sum(W_in[NgramOOV(word2)],axis=0)
    word2_norm = np.sqrt(np.sum(np.square(word2)))

    sim = np.dot(word1/word1_norm,word2/word2_norm)
    return sim

def Rank_Correlation(data):
    # data : [[word1,word2],score]
    score_1=[]
    for i in data:
        score_1.append(i[1])
        
    rank=1
    for i in np.argsort(score_1*np.array(-1)):
        data[i].append(rank)  # data : [[word1,word2],score,rank_1]
        rank+=1
        
    score_2=[]
    for i in data:
        word1,word2 = i[0]
        score_2.append(Similarity(word1,word2))
    
    rank=1
    for i in np.argsort(np.array(score_2)*np.array(-1)):
        data[i].append(rank)  # data : [[word1,word2],score,rank_1,rank_2]
        rank+=1

    d_rank=0
    for i in data:
        d_rank+=np.square(i[2]-i[3])
    
    return 1-d_rank*6/(len(data)*(len(data)**2-1))




# Word Analogy
test_data = open('C:/Users/조강/Desktop/Word2Vec/A. Data/Efficient Estimation of Word Representations in Vector Space dataset.txt','r',encoding='utf-8')
raw=[]
for lines in test_data.readlines()[1:]:
    raw.append(lines)
    
test_pair=[]
for lines in raw:
    if ':' in lines:
        continue
    else:
        test_pair.append(lines.split())

semantic = test_pair[:8869] 
syntatic = test_pair[8869:]

wor=dict()
dd=dict()
for i in word_dict:
    if freq_dict[i] >1000:
        wor[i]=len(wor)
        dd[len(dd)]=i
len(wor)


input_weight=[]


for i in wor:
    input_weight.append(np.sum(W_in[word_to_ngram[word_dict[i]]],axis=0))
input_weight=np.array(input_weight)

norm_all = np.sqrt(np.sum(np.square(input_weight), 1, keepdims=True))
all_ = input_weight/norm_all


def NgramOOV(word):
    OVVset=[]
    for case in Ngram2_6(word):
        if case in ngram_dict:
            OVVset.append(ngram_dict[case])
            
    return OVVset


def Eval(name,pair_data,all_):

    score=0
    not_=0

    running=0

    for word1, word2, word3, word4 in pair_data:
        
        word1 = word1.lower()
        word2 = word2.lower()
        word3 = word3.lower()
        word4 = word4.lower()
                      
        running+=1
        if word1 in word_dict:
            set1 = word_to_ngram[word_dict[word1]]
        else:
            set1 = NgramOOV(word1)

            
        if word2 in word_dict:
            set2 = word_to_ngram[word_dict[word2]]
        else:
            set2 = NgramOOV(word2)
            
        if word3 in word_dict:
            set3 = word_to_ngram[word_dict[word3]]
        else:
            set3 = NgramOOV(word3)
            
        if word4 in word_dict:
            set4 = word_to_ngram[word_dict[word4]]
        else:
            set4 = NgramOOV(word4)

        testing = np.sum(W_in[set2],axis=0)                  -np.sum(W_in[set1],axis=0)                  +np.sum(W_in[set3],axis=0)

        norm_testing = np.sqrt(np.sum(np.square(testing)))
        test = testing/norm_testing

        Cosine = np.dot(all_,test)

        sorting = np.argsort(Cosine*np.array(-1))[:4]
        top_word=[]
        
        for top_ in sorting:
            top_word.append(dd[top_].lower())

        if word4.lower() in top_word:
            score+=1
            #print('%d / %d' % (score,running))



    print(" %s Test - %03f %%" % (name, score/len(pair_data)*100))
    print("    -> CAN'T TESTING (NOT WORD) :",not_)
    print("    -> Adjusting Test : %03f %%" % ((score)/(len(pair_data)-not_)*100))
#Eval("Semantic",semantic,all_)
#Eval("Syntatic",syntatic,all_) 


# In[ ]:

D = 300 # embedding
N = len(ngram_dict)
V = len(word_dict)

W_in = np.random.uniform(-0.01,0.01,(N,D))
W_out = np.random.uniform(-0.01,0.01,(V,D)) # random.randn으로하면 안된다.

import random
num_negative=5
learning_rate=0.025
lamba=0.0075
word_count=0

data_path = os.listdir()

shuffle_File=[num for num in range(len(data_path))]
random.shuffle(shuffle_File)


NUMBER_DATA=0
for num_path in shuffle_File[NUMBER_DATA:]:
    Data_RunningTime=time.time()
    print("Training Data < %d > - %s / %s Data Path" % (NUMBER_DATA,num_path,len(data_path)))

    data_name = data_path[num_path]
    data_raw = open(data_name,'r',encoding='utf-8')

    data=[]
    for lines in data_raw.readlines():
        data.append(lines)

    random.shuffle(data)

    sss=0
    t1=time.time()
    
    for lines in data:
        sentence = SentenceSubsampling(lines)
        sss+=1
        # Update lr

        for position_ in range(len(sentence)):
            R_window = RandomWindow()
            target, contexts = PairWord(sentence,position_)

            for context in contexts:
                word_count+=1

                negative_sample=[]
                for _ in range(num_negative):
                    next_random=next_random*25214903917+11
                    next_random=c_ulonglong(next_random).value # unsigned long long
                    negative_sample.append(unigram_table[int(bin(next_random >> 16),2) % table_size])

                negative_sample

                output_indexs=[context]+negative_sample

                # forward
                target_ngram=word_to_ngram[target]
                hidden = np.sum(W_in[target_ngram],axis=0)
                output = np.matmul(W_out[output_indexs],hidden)
                logit = [SIGMOID(out) for out in output]


                # backward
                logit[0]-=1
#                W_in[target_ngram] -= learning_rate*(np.matmul(W_out[output_indexs].T,logit))
#                W_out[output_indexs] -= learning_rate*(np.outer(logit,hidden))
#
                W_in[target_ngram] -= learning_rate*(np.matmul(W_out[output_indexs].T,logit)                                       +lamba*W_in[target_ngram])
                W_out[output_indexs] -= learning_rate*(np.outer(logit,hidden)+lamba*W_out[output_indexs])

        if word_count>300000:
            learning_rate*=0.9995
            word_count=0

        if sss % 100000 == 0:
            print(sss,'/',len(data))
    
    print("LEARNIGN_RATE :",learning_rate)
    print(int(time.time()-t1),'의 시간이 걸립니다.')
    NUMBER_DATA+=1
    
    input_weight=[]


    for i in wor:
        input_weight.append(np.sum(W_in[word_to_ngram[word_dict[i]]],axis=0))
    input_weight=np.array(input_weight)

    norm_all = np.sqrt(np.sum(np.square(input_weight), 1, keepdims=True))
    all_ = input_weight/norm_all

    Eval("Semantic",semantic,all_)
    Eval("Syntatic",syntatic,all_)   
    
    
    # WS353
    f=open(r'C:\Users\조강\Desktop\FastText\A. Data\WS353.txt','r',encoding='utf-8')
    ws=[]
    for i in f.readlines():
        s=i.split(';')
        ws.append([[s[1],s[2]],float(s[3])])

    print("WS353 Correlation :",Rank_Correlation(ws))
    
    
    # RW
    f=open(r'C:\Users\조강\Desktop\FastText\A. Data\RW_test.txt','r',encoding='utf-8')
    rw=[]
    for i in f.readlines():
        s=i.split('\t')
        rw.append([[s[0],s[1]],float(s[2])])
    print("RW Correlation :",Rank_Correlation(rw))


# In[ ]:

# 계산방식 ( word 연산 - semantic )
#  2-1+3 -- 4 = 67.2
#  2-1+4 -- 3 = 40
#  4-3+1 -- 2 = 67.6

