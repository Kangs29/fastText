
# coding: utf-8

# In[2]:

import os
import re
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip()

os.chdir(r'C:\Users\조강\Desktop\FastText\A. Data')

# AG Data hyperparamters
AG=dict()
AG["Max_length(Unigram)"]=50
AG["Max_length(Bigram)"]=100
AG["Path"]="ag_news_csv"
AG["Class"] = 4
AG["Learning_rate"]=0.0005
AG["Batch"]=256

# Sogou Data hyperparamters
Sogou=dict()
Sogou["Max_length(Unigram)"]=1000
Sogou["Max_length(Bigram)"]=1500
Sogou["Path"]="sogou_news_csv"
Sogou["Class"]=5
Sogou["Learning_rate"]=0.0003
Sogou["Batch"]=1024

# Amazon Full Data hyperparamters
Amz_F=dict()
Amz_F["Max_length(Unigram)"]=100
Amz_F["Max_length(Bigram)"]=500
Amz_F["Path"]="amazon_review_full_csv"
Amz_F["Class"]=5
Amz_F["Learning_rate"]=0.0001
Amz_F["Batch"]=1024

# Amazon Polarity Data hyperparamters
Amz_P=dict()
Amz_P["Max_length(Unigram)"]=100
Amz_P["Max_length(Bigram)"]=350
Amz_P["Path"]="amazon_review_polarity_csv"
Amz_P["Class"]=2
Amz_P["Learning_rate"]=0.0001
Amz_P["Batch"]=1024

# DBPedia Data hyperparamters
DBP=dict()
DBP["Max_length(Unigram)"]=50
DBP["Max_length(Bigram)"]=100
DBP["Path"]="dbpedia_csv"
DBP["Class"]=14
DBP["Learning_rate"]=0.0005
DBP["Batch"]=256

# Yahoo answer Data hyperparamters
Yah_A=dict()
Yah_A["Max_length(Unigram)"]=150
Yah_A["Max_length(Bigram)"]=300
Yah_A["Path"]="yahoo_answers_csv"
Yah_A["Class"]=10
Yah_A["Learning_rate"]=0.0001
Yah_A["Batch"]=1024

# Yelp Polarity Data hyperparamters
Yelp_P=dict()
Yelp_P["Max_length(Unigram)"]=250
Yelp_P["Max_length(Bigram)"]=500
Yelp_P["Path"]="yelp_review_polarity_csv"
Yelp_P["Class"]=2
Yelp_P["Learning_rate"]=0.00025
Yelp_P["Batch"]=1024

# Yelp Full Data hyperparamters
Yelp_F=dict()
Yelp_F["Max_length(Unigram)"]=250
Yelp_F["Max_length(Bigram)"]=500
Yelp_F["Path"]="yelp_review_full_csv"
Yelp_F["Class"]=5
Yelp_F["Learning_rate"]=0.00025
Yelp_F["Batch"]=1024


DataName=dict()
DataName["AG"]=AG
DataName["Sogou"]=Sogou
DataName["DBP"]=DBP
DataName["Yelp P."]=Yelp_P
DataName["Yelp F."]=Yelp_F
DataName["Yah. A."]=Yah_A
DataName["Amz F."]=Amz_F
DataName["Amz P."]=Amz_P



def RawData(PATH):
    # Loading train and test data
    # return [sentence,label] (not split sentence) (because of efficient memory)
    
    # Loading train data
    print("The train data is being loaded")
    train_open = open(PATH+"/train.csv",'r',encoding='utf-8')
    train=[[clean_str(lines)[4:],int(clean_str(lines)[0])] for lines in train_open.readlines()] # clean_str : data prerpreocessing

    # Loading test data
    print("The test data is being loaded")
    test_open = open(PATH+"/test.csv",'r',encoding='utf-8')
    test=[[clean_str(lines)[4:],int(clean_str(lines)[0])] for lines in test_open.readlines()] # clean_str : data prerpreocessing

    random.shuffle(train)
    
    return train, test
  

def Dictionary(train, test, Bigrams=False):

    # Making Bigram Function
    def Bigram(x):
        edit=[]
        for index in range(len(x)-1):
            edit.append('%s_%s' % (x[index],x[index+1]))
        return edit   

    
    word_dict = dict()
    word_dict["#PAD"] = 0 # For padding (mini-batch)
    
    # Unigram
    if Bigrams == False:
        print("< Unigram > Word dictionary is being made")
        for lines in train:
            sentence = lines[0].split() # train sentece split

            for word in sentence:
                if not word in word_dict:
                    word_dict[word]=len(word_dict) # making dictionary

        # Adding test words
        for lines in test:
            sentence = lines[0].split() # test sentence split

            for word in sentence:
                if not word in word_dict:
                    word_dict[word]=len(word_dict) # making dictionary
        
        return word_dict

    else:
        print("< Unigram + Bigram > Word dictionary is being made")
        for lines in train:
            sentence = lines[0].split() # train sentece split
            sentence += Bigram(sentence) # Adding Bigram 

            for word in sentence:
                if not word in word_dict:
                    word_dict[word]=len(word_dict) # making dictionary

        # Adding test words
        for lines in test:
            sentence = lines[0].split() # test sentence split
            sentence += Bigram(sentence) # Adding Bigram        

            for word in sentence:
                if not word in word_dict:
                    word_dict[word]=len(word_dict) # making dictionary

        return word_dict


def Padding(data, length, word_dict,Bigrams=False):

    # Making Bigram Function
    def Bigram(x):
        edit=[]
        for index in range(len(x)-1):
            edit.append('%s_%s' % (x[index],x[index+1]))
        return edit   
    
    sentence_index=[]
    
    if Bigrams==False:
        print("< Unigram > The data spliting is being loaded")
        
        # Unigram
        for lines in data:
            sentence=lines[0].split() # sentence split
            label = lines[1]-1 # because of difference between label and index

            edit=[]
            for word in sentence:
                edit.append(word_dict[word])

            sentence_index.append([edit[:length]+[0]*(length-len(edit)),label])
            
        return sentence_index

    else:
        print("< Unigram + Bigram > The data spliting is being loaded")

        # Unigram + Bigram
        for lines in data:
            sentence=lines[0].split() # sentence split
            sentence += Bigram(sentence) # Adding Bigram   
            label = lines[1]-1 # because of difference between label and index

            edit=[]
            for word in sentence:
                edit.append(word_dict[word])

            sentence_index.append([edit[:length]+[0]*(length-len(edit)),label])
        
        return sentence_index
    
    

def Spliting(train,test,valid_rate):

    print("Split train - dev")
    valid_rate = 0.05

    random.shuffle(train) # For making data sequences randomly
    train_ = train[:int(len(train)*(1-valid_rate))] # train set 
    valid_ = train[int(len(train)*(1-valid_rate)):] # validation set
    test_  = test                                   # test set
    
    return train_, valid_, test_


# In[11]:

def fastText_train(fastText_model, params, train, valid, test, lr=0.0005, epochs=5, batch=256):
       
    print("Training fastText model for text classification\n")
    print("Learning Rate :",lr)
    print("Epoch :",epochs)
    print("Batch :",batch)
    print("Opimizer : Adam")
    print("Loss : CrossEntropy")
    
    
    model=fastText_model(**params)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    loss_function = nn.CrossEntropyLoss()                               # loss function : Cross Entropy
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)      # optimizer :adam

    
    for epoch in range(epochs):
        for num in range(int(len(train)/batch)+1):
            model.train()                                                   # using dropout
        
            train_batch = random.sample(train,batch)        # mini-batch : 50이러한 점에서 나라별 언어가 얼마나 리치한지 구별했습니다
            train_input, train_label = zip(*train_batch)

            train_logits, train_probs, train_classes = model(train_input)   # training

            losses = loss_function(train_logits, torch.tensor(train_label)) # calculate loss
            optimizer.zero_grad()                                           # gradient to zero
            losses.backward()                                               # load backward function
    #        nn.utils.clip_grad_norm_(parameters, 1)
            optimizer.step()                                                # update parameters


        train_accuracy = torch.sum(torch.tensor([train_classes[i]==train_label[i] 
                               for i in range(len(train_classes))]), dtype=torch.float)/batch

        model.eval()
        valid_input, valid_label = zip(*valid)

        valid_logits, valid_probs, valid_classes = model(valid_input)
        valid_accuracy = torch.sum(torch.tensor([valid_classes[i]==valid_label[i]
                                               for i in range(len(valid_classes))]), dtype=torch.float)/len(valid)
        print("Epoch :",epoch,
            "-- Loss :",round(float(losses),4),
            " Train_accuracy :",round(float(train_accuracy),4),
            " Valid_accuracy :",round(float(valid_accuracy),4))

    test_ac=[]
    model.eval()
    for ns in range(10):
        test_s=test[int(len(test)/10)*ns:int(len(test)/10)*(ns+1)]
        test_input, test_label = zip(*test_s)
        test_logits, test_probs, test_classes = model(test_input)
        test_accuracy = torch.sum(torch.tensor([test_classes[i]==test_label[i] 
                                                for i in range(len(test_classes))]), dtype=torch.float)/(int(len(test_s)))

        test_ac.append(test_accuracy)

    print("                   +++++++++++++++++++++++++++++++")
    print("                     - Test_accuracy :",round(float(torch.sum(torch.tensor(test_ac))/10),4))
    print("                   +++++++++++++++++++++++++++++++")
        
    return None


def fastText_eval(DataName,name,bigram=False):
    
    class fastText(nn.Module):
        def __init__(self,**params):
            super(fastText,self).__init__()

            # parameters
            self.voca_size = params["voca_size"]
            self.embedding_size = params["embedding_size"]
            self.hidden_size = params["hidden_size"]
            self.num_classes = params["num_classes"]

            self.embedding = nn.Embedding(self.voca_size,self.embedding_size)
            self.embedding.weight.data.uniform_(-0.01,0.01)

            self.aggregation = nn.Linear(self.hidden_size, self.num_classes)
            self.aggregation.weight.data.uniform_(-0.01,0.01)
            self.aggregation.bias.data.uniform_(-0.01,0.01)

        def forward(self, inputs):
            Lookup = self.embedding(torch.tensor(inputs))
            text_level =  torch.sum(Lookup,dim=1)
            test_level = F.dropout(text_level,0.5,training=self.training)
            outputs = self.aggregation(text_level)
    #        outputs = nn.BatchNorm1d(256)(outputs)

            probs = F.softmax(outputs)
            classes = torch.max(probs, 1)[1]

            return outputs, probs, classes    
    
    
    
    print("Learning Target :",name)
    print("Bigram ? :", bigram)
    print("\n")
    
    Route=DataName[name]
    if bigram==False:        
        print("fastText for Text Classification")
        train_raw, test_raw = RawData(Route["Path"])
        word_dict=Dictionary(train_raw, test_raw,Bigrams=False)
        train_split = Padding(train_raw,Route["Max_length(Unigram)"],word_dict)
        test_split = Padding(test_raw,Route["Max_length(Unigram)"],word_dict)
        train,valid,test = Spliting(train_split,test_split,valid_rate=0.05)

    else:
        print("fastText for Text Classification")
        train_raw, test_raw = RawData(Route["Path"])
        word_dict=Dictionary(train_raw, test_raw,True)
        train_split = Padding(train_raw,Route["Max_length(Bigram)"],word_dict,True)
        test_split = Padding(test_raw,Route["Max_length(Bigram)"],word_dict,True)
        train,valid,test = Spliting(train_split,test_split,valid_rate=0.05)



    params=dict()
    params["voca_size"]=len(word_dict)
    params["embedding_size"]=10
    params["hidden_size"]=10
    params["num_classes"]=Route["Class"]
    
    # 데이터 로딩 여부 True False 

    print("\n")
    print(" - Data Summary")
    print("    The number of train :",len(train))
    print("    The number of valid :",len(valid))
    print("    The number of test :",len(test))
    print("    Thu number of word :",len(word_dict))
    print("\n")
    fastText_train(fastText,params,train, valid, test,lr=Route["Learning_rate"],batch=Route["Batch"])
    
    del train_raw, test_raw, word_dict, train_split, test_split, train, valid, test
    return print("\nEnd")


# In[12]:

fastText_eval(DataName,"AG",bigram=False)


# In[ ]:

# For GPU


# In[ ]:

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from google.colab import drive
drive.mount('/content/drive')


# In[ ]:

import os
import re
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip()

# AG Data hyperparamters
AG=dict()
AG["Max_length(Unigram)"]=50
AG["Max_length(Bigram)"]=100
AG["Path"]="ag"
AG["Class"] = 4
AG["Learning_rate"]=0.0005
AG["Batch"]=256

# Sogou Data hyperparamters
Sogou=dict()
Sogou["Max_length(Unigram)"]=1000
Sogou["Max_length(Bigram)"]=1500
Sogou["Path"]="sogou"
Sogou["Class"]=5
Sogou["Learning_rate"]=0.0003
Sogou["Batch"]=1024

# Amazon Full Data hyperparamters
Amz_F=dict()
Amz_F["Max_length(Unigram)"]=100
Amz_F["Max_length(Bigram)"]=500
Amz_F["Path"]="amzf"
Amz_F["Class"]=5
Amz_F["Learning_rate"]=0.0001
Amz_F["Batch"]=1024

# Amazon Polarity Data hyperparamters
Amz_P=dict()
Amz_P["Max_length(Unigram)"]=100
Amz_P["Max_length(Bigram)"]=350
Amz_P["Path"]="amzp"
Amz_P["Class"]=2
Amz_P["Learning_rate"]=0.0001
Amz_P["Batch"]=1024

# DBPedia Data hyperparamters
DBP=dict()
DBP["Max_length(Unigram)"]=50
DBP["Max_length(Bigram)"]=100
DBP["Path"]="dbp"
DBP["Class"]=14
DBP["Learning_rate"]=0.0005
DBP["Batch"]=256

# Yahoo answer Data hyperparamters
Yah_A=dict()
Yah_A["Max_length(Unigram)"]=150
Yah_A["Max_length(Bigram)"]=300
Yah_A["Path"]="yah"
Yah_A["Class"]=10
Yah_A["Learning_rate"]=0.0001
Yah_A["Batch"]=1024

# Yelp Polarity Data hyperparamters
Yelp_P=dict()
Yelp_P["Max_length(Unigram)"]=250
Yelp_P["Max_length(Bigram)"]=500
Yelp_P["Path"]="ypp"
Yelp_P["Class"]=2
Yelp_P["Learning_rate"]=0.00025
Yelp_P["Batch"]=1024

# Yelp Full Data hyperparamters
Yelp_F=dict()
Yelp_F["Max_length(Unigram)"]=250
Yelp_F["Max_length(Bigram)"]=500
Yelp_F["Path"]="ypf"
Yelp_F["Class"]=5
Yelp_F["Learning_rate"]=0.00025
Yelp_F["Batch"]=1024


DataName=dict()
DataName["AG"]=AG
DataName["Sogou"]=Sogou
DataName["DBP"]=DBP
DataName["Yelp P."]=Yelp_P
DataName["Yelp F."]=Yelp_F
DataName["Yah. A."]=Yah_A
DataName["Amz F."]=Amz_F
DataName["Amz P."]=Amz_P



def RawData(PATH):
    # Loading train and test data
    # return [sentence,label] (not split sentence) (because of efficient memory)
    
    # Loading train data
    print("The train data is being loaded")
    train_open = open("/content/drive/My Drive/train_"+PATH+".csv",'r',encoding='utf-8')
    train=[[clean_str(lines)[4:],int(clean_str(lines)[0])] for lines in train_open.readlines()] # clean_str : data prerpreocessing

    # Loading test data
    print("The test data is being loaded")
    test_open = open("/content/drive/My Drive/test_"+PATH+".csv",'r',encoding='utf-8')
    test=[[clean_str(lines)[4:],int(clean_str(lines)[0])] for lines in test_open.readlines()] # clean_str : data prerpreocessing

    random.shuffle(train)
    
    return train, test
  

def Dictionary(train, test, Bigrams=False):

    # Making Bigram Function
    def Bigram(x):
        edit=[]
        for index in range(len(x)-1):
            edit.append('%s_%s' % (x[index],x[index+1]))
        return edit   

    
    word_dict = dict()
    word_dict["#PAD"] = 0 # For padding (mini-batch)
    
    # Unigram
    if Bigrams == False:
        print("< Unigram > Word dictionary is being made")
        for lines in train:
            sentence = lines[0].split() # train sentece split

            for word in sentence:
                if not word in word_dict:
                    word_dict[word]=len(word_dict) # making dictionary

        # Adding test words
        for lines in test:
            sentence = lines[0].split() # test sentence split

            for word in sentence:
                if not word in word_dict:
                    word_dict[word]=len(word_dict) # making dictionary
        
        return word_dict

    else:
        print("< Unigram + Bigram > Word dictionary is being made")
        for lines in train:
            sentence = lines[0].split() # train sentece split
            sentence += Bigram(sentence) # Adding Bigram 

            for word in sentence:
                if not word in word_dict:
                    word_dict[word]=len(word_dict) # making dictionary

        # Adding test words
        for lines in test:
            sentence = lines[0].split() # test sentence split
            sentence += Bigram(sentence) # Adding Bigram        

            for word in sentence:
                if not word in word_dict:
                    word_dict[word]=len(word_dict) # making dictionary

        return word_dict


def Padding(data, length, word_dict,Bigrams=False):

    # Making Bigram Function
    def Bigram(x):
        edit=[]
        for index in range(len(x)-1):
            edit.append('%s_%s' % (x[index],x[index+1]))
        return edit   
    
    sentence_index=[]
    
    if Bigrams==False:
        print("< Unigram > The data spliting is being loaded")
        
        # Unigram
        for lines in data:
            sentence=lines[0].split() # sentence split
            label = lines[1]-1 # because of difference between label and index

            edit=[]
            for word in sentence:
                edit.append(word_dict[word])

            sentence_index.append([edit[:length]+[0]*(length-len(edit)),label])
            
        return sentence_index

    else:
        print("< Unigram + Bigram > The data spliting is being loaded")

        # Unigram + Bigram
        for lines in data:
            sentence=lines[0].split() # sentence split
            sentence += Bigram(sentence) # Adding Bigram   
            label = lines[1]-1 # because of difference between label and index

            edit=[]
            for word in sentence:
                edit.append(word_dict[word])

            sentence_index.append([edit[:length]+[0]*(length-len(edit)),label])
        
        return sentence_index
    
    

def Spliting(train,test,valid_rate):

    print("Split train - dev")
    valid_rate = 0.05

    random.shuffle(train) # For making data sequences randomly
    train_ = train[:int(len(train)*(1-valid_rate))] # train set 
    valid_ = train[int(len(train)*(1-valid_rate)):] # validation set
    test_  = test                                   # test set
    
    return train_, valid_, test_


# In[ ]:

def fastText_train(fastText_model, params, train, valid, test, lr=0.0005, epochs=5, batch=256):
       
    print("Training fastText model for text classification\n")
    print("Learning Rate :",lr)
    print("Epoch :",epochs)
    print("Batch :",batch)
    print("Opimizer : Adam")
    print("Loss : CrossEntropy")
    
    
    model=fastText_model(**params).to(device)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    loss_function = nn.CrossEntropyLoss()                               # loss function : Cross Entropy
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)      # optimizer :adam

   
    for epoch in range(epochs):
        for num in range(int(len(train)/batch)+1):
            model.train()                                                   # using dropout

            train_batch = random.sample(train,batch)        # mini-batch : 50이러한 점에서 나라별 언어가 얼마나 리치한지 구별했습니다
            train_input, train_label = zip(*train_batch)

            train_logits, train_probs, train_classes = model(train_input)   # training

            losses = loss_function(train_logits, torch.tensor(train_label).to(device)) # calculate loss
            optimizer.zero_grad()                                           # gradient to zero
            losses.backward()                                               # load backward function
    #        nn.utils.clip_grad_norm_(parameters, 1)
            optimizer.step()                                                # update parameters


        train_accuracy = torch.sum(torch.tensor([train_classes[i]==train_label[i] 
                               for i in range(len(train_classes))]).to(device), dtype=torch.float)/batch

        model.eval()
        valid_input, valid_label = zip(*valid)

        valid_logits, valid_probs, valid_classes = model(valid_input)
        valid_accuracy = torch.sum(torch.tensor([valid_classes[i]==valid_label[i]
                                               for i in range(len(valid_classes))]).to(device), dtype=torch.float)/len(valid)
        print("Epoch :",epoch,
            "-- Loss :",round(float(losses),4),
            " Train_accuracy :",round(float(train_accuracy),4),
            " Valid_accuracy :",round(float(valid_accuracy),4))


    test_ac=[]
    model.eval()
    for ns in range(10):
        test_s=test[int(len(test)/10)*ns:int(len(test)/10)*(ns+1)]
        test_input, test_label = zip(*test_s)
        test_logits, test_probs, test_classes = model(test_input)
        test_accuracy = torch.sum(torch.tensor([test_classes[i]==test_label[i] 
                                                for i in range(len(test_classes))]).to(device), dtype=torch.float)/(int(len(test_s)))

        test_ac.append(test_accuracy)

    print("                   +++++++++++++++++++++++++++++++")
    print("                     - Test_accuracy :",round(float(torch.sum(torch.tensor(test_ac).to(device))/10),4))
    print("                   +++++++++++++++++++++++++++++++")
        
    return None




def fastText_eval(DataName,name,bigram=False):
    
    class fastText(nn.Module):
        def __init__(self,**params):
            super(fastText,self).__init__()

            # parameters
            self.voca_size = params["voca_size"]
            self.embedding_size = params["embedding_size"]
            self.hidden_size = params["hidden_size"]
            self.num_classes = params["num_classes"]

            self.embedding = nn.Embedding(self.voca_size,self.embedding_size)
            self.embedding.weight.data.uniform_(-0.01,0.01).to(device)

            self.aggregation = nn.Linear(self.hidden_size, self.num_classes)
    #        torch.nn.init.xavier_uniform_(self.aggregation.weight)
            self.aggregation.weight.data.uniform_(-0.01,0.01).to(device)
            self.aggregation.bias.data.uniform_(-0.01,0.01).to(device)

        def forward(self, inputs):
            Lookup = self.embedding(torch.tensor(inputs).to(device))
            text_level =  torch.sum(Lookup,dim=1)
            test_level = F.dropout(text_level,0.5,training=self.training)
            outputs = self.aggregation(text_level)
    #        outputs = nn.BatchNorm1d(1024)(outputs)

            probs = F.softmax(outputs)
            classes = torch.max(probs, 1)[1]

            return outputs, probs, classes
    
    
    print("Learning Target :",name)
    print("Bigram ? :", bigram)
    print("\n")
    
    Route=DataName[name]
    if bigram==False:        
        print("fastText for Text Classification")
        train_raw, test_raw = RawData(Route["Path"])
        word_dict=Dictionary(train_raw, test_raw,Bigrams=False)
        train_split = Padding(train_raw,Route["Max_length(Unigram)"],word_dict)
        test_split = Padding(test_raw,Route["Max_length(Unigram)"],word_dict)
        train,valid,test = Spliting(train_split,test_split,valid_rate=0.05)

    else:
        print("fastText for Text Classification")
        train_raw, test_raw = RawData(Route["Path"])
        word_dict=Dictionary(train_raw, test_raw,True)
        train_split = Padding(train_raw,Route["Max_length(Bigram)"],word_dict,True)
        test_split = Padding(test_raw,Route["Max_length(Bigram)"],word_dict,True)
        train,valid,test = Spliting(train_split,test_split,valid_rate=0.05)



    params=dict()
    params["voca_size"]=len(word_dict)
    params["embedding_size"]=10
    params["hidden_size"]=10
    params["num_classes"]=Route["Class"]
    
    # 데이터 로딩 여부 True False 

    print("\n")
    print(" - Data Summary")
    print("    The number of train :",len(train))
    print("    The number of valid :",len(valid))
    print("    The number of test :",len(test))
    print("    Thu number of word :",len(word_dict))
    print("\n")
    fastText_train(fastText,params,train, valid, test,lr=Route["Learning_rate"],batch=Route["Batch"])
    
    del train_raw, test_raw, word_dict, train_split, test_split, train, valid, test
    return print("\nEnd")


# In[ ]:

fastText_eval(DataName,"AG",bigram=True)

