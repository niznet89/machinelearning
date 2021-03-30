#!/usr/bin/env python
# coding: utf-8

# ## Guided Project: Building a Spam Filter with Naive Bayes
# 
# Welcome to a project that I'm paticuarly excited about. In this we're going to use what has been learned so far to create a spam filter (based on a data set of 5,572 messages - that have been classified by humans). 
# 
# This is exciting to me as it's my first foray into what could technically be classified as a Machine Learning algorithim. Let's get into it!

# In[1]:


# Import the appropriate library's
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt


# In[2]:


# Import the data into a Panda's Dataframe

spam_collection = pd.read_csv('SMSSpamCollection_1.txt', sep='\t', header=None, names=['Label', 'SMS'])


# In[3]:


# Exploring the data set a little

print(spam_collection.describe())
spam_collection.head(15)


# In[4]:


# Finding out what percentage is Spam vs Non-spam

spam_collection['Label'].value_counts(normalize=True)*100


# It appears that 86.5% of messages are legitamate while 13.4% are spam. 

# ### Creating the Training and Test Set

# The next stage is allocating diffent parts of the data as a training set and a test set. To allocate appropriately, we're going to split it 80 / 20 (Pareto's principle anyone?). 
# 
# The first step is to randomly assign a training & test data set then compare the percentages of spam / non-spam to determine if they're similar. 
# 
# The first step will be creating a randomized dataframe before splitting the training set by 80% & test set by 20%.

# In[5]:


randomized_spam = spam_collection.sample(frac=1, random_state=1)


training_set = randomized_spam.iloc[0 : round(len(randomized_spam)*0.8), :].reset_index()
test_set = randomized_spam.iloc[round(len(randomized_spam)*0.8) : len(randomized_spam), :].reset_index()

print(training_set.shape)
print(test_set.shape)


# In[6]:


# Looking to see if the percentages align with the original data set

print(training_set['Label'].value_counts(normalize=True)*100)
print('\n')
print(test_set['Label'].value_counts(normalize=True)*100)


# 
# 
# It appears that the new training & test sets align percentage wise as the original data set. Good news!
# 
# 

# ### Letter Case and Punctuation

# To begin the actual data exploration process, we're going to remove any characters that are not letters or numbers & make all the messages lower case.

# In[7]:


# Removing characters that are not letters or numbers

training_set['SMS'] = training_set['SMS'].str.replace('\W', ' ').str.lower()
test_set['SMS'] = test_set['SMS'].str.replace('\W', ' ').str.lower()


# In[8]:


training_set.head(10)


# In[9]:


test_set.head(10)


# ### Creating the Vocabulary

# Now that we've removed the punctuation and changed the characters to lower case, we're going to create a vocabulary to cross reference and build up a statistical model. 
# 
# To do this, we're going to create a frequency table to count each individual word and apply the algorithim. 

# In[10]:


# Initializing an empty list and splitting the string before each space character 

training_set['SMS'] = training_set['SMS'].str.split()


# In[11]:


# Creating empty list then appending each individual word found in the data set to the list

vocabulary = []

for training in training_set['SMS']:
    for nested in training:
        vocabulary.append(nested)


vocabulary_unique = list(set(vocabulary))

len(vocabulary_unique)
        


# It appears we have 7783 unique words!

# ### Creating a Final training set (with frequency tables)
#     
# 

# In the next step we're going to create a frequency table that creates a word (which will act as the column in the Dataframe), then creates a list the length of the training set (these will be 0's for now). 
# 
# From there, we looped through the training set to get the SMS & index for the SMS and tally each SMS with the number of words in the vocabulary.
# 
# Finally, we're going to concatanate this new Dataframe with the original Training set so we have the Labal (spam or not spam), SMS and counts of each vocab word in the same dataframe. 

# In[12]:


word_counts_per_sms = {}

for v in vocabulary_unique:
    word_counts_per_sms[v] = len(training_set['SMS'])*[0]
    
for index, sms in enumerate(training_set['SMS']):
    for word in sms:
        word_counts_per_sms[word][index] += 1
    
words_sms = pd.DataFrame(word_counts_per_sms)    


# In[13]:


final_training_set = pd.concat([training_set, words_sms], axis=1)

final_training_set.head(10)


# ### Calculating Constants 

# We're now going to start building the Spam filter with the Naive Bayes algorithm. We're going to have to figure out the probability of a message being 'Spam' or 'Not Spam' (Ham). 
# 
# We're also going to need to use Laplace smoothing to smooth out every value so the equation isn't multiplied by zero. 
# 
# We'll start out by calculating:
# - P(Spam) and P(Ham)
# - Nspam, N, Nvocab

# In[14]:


spam_df = final_training_set[final_training_set['Label'] == 'spam']['SMS']
ham_df = final_training_set[final_training_set['Label'] == 'ham']['SMS']

alpha = 1

n_spam = spam_df.apply(len).sum()
n_ham = ham_df.apply(len).sum()
n_vocab = len(vocabulary_unique) 


prob_spam = len(final_training_set[final_training_set['Label'] == 'spam']) / len(final_training_set)
prob_ham = len(final_training_set[final_training_set['Label'] == 'ham']) / len(final_training_set)


# ### Calculating Parameters

# We are now going to go through the process of calculating the individual probabilities of each word given spam / not-spam to help speed up the calculation so it doesn't need to be done ad-hoc. When the algorithm needs to call the values for a given message it will be able to call it from a given variable. 
# 
# We'll do this by initializing two dictionaries (for spam & non-spam) where each word in the vocabulary will have a probability given it's parameter.

# In[15]:


spam_vocab = {word: 0 for word in vocabulary_unique}
ham_vocab = {word: 0 for word in vocabulary_unique}

spam_training_set = final_training_set[final_training_set['Label'] == 'spam']
ham_training_set = final_training_set[final_training_set['Label'] == 'ham']

for s in vocabulary_unique:
    sum_column = spam_training_set[s].sum()
    
    prob_word = (sum_column + alpha) / (n_spam + alpha * n_vocab)
    spam_vocab[s] = prob_word
    
for s in vocabulary_unique:
    sum_column = ham_training_set[s].sum()
    prob_word = (sum_column + alpha) / (n_ham + alpha * n_vocab)
    ham_vocab[s] = prob_word  
    
  
    


# ### Classifying a new message 

# Now that we've created the constants & parameters we have the neccasary data to classify a message as spam or not-spam.
# 
# Our first order of events creating the core of the Naive Bayes algorithim, a function that defines the percentage chance that it falls as either spam / not spam. If the probability falls on either side, we'll have this function spit out a message letting the user know that its one or the other.

# In[16]:


import re

def classify(message):

    message = re.sub('\W', ' ', message)
    message = message.lower()
    message = message.split()


    p_spam_given_message = prob_spam
    for m in message:
        if m in spam_vocab:
            p_spam_given_message *= spam_vocab[m] 
        
    
    p_ham_given_message = prob_ham
    for m in message:
        if m in ham_vocab:
            p_ham_given_message *= ham_vocab[m]
        
        

    print('P(Spam|message):', p_spam_given_message)
    print('P(Ham|message):', p_ham_given_message)

    if p_ham_given_message > p_spam_given_message:
        print('Label: Ham')
    elif p_ham_given_message < p_spam_given_message:
        print('Label: Spam')
    else:
        print('Equal proabilities, have a human classify this!')
        


# Now that we've built the function we're going to classify two messages:
# 
# - "Hey mate, on my way back home. Do you need me to pick up anything from Woolies?"
# - 'WINNER!! This is the secret code to unlock the money: C3421.'

# In[17]:


classify("Hey mate, on my way back home. Do you need me to pick up anything from Woolies?")


# In[19]:


classify('WINNER!! This is the secret code to unlock the money: C3421.')


# ### Measure the Spam filters accuracy

# Now that we've inputted two sample messages and can see that we have a working function that we can input into the training data set and compare it to the test set. 
# 
# First, we're going to alter the initial function to 'return' instead of 'printing' the values so we can apply it the Dataframe. 
# 
# 

# In[20]:


def classify_test_set(message):

    message = re.sub('\W', ' ', message)
    message = message.lower()
    message = message.split()


    p_spam_given_message = prob_spam
    for m in message:
        if m in spam_vocab:
            p_spam_given_message *= spam_vocab[m] 
        
    
    p_ham_given_message = prob_ham
    for m in message:
        if m in ham_vocab:
            p_ham_given_message *= ham_vocab[m]
        
        

    print('P(Spam|message):', p_spam_given_message)
    print('P(Ham|message):', p_ham_given_message)

    if p_ham_given_message > p_spam_given_message:
        return 'ham'
    elif p_ham_given_message < p_spam_given_message:
        return 'spam'
    else:
        return 'needs human classification'
        


# In[21]:


test_set['predicted'] = test_set['SMS'].apply(classify_test_set)


# In[23]:


test_set.head()


# In[41]:


correct = 0
total = len(test_set['Label'])

for index, row in test_set.iterrows():
    if row[1] == row[3]:
        correct += 1

accuracy = (correct / total) * 100

print("The accuracy of this Spam filter (using the Naive Bayes algorithm) is {0:.4f}%".format(accuracy))


# ### Conclusion

# Through building the algorithim and applying it to a couple test texts it can (pretty quickly) determine whether it is spam or not spam. 
# 
# Upoon applying it to the test set we also found that it is pretty accurate with 98.7% accuracy. In that context, we should be able to apply it to a data set with pretty good accuracy. 
# 
# 
