from gramformer import Gramformer
import torch
import docx2txt
from nltk.tokenize import PunktSentenceTokenizer
import pandas as pd
import re
from rapidfuzz import fuzz
from termcolor import colored
from tkinter import filedialog
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

file = r"C:\Users\chris\Documents\Transgola\Clients\PROJECTS\2021\401210621_TM_HS\Translation\MU COPY_HLA_P.1689_Ictiofauna_20210621 en.docx"
  
print( "\n")
print("Checking")
print( "\n")
text = docx2txt.process(file)
sent_tokenizer = PunktSentenceTokenizer(text)
sents = sent_tokenizer.tokenize(text)
sents = set(sents)


def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(1212)


gf = Gramformer(models = 2, use_gpu=False) # 0=detector, 1=highlighter, 2=corrector, 3=all 

influent_sentences = sents

inputs = []
corrections = []

n = 1
for influent_sentence in influent_sentences:
    corrected_sentence = gf.correct(influent_sentence)
    inputs.append(influent_sentence)
    corrections.append(corrected_sentence[0])
    print(f"{n}/{len(influent_sentences)} processed")
    n += 1

    
x_list = []
y_list = []
score = []

for x,y in zip(inputs, corrections):
    fuzz.ratio(x, y)
    score.append(fuzz.ratio(x, y))
    x_list.append(x)
    y_list.append(y)
    
# remove consecutive blank lines

x_list1 = []  
for x in x_list:

    xn = re.sub(r'\n\s*\n', '\n\n', x)
    x_list1.append(xn)
    
y_list1 = []  
for y in y_list:

    yn = re.sub(r'\n\s*\n', '\n\n', y)
    y_list1.append(yn)
    
data_tuples = list(zip(x_list1,y_list1,score))

results = pd.DataFrame(data_tuples, columns=['X','Y', 'Score'])  

results = results.sort_values(by=['Score'], ascending=False)
results = results[results['Score'] < 100 ]

x_list3 = list(results['X'])
y_list3 = list(results['Y'])
        
    
# uncommon words

diffs = []


def find(X, Y):
    count = {}
    for word in X.split():
        count[word] = count.get(word, 0) + 1

    for word in Y.split():
        count[word] = count.get(word, 0) + 1
    return [word for word in count if count[word] == 1]



for X,Y in zip(x_list3, y_list3):
    diffs.append((find(X, Y)))
    
diffsList = [' '.join(x) for x in diffs]
results['Diffs'] = diffsList
results = results[['Score', 'X', 'Y', 'Diffs']]


resultsXlist = results['X'].tolist()
resultsYlist = results['Y'].tolist()
resultDIFFSYlist = results['Diffs'].tolist()
resultSCORElist  = results['Score'].tolist()




n = 0
while n <= len(resultsXlist) - 1:

    text1 = resultsXlist[n]  
    text2 = resultsYlist[n] 
    l1 = resultDIFFSYlist[n].split()

    
    
    formattedText1 = []
    for t in text1.split():
        if t in l1:
            formattedText1.append(colored(t,'red', attrs=['bold']))
        else: 
            formattedText1.append(t)

    
    formattedText2 = []
    for t in text2.split():
        if t in l1:
            formattedText2.append(colored(t,'red', attrs=['bold']))
        else: 
            formattedText2.append(t)
 
    print( "\n")
    print(colored(resultSCORElist[n], 'green'))
    print(colored(l1, 'blue'))
    print( "\n")
    print(" ".join(formattedText1))
    print( "\n")
    print(" ".join(formattedText2))
    print( "\n")
    print( "\n")
    
    n += 1
