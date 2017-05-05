from textblob.classifiers import NaiveBayesClassifier
import nltk.data
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from itertools import cycle

import re
def untokenize(words):
    text = ' '.join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace("can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
fp = open("./test_data/test.txt")
data = fp.read()

a = []
a = (tokenizer.tokenize(data))

stop_words = set(stopwords.words('english'))

stop_words.update(('and','I','Describe','system','Compare','Infer','Tell','Name','List','steps','Enumerate','scenarios','A','And','application','So','arnt','This','When','It','works','constraints','variations','many','restrictions','Many','so','clarification','importance','discuss','difference','method','advantages','contrast','medium','details','detail','services','structure','responsibilities','Justify','features','diagram','Suppose','suppose','issues','cant','Yes','yes','No','no','These','these','use','Do','type','create','feature','True-justify','share','default','name','Write','Explain','How','benefits','data','Discuss','uses','example'))

fp1 = open("./train_data/train_cpp.txt")
lines = fp1.read()
word_tokens = word_tokenize(lines)

filtered_sentence = [w for w in word_tokens if not w in stop_words]
filtered_sentence1=untokenize(filtered_sentence)
train1=[]
train=[]
is_noun = lambda pos: pos[:2] == 'NN'
tokenized = nltk.word_tokenize(filtered_sentence1)
nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 
sub = ["c++"]
train1=zip(nouns,cycle(sub))

fp1 = open("./train_data/train_cn.txt")
lines = fp1.read()
word_tokens = word_tokenize(lines)

filtered_sentence = [w for w in word_tokens if not w in stop_words]
filtered_sentence1=untokenize(filtered_sentence)
is_noun = lambda pos: pos[:2] == 'NN'
tokenized = nltk.word_tokenize(filtered_sentence1)
nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 
sub = ["computer networks"]
train2=zip(nouns,cycle(sub))

fp1 = open("./train_data/train_os.txt")
lines = fp1.read()
word_tokens = word_tokenize(lines)

filtered_sentence = [w for w in word_tokens if not w in stop_words]
filtered_sentence1=untokenize(filtered_sentence)
is_noun = lambda pos: pos[:2] == 'NN'
tokenized = nltk.word_tokenize(filtered_sentence1)
nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 
sub = ["operating systems"]
train3=zip(nouns,cycle(sub))

fp1 = open("./train_data/train_dm.txt")
lines = fp1.read()
word_tokens = word_tokenize(lines)

filtered_sentence = [w for w in word_tokens if not w in stop_words]
filtered_sentence1=untokenize(filtered_sentence)
is_noun = lambda pos: pos[:2] == 'NN'
tokenized = nltk.word_tokenize(filtered_sentence1)
nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 
sub = ["data mining"]
train4=zip(nouns,cycle(sub))

train=train1+train2+train3+train4

fp1 = open("./test_data/test.txt")
lines = fp1.read()
sent_tokens = sent_tokenize(lines);
sub= ["c++","computer networks","operating systems","c++","computer networks","operating systems","data mining","data mining","data mining","data mining","c++","computer networks","data mining","data mining","operating systems","computer networks","data mining","data mining","computer networks","data mining","c++","c++","computer networks","computer networks","data mining","data mining","operating systems","data mining"]
test= zip(sent_tokens,sub)

cl = NaiveBayesClassifier(train)

res = []
for i in a:
	res.append((cl.classify(i)))

count_cpp=0
count_dm=0
count_cn=0
count_os=0
count=0

for k in res:
   if k=='c++':
       count_cpp+= 1
   elif k=='data mining':
       count_dm+=1
   elif k=='computer networks':
       count_cn+=1
   elif k=='operating systems':
       count_os+=1
   else:
       count+=1        

tot = count_os+count_cn+count_dm+count_cpp+count

print ("Percentage of operating systems: " + str(float(count_os)/tot * 100)) + "%"
print ("Percentage of computer networks: " + str(float(count_cn)/tot * 100)) + "%"
print ("Percentage of data mining and ware housing: " + str(float(count_dm)/tot * 100)) + "%"
print ("Percentage of c++: " + str(float(count_cpp )/tot * 100)) + "%"

print ("accuracy:" + str(cl.accuracy(test)*100)) + "%"