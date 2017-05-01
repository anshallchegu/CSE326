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
