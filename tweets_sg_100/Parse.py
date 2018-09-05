# -*- coding: utf8 -*-
import gensim
import re
import io

# load the model
model = gensim.models.Word2Vec.load('tweets_sg_100')

# Clean/Normalize Arabic Text
def clean_str(text):
    search = ["أ","إ","آ","ة","_","-","/",".","،"," و "," يا ",'"',"ـ","'","ى","\\",'\n', '\t','&quot;','?','؟','!']
    replace = ["ا","ا","ا","ه"," "," ","","",""," و"," يا","","","","ي","",' ', ' ',' ',' ? ',' ؟ ',' ! ']
    
    #remove tashkeel
    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(p_tashkeel,"", text)
    
    #remove longation
    p_longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    text = re.sub(p_longation, subst, text)
    
    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')
    
    for i in range(0, len(search)):
        text = text.replace(search[i], replace[i])
    
    #trim    
    text = text.strip()

    return text

# python 3.X
word = clean_str(u'سعيد')
# python 2.7
# word = clean_str('القاهرة'.decode('utf8', errors='ignore'))

	
# get a word vector
word_vector = model.wv[ word ]

with io.open('AraWordVec.txt', 'w', encoding='utf8') as outFile:
    for word in model.wv.index2word[1:]:
        outFile.write(word)
        #outFile.write(' ')
        for value in model.wv[ word ]:
            outFile.write(' ')
            outFile.write(str(value))
        outFile.write('\n')
# find and print the most similar terms to a word
most_similar = model.wv.most_similar( word )
for term, score in most_similar:
	print(term, score)
