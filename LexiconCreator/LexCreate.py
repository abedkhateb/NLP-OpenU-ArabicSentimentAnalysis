# -*- coding: utf8 -*-
import xml.etree.ElementTree as et
import re
import io

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

e = et.parse('awn.xml').getroot()
#p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
#text = re.sub(p_tashkeel,"", text)
wordDict = {}
for item in e.findall('item'):
    wordDict[item.get('itemid')] = clean_str(item.get('name')).replace(' ', '_')

linksDict = {}
for link in e.findall('link'):
    if link.get('type') == 'related_to' or link.get('type') == 'has_hyponym' or link.get('type') == 'has_holo_part' or link.get('type') == 'near_synonym':  
        link1 = link.get('link1')
        link2 = link.get('link2')
        if link1 in linksDict.keys():
            linksDict[link1].append(link2)
        else:
                linksDict[link1] = []
                linksDict[link1].append(link2)

with io.open('AraSynonyms.txt', 'w', encoding='utf8') as outFile:
    for key in linksDict.keys():
        outFile.write(wordDict[key])
        outFile.write(' ')
        for link in linksDict[key]:
            outFile.write(wordDict[link])
            outFile.write(' ')
        outFile.write('\n')

print(e.tag)