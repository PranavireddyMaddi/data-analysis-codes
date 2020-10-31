import requests   # Importing requests to extract content from a url
from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scrapping...used to scrap specific content 
import re 

#import nltk
#from nltk.corpus import stopwords

import matplotlib.pyplot as plt
from wordcloud import WordCloud

# creating empty reviews list 
Ex_reviews=[]
#forest = ["the","king","of","jungle"]

for i in list(range(1,20)):
  ip=[]  
  #url="https://www.amazon.in/Apple-MacBook-Air-13-3-inch-Integrated/product-reviews/B073Q5R6VR/ref=cm_cr_arp_d_paging_btm_2?showViewpoints=1&pageNumber="+str(i)
  url = "https://www.amazon.in/OnePlus-Display-Storage-4000mAH-Battery/product-reviews/B07HGJK535/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber="+str(i)
  response = requests.get(url)
  soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 
  reviews = soup.findAll("span",attrs={"class","a-size-base review-text review-text-content"})# Extracting the content under specific tags  
  for j in range(len(reviews)):
    ip.append(reviews[j].text)
    
  Ex_reviews= Ex_reviews+ip # adding the reviews of one page to empty list which in future contains all the reviews



# writng reviews in a text file 
with open("oneplus.txt","w",encoding='utf8') as output:
    output.write(str(Ex_reviews))
    
    
with open("C:\\Users\\Hi\\HP.txt","r",encoding='utf8') as k:
    HP_reviews = k.read()
    
import os
os.getcwd()    
    
    
# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(Ex_reviews)



# Removing unwanted symbols incase if exists

ip_rev_string = re.sub("[^A-Za-z" "]+"," ",ip_rev_string).lower()
#ip_rev_string = re.sub("[0-9" "]+"," ",ip_rev_string)

s="this is awesome"
s.split(" ")
# words that contained in iphone 7 reviews
ip_reviews_words = ip_rev_string.split(" ")

with open("D:\\CQ\\Textmining\\sw.txt","r") as sw:
    stopwords = sw.read()

stopwords = stopwords.split("\n")
stopwords.extend(["oneplus","phone","mobile","amazon","product"])

ip_new_words=[]
for i in range(len(ip_reviews_words)):
    if(ip_reviews_words[i] not in stopwords):
        ip_new_words.append(ip_reviews_words[i])

temp = ["this","is","awsome","Data","Science"]
[i for i in temp if i in "is"]

ip_reviews_words = [i for i in ip_reviews_words if not i in stopwords]
#ip_reviews_words =  [i for i in ip_reviews_words if not i in "iphone"]



# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(ip_reviews_words)

# WordCloud can be performed on the string inputs. That is the reason we have combined 
# entire reviews into single paragraph
# Simple word cloud


wordcloud_ip = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_rev_string)

plt.imshow(wordcloud_ip)

# positive words # Choose the path for +ve words stored in system
with open("D:\CQ\Textmining\\positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")
  
poswords = poswords[36:]



# negative words  Choose path for -ve words stored in system
with open("D:\CQ\Textmining\\negative-words.txt","r") as neg:
  negwords = neg.read().split("\n")

negwords = negwords[37:]

# negative word cloud
# Choosing the only words which are present in negwords
negative_string= " ".join ([w for w in ip_reviews_words if w in negwords])


wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(negative_string)

plt.imshow(wordcloud_neg_in_neg)

# Positive word cloud
# Choosing the only words which are present in positive words
positive_string = " ".join ([w for w in ip_reviews_words if w in poswords])
wordcloud_pos_in_pos = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(positive_string)

plt.imshow(wordcloud_pos_in_pos)

nltk 

# Unique words 
unique_words = list(set(ip_reviews_words))


################# IMDB reviews extraction ######################## Time Taking process as this program is operating the web page while extracting 
############# the data we need to use time library in order sleep and make it to extract for that specific page 
#### We need to install selenium for python
#### pip install selenium
#### time library to sleep the program for few seconds 

from selenium import webdriver
browser = webdriver.Chrome()
from bs4 import BeautifulSoup as bs
#page = "http://www.imdb.com/title/tt0944947/reviews?ref_=tt_urv"
page = "http://www.imdb.com/title/tt6294822/reviews?ref_=tt_urv"
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import ElementNotVisibleException
browser.get(page)
import time
reviews = []
i=1
while (i>0):
    #i=i+25
    try:
        button = browser.find_element_by_xpath('//*[@id="load-more-trigger"]')
        button.click()
        time.sleep(5)
        ps = browser.page_source
        soup=bs(ps,"html.parser")
        rev = soup.findAll("div",attrs={"class","text"})
        reviews.extend(rev)
    except NoSuchElementException:
        break
    except ElementNotVisibleException:
        break
        

##### If we want only few recent reviews you can either press cntrl+c to break the operation in middle but the it will store 
##### Whatever data it has extracted so far #######
len(reviews)
len(list(set(reviews)))


import re 
cleaned_reviews= re.sub('[^A-Za-z0-9" "]+', '', reviews)

f = open("reviews.txt","w")
f.write(cleaned_reviews)
f.close()

with open("The_Post.text","w") as fp:
    fp.write(str(reviews))



len(soup.find_all("p"))
