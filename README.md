## Text/Meeting Summarization
There are broadly two different approaches that are used for text summarization:  
## Extractive Summarization  
We identify the important sentences or phrases from the original text and extract only those from the text. Those extracted sentences would be our summary. Some of the methods are:  

1. Sentence Scoring based on Word Frequency
2. TextRank using Universal Sentence Encoder
## Abstractive Summarization  
Here, we generate new sentences from the original text. This is in contrast to the extractive approach we saw earlier where we used only the sentences that were present. The sentences generated through abstractive summarization might not be present in the original text.   

## Sentence Scoring based on Word Frequency
### Code for preprocessing (Removing stopwords and lemmatization)
```ruby
'''
pre processing steps on the entire dataset
'''
# importing customized stopwords from customized_stopwords.txt

with open ('customized_stopwords', 'rb') as fp:
    customized_stopwords = pickle.load(fp)
more_stop_words = ['sounds','works','thinking','talking','dream','honestly','sofia','francis','simon','presently','month','wanna','longer','alternatively','hear','issue','options','difference','wouldn','morning','current','worry','short','school','plan','guest','bring','depend','latest','mention','earlier','read','simple','spend','include','friend','question','couldn','option','happen','finish','start','tomorrow','work','agree','think','middle','dicide','write','haven','understand','print','call','return','talk','happen']   
customized_stopwords=more_stop_words + customized_stopwords  
#stemmer = SnowballStemmer("english")
def lemmatize(word):                                    # input is a word that is to be converted to root word for verb
    return WordNetLemmatizer().lemmatize(word, pos = 'v')

def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if (token not in gensim.parsing.preprocessing.STOPWORDS) and (len(token) > 4) and (token not in customized_stopwords):
            token = lemmatize(token)
            if token not in customized_stopwords:
                result.append(lemmatize(token))
                
    return result

```
### Code for preprocessing (Removing stopwords only)
```ruby
'''
Removing only stopwords(no lemmatisation)
'''
# importing customized stopwords from customized_stopwords.txt
with open ('customized_stopwords', 'rb') as fp:
    customized_stopwords = pickle.load(fp)
more_stop_words = ['sounds','works','thinking','talking','dream','honestly','sofia','francis','simon','presently','month','wanna','longer','alternatively','hear','issue','options','difference','wouldn','morning','current','worry','short','school','plan','guest','bring','depend','latest','mention','earlier','read','simple','spend','include','friend','question','couldn','option','happen','finish','start','tomorrow','work','agree','think','middle','dicide','write','haven','understand','print','call','return','talk','happen']   
customized_stopwords=more_stop_words + customized_stopwords  


def preprocess_stopwords(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if (token not in gensim.parsing.preprocessing.STOPWORDS) and (len(token) > 4) and (token not in customized_stopwords):
            #token = lemmatize(token)
            if token not in customized_stopwords:
                result.append((token))
                
    return result

```
### Code for generating word cloud for each meeting
```ruby
# Word cloud of a meeting
def get_word_cloud(combined_words):
    #combined_words
    cleaned_combined_words = []
    for word in combined_words.split(" "):
        cleaned_combined_words += preprocess(word)
    cleaned_combined_words = " ".join(cleaned_combined_words)
    
    wordcloud = WordCloud(width=700, height=300, random_state=21, max_font_size=110).generate(cleaned_combined_words)
    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title("Some frequent words used in the headlines", weight='bold', fontsize=14)
    plt.show()

```
## Sentence scoring based on word frequency(Spacy)

```ruby
def get_spacy_summary(text,percent=.02):
    '''
    input : text -> text to be summarized, 
            percent -> ratio to which summary is needed
    returns : summary of text
    '''

    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    word_frequency = collections.Counter(preprocess_stopwords(text)) # word frequency dictionary
    
    max_frequency = max(word_frequency.values())

    for word in word_frequency.keys():
        word_frequency[word] =word_frequency[word]/ max_frequency             # normalising the word frequency
    sentence_tokens = [sent for sent in doc.sents]                            # making sentence tokens
    sentence_score = {}
    for sent in sentence_tokens:
        sentence_score[sent] = 0
        word_count_in_sentence = len(sent)
        #print (sent,str(word_count_in_sentence))
        for word in sent :
            
            if word.text.lower() in word_frequency.keys():
                #print (word)
                sentence_score[sent] += word_frequency[word.text.lower()] # calculating the sentence score
            
                
        #print(sentence_score[sent],sent,str(word_count_in_sentence))
        #if word_count_in_sentence > 0:
        #sentence_score[sent] = sentence_score[sent] // word_count_in_sentence # normalizing the sentence score based on length
        #print(sentence_score[sent],sent,str(word_count_in_sentence))
    total_score = 0
    for sent in sentence_score:
        #print(sentence_score[sent])
        #print(sent)
        total_score += sentence_score[sent]
    average_score = total_score/len(sentence_score)
    #print(total_score,len(sentence_score),average_score)
        
    
    select_length = int(len(sentence_tokens)*percent)
    summary = nlargest(select_length,sentence_score,key = sentence_score.get) # summary for the needed percent

    return summary

```
## Using LDA for Topic modelling for a meeting
```ruby
def print_lda_bow_result(docs,num_of_topics):
    cleaned_docs = []
    np.random.seed(100)
    for doc in docs:
        for word in doc:
            cd = preprocess(word)
            cleaned_docs.append(cd)

    
    # Create dictionary

    dictionary = gensim.corpora.Dictionary(cleaned_docs)
    #dictionary.filter_extremes(no_below=1, no_above=0.5, keep_n=100000) # optional

    # Create Term Document Frequency or the Bag of Words
    bow_corpus = [dictionary.doc2bow(doc) for doc in cleaned_docs]

    # create model
    ldamodels = gensim.models.ldamodel.LdaModel(bow_corpus, num_topics = num_of_topics,id2word=dictionary, passes=50)
    for i in ldamodels.print_topics(num_words = 18): 
        for j in i: print (j)
```
## Cosine similarity of sentences
Thanks to :
https://towardsdatascience.com/understand-text-summarization-and-create-your-own-summarizer-in-python-b26a9f09fc70

```ruby
# Cosine similarity of senences

def get_cosine_sentence_similarity_summary(text,top_n):
    def read_article():    
        filedata = text
        article = filedata.split(". ")
        sentences = []
        for sentence in article:
            sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
        sentences.pop() 
        return sentences

    def sentence_similarity(sent1, sent2, stopwords=None):
        if stopwords is None:
            stopwords = []
        sent1 = [w.lower() for w in sent1]
        sent2 = [w.lower() for w in sent2]
        all_words = list(set(sent1 + sent2))
        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)

        # build the vector for the first sentence
        for w in sent1:
            if w in stopwords:
                continue
            vector1[all_words.index(w)] += 1

        # build the vector for the second sentence
        for w in sent2:
            if w in stopwords:
                continue
            vector2[all_words.index(w)] += 1
        #print(cosine_distance(vector1,vector2))
        if math.isnan(cosine_distance(vector1,vector2)):
            dist = 0
        else:
            dist = cosine_distance(vector1,vector2)
        return dist

    def build_similarity_matrix(sentences, stop_words):
        # Create an empty similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))

        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2: #ignore if both are same sentences
                    continue 
                similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

        return similarity_matrix


    def generate_summary():
        #stop_words = stopwords.words('english')
        stop_words = customized_stopwords
        summarize_text = []

        # Step 1 - Read text anc split it
        sentences =  read_article()
        #print (sentences)
        # Step 2 - Generate Similary Martix across sentences
        sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)
        ##print(sentence_similarity_martix)
        # Step 3 - Rank sentences in similarity martix
        sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
        scores = nx.pagerank(sentence_similarity_graph)
        ##print(scores)
        # Step 4 - Sort the rank and pick top sentences
        ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
        ##print("***********")
        ##print("Indexes of top ranked_sentence order are ", ranked_sentence)    
        #'''
        for i in range(top_n):
          summarize_text.append(" ".join(ranked_sentence[i][1]))

        # Step 5 - Offcourse, output the summarize texr
        print("Summarize Text from cosine similarity of sentences: \n", ". ".join(summarize_text))
        #'''
    # let's begin

    generate_summary()
```
## The caller code to generate meeting summary and some other details of each meeting
```ruby
# reading all documents
combined_words = ""
doc_wise_combined_words = []
docs = []
i=0
for transcript_file_name in glob.iglob('./transcripts/train//*.*', recursive=True):
    print("\n")
    print(os.path.basename(transcript_file_name))
    data = open(transcript_file_name).readlines()
    speaker_data = {line.split(":")[0]:line.split(":")[1] for line in data}
    words_in_file = ""
    speaker_dic ={}
    for name,words in  speaker_data.items():
        words = words.replace("\n","").lower()
        words_in_file = words_in_file + words
        if name.split("_")[0] in speaker_dic:
            speaker_dic[name.split("_")[0]] += words
        else:
            speaker_dic[name.split("_")[0]] = words
    print("Words:",str(len(words_in_file)))
    i+=1
    combined_words += words_in_file
    doc_wise_combined_words.append(words_in_file)
    print(collections.Counter(preprocess(words_in_file)).most_common(10))  # most common top 10 words
    print("\nLDA Results*********************")
    print_lda_bow_result([[words_in_file]],3)                              # lda result
    get_word_cloud(words_in_file)                                          # printing word cloud
    print("Summary results from sentence score based on word frequency")
    print(get_spacy_summary(words_in_file,0.02))                           # printing spacy summary
    print("\n")
    get_cosine_sentence_similarity_summary(words_in_file,50)               # printing sentence cosine similarity summary
    docs.append([words_in_file])
print ( "Total Transcripts : ",str(i))
cleaned_docs = []
for doc in docs:
    for word in doc:
        cd = preprocess(word)
        cleaned_docs.append(cd)
```
Thanks for reading!





