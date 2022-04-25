from config import *

# read data
def parse_data(file):
    for l in open(file,'r'):
        yield json.loads(l)

data = list(parse_data('G:\\NLP Projects\\Sarcasm Detection\\data\\Sarcasm_Headlines_Dataset_v2.json'))

# convert to pandas dataframe
df = pd.DataFrame(data)
df.drop('article_link', axis = 1, inplace=True)
#print(df.head())

labels = list(df.is_sarcastic)
sentences = list(df.headline)
print('Number of sentences and labels: ', len(labels), len(sentences))

x_train, x_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2)
print('Train and Test set distribution: ', len(x_train), len(x_test), len(y_train), len(y_test))

# tokinizing the texts
tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_token)
tokenizer.fit_on_texts(x_train)
word_index = tokenizer.word_index
#print(wored_index)

# pdding
train_sequences = tokenizer.texts_to_sequences(x_train)
padded_train_sequences = pad_sequences(train_sequences, maxlen = max_length, padding = padding_type)

test_sequences = tokenizer.texts_to_sequences(x_test)
paddes_test_sentences = pad_sequences(test_sequences, maxlen = max_length, padding = padding_type)