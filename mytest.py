import pandas as pd 
from cdqa.utils.download import download_squad, download_model, download_bnpp_data
from ast import literal_eval
from cdqa.pipeline import QAPipeline
# directory = './models/'

# Downloading data
# download_squad(dir=directory)
# download_bnpp_data(dir=directory)

# Downloading pre-trained BERT fine-tuned on SQuAD 1.1
# download_model('bert-squad_1.1', dir=directory)

# Downloading pre-trained DistilBERT fine-tuned on SQuAD 1.1
# download_model('distilbert-squad_1.1', dir=directory)

df2 = pd.read_csv('./test.csv', converters={'paragraphs': literal_eval})

#print(df2)
# df = pd.read_csv('./test.csv')

cdqa_pipeline = QAPipeline(reader='./models/bert_qa.joblib')
cdqa_pipeline.fit_retriever(df=df2)

query = 'What can I expect from an MRI?'
prediction = cdqa_pipeline.predict(query)

print('query: {}'.format(query))
print('answer: {}'.format(prediction[0]))
print('title: {}'.format(prediction[1]))
print('paragraph: {}'.format(prediction[2]))