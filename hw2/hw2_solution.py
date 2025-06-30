import os
import numpy as np
import requests
from fastembed import TextEmbedding

os.environ["FASTEMBED_CACHE_PATH"] = "cache/"
model_name = "jinaai/jina-embeddings-v2-small-en"
emb_model = TextEmbedding(model_name = model_name)

text = "I just discovered the course. Can I join now?"
vector = next(emb_model.embed(text))
# print(np.min((vector))) # Answer 1
# print(np.linalg.norm(vector))
# print(vector.dot(vector))

another_text = "Can I still join the course after the start date?"
another_vector = next(emb_model.embed(another_text))
# print(vector.dot(another_vector)) # Answer 2

documents = [
 {'text': "Yes, even if you don't register, you're still eligible to submit the homeworks.\nBe aware, however, that there will be deadlines for turning in the final projects. So don't leave everything for the last minute.",
  'section': 'General course-related questions',
  'question': 'Course - Can I still join the course after the start date?',
  'course': 'data-engineering-zoomcamp'},
 {'text': 'Yes, we will keep all the materials after the course finishes, so you can follow the course at your own pace after it finishes.\nYou can also continue looking at the homeworks and continue preparing for the next cohort. I guess you can also start working on your final capstone project.',
  'section': 'General course-related questions',
  'question': 'Course - Can I follow the course after it finishes?',
  'course': 'data-engineering-zoomcamp'},
 {'text': "The purpose of this document is to capture frequently asked technical questions\nThe exact day and hour of the course will be 15th Jan 2024 at 17h00. The course will start with the first  “Office Hours'' live.1\nSubscribe to course public Google Calendar (it works from Desktop only).\nRegister before the course starts using this link.\nJoin the course Telegram channel with announcements.\nDon’t forget to register in DataTalks.Club's Slack and join the channel.",
  'section': 'General course-related questions',
  'question': 'Course - When will the course start?',
  'course': 'data-engineering-zoomcamp'},
 {'text': 'You can start by installing and setting up all the dependencies and requirements:\nGoogle cloud account\nGoogle Cloud SDK\nPython 3 (installed with Anaconda)\nTerraform\nGit\nLook over the prerequisites and syllabus to see if you are comfortable with these subjects.',
  'section': 'General course-related questions',
  'question': 'Course - What can I do before the course starts?',
  'course': 'data-engineering-zoomcamp'},
 {'text': 'Star the repo! Share it with friends if you find it useful ❣️\nCreate a PR if you see you can improve the text or the structure of the repository.',
  'section': 'General course-related questions',
  'question': 'How can we contribute to the course?',
  'course': 'data-engineering-zoomcamp'}]
# texts_embs = [next(emb_model.embed(doc['text'])) for doc in documents]
# text_array = np.array(texts_embs)
# print(np.argmax(text_array.dot(another_vector))) # Answer 3


full_texts_embs = [next(emb_model.embed(doc['question'] + " " + doc['text'])) for doc in documents]
full_text_array = np.array(full_texts_embs)
# print(np.argmax(full_text_array.dot(another_vector))) # Answer 4


# print(set([model["dim"] for model in TextEmbedding.list_supported_models()])) # Answer 5

# ------------------------------

docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()


documents = []
for course in documents_raw:
    course_name = course['course']
    if course_name != 'machine-learning-zoomcamp':
        continue
    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)

model_name = "BAAI/bge-small-en"
emb_model = TextEmbedding(model_name = model_name)

full_texts_embs = [next(emb_model.embed(doc['question'] + " " + doc['text'])) for doc in documents]
full_text_array = np.array(full_texts_embs)


# -----------------
from qdrant_client import QdrantClient, models

client = QdrantClient("http://localhost:6333") #connecting to local Qdrant instance

EMBEDDING_DIMENSIONALITY = 384

# Define the collection name
collection_name = "zoomcamp-rag"

# Create the collection with specified vector parameters
client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=EMBEDDING_DIMENSIONALITY,  # Dimensionality of the vectors
        distance=models.Distance.COSINE  # Distance metric for similarity search
    )
)

points = []
id = 0
model_handle = "BAAI/bge-small-en"

for course in documents_raw:
    for doc in course['documents']:

        point = models.PointStruct(
            id=id,
            vector=models.Document(text=doc['question'] + ' ' + doc['text'], model=model_handle), #embed text locally with "jinaai/jina-embeddings-v2-small-en" from FastEmbed
            payload={
                "text": doc['text'],
                "section": doc['section'],
                "course": course['course']
            } #save all needed metadata fields
        )
        points.append(point)

        id += 1


client.upsert(
    collection_name=collection_name,
    points=points
)

def search(query, limit=5):

    results = client.query_points(
        collection_name=collection_name,
        query=models.Document( #embed the query text locally with "jinaai/jina-embeddings-v2-small-en"
            text=query,
            model=model_handle 
        ),
        limit=limit, # top closest matches
        with_payload=True #to get metadata in the results
    )

    return results

query = "I just discovered the course. Can I join now?"
print(search(query)) # Answer 6