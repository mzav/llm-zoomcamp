from elasticsearch import Elasticsearch, helpers
import requests 
import tiktoken

docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)
        doc.pop("section", None)

# Connect to Elasticsearch (adjust for security if needed)
es = Elasticsearch("https://localhost:9200", 
basic_auth=("elastic", ""),
verify_certs=False
)

# Define index name
index_name = "course-qa"

# 1. Delete the index if it exists (for fresh start)
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)

# 2. Create index with mappings
mapping = {
    "mappings": {
        "properties": {
            "course": {"type": "keyword"},
            "question": {"type": "text"},
            "text": {"type": "text"}
        }
    }
}

es.indices.create(index=index_name, body=mapping)

# 3. Prepare for bulk indexing
bulk_docs = [
    {"_index": index_name, "_source": doc} for doc in documents
]

helpers.bulk(es, bulk_docs)
es.indices.refresh(index=index_name)

# 4. Execute boosted multi_match search
query_text = "How do copy a file to a Docker container?"

response = es.search(index=index_name, query={"match_all": {}})

response = es.search(
    index=index_name,
    size=3,
    query={
        "bool": {
            "must": {
                "multi_match": {
                    "query": query_text,
                    "fields": ["question^4", "text"],
                    "type": "best_fields"
                }
            },
            "filter": {
                "term": {
                    "course": "machine-learning-zoomcamp"
                }
            }
        }
    }
)

# 5. Print top hits
# print("Top search results:")
contexts=[]
for hit in response["hits"]["hits"]:
    # print(f"- Score: {hit['_score']}")
    # print(f"  Question: {hit['_source']['question']}")
    # print(f"  Text: {hit['_source']['text']}")
    # print()

    question = hit['_source']['question']
    text = hit['_source']['text']
    context = f"""
    Q: {question}
    A: {text}
    """.strip()

    contexts.append(context)

CONTEXT_ALL = ("\n\n").join(contexts)
QUESTION = "How do copy a file to a Docker container?"

prompt_template = f"""
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {QUESTION}

CONTEXT:
{CONTEXT_ALL}
""".strip()

print(prompt_template)
print(len(prompt_template))

encoding = tiktoken.encoding_for_model("gpt-4o")
tokens = encoding.encode(prompt_template)
print(len(tokens))
