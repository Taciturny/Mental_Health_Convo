from elasticsearch import Elasticsearch

class ElasticsearchService:
    def __init__(self, hosts=None):
        self.client = Elasticsearch(hosts)

    def create_index(self, index_name, settings):
        self.client.indices.create(index=index_name, body=settings)

    def index_document(self, index_name, doc_id, document):
        self.client.index(index=index_name, id=doc_id, body=document)

    def search(self, index_name, query):
        return self.client.search(index=index_name, body=query)
