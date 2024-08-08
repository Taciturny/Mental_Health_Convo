# Mental_Health_Convo

### Project In Progress

```
mental_health_conversation/
├── app/
│   ├── __init__.py
│   ├── streamlit.py
│   ├── routes.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── llm_model.py
│   │   
│   ├── services/
│   │   ├── __init__.py
│   │   ├── weaviate_service.py
│   │   ├── elasticsearch_service.py (still considering)
│   │   ├── semantic_search.py
│   │   
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── config.py
│   │   ├── weaviate_schema.py
│   │   ├── elasticsearch_schema.py (still considering)
|   |
|   ├── monitoring/
|   |   ├── response_generator.py
|   |
|   ├── tests/unit/intergration
│   ├── test_chatbot.py
│   ├── llm_model.py
│   ├── response_generator.py
│   ├── weaviate_service.py
│   └── semantic_search.py
├── Dockerfile
├── docker-compose.yml
├── data_ingestion_script_py
├── requirements.txt
├── README.md
```
```
mental_health_conversation/
├── app/
│   ├── __init__.py
│   ├── streamlit.py
│   ├── main.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── llm_model.py
│   │   
│   ├── services/
│   │   ├── __init__.py
│   │   ├── weaviate.py
│   │   ├── semantic_search.py
│   │   
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── config.py
│   │   ├── weaviate_schema.py
|   |
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── db.py
│   │   ├── metrics.py
│   │   ├── prometheus_client.py
│   │   ├── logger.py
│   │   ├── health_check.py
│   │   ├── performance_tracker.py
│   │   ├── model_metrics.py
│   │   ├── user_metrics.py
│   │   └── dashboard_configs/
│   │       ├── grafana_dashboard.json
│   │       └── prometheus.yml
│   │   
|   ├── tests/unit/intergration
│   ├── test_chatbot.py
│   ├── llm_model.py
│   ├── response_generator.py
│   ├── weaviate_service.py
│   └── semantic_search.py
├── Dockerfile
├── docker-compose.yml
├── data_ingestion_script_py
├── requirements.txt
├── README.md
```

streamlit
transformers
weaviate-client
scikit-learn
uvicornfastapi
torch
bitsandbytes
accelerate
elasticsearch
requests
streamlit
sqlalchemy
psycopg2-binary
mlflow
rogue

