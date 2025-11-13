"""Query Router for Modular RAG - automatically determines query type and selects optimal search strategy"""

from typing import Dict, List, Literal
from enum import Enum
from utils.logger_config import setup_logger

logger = setup_logger("query_router")


class QueryType(Enum):
    """Query types for classification"""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    PROCEDURAL = "procedural"
    CONCEPTUAL = "conceptual"
    COMPARISON = "comparison"


class SearchStrategy(Enum):
    """Search strategies"""
    PRECISE = "precise"
    BROAD = "broad"
    COMPREHENSIVE = "comprehensive"


class QueryRouter:
    """Query router for Modular RAG - analyzes queries and selects optimal search strategy"""

    def __init__(self):
        self.keywords = {
            QueryType.FACTUAL: [
                'кто', 'что', 'где', 'когда', 'какой', 'какая', 'какие',
                'сколько', 'дата', 'дедлайн', 'процент', 'команды'
            ],
            QueryType.ANALYTICAL: [
                'почему', 'зачем', 'как работает', 'причина', 'объясни',
                'разберись', 'проанализируй', 'в чем суть'
            ],
            QueryType.PROCEDURAL: [
                'как сделать', 'как создать', 'шаги', 'инструкция',
                'руководство', 'tutorial', 'как использовать'
            ],
            QueryType.CONCEPTUAL: [
                'что такое', 'определение', 'концепция', 'понятие',
                'термин', 'смысл', 'значение'
            ],
            QueryType.COMPARISON: [
                'сравни', 'различие', 'отличие', 'сходство', 'vs',
                'лучше', 'хуже', 'преимущество', 'недостаток'
            ]
        }

        self.strategy_configs = {
            SearchStrategy.PRECISE: {
                'top_k': 3,
                'similarity_threshold': 0.0,
                'description': 'Precise search for factual questions'
            },
            SearchStrategy.BROAD: {
                'top_k': 7,
                'similarity_threshold': 0.0,
                'description': 'Broad search for analytical questions'
            },
            SearchStrategy.COMPREHENSIVE: {
                'top_k': 10,
                'similarity_threshold': 0.0,
                'description': 'Comprehensive search for comparisons and complex questions'
            }
        }

        self.type_to_strategy = {
            QueryType.FACTUAL: SearchStrategy.PRECISE,
            QueryType.ANALYTICAL: SearchStrategy.BROAD,
            QueryType.PROCEDURAL: SearchStrategy.BROAD,
            QueryType.CONCEPTUAL: SearchStrategy.PRECISE,
            QueryType.COMPARISON: SearchStrategy.COMPREHENSIVE
        }

    def classify_query(self, query: str) -> QueryType:
        """Classifies query by type using keyword matching"""
        query_lower = query.lower()
        scores = {query_type: 0 for query_type in QueryType}

        for query_type, keywords in self.keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    scores[query_type] += 1

        max_score = max(scores.values())
        if max_score == 0:
            logger.info(f"Query '{query}' не классифицирован, используется ANALYTICAL")
            return QueryType.ANALYTICAL

        best_type = max(scores.items(), key=lambda x: x[1])[0]
        logger.info(f"Query '{query}' классифицирован как {best_type.value}")
        return best_type

    def route(self, query: str) -> Dict:
        """Determines optimal search strategy for query"""
        query_type = self.classify_query(query)
        strategy = self.type_to_strategy[query_type]
        config = self.strategy_configs[strategy]

        result = {
            'query': query,
            'query_type': query_type.value,
            'strategy': strategy.value,
            'top_k': config['top_k'],
            'similarity_threshold': config['similarity_threshold'],
            'description': config['description']
        }

        logger.info(
            f"Routing: {query_type.value} -> {strategy.value} "
            f"(top_k={config['top_k']}, threshold={config['similarity_threshold']})"
        )

        return result

    def explain_routing(self, query: str) -> str:
        """Returns human-readable explanation of strategy selection"""
        routing_result = self.route(query)

        explanation = (
            f"📊 Query Routing Analysis:\n"
            f"• Query Type: {routing_result['query_type']}\n"
            f"• Strategy: {routing_result['strategy']}\n"
            f"• Top K: {routing_result['top_k']} documents\n"
            f"• Similarity Threshold: {routing_result['similarity_threshold']}\n"
            f"• Description: {routing_result['description']}"
        )

        return explanation


if __name__ == "__main__":
    router = QueryRouter()

    test_queries = [
        "Какой дедлайн финального проекта?",
        "Что такое Multi-Representation индексирование?",
        "Сравни Multi-representation и RAPTOR",
        "Как создать RAG систему?",
        "Почему ColBERT лучше обычных embeddings?"
    ]

    print("=== Query Routing Examples ===\n")
    for query in test_queries:
        print(f"Query: {query}")
        print(router.explain_routing(query))
        print("-" * 80 + "\n")
