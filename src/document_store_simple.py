import json
import os
from typing import List, Dict, Optional
import numpy as np
from src.openai_service import OpenAIService
from utils.logger_config import setup_logger

class DocumentStore:
    """Simplified version of document storage without FAISS - uses only NumPy"""

    def __init__(self, storage_dir: str = "documents"):
        self.logger = setup_logger("document_store")
        self.storage_dir = storage_dir
        self.openai_service = OpenAIService()
        self.documents: List[Dict] = []
        self.embedding_dim = 1536
        self.user_index: Dict[int, List[int]] = {}

        os.makedirs(storage_dir, exist_ok=True)
        self.metadata_file = os.path.join(storage_dir, "metadata.json")

        self._load_documents()
        self._build_user_index()
        self.logger.info(f"DocumentStore инициализирован. Загружено документов: {len(self.documents)}")
    
    def _build_user_index(self):
        """Builds index user_id -> document indices for fast lookup"""
        self.user_index = {}
        for idx, doc in enumerate(self.documents):
            user_id = doc.get('user_id')
            if user_id is not None:
                if user_id not in self.user_index:
                    self.user_index[user_id] = []
                self.user_index[user_id].append(idx)
        self.logger.debug(f"Построен индекс для {len(self.user_index)} пользователей")

    def _load_documents(self):
        """Loads document metadata from file"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                self.logger.info(f"Загружено {len(self.documents)} документов из {self.metadata_file}")
            except Exception as e:
                self.logger.error(f"Ошибка при загрузке документов: {e}", exc_info=True)
                self.documents = []
    
    def _save_documents(self):
        """Saves document metadata to file"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Метаданные документов сохранены в {self.metadata_file}")
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении документов: {e}", exc_info=True)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculates cosine similarity between two vectors"""
        try:
            v1 = np.array(vec1, dtype=np.float32)
            v2 = np.array(vec2, dtype=np.float32)
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(dot_product / (norm1 * norm2))
        except Exception as e:
            print(f"Ошибка при вычислении косинусного сходства: {e}")
            return 0.0
    
    def add_document(self, document_id: str, title: str, full_text: str, user_id: int) -> bool:
        """Adds document: summarizes, creates embedding, saves full text"""
        try:
            self.logger.info(f"Добавление документа '{title}' (user_id: {user_id}, длина: {len(full_text)} символов)")

            summary = self.openai_service.summarize_document(full_text)
            embedding = self.openai_service.get_embedding(summary)

            if not embedding:
                self.logger.error("Не удалось получить эмбеддинг для документа")
                return False

            text_file = os.path.join(self.storage_dir, f"{document_id}.txt")
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(full_text)

            doc = {
                "id": document_id,
                "title": title,
                "summary": summary,
                "embedding": embedding,
                "text_file": text_file,
                "user_id": user_id,
                "created_at": str(os.path.getctime(text_file) if os.path.exists(text_file) else "")
            }

            self.documents.append(doc)
            self._save_documents()

            doc_index = len(self.documents) - 1
            if user_id not in self.user_index:
                self.user_index[user_id] = []
            self.user_index[user_id].append(doc_index)

            self.logger.info(f"Документ '{title}' успешно добавлен (ID: {document_id})")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при добавлении документа: {e}", exc_info=True)
            return False
    
    def search_documents(self, query: str, user_id: int, top_k: int = 3,
                         similarity_threshold: float = 0.0) -> List[Dict]:
        """Searches for relevant documents by query using cosine similarity"""
        if not self.documents:
            self.logger.warning("Нет документов для поиска")
            return []

        try:
            self.logger.info(
                f"Поиск документов для запроса: '{query[:50]}...' "
                f"(user_id: {user_id}, top_k: {top_k}, threshold: {similarity_threshold})"
            )

            if user_id not in self.user_index or not self.user_index[user_id]:
                self.logger.warning(f"У пользователя {user_id} нет документов")
                return []

            query_embedding = self.openai_service.get_embedding(query)
            if not query_embedding:
                self.logger.error("Не удалось получить эмбеддинг для запроса")
                return []

            similarities = []
            for doc_idx in self.user_index[user_id]:
                doc = self.documents[doc_idx]

                if 'embedding' not in doc or not doc['embedding']:
                    continue

                similarity = self._cosine_similarity(query_embedding, doc['embedding'])

                if similarity >= similarity_threshold:
                    similarities.append((similarity, doc))

            self.logger.info(
                f"Найдено {len(similarities)} документов пользователя "
                f"(после фильтрации по threshold={similarity_threshold})"
            )

            similarities.sort(key=lambda x: x[0], reverse=True)

            results = []
            for similarity, doc in similarities[:top_k]:
                full_text = ""
                text_file = doc.get('text_file', '')
                if text_file and os.path.exists(text_file):
                    try:
                        with open(text_file, 'r', encoding='utf-8') as f:
                            full_text = f.read()
                    except Exception as e:
                        self.logger.error(f"Ошибка при чтении файла {text_file}: {e}")
                        full_text = doc.get('summary', '')
                else:
                    full_text = doc.get('summary', '')

                results.append({
                    "title": doc.get('title', 'Без названия'),
                    "summary": doc.get('summary', ''),
                    "full_text": full_text,
                    "similarity": similarity,
                    "distance": 1.0 - similarity
                })

            self.logger.info(f"Возвращено {len(results)} релевантных документов")
            return results
        except Exception as e:
            self.logger.error(f"Ошибка при поиске документов: {e}", exc_info=True)
            return []
    
    def get_user_documents(self, user_id: int) -> List[Dict]:
        """Returns list of all user documents"""
        if user_id not in self.user_index:
            return []
        return [self.documents[idx] for idx in self.user_index[user_id]]
    
    def delete_document(self, document_id: str, user_id: int) -> bool:
        """Deletes document"""
        try:
            self.logger.info(f"Удаление документа (ID: {document_id}, user_id: {user_id})")

            doc_to_delete = None
            for doc in self.documents:
                if doc['id'] == document_id and doc.get('user_id') == user_id:
                    doc_to_delete = doc
                    break

            if doc_to_delete:
                if os.path.exists(doc_to_delete['text_file']):
                    os.remove(doc_to_delete['text_file'])
                    self.logger.info(f"Файл документа {doc_to_delete['text_file']} удален")

                doc_idx = self.documents.index(doc_to_delete)
                self.documents.remove(doc_to_delete)

                if user_id in self.user_index:
                    self.user_index[user_id] = [
                        idx if idx < doc_idx else idx - 1
                        for idx in self.user_index[user_id] if idx != doc_idx
                    ]
                    if not self.user_index[user_id]:
                        del self.user_index[user_id]

                self._save_documents()
                self.logger.info(f"Документ '{doc_to_delete.get('title', 'N/A')}' успешно удален")
                return True

            self.logger.warning(f"Документ с ID {document_id} не найден для user_id {user_id}")
            return False
        except Exception as e:
            self.logger.error(f"Ошибка при удалении документа: {e}", exc_info=True)
            return False

