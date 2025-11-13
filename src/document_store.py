import json
import os
from typing import List, Dict, Optional
import numpy as np
import faiss
from src.openai_service import OpenAIService

class DocumentStore:
    def __init__(self, storage_dir: str = "documents"):
        self.storage_dir = storage_dir
        self.openai_service = OpenAIService()
        self.documents: List[Dict] = []
        self.index: Optional[faiss.Index] = None
        self.embedding_dim = 1536
        self._index_to_doc_map: List[int] = []

        os.makedirs(storage_dir, exist_ok=True)
        self.metadata_file = os.path.join(storage_dir, "metadata.json")

        self._load_documents()
        self._build_index()
    
    def _load_documents(self):
        """Загружает метаданные документов из файла"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
            except Exception as e:
                print(f"Ошибка при загрузке документов: {e}")
                self.documents = []
    
    def _save_documents(self):
        """Сохраняет метаданные документов в файл"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка при сохранении документов: {e}")
    
    def _build_index(self):
        """Builds FAISS index for embedding-based search"""
        if not self.documents:
            self.index = None
            return

        if self.index is None:
            self.index = faiss.IndexFlatL2(self.embedding_dim)

        self.index.reset()

        embeddings = []
        valid_doc_indices = []

        for i, doc in enumerate(self.documents):
            if 'embedding' in doc and doc['embedding']:
                embedding = doc['embedding']
                if isinstance(embedding, list) and len(embedding) == self.embedding_dim:
                    embeddings.append(embedding)
                    valid_doc_indices.append(i)

        if embeddings:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            self.index.add(embeddings_array)
            self._index_to_doc_map = valid_doc_indices
        else:
            self.index = None
            self._index_to_doc_map = []
    
    def add_document(self, document_id: str, title: str, full_text: str, user_id: int) -> bool:
        """Adds document: summarizes, creates embedding, saves full text"""
        try:
            summary = self.openai_service.summarize_document(full_text)
            embedding = self.openai_service.get_embedding(summary)

            if not embedding:
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
            self._build_index()
            
            return True
        except Exception as e:
            print(f"Ошибка при добавлении документа: {e}")
            return False
    
    def search_documents(self, query: str, user_id: int, top_k: int = 3) -> List[Dict]:
        """Searches for relevant documents by query"""
        if not self.documents or self.index is None or not self._index_to_doc_map:
            return []

        try:
            query_embedding = self.openai_service.get_embedding(query)
            if not query_embedding:
                return []

            search_k = min(top_k * 5, len(self._index_to_doc_map))
            query_vector = np.array([query_embedding], dtype=np.float32)
            distances, indices = self.index.search(query_vector, search_k)

            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self._index_to_doc_map):
                    doc_idx = self._index_to_doc_map[idx]
                    if doc_idx < len(self.documents):
                        doc = self.documents[doc_idx]
                        if doc.get('user_id') == user_id:
                            full_text = ""
                            text_file = doc.get('text_file', '')
                            if text_file and os.path.exists(text_file):
                                try:
                                    with open(text_file, 'r', encoding='utf-8') as f:
                                        full_text = f.read()
                                except Exception as e:
                                    print(f"Ошибка при чтении файла {text_file}: {e}")
                                    full_text = doc.get('summary', '')
                            else:
                                full_text = doc.get('summary', '')

                            results.append({
                                "title": doc.get('title', 'Без названия'),
                                "summary": doc.get('summary', ''),
                                "full_text": full_text,
                                "distance": float(distance)
                            })

                            if len(results) >= top_k:
                                break
            
            return results
        except Exception as e:
            print(f"Ошибка при поиске документов: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_user_documents(self, user_id: int) -> List[Dict]:
        """Возвращает список всех документов пользователя"""
        return [doc for doc in self.documents if doc.get('user_id') == user_id]
    
    def delete_document(self, document_id: str, user_id: int) -> bool:
        """Deletes document"""
        try:
            doc_to_delete = None
            for doc in self.documents:
                if doc['id'] == document_id and doc.get('user_id') == user_id:
                    doc_to_delete = doc
                    break

            if doc_to_delete:
                if os.path.exists(doc_to_delete['text_file']):
                    os.remove(doc_to_delete['text_file'])

                self.documents.remove(doc_to_delete)
                self._save_documents()
                self._build_index()
                return True
            
            return False
        except Exception as e:
            print(f"Ошибка при удалении документа: {e}")
            return False

