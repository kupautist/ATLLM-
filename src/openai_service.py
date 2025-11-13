import openai
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
from utils.logger_config import setup_logger
from utils.retry_handler import RetryHandler

load_dotenv()

class OpenAIService:
    def __init__(self):
        self.logger = setup_logger("openai_service")
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.logger.info(f"OpenAI Service инициализирован с моделью {self.model}")
    
    @RetryHandler.exponential_backoff(max_retries=3)
    def summarize_document(self, text: str, max_length: int = 500) -> str:
        """Summarizes a document to create a brief description"""
        try:
            max_input_chars = 16000
            if len(text) > max_input_chars:
                text_to_summarize = text[:max_input_chars] + "... [документ обрезан для суммаризации]"
                self.logger.info(f"Текст обрезан с {len(text)} до {max_input_chars} символов для суммаризации")
            else:
                text_to_summarize = text

            self.logger.debug(f"Начало суммаризации документа (длина: {len(text_to_summarize)} символов)")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Ты помощник, который создает краткие и информативные резюме документов. Фокусируйся на ключевых идеях и важных деталях."},
                    {"role": "user", "content": f"Создай краткое резюме следующего документа (максимум {max_length} символов):\n\n{text_to_summarize}"}
                ],
                temperature=0.3,
                max_tokens=500
            )
            summary = response.choices[0].message.content.strip()
            self.logger.info("Суммаризация документа успешно выполнена")
            return summary
        except Exception as e:
            self.logger.error(f"Ошибка при суммаризации: {e}", exc_info=True)
            return text[:max_length] if len(text) > max_length else text
    
    @RetryHandler.exponential_backoff(max_retries=3)
    def get_embedding(self, text: str) -> List[float]:
        """Gets embedding for text"""
        try:
            self.logger.debug(f"Получение эмбеддинга для текста (длина: {len(text)} символов)")
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            self.logger.info("Эмбеддинг успешно получен")
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Ошибка при получении эмбеддинга: {e}", exc_info=True)
            return []
    
    def estimate_tokens(self, text: str) -> int:
        """Estimates token count using 1 token ≈ 4 characters approximation"""
        return len(text) // 4

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncates text to maximum token count"""
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "... [текст обрезан]"

    def extract_relevant_chunks(self, text: str, query: str, max_chunks: int = 3, chunk_size: int = 2000) -> List[str]:
        """Extracts text chunks most relevant to the query using keyword matching and scoring"""
        query_words = set(query.lower().split())

        stop_words = {'и', 'в', 'на', 'с', 'по', 'для', 'от', 'к', 'из', 'что', 'как', 'это', 'а', 'но', 'или'}
        query_words = {w for w in query_words if w not in stop_words and len(w) > 2}

        paragraphs = text.split('\n\n')

        scored_paragraphs = []
        for para in paragraphs:
            if not para.strip():
                continue
            para_lower = para.lower()
            score = sum(1 for word in query_words if word in para_lower)
            score += sum(para_lower.count(word) for word in query_words) * 0.1
            if score > 0:
                scored_paragraphs.append((score, para.strip()))

        if not scored_paragraphs:
            sentences = text.replace('\n', ' ').split('.')
            for sentence in sentences:
                if not sentence.strip():
                    continue
                sentence_lower = sentence.lower()
                score = sum(1 for word in query_words if word in sentence_lower)
                if score > 0:
                    scored_paragraphs.append((score, sentence.strip() + '.'))

        scored_paragraphs.sort(key=lambda x: x[0], reverse=True)

        chunks = []
        total_length = 0
        max_total_length = chunk_size * max_chunks

        for _, para in scored_paragraphs:
            if total_length + len(para) <= max_total_length and len(chunks) < max_chunks:
                chunks.append(para)
                total_length += len(para)
            elif len(chunks) < max_chunks:
                remaining = max_total_length - total_length
                if remaining > 500:
                    chunks.append(para[:remaining] + "...")
                    break
            else:
                break

        if not chunks:
            chunks.append(text[:chunk_size])

        return chunks
    
    @RetryHandler.exponential_backoff(max_retries=3)
    def generate_answer(self, query: str, context: str, conversation_history: List[Dict[str, str]] = None, max_context_tokens: int = 60000) -> str:
        """Generates an answer based on context and conversation history"""
        try:
            context_tokens = self.estimate_tokens(context)
            if context_tokens > max_context_tokens:
                context = self.truncate_text(context, max_context_tokens)
                self.logger.warning(f"Контекст обрезан с {context_tokens} до ~{max_context_tokens} токенов (экономия токенов)")

            system_prompt = "Ты помощник для университетских документов. Отвечай кратко и точно на основе контекста. Используй историю диалога для понимания контекста разговора."

            messages = [{"role": "system", "content": system_prompt}]

            if conversation_history:
                recent_history = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
                messages.extend(recent_history)
                self.logger.debug(f"Добавлено {len(recent_history)} сообщений из истории")

            current_message = f"Контекст из документов:\n\n{context}\n\nВопрос: {query}"
            messages.append({"role": "user", "content": current_message})

            self.logger.info(f"Генерация ответа на запрос: '{query[:50]}...' (с историей: {bool(conversation_history)})")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=800
            )
            answer = response.choices[0].message.content.strip()
            self.logger.info("Ответ успешно сгенерирован")
            return answer
        except Exception as e:
            error_msg = str(e)
            if "context_length_exceeded" in error_msg:
                self.logger.error(f"Превышен лимит длины контекста: {error_msg}")
                return "❌ Документы слишком большие. Попробуй задать более конкретный вопрос или загрузи документы меньшего размера."
            self.logger.error(f"Ошибка при генерации ответа: {e}", exc_info=True)
            return "Извините, произошла ошибка при генерации ответа."

