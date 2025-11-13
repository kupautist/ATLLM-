import json
import os
from typing import List, Dict, Optional
from collections import defaultdict
from utils.logger_config import setup_logger

class ConversationManager:
    """Manager for handling user conversation history"""

    def __init__(self, storage_dir: str = "conversations", max_history: int = 10):
        self.logger = setup_logger("conversation_manager")
        self.storage_dir = os.getenv("CONVERSATIONS_DIR", storage_dir)
        self.max_history = max_history
        self.conversations: Dict[int, List[Dict[str, str]]] = defaultdict(list)

        os.makedirs(self.storage_dir, exist_ok=True)
        self._load_conversations()
        self.logger.info(f"ConversationManager инициализирован (max_history: {max_history})")

    def _get_conversation_file(self, user_id: int) -> str:
        """Returns the path to the history file for a user"""
        return os.path.join(self.storage_dir, f"user_{user_id}.json")

    def _load_conversations(self):
        """Loads conversation histories from files"""
        try:
            for filename in os.listdir(self.storage_dir):
                if not filename.startswith("user_") or not filename.endswith(".json"):
                    continue

                filepath = os.path.join(self.storage_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        user_id = data.get('user_id')
                        messages = data.get('messages', [])
                        if user_id:
                            self.conversations[user_id] = messages
                except Exception as e:
                    self.logger.warning(f"Ошибка при загрузке истории из {filename}: {e}")

            self.logger.info(f"Загружено историй для {len(self.conversations)} пользователей")
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке историй: {e}", exc_info=True)

    def _save_conversation(self, user_id: int):
        """Saves conversation history for a user"""
        try:
            filepath = self._get_conversation_file(user_id)
            data = {
                'user_id': user_id,
                'messages': self.conversations[user_id]
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            self.logger.debug(f"История сохранена для user_id {user_id}")
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении истории для user_id {user_id}: {e}", exc_info=True)

    def add_user_message(self, user_id: int, message: str):
        """Adds user message to history"""
        self.conversations[user_id].append({
            "role": "user",
            "content": message
        })

        if len(self.conversations[user_id]) > self.max_history * 2:
            self.conversations[user_id] = self.conversations[user_id][2:]
            self.logger.debug(f"История обрезана для user_id {user_id}")

        self._save_conversation(user_id)
        self.logger.debug(f"Добавлено сообщение пользователя для user_id {user_id}")

    def add_assistant_message(self, user_id: int, message: str):
        """Adds assistant response to history"""
        self.conversations[user_id].append({
            "role": "assistant",
            "content": message
        })

        self._save_conversation(user_id)
        self.logger.debug(f"Добавлен ответ ассистента для user_id {user_id}")

    def get_history(self, user_id: int, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Retrieves conversation history for a user"""
        history = self.conversations.get(user_id, [])

        if limit:
            history = history[-limit:]

        self.logger.debug(f"Получена история для user_id {user_id} (сообщений: {len(history)})")
        return history

    def clear_history(self, user_id: int) -> bool:
        """Clears conversation history for a user"""
        try:
            if user_id in self.conversations:
                self.conversations[user_id] = []

                filepath = self._get_conversation_file(user_id)
                if os.path.exists(filepath):
                    os.remove(filepath)

                self.logger.info(f"История очищена для user_id {user_id}")
                return True
            else:
                self.logger.debug(f"История для user_id {user_id} не найдена")
                return False
        except Exception as e:
            self.logger.error(f"Ошибка при очистке истории для user_id {user_id}: {e}", exc_info=True)
            return False

    def get_stats(self, user_id: int) -> Dict[str, int]:
        """Retrieves statistics for user's history"""
        history = self.conversations.get(user_id, [])
        user_messages = sum(1 for msg in history if msg['role'] == 'user')
        assistant_messages = sum(1 for msg in history if msg['role'] == 'assistant')

        return {
            'total_messages': len(history),
            'user_messages': user_messages,
            'assistant_messages': assistant_messages
        }

    def format_history_for_openai(self, user_id: int, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Formats history for passing to OpenAI API"""
        return self.get_history(user_id, limit)
