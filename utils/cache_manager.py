import json
import os
import hashlib
import time
from typing import Optional, Any
from utils.logger_config import setup_logger

class CacheManager:
    """Cache manager for question answers"""

    def __init__(self, cache_dir: str = "cache", ttl: int = 3600):
        self.logger = setup_logger("cache_manager")
        self.cache_dir = os.getenv("CACHE_DIR", cache_dir)
        self.ttl = ttl

        os.makedirs(self.cache_dir, exist_ok=True)
        self.logger.info(f"CacheManager инициализирован (TTL: {ttl}s, dir: {self.cache_dir})")

    def _generate_key(self, query: str, context: str, user_id: int) -> str:
        """Generates a unique cache key based on query, context, and user_id"""
        composite_key = f"{user_id}:{query}:{context}"
        return hashlib.sha256(composite_key.encode('utf-8')).hexdigest()

    def _get_cache_path(self, key: str) -> str:
        """Returns the path to the cache file"""
        return os.path.join(self.cache_dir, f"{key}.json")

    def get(self, query: str, context: str, user_id: int) -> Optional[str]:
        """Retrieves answer from cache if it exists and is not expired"""
        key = self._generate_key(query, context, user_id)
        cache_path = self._get_cache_path(key)

        if not os.path.exists(cache_path):
            self.logger.debug(f"Кэш не найден для ключа {key[:16]}...")
            return None

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            timestamp = cache_data.get('timestamp', 0)
            current_time = time.time()

            if current_time - timestamp > self.ttl:
                self.logger.info(f"Кэш устарел для ключа {key[:16]}... (возраст: {current_time - timestamp:.0f}s)")
                os.remove(cache_path)
                return None

            answer = cache_data.get('answer')
            self.logger.info(f"Кэш найден для ключа {key[:16]}... (возраст: {current_time - timestamp:.0f}s)")
            return answer

        except Exception as e:
            self.logger.error(f"Ошибка при чтении кэша: {e}", exc_info=True)
            try:
                os.remove(cache_path)
            except:
                pass
            return None

    def set(self, query: str, context: str, user_id: int, answer: str) -> bool:
        """Saves answer to cache"""
        key = self._generate_key(query, context, user_id)
        cache_path = self._get_cache_path(key)

        try:
            cache_data = {
                'query': query,
                'answer': answer,
                'timestamp': time.time(),
                'user_id': user_id
            }

            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"Ответ сохранен в кэш для ключа {key[:16]}...")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка при сохранении в кэш: {e}", exc_info=True)
            return False

    def clear_expired(self) -> int:
        """Removes all expired cache entries"""
        deleted_count = 0
        current_time = time.time()

        try:
            for filename in os.listdir(self.cache_dir):
                if not filename.endswith('.json'):
                    continue

                cache_path = os.path.join(self.cache_dir, filename)
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)

                    timestamp = cache_data.get('timestamp', 0)
                    if current_time - timestamp > self.ttl:
                        os.remove(cache_path)
                        deleted_count += 1
                except Exception as e:
                    self.logger.warning(f"Ошибка при обработке {filename}: {e}")
                    try:
                        os.remove(cache_path)
                        deleted_count += 1
                    except:
                        pass

            if deleted_count > 0:
                self.logger.info(f"Удалено {deleted_count} устаревших записей кэша")

        except Exception as e:
            self.logger.error(f"Ошибка при очистке кэша: {e}", exc_info=True)

        return deleted_count

    def clear_all(self) -> int:
        """Deletes all cache"""
        deleted_count = 0

        try:
            for filename in os.listdir(self.cache_dir):
                if not filename.endswith('.json'):
                    continue

                cache_path = os.path.join(self.cache_dir, filename)
                try:
                    os.remove(cache_path)
                    deleted_count += 1
                except Exception as e:
                    self.logger.warning(f"Ошибка при удалении {filename}: {e}")

            self.logger.info(f"Весь кэш очищен. Удалено {deleted_count} записей")

        except Exception as e:
            self.logger.error(f"Ошибка при полной очистке кэша: {e}", exc_info=True)

        return deleted_count

    def get_cache_stats(self) -> dict:
        """Returns cache statistics"""
        total_count = 0
        expired_count = 0
        current_time = time.time()
        total_size = 0

        try:
            for filename in os.listdir(self.cache_dir):
                if not filename.endswith('.json'):
                    continue

                cache_path = os.path.join(self.cache_dir, filename)
                total_count += 1
                total_size += os.path.getsize(cache_path)

                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)

                    timestamp = cache_data.get('timestamp', 0)
                    if current_time - timestamp > self.ttl:
                        expired_count += 1
                except:
                    expired_count += 1

            stats = {
                'total_entries': total_count,
                'valid_entries': total_count - expired_count,
                'expired_entries': expired_count,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2)
            }

            self.logger.debug(f"Статистика кэша: {stats}")
            return stats

        except Exception as e:
            self.logger.error(f"Ошибка при получении статистики кэша: {e}", exc_info=True)
            return {
                'total_entries': 0,
                'valid_entries': 0,
                'expired_entries': 0,
                'total_size_bytes': 0,
                'total_size_mb': 0.0
            }
