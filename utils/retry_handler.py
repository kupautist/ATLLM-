import time
import functools
from typing import Callable, Any, Tuple, Type
from logger_config import setup_logger
import openai

logger = setup_logger("retry_handler")

class RetryHandler:
    """Retry handler for API calls with exponential backoff"""

    @staticmethod
    def exponential_backoff(
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        exceptions: Tuple[Type[Exception], ...] = (
            openai.APIError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.APITimeoutError,
        )
    ) -> Callable:
        """Decorator for retries with exponential backoff"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                delay = initial_delay
                last_exception = None

                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e

                        if isinstance(e, openai.RateLimitError):
                            logger.warning(f"Rate limit достигнут. Попытка {attempt + 1}/{max_retries}")
                        elif isinstance(e, openai.APIConnectionError):
                            logger.warning(f"Ошибка подключения к API. Попытка {attempt + 1}/{max_retries}")
                        elif isinstance(e, openai.APITimeoutError):
                            logger.warning(f"Timeout API. Попытка {attempt + 1}/{max_retries}")
                        else:
                            logger.warning(f"Ошибка API: {e}. Попытка {attempt + 1}/{max_retries}")

                        if attempt < max_retries - 1:
                            current_delay = min(delay * (exponential_base ** attempt), max_delay)
                            logger.info(f"Ожидание {current_delay:.2f} секунд перед следующей попыткой...")
                            time.sleep(current_delay)
                    except Exception as e:
                        logger.error(f"Неожиданная ошибка (не повторяется): {e}", exc_info=True)
                        raise

                logger.error(f"Все {max_retries} попыток исчерпаны. Последняя ошибка: {last_exception}")
                raise last_exception

            return wrapper
        return decorator

    @staticmethod
    def with_fallback(fallback_value: Any = None) -> Callable:
        """Decorator for returning fallback value on error"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Ошибка в {func.__name__}: {e}. Возвращаем fallback значение.", exc_info=True)
                    return fallback_value
            return wrapper
        return decorator
