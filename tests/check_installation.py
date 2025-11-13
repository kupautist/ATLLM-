#!/usr/bin/env python3
"""Скрипт для проверки установки зависимостей"""

import sys

def check_package(package_name, import_name=None):
    """Проверяет, установлен ли пакет"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name} - установлен")
        return True
    except ImportError:
        print(f"❌ {package_name} - НЕ установлен")
        return False

def main():
    print("=" * 50)
    print("Проверка установки зависимостей")
    print("=" * 50)
    print()
    
    packages = [
        ("python-telegram-bot", "telegram"),
        ("openai", "openai"),
        ("numpy", "numpy"),
        ("python-dotenv", "dotenv"),
        ("aiofiles", "aiofiles"),
    ]
    
    results = []
    for package, import_name in packages:
        results.append(check_package(package, import_name))
    
    print()
    
    # Проверка FAISS (опционально)
    faiss_installed = check_package("faiss-cpu", "faiss")
    results.append(faiss_installed)
    
    print()
    print("=" * 50)
    
    if all(results[:-1]):  # Все основные пакеты установлены
        if faiss_installed:
            print("✅ Все пакеты установлены (включая FAISS)")
            print("✅ Основной бот запускается командой: python bot_simple.py")
            print("   (FAISS можно использовать при кастомизации хранилища)")
        else:
            print("✅ Основные пакеты установлены (без FAISS)")
            print("✅ Запускай: python bot_simple.py")
        print()
        print("Следующий шаг: настрой .env файл и запусти бота")
        return 0
    else:
        print("❌ Не все пакеты установлены")
        print()
        print("Установи недостающие пакеты:")
        if not faiss_installed:
            print("  pip install -r requirements_simple.txt  # Без FAISS")
        else:
            print("  pip install -r requirements.txt  # С FAISS")
        return 1

if __name__ == "__main__":
    sys.exit(main())

