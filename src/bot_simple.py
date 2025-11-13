import os
import uuid
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from src.document_store_simple import DocumentStore  # Using simplified version
from src.openai_service import OpenAIService
from src.pdf_extractor import PDFExtractor
from dotenv import load_dotenv
from utils.logger_config import setup_logger
from utils.cache_manager import CacheManager
from utils.conversation_manager import ConversationManager
from utils.query_router import QueryRouter

load_dotenv()

class UniversityDocumentBot:
    def __init__(self):
        self.logger = setup_logger("telegram_bot")
        self.document_store = DocumentStore()
        self.openai_service = OpenAIService()
        self.cache_manager = CacheManager(ttl=3600)
        self.conversation_manager = ConversationManager(max_history=10)
        self.query_router = QueryRouter()
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")

        if not self.token:
            self.logger.error("TELEGRAM_BOT_TOKEN не найден в переменных окружения!")
            raise ValueError("TELEGRAM_BOT_TOKEN не найден в переменных окружения!")

        self.application = Application.builder().token(self.token).build()
        self._setup_handlers()
        self.logger.info("Бот успешно инициализирован")
    
    def _setup_handlers(self):
        """Sets up command and message handlers"""
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("docs", self.list_documents))
        self.application.add_handler(CommandHandler("add_doc", self.add_text_document))
        self.application.add_handler(CommandHandler("delete", self.delete_document))
        self.application.add_handler(CommandHandler("clear", self.clear_history))
        self.application.add_handler(CommandHandler("routing", self.explain_routing))
        self.application.add_handler(MessageHandler(filters.Document.ALL, self.handle_document))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler for /start command"""
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"
        self.logger.info(f"Пользователь {username} (ID: {user_id}) запустил команду /start")
        welcome_message = (
            "👋 Привет! Я умный помощник для работы с университетскими документами.\n\n"
            "📚 Что я умею:\n"
            "• Загружать и анализировать документы (лекции, политики, заметки)\n"
            "• Отвечать на вопросы на основе загруженных документов\n"
            "• Быстро находить нужную информацию в длинных текстах\n\n"
            "📝 Как использовать:\n"
            "1. Отправь мне документ (TXT, PDF или текст)\n"
            "2. Задай вопрос о документе\n"
            "3. Получи краткий и точный ответ!\n\n"
            "Команды:\n"
            "/help - помощь\n"
            "/docs - список твоих документов\n"
            "/delete <ID> - удалить документ\n"
            "/clear - очистить историю диалога\n"
            "/routing - информация о query routing"
        )
        await update.message.reply_text(welcome_message)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler for /help command"""
        help_text = (
            "📖 Помощь по использованию бота:\n\n"
            "📤 Загрузка документов:\n"
            "Отправь текстовый файл (.txt), PDF файл (.pdf) или напиши текст сообщением.\n"
            "Бот автоматически:\n"
            "• Извлечет текст из PDF (если нужно)\n"
            "• Создаст краткое резюме документа\n"
            "• Сохранит документ для поиска\n"
            "• Подготовит его для ответов на вопросы\n\n"
            "❓ Задать вопрос:\n"
            "Напиши любой вопрос о загруженных документах.\n"
            "Бот найдет релевантную информацию и даст точный ответ.\n\n"
            "📋 Команды:\n"
            "/start - начать работу\n"
            "/docs - показать список твоих документов\n"
            "/add_doc - добавить текст как документ\n"
            "/delete <ID> - удалить документ\n"
            "/clear - очистить историю диалога\n"
            "/routing - информация о query routing (Modular RAG)\n"
            "/help - эта справка\n\n"
            "💡 Примеры вопросов:\n"
            "• \"Какие правила оценивания в курсе?\"\n"
            "• \"Что говорится о дедлайнах?\"\n"
            "• \"Объясни тему X из лекций\""
        )
        await update.message.reply_text(help_text)
    
    async def list_documents(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler for /docs command - shows user's document list"""
        user_id = update.effective_user.id
        documents = self.document_store.get_user_documents(user_id)
        
        if not documents:
            await update.message.reply_text("📭 У тебя пока нет загруженных документов.\n\nОтправь текстовый файл (TXT), PDF или текст, чтобы начать!")
            return
        
        message = "📚 Твои документы:\n\n"
        for i, doc in enumerate(documents, 1):
            message += f"{i}. {doc.get('title', 'Без названия')}\n"
            message += f"   ID: {doc.get('id', 'N/A')}\n\n"
        
        message += "🗑 Чтобы удалить документ, используй команду:\n/delete <ID>"
        
        await update.message.reply_text(message)
    
    async def add_text_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler for /add_doc command - adds text as document"""
        user_id = update.effective_user.id

        # Check if there's text after the command
        if not context.args:
            await update.message.reply_text(
                "📝 Использование: /add_doc <название документа>\n\n"
                "После отправки команды отправь текст документа следующим сообщением.\n"
                "Или используй: /add_doc Название документа <текст документа>"
            )
            return

        if len(context.args) > 1:
            title = context.args[0]
            text = " ".join(context.args[1:])
        else:
            title = " ".join(context.args)
            context.user_data['waiting_for_doc_text'] = True
            context.user_data['doc_title'] = title
            await update.message.reply_text(
                f"📝 Ожидаю текст документа для \"{title}\".\n\n"
                f"Отправь текст следующим сообщением."
            )
            return

        await self._process_text_document(update, user_id, title, text)
    
    async def _process_text_document(self, update: Update, user_id: int, title: str, text: str):
        """Processes adding a text document"""
        await update.message.reply_text("⏳ Обрабатываю документ...")
        
        try:
            doc_id = str(uuid.uuid4())
            success = self.document_store.add_document(
                document_id=doc_id,
                title=title,
                full_text=text,
                user_id=user_id
            )
            
            if success:
                await update.message.reply_text(
                    f"✅ Документ успешно загружен!\n\n"
                    f"📄 Название: {title}\n"
                    f"🆔 ID: {doc_id}\n\n"
                    f"Теперь ты можешь задавать вопросы о этом документе!"
                )
            else:
                await update.message.reply_text(
                    "❌ Ошибка при обработке документа. Попробуй еще раз."
                )
        except Exception as e:
            print(f"Ошибка при обработке текстового документа: {e}")
            await update.message.reply_text(
                "❌ Произошла ошибка при обработке документа. Попробуй еще раз."
            )
    
    async def delete_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler for /delete command - deletes document by ID"""
        user_id = update.effective_user.id
        
        if not context.args:
            await update.message.reply_text(
                "🗑 Использование: /delete <ID документа>\n\n"
                "Получить список документов с ID можно командой /docs."
            )
            return
        
        doc_id = context.args[0].strip()
        if not doc_id:
            await update.message.reply_text(
                "❌ Укажи корректный ID документа.\n\n"
                "Пример: /delete 123e4567-e89b-12d3-a456-426614174000"
            )
            return
        
        success = self.document_store.delete_document(doc_id, user_id)
        if success:
            await update.message.reply_text(
                f"🗑 Документ с ID {doc_id} успешно удален.\n\n"
                "Чтобы убедиться, можешь снова посмотреть список документов: /docs"
            )
        else:
            await update.message.reply_text(
                "❌ Не удалось найти документ с таким ID.\n\n"
                "Убедись, что указал правильный ID и документ загружен тобой.\n"
                "Получить список твоих документов: /docs"
            )

    async def clear_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler for /clear command - clears conversation history"""
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"

        self.logger.info(f"Пользователь {username} (ID: {user_id}) запустил команду /clear")

        stats = self.conversation_manager.get_stats(user_id)
        success = self.conversation_manager.clear_history(user_id)

        if success and stats['total_messages'] > 0:
            await update.message.reply_text(
                f"🧹 История диалога очищена!\n\n"
                f"Удалено сообщений: {stats['total_messages']}\n"
                f"Теперь я начну новый диалог с чистого листа."
            )
        else:
            await update.message.reply_text(
                "📭 История диалога пуста.\n\n"
                "Продолжай задавать вопросы!"
            )

    async def explain_routing(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler for /routing command - explains how query routing works"""
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"

        self.logger.info(f"Пользователь {username} (ID: {user_id}) запустил команду /routing")

        if context.args and len(context.args) > 0:
            test_query = " ".join(context.args)
            explanation = self.query_router.explain_routing(test_query)
            await update.message.reply_text(explanation)
        else:
            info_text = (
                "🧭 Query Routing для Modular RAG\n\n"
                "Бот автоматически анализирует твой запрос и выбирает оптимальную стратегию поиска:\n\n"
                "📌 Типы запросов:\n"
                "• FACTUAL - фактологические вопросы (кто, что, где, когда)\n"
                "• ANALYTICAL - аналитические вопросы (почему, как работает)\n"
                "• PROCEDURAL - процедурные вопросы (как сделать, шаги)\n"
                "• CONCEPTUAL - концептуальные вопросы (что такое, определение)\n"
                "• COMPARISON - сравнительные вопросы (различия, сходства)\n\n"
                "🎯 Стратегии поиска:\n"
                "• PRECISE - точный поиск (top_k=3, threshold=0.0)\n"
                "• BROAD - широкий поиск (top_k=7, threshold=0.0)\n"
                "• COMPREHENSIVE - полный поиск (top_k=10, threshold=0.0)\n\n"
                "💡 Использование:\n"
                "/routing <запрос> - анализ конкретного запроса\n\n"
                "Примеры:\n"
                "/routing Какой дедлайн проекта?\n"
                "/routing Что такое RAPTOR?\n"
                "/routing Сравни ColBERT и обычные embeddings"
            )
            await update.message.reply_text(info_text)

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler for document uploads (supports TXT and PDF)"""
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"
        document = update.message.document
        filename = document.file_name or ""

        self.logger.info(f"Пользователь {username} (ID: {user_id}) загружает документ: {filename}")

        file_type = "PDF" if filename.lower().endswith('.pdf') else "текстовый файл"
        await update.message.reply_text(f"⏳ Обрабатываю {file_type}...")

        try:
            file = await context.bot.get_file(document.file_id)
            file_content = await file.download_as_bytearray()
            file_bytes = bytes(file_content)

            text = None
            pdf_extractor = PDFExtractor()

            if PDFExtractor.is_pdf(file_bytes, filename):
                await update.message.reply_text("📄 Извлекаю текст из PDF...")
                text = pdf_extractor.extract_text_from_pdf(file_bytes)

                if text is None or not text.strip():
                    await update.message.reply_text(
                        "❌ Не удалось извлечь текст из PDF. Возможно, файл защищен паролем или содержит только изображения."
                    )
                    return

                text_length = len(text)
                await update.message.reply_text(
                    f"✅ Извлечено {text_length} символов из PDF. Обрабатываю..."
                )
            else:
                try:
                    text = file_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        text = file_bytes.decode('windows-1251')
                    except UnicodeDecodeError:
                        text = file_bytes.decode('utf-8', errors='ignore')
            
            if not text or not text.strip():
                await update.message.reply_text(
                    "❌ Файл пуст или не удалось извлечь текст. Попробуй другой файл."
                )
                return

            title = filename or f"Документ {uuid.uuid4().hex[:8]}"
            if title.endswith('.pdf') or title.endswith('.txt'):
                title = title.rsplit('.', 1)[0]

            doc_id = str(uuid.uuid4())
            success = self.document_store.add_document(
                document_id=doc_id,
                title=title,
                full_text=text,
                user_id=user_id
            )
            
            if success:
                file_type_emoji = "📕" if PDFExtractor.is_pdf(file_bytes, filename) else "📄"
                await update.message.reply_text(
                    f"✅ Документ успешно загружен!\n\n"
                    f"{file_type_emoji} Название: {title}\n"
                    f"🆔 ID: {doc_id}\n"
                    f"📊 Размер: {len(text)} символов\n\n"
                    f"Теперь ты можешь задавать вопросы о этом документе!"
                )
            else:
                await update.message.reply_text(
                    "❌ Ошибка при обработке документа. Попробуй еще раз."
                )
        except Exception as e:
            self.logger.error(f"Ошибка при обработке документа {filename}: {e}", exc_info=True)
            await update.message.reply_text(
                f"❌ Произошла ошибка при обработке документа: {str(e)}\n\n"
                f"Убедись, что файл в поддерживаемом формате (TXT или PDF)."
            )
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler for text messages (user questions)"""
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"
        query = update.message.text.strip()

        self.logger.info(f"Пользователь {username} (ID: {user_id}) отправил сообщение: '{query[:50]}...'")

        if context.user_data.get('waiting_for_doc_text'):
            title = context.user_data.pop('doc_title', 'Документ')
            context.user_data.pop('waiting_for_doc_text', None)
            await self._process_text_document(update, user_id, title, query)
            return

        user_docs = self.document_store.get_user_documents(user_id)
        if not user_docs:
            await update.message.reply_text(
                "📭 У тебя пока нет загруженных документов.\n\n"
                "Отправь текстовый файл (TXT) или PDF, или используй /add_doc для добавления текста!"
            )
            return

        await update.message.reply_text("🤔 Ищу ответ в документах...")

        try:
            routing_result = self.query_router.route(query)
            self.logger.info(
                f"Query routing: type={routing_result['query_type']}, "
                f"strategy={routing_result['strategy']}, "
                f"top_k={routing_result['top_k']}, "
                f"threshold={routing_result['similarity_threshold']}"
            )

            relevant_docs = self.document_store.search_documents(
                query,
                user_id,
                top_k=routing_result['top_k'],
                similarity_threshold=routing_result['similarity_threshold']
            )
            
            if not relevant_docs:
                await update.message.reply_text(
                    "😕 Не нашел релевантной информации в твоих документах.\n\n"
                    "Попробуй переформулировать вопрос или загрузи больше документов."
                )
                return

            context_parts = []
            max_tokens_per_doc = 10000
            max_total_tokens = 60000
            
            total_tokens = 0
            for doc in relevant_docs:
                if total_tokens >= max_total_tokens:
                    break

                doc_summary = doc.get('summary', '')
                full_text = doc.get('full_text', '')

                if len(full_text) < 5000:
                    doc_context = f"Документ: {doc['title']}\nРезюме: {doc_summary}\n\nСодержание:\n{full_text}"
                else:
                    relevant_chunks = self.openai_service.extract_relevant_chunks(
                        full_text,
                        query,
                        max_chunks=2,
                        chunk_size=1500
                    )
                    chunks_text = "\n\n".join([f"[Часть {i+1}]\n{chunk}" for i, chunk in enumerate(relevant_chunks)])
                    doc_context = f"Документ: {doc['title']}\nРезюме: {doc_summary}\n\nРелевантные части:\n{chunks_text}"

                doc_tokens = self.openai_service.estimate_tokens(doc_context)
                if doc_tokens > max_tokens_per_doc:
                    doc_context = self.openai_service.truncate_text(doc_context, max_tokens_per_doc)
                    doc_tokens = max_tokens_per_doc

                if total_tokens + doc_tokens <= max_total_tokens:
                    context_parts.append(doc_context)
                    total_tokens += doc_tokens
                else:
                    remaining_tokens = max_total_tokens - total_tokens
                    if remaining_tokens > 1000:
                        partial_context = self.openai_service.truncate_text(doc_context, remaining_tokens)
                        context_parts.append(partial_context + "\n[Документ обрезан из-за ограничения размера]")
                    break

            context = "\n\n---\n\n".join(context_parts)
            conversation_history = self.conversation_manager.get_history(user_id, limit=6)

            cached_answer = self.cache_manager.get(query, context, user_id)
            if cached_answer:
                self.logger.info(f"Возвращен кэшированный ответ для пользователя {user_id}")
                answer = cached_answer
                is_cached = True
            else:
                answer = self.openai_service.generate_answer(
                    query,
                    context,
                    conversation_history=conversation_history,
                    max_context_tokens=60000
                )
                self.cache_manager.set(query, context, user_id, answer)
                is_cached = False

            self.conversation_manager.add_user_message(user_id, query)
            self.conversation_manager.add_assistant_message(user_id, answer)

            docs_used = len(context_parts)
            response = f"📚 Ответ на основе твоих документов:\n\n{answer}\n\n"

            response += f"🧭 Query Type: {routing_result['query_type'].upper()}\n"
            response += f"🎯 Strategy: {routing_result['strategy'].upper()}\n"
            response += f"📄 Использовано документов: {docs_used}/{routing_result['top_k']}"

            if is_cached:
                response += " 💾"

            if docs_used < len(relevant_docs):
                response += f"\n⚠️ Из {len(relevant_docs)} найденных документов использовано {docs_used} (остальные обрезаны для экономии токенов)"
            
            await update.message.reply_text(response)
            
        except Exception as e:
            self.logger.error(f"Ошибка при обработке вопроса: {e}", exc_info=True)
            await update.message.reply_text(
                "❌ Произошла ошибка при обработке вопроса. Попробуй еще раз."
            )
    
    def run(self):
        """Starts the bot"""
        self.logger.info("🚀 Бот запущен и готов к работе!")
        print("🚀 Бот запущен!")
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    bot = UniversityDocumentBot()
    bot.run()

