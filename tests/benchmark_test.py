"""Automatic benchmarking system for RAG system - tests retrieval accuracy and answer quality"""
import json
import time
from typing import List, Dict, Tuple
from src.document_store_simple import DocumentStore
from src.openai_service import OpenAIService
from utils.logger_config import setup_logger
import statistics

class RAGBenchmark:
    """System for automatic RAG benchmarking"""

    def __init__(self, document_store: DocumentStore, openai_service: OpenAIService):
        self.logger = setup_logger("benchmark")
        self.document_store = document_store
        self.openai_service = openai_service
        self.results = []

    def load_test_dataset(self, filepath: str) -> List[Dict]:
        """Loads test dataset from JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            self.logger.info(f"Загружено {len(dataset)} тестовых вопросов")
            return dataset
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке датасета: {e}")
            return []

    def calculate_retrieval_precision(self, retrieved_docs: List[Dict], expected_titles: List[str]) -> float:
        """Calculates precision: relevant retrieved / total retrieved"""
        if not retrieved_docs:
            return 0.0

        retrieved_titles = [doc.get('title', '') for doc in retrieved_docs]
        relevant_retrieved = sum(1 for title in retrieved_titles if any(exp in title for exp in expected_titles))

        precision = relevant_retrieved / len(retrieved_docs)
        return precision

    def calculate_retrieval_recall(self, retrieved_docs: List[Dict], expected_titles: List[str]) -> float:
        """Calculates recall: relevant retrieved / total relevant"""
        if not expected_titles:
            return 1.0

        retrieved_titles = [doc.get('title', '') for doc in retrieved_docs]
        relevant_retrieved = sum(1 for exp in expected_titles if any(exp in title for title in retrieved_titles))

        recall = relevant_retrieved / len(expected_titles)
        return recall

    def calculate_f1_score(self, precision: float, recall: float) -> float:
        """Calculates F1-score"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def evaluate_answer_quality(self, generated_answer: str, ground_truth: str) -> Dict[str, float]:
        """Evaluates answer quality using keyword overlap, length similarity, and semantic similarity"""
        gen_keywords = set(generated_answer.lower().split())
        truth_keywords = set(ground_truth.lower().split())
        keyword_overlap = len(gen_keywords & truth_keywords) / max(len(truth_keywords), 1)

        len_similarity = min(len(generated_answer), len(ground_truth)) / max(len(generated_answer), len(ground_truth), 1)
        semantic_score = self._evaluate_semantic_similarity(generated_answer, ground_truth)

        return {
            'keyword_overlap': keyword_overlap,
            'length_similarity': len_similarity,
            'semantic_similarity': semantic_score
        }

    def _evaluate_semantic_similarity(self, generated: str, ground_truth: str) -> float:
        """Uses LLM to evaluate semantic similarity of answers"""
        try:
            prompt = f"""Оцени семантическую схожесть двух ответов по шкале от 0 до 1.
0 = полностью разные ответы
1 = идентичные по смыслу ответы

Ответ 1 (сгенерированный): {generated}

Ответ 2 (эталонный): {ground_truth}

Верни только число от 0 до 1."""

            response = self.openai_service.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Ты эксперт по оценке качества текстов."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=10
            )

            score_text = response.choices[0].message.content.strip()
            score = float(score_text.split()[0])
            return min(max(score, 0.0), 1.0)
        except Exception as e:
            self.logger.warning(f"Ошибка при оценке семантической схожести: {e}")
            return 0.5

    def run_single_test(self, test_case: Dict, user_id: int) -> Dict:
        """Runs a single test"""
        question = test_case['question']
        expected_titles = test_case.get('expected_doc_titles', [])
        ground_truth = test_case.get('ground_truth_answer', '')

        self.logger.info(f"Тестирование вопроса: {question}")

        start_time = time.time()
        retrieved_docs = self.document_store.search_documents(question, user_id, top_k=3)
        retrieval_time = time.time() - start_time

        precision = self.calculate_retrieval_precision(retrieved_docs, expected_titles)
        recall = self.calculate_retrieval_recall(retrieved_docs, expected_titles)
        f1_score = self.calculate_f1_score(precision, recall)

        if retrieved_docs:
            context = "\n\n".join([doc['full_text'][:2000] for doc in retrieved_docs])
            start_time = time.time()
            generated_answer = self.openai_service.generate_answer(question, context)
            generation_time = time.time() - start_time

            # Evaluate answer quality
            if ground_truth:
                answer_quality = self.evaluate_answer_quality(generated_answer, ground_truth)
            else:
                answer_quality = None
        else:
            generated_answer = "Не найдено релевантных документов"
            generation_time = 0
            answer_quality = None

        result = {
            'question': question,
            'retrieval_precision': precision,
            'retrieval_recall': recall,
            'f1_score': f1_score,
            'retrieval_time': retrieval_time,
            'generation_time': generation_time,
            'total_time': retrieval_time + generation_time,
            'retrieved_docs_count': len(retrieved_docs),
            'generated_answer': generated_answer,
            'answer_quality': answer_quality
        }

        return result

    def run_benchmark(self, test_dataset: List[Dict], user_id: int) -> Dict:
        """
        Runs full benchmark on dataset

        Returns:
            Dictionary with aggregated metrics
        """
        self.logger.info(f"Запуск бенчмарка на {len(test_dataset)} тестовых вопросах")
        self.results = []

        for i, test_case in enumerate(test_dataset, 1):
            self.logger.info(f"Тест {i}/{len(test_dataset)}")
            result = self.run_single_test(test_case, user_id)
            self.results.append(result)

        # Aggregate results
        aggregated = self._aggregate_results()
        return aggregated

    def _aggregate_results(self) -> Dict:
        """Aggregates results from all tests"""
        if not self.results:
            return {}

        precisions = [r['retrieval_precision'] for r in self.results]
        recalls = [r['retrieval_recall'] for r in self.results]
        f1_scores = [r['f1_score'] for r in self.results]
        retrieval_times = [r['retrieval_time'] for r in self.results]
        generation_times = [r['generation_time'] for r in self.results]
        total_times = [r['total_time'] for r in self.results]

        # Aggregate answer quality metrics
        answer_qualities = [r['answer_quality'] for r in self.results if r['answer_quality']]
        if answer_qualities:
            avg_keyword_overlap = statistics.mean([q['keyword_overlap'] for q in answer_qualities])
            avg_semantic_similarity = statistics.mean([q['semantic_similarity'] for q in answer_qualities])
        else:
            avg_keyword_overlap = 0.0
            avg_semantic_similarity = 0.0

        aggregated = {
            'total_tests': len(self.results),
            'avg_retrieval_precision': statistics.mean(precisions),
            'avg_retrieval_recall': statistics.mean(recalls),
            'avg_f1_score': statistics.mean(f1_scores),
            'avg_retrieval_time': statistics.mean(retrieval_times),
            'avg_generation_time': statistics.mean(generation_times),
            'avg_total_time': statistics.mean(total_times),
            'avg_keyword_overlap': avg_keyword_overlap,
            'avg_semantic_similarity': avg_semantic_similarity,
            'max_retrieval_time': max(retrieval_times),
            'min_retrieval_time': min(retrieval_times),
        }

        return aggregated

    def save_results(self, filepath: str):
        """Saves benchmark results to JSON"""
        try:
            output = {
                'aggregated_metrics': self._aggregate_results(),
                'individual_results': self.results
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            self.logger.info(f"Результаты бенчмарка сохранены в {filepath}")
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении результатов: {e}")

    def print_summary(self):
        """Prints brief summary of results"""
        if not self.results:
            print("Нет результатов для отображения")
            return

        aggregated = self._aggregate_results()

        print("\n" + "=" * 60)
        print("📊 РЕЗУЛЬТАТЫ БЕНЧМАРКА RAG СИСТЕМЫ")
        print("=" * 60)
        print(f"\n📈 Метрики Retrieval:")
        print(f"  • Precision: {aggregated['avg_retrieval_precision']:.2%}")
        print(f"  • Recall: {aggregated['avg_retrieval_recall']:.2%}")
        print(f"  • F1-Score: {aggregated['avg_f1_score']:.2%}")
        print(f"\n⚡ Метрики производительности:")
        print(f"  • Среднее время retrieval: {aggregated['avg_retrieval_time']:.3f}s")
        print(f"  • Среднее время генерации: {aggregated['avg_generation_time']:.3f}s")
        print(f"  • Среднее общее время: {aggregated['avg_total_time']:.3f}s")
        print(f"\n📝 Качество ответов:")
        print(f"  • Keyword Overlap: {aggregated['avg_keyword_overlap']:.2%}")
        print(f"  • Semantic Similarity: {aggregated['avg_semantic_similarity']:.2%}")
        print(f"\n📊 Общая статистика:")
        print(f"  • Всего тестов: {aggregated['total_tests']}")
        print("=" * 60 + "\n")


def run_example_benchmark():
    """Example of running benchmark"""
    from src.document_store_simple import DocumentStore
    from src.openai_service import OpenAIService

    # Initialization
    doc_store = DocumentStore()
    openai_service = OpenAIService()
    benchmark = RAGBenchmark(doc_store, openai_service)

    # Load test dataset
    test_dataset = benchmark.load_test_dataset('test_dataset.json')

    if test_dataset:
        # Run benchmark (using test user with ID 999999)
        results = benchmark.run_benchmark(test_dataset, user_id=999999)

        # Print results
        benchmark.print_summary()

        # Save results
        benchmark.save_results('benchmark_results.json')
    else:
        print("Тестовый датасет не найден. Создайте test_dataset.json")


if __name__ == "__main__":
    run_example_benchmark()
