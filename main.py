import argparse
import logging
import os
import re
import markdown
import fitz  # PyMuPDF
from openai import OpenAI
from typing import List, Optional
import time

# Конфигурация логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Конфигурация OpenAI
OPENAI_API_KEY = "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"

class TextProcessor:
    def __init__(self, api_key: str):
        """Инициализирует процессор текста с клиентом OpenAI."""
        self.blocks = []
        try:
            self.client = OpenAI(api_key=api_key)
            logger.info("Клиент OpenAI успешно инициализирован.")
        except Exception as e:
            logger.error(f"Ошибка инициализации клиента OpenAI: {str(e)}")
            raise Exception("Не удалось инициализировать клиент OpenAI. Проверьте API-ключ и соединение.")

    def read_text_file(self, file_path: str) -> str:
        """Читает текст из файлов .txt или .md."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                if file_path.endswith('.md'):
                    content = markdown.markdown(content)
                self.blocks = [para.strip() for para in content.split('\n\n') if para.strip() and not para.strip().isdigit()]
                logger.info(f"Извлечено {len(self.blocks)} блоков из текстового файла {file_path}")
                return '\n\n'.join(self.blocks)
        except Exception as e:
            logger.error(f"Ошибка при чтении текстового файла {file_path}: {e}")
            return ""

    def read_docx_file(self, file_path: str) -> str:
        """Читает текст из файлов .docx."""
        try:
            import docx
            doc = docx.Document(file_path)
            self.blocks = [para.text.strip() for para in doc.paragraphs if para.text.strip() and not para.text.strip().isdigit()]
            logger.info(f"Извлечено {len(self.blocks)} блоков из файла DOCX {file_path}")
            return '\n\n'.join(self.blocks)
        except Exception as e:
            logger.error(f"Ошибка при чтении файла DOCX {file_path}: {e}")
            return ""

    def read_pdf_file(self, file_path: str, output_original: str = 'output/original.txt') -> str:
        """Извлекает текст из PDF, разбивает на абзацы и сохраняет в original.txt."""
        try:
            doc = fitz.open(file_path)
            blocks = []
            block_number = 1

            for page in doc:
                page_blocks = page.get_text("blocks", sort=False)
                left_column = []
                right_column = []
                page_width = page.rect.width
                column_threshold = page_width / 2

                for block in page_blocks:
                    x0, y0, x1, y1, block_text = block[:5]
                    block_text = block_text.strip()
                    if block_text and not block_text.isdigit():
                        if x0 < column_threshold:
                            left_column.append((y0, block_text))
                        else:
                            right_column.append((y0, block_text))

                left_column.sort(key=lambda x: x[0])
                right_column.sort(key=lambda x: x[0])

                for _, block_text in left_column:
                    blocks.append({'number': block_number, 'text': block_text})
                    block_number += 1
                for _, block_text in right_column:
                    blocks.append({'number': block_number, 'text': block_text})
                    block_number += 1

            os.makedirs(os.path.dirname(output_original), exist_ok=True)
            with open(output_original, 'w', encoding='utf-8') as f:
                for block in blocks:
                    if len(block['text']) > 2:
                        f.write(f"Блок {block['number']}: {block['text']}\n\n")

            doc.close()
            self.blocks = [block['text'] for block in blocks]
            logger.info(f"Извлечено {len(self.blocks)} блоков из файла PDF {file_path}, сохранено в {output_original}")
            return '\n\n'.join(self.blocks)
        except Exception as e:
            logger.error(f"Ошибка при чтении файла PDF {file_path}: {e}")
            return ""

    def read_input_file(self, file_path: str, output_original: str = 'output/original.txt') -> str:
        """Читает входной файл в зависимости от его расширения."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".txt", ".md"]:
            return self.read_text_file(file_path)
        elif ext == ".docx":
            return self.read_docx_file(file_path)
        elif ext == ".pdf":
            return self.read_pdf_file(file_path, output_original)
        else:
            logger.error(f"Неподдерживаемый формат файла: {ext}")
            return ""

    def clean_block(self, block: str) -> str:
        """Очищает блок текста, удаляя все после символов '===' или '<s>'."""
        cleaned = re.split(r'===|<s>', block)[0].strip()
        logger.debug(f"Очищенный блок: {cleaned[:50]}...")
        return cleaned

    def split_block(self, block: str, max_length: int = 500) -> List[str]:
        """Разделяет длинный блок текста на части, если он превышает max_length символов."""
        if len(block) <= max_length:
            return [block]
        words = block.split()
        chunks = []
        current_chunk = []
        current_length = 0
        for word in words:
            if current_length + len(word) + 1 > max_length:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word) + 1
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        logger.debug(f"Блок разделён на {len(chunks)} частей")
        return chunks

    def paraphrase_block(self, block: str, theme: str, block_index: int) -> str:
        """Перефразирует отдельный блок текста с использованием API OpenAI."""
        chunks = self.split_block(block, max_length=500)
        paraphrased_chunks = []
        for chunk_idx, chunk in enumerate(chunks, 1):
            prompt = f"""
            Перефразируйте текст на русском языке, строго сохраняя академический стиль и точную научную терминологию для публикации в области {theme}. 
            Следуйте этим правилам:
            1. Сохраняйте исходный смысл, не добавляйте новых фактов и не удаляйте существующую информацию.
            2. Поддерживайте структуру текста: сохраняйте количество предложений и их порядок, избегая излишнего переструктурирования.
            3. Если текст начинается или заканчивается дефисом, оставьте неизменным. Если в начале блока первое слово не понятно и с маленькой буквы, оставь его неизменным.
            4. Если текст слишком короток, отсутствует или содержит бессмысленные знаки или цифры, верните его неизменным.
            5. Удалите префиксы типа "Блок n", если они присутствуют.
            6. Верните только перефразированный текст без дополнительных комментариев.
            7. Перефразированный текст должен быть близок по объёму к оригиналу (±10% слов).
            Текст: {chunk}
            """
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "Вы — эксперт по перефразированию научных текстов на русском языке в академическом стиле."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2000,
                    temperature=0.4
                )
                paraphrased_text = response.choices[0].message.content.strip()
                if not paraphrased_text:
                    logger.warning(f"Часть {chunk_idx} блока {block_index}: пустой ответ от API, возвращается оригинал")
                    paraphrased_chunks.append(chunk)
                else:
                    paraphrased_chunks.append(paraphrased_text)
                    logger.info(f"Часть {chunk_idx} блока {block_index} успешно перефразирована")
            except Exception as e:
                logger.error(f"Ошибка перефразирования части {chunk_idx} блока {block_index}: {str(e)}")
                if "rate_limit_exceeded" in str(e) or "insufficient_quota" in str(e):
                    logger.info("Превышен лимит запросов или квота. Ожидание 30 секунд...")
                    time.sleep(30)
                    try:
                        response = self.client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": "Вы — эксперт по перефразированию научных текстов на русском языке в академическом стиле."},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=2000,
                            temperature=0.4
                        )
                        paraphrased_text = response.choices[0].message.content.strip()
                        if not paraphrased_text:
                            logger.warning(f"Часть {chunk_idx} блока {block_index}: пустой ответ при повторной попытке, возвращается оригинал")
                            paraphrased_chunks.append(chunk)
                        else:
                            paraphrased_chunks.append(paraphrased_text)
                            logger.info(f"Часть {chunk_idx} блока {block_index} успешно перефразирована после повторной попытки")
                    except Exception as retry_e:
                        logger.error(f"Повторная попытка не удалась для части {chunk_idx} блока {block_index}: {retry_e}")
                        paraphrased_chunks.append(chunk)
                else:
                    paraphrased_chunks.append(chunk)
        return " ".join(paraphrased_chunks)

    def process_text(self, text: str, theme: str) -> str:
        """Обрабатывает текст: перефразирует блоки."""
        blocks = [block.strip() for block in text.split('\n\n') if block.strip()]
        processed_blocks = []
        for i, block in enumerate(blocks, 1):
            if not block or len(block) <= 2:
                logger.info(f"Блок {i} пустой или слишком короткий, пропущен")
                continue
            cleaned_block = self.clean_block(block)
            if not cleaned_block:
                logger.info(f"Блок {i} после очистки пуст, пропущен")
                continue
            paraphrased_block = self.paraphrase_block(cleaned_block, theme, i)
            if paraphrased_block == cleaned_block:
                logger.warning(f"Блок {i} не был перефразирован, сохранён оригинал")
            processed_blocks.append(f"Блок {i}: {paraphrased_block}")
        return '\n\n'.join(processed_blocks) if processed_blocks else ""

    def save_file(self, content: str, output_path: str):
        """Сохраняет перефразированный текст в .txt."""
        try:
            output_dir = os.path.dirname(output_path) or '.'
            os.makedirs(output_dir, exist_ok=True)
            if not os.access(output_dir, os.W_OK):
                raise PermissionError(f"Нет прав на запись в директорию: {output_dir}")
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except PermissionError:
                    raise Exception(f"Ошибка доступа: невозможно перезаписать {output_path}")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Текст сохранён в {output_path}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении файла {output_path}: {e}")
            raise

    def process(self, input_path: str, output_report_path: str, theme: str = "РЕНТГЕНОДИАГНОСТИКА ЗАБОЛЕВАНИЙ КОСТЕЙ И СУСТАВОВ"):
        """Основная функция обработки: чтение, перефразирование и сохранение."""
        try:
            text = self.read_input_file(input_path)
            if not text:
                raise Exception("Не удалось извлечь текст из файла")
            processed_text = self.process_text(text, theme)
            if not processed_text:
                raise Exception("Не удалось обработать текст")
            self.save_file(processed_text, output_report_path)
            return True, "Обработка успешно завершена"
        except Exception as e:
            logger.error(f"Ошибка обработки: {str(e)}")
            return False, f"Ошибка: {str(e)}"

def main():
    """Основная функция для обработки текста из файла или аргументов командной строки."""
    parser = argparse.ArgumentParser(description="Обработка текста: перефразирование с использованием API OpenAI.")
    parser.add_argument("--input-file", type=str, default="input/Кости_глава_1.pdf",
                        help="Путь к входному файлу (.pdf, .txt, .md, .docx) (по умолчанию: input/Кости_глава_1.pdf)")
    parser.add_argument("--output-file", type=str, default="output/paraphrased.txt",
                        help="Путь к выходному файлу .txt (по умолчанию: output/paraphrased.txt)")
    parser.add_argument("--theme", type=str, default=None,
                        help="Тематика текста (по умолчанию: запрашивается у пользователя)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API-ключ OpenAI (по умолчанию: запрашивается у пользователя)")
    args = parser.parse_args()

    try:
        # Запрашиваем тему, если не указана в аргументах
        theme = args.theme if args.theme else input("Введите тематику текста: ").strip()
        if not theme:
            theme = "РЕНТГЕНОДИАГНОСТИКА ЗАБОЛЕВАНИЙ КОСТЕЙ И СУСТАВОВ"
            logger.info(f"Тема не указана, используется значение по умолчанию: {theme}")

        # Запрашиваем API-ключ, если не указан в аргументах
        api_key = args.api_key if args.api_key else input("Введите API-ключ OpenAI: ").strip()
        if not api_key:
            api_key = OPENAI_API_KEY
            logger.info("API-ключ не указан, используется значение по умолчанию")

        processor = TextProcessor(api_key=api_key)
        success, message = processor.process(args.input_file, args.output_file, theme)
        logger.info(message)
    except Exception as e:
        logger.error(f"Ошибка в main: {str(e)}")
        logger.info("Рекомендации по устранению ошибки:")
        logger.info("1. Убедитесь, что API-ключ OpenAI действителен: https://platform.openai.com/")
        logger.info("2. Проверьте наличие интернет-соединения.")
        logger.info("3. Убедитесь, что у вас достаточно квоты для API OpenAI.")
        logger.info("4. Проверьте, что установлены все библиотеки: `pip install -r requirements.txt`")

if __name__ == "__main__":
    main()