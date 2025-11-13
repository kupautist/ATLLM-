import io
from typing import Optional
import pypdf

class PDFExtractor:
    """Class for extracting text from PDF files"""

    @staticmethod
    def extract_text_from_pdf(pdf_bytes: bytes) -> Optional[str]:
        """Extracts text from a PDF file"""
        try:
            pdf_file = io.BytesIO(pdf_bytes)
            pdf_reader = pypdf.PdfReader(pdf_file)

            text_parts = []
            total_pages = len(pdf_reader.pages)

            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_parts.append(page_text)
                except Exception as e:
                    print(f"Ошибка при извлечении текста со страницы {page_num}: {e}")
                    continue

            if not text_parts:
                return None

            full_text = "\n\n".join(text_parts)

            import re
            full_text = re.sub(r' +', ' ', full_text)
            full_text = re.sub(r'\n{3,}', '\n\n', full_text)

            return full_text.strip()

        except pypdf.errors.PdfReadError as e:
            print(f"Ошибка при чтении PDF: {e}")
            return None
        except Exception as e:
            print(f"Неожиданная ошибка при извлечении текста из PDF: {e}")
            return None
    
    @staticmethod
    def is_pdf(file_bytes: bytes, filename: Optional[str] = None) -> bool:
        """Checks if the file is a PDF by extension or magic bytes"""
        if filename:
            filename_lower = filename.lower()
            if filename_lower.endswith('.pdf'):
                return True

        try:
            if file_bytes.startswith(b'%PDF-'):
                return True
        except:
            pass

        return False

