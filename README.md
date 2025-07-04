# PDF Translator

This script translates the text content of a PDF file from a source language to a target language, attempting to preserve the original layout, fonts, and colors.

## Features

- Extracts text from PDFs, including using Optical Character Recognition (OCR) for scanned images.
- Translates text using the `deep-translator` library.
- Rebuilds the PDF with the translated text, overlaying it on the original page structure.
- Attempts to preserve the original text's font, color, and size.
- Graceful font handling:
    1.  **Local Fonts:** Prioritizes using font files (`.ttf`, `.otf`) placed in a local `fonts` directory.
    2.  **Embedded Fonts:** Extracts and uses fonts directly from the source PDF.
    3.  **Built-in Fonts:** Falls back to standard PDF fonts (like Helvetica, Times, Courier) if the original font cannot be found or embedded.

## Requirements

- Python 3.6+
- Tesseract OCR engine. You must install it on your system.
    - **Debian/Ubuntu:** `sudo apt-get install tesseract-ocr`
    - **macOS (Homebrew):** `brew install tesseract`
    - **Windows:** Download from the official [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) page.

The required Python packages are listed in `requirements.txt`.

## Installation

1.  **Clone the repository or download the files.**

2.  **Install Tesseract OCR** for your operating system (see above).

3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4.  **Install the Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the script from your terminal:

```bash
python pdf_translator.py <input_pdf_path> <output_pdf_path> --target_lang <language_code>
```

**Arguments:**

-   `input_pdf_path`: The path to the PDF file you want to translate.
-   `output_pdf_path`: The path where the translated PDF will be saved.
-   `--target_lang`: (Optional) The language code to translate the text into (e.g., 'en' for English, 'es' for Spanish). Defaults to 'en'.
-   `--source_lang`: (Optional) The language code of the source text. If not provided, the translator will attempt to auto-detect it.

**Example:**

```bash
python pdf_translator.py my_document.pdf translated_document.pdf --target_lang fr
```

## Improving Font Quality

For the best visual results, the script needs access to the same fonts used in the original PDF.

1.  **Run the script once.** When the script runs, it will print messages to the console if it cannot find a specific font.
    ```
    INFO: Font 'Arial-BoldMT' not found. For best results, place a corresponding font file in 'fonts'. Falling back to 'helv-bold'.
    ```

2.  **Create the `fonts` directory.** If it doesn't exist, the script will create a directory named `fonts` in the same folder.

3.  **Add the missing fonts.** Find the required font files (`.ttf` or `.otf`) on your system or download them and place them inside the `fonts` directory. The script will automatically detect and use them on the next run.
