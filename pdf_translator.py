import os
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/5/tessdata'
import fitz  # PyMuPDF
from langdetect import detect
import onnxruntime as ort
from PIL import Image
import pytesseract
import io
import requests
import zipfile
from deep_translator import GoogleTranslator
from tqdm import tqdm
import argparse
from transformers import pipeline, AutoTokenizer
from optimum.intel import OVModelForSeq2SeqLM
import re

# 1. Load PDF and extract text/layout
# 2. Detect or accept source language
# 3. Translate text using a local model
# 4. Rebuild PDF with translated text, preserving layout and fonts

def extract_pdf_content(input_pdf):
    doc = fitz.open(input_pdf)
    pages_content = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")['blocks']
        page_items = []
        for block in blocks:
            if block['type'] == 0:  # text block
                for line in block['lines']:
                    for span in line['spans']:
                        item = {
                            'text': span['text'],
                            'bbox': span['bbox'],
                            'font': span['font'],
                            'size': span['size'],
                            'flags': span['flags'],
                            'color': span['color'],
                            'origin': (span['origin'][0], span['origin'][1]) if 'origin' in span else None
                        }
                        page_items.append(item)
            elif block['type'] == 1:  # image block
                # You can extract images if needed
                pass
        pages_content.append(page_items)
    return pages_content

def extract_pdf_content_with_ocr(input_pdf):
    doc = fitz.open(input_pdf)
    pages_content = []

    # Extract font buffers to embed them later
    font_buffers = {}
    # Iterate through pages to get fonts for older PyMuPDF versions
    for page in doc:
        for font in page.get_fonts(full=True):
            xref = font[0]
            name = font[4] # Corrected index for font name
            if name not in font_buffers:
                try:
                    font_info = doc.extract_font(xref)
                    if len(font_info) >= 5 and font_info[4]:
                        buffer = font_info[4]
                        font_buffers[name] = buffer
                except Exception as e:
                    print(f"Warning: Could not extract font '{name}' (xref={xref}): {e}")

    for page_num in tqdm(range(len(doc)), desc="Extracting content"): 
        page = doc[page_num]
        # Revert to simple span-by-span extraction
        blocks = page.get_text("dict")['blocks']
        page_items = []
        for block in blocks:
            if block['type'] == 0:  # text block
                for line in block['lines']:
                    for span in line['spans']:
                        item = {
                            'text': span['text'],
                            'bbox': span['bbox'],
                            'font': span['font'],
                            'size': span['size'],
                            'flags': span['flags'],
                            'color': span['color'],
                            'origin': (span['origin'][0], span['origin'][1]) if 'origin' in span else None,
                            'type': 'text'
                        }
                        page_items.append(item)
        
        # Extract images and perform OCR
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            img_rect = page.get_image_bbox(img_info)
            base_image = doc.extract_image(xref)
            image_bytes = base_image['image']
            image = Image.open(io.BytesIO(image_bytes))
            ocr_data = pytesseract.image_to_data(image, lang='fra', output_type=pytesseract.Output.DICT)
            
            lines = {}
            for i in range(len(ocr_data['text'])):
                if ocr_data['text'][i].strip():
                    x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                    x0 = img_rect.x0 + (x / image.width) * img_rect.width
                    y0 = img_rect.y0 + (y / image.height) * img_rect.height
                    x1 = img_rect.x0 + ((x + w) / image.width) * img_rect.width
                    y1 = img_rect.y0 + ((y + h) / image.height) * img_rect.height
                    line_num = ocr_data['line_num'][i]
                    if line_num not in lines:
                        lines[line_num] = []
                    lines[line_num].append({'text': ocr_data['text'][i], 'bbox': (x0, y0, x1, y1)})
            
            for line_num in sorted(lines.keys()):
                line_words = lines[line_num]
                line_text = " ".join([word['text'] for word in line_words])
                x0 = min(word['bbox'][0] for word in line_words)
                y0 = min(word['bbox'][1] for word in line_words)
                x1 = max(word['bbox'][2] for word in line_words)
                y1 = max(word['bbox'][3] for word in line_words)
                bbox = (x0, y0, x1, y1)
                
                item = {
                    'text': line_text,
                    'bbox': bbox,
                    'font': 'helv', # Default font for OCR text
                    'size': (y1 - y0) * 0.8, # Estimate font size
                    'color': 0, # Default color (black)
                    'type': 'ocr'
                }
                page_items.append(item)

        pages_content.append(page_items)
    return pages_content, font_buffers


def detect_language(pages_content, source_lang=None):
    if source_lang:
        return source_lang
    # Concatenate text from the first few pages for detection
    text_sample = " ".join([item['text'] for page in pages_content[:3] for item in page if item['text'].strip()])
    return detect(text_sample)

def translate_texts(texts, src_lang, tgt_lang, device="CPU"):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    translator = None
    loaded_device = None

    # Attempt to load the local model, with fallback from specified device to CPU
    devices_to_try = [d.strip() for d in device.split(',')]
    if "CPU" not in devices_to_try:
        devices_to_try.append("CPU")

    for dev in devices_to_try:
        try:
            print(f"Attempting to load model on {dev}...")
            model = OVModelForSeq2SeqLM.from_pretrained(model_name, export=True, compile=True, device=dev)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            translator = pipeline("translation", model=model, tokenizer=tokenizer)
            loaded_device = dev
            print(f"Successfully loaded model on {dev}.")
            break  # Success, exit the loop
        except Exception as e:
            print(f"Warning: Could not load the model on {dev}. Error: {e}")
            if dev == devices_to_try[-1]: # If this was the last device to try
                print("Local model loading failed on all available devices.")
            else:
                print("Attempting to fall back to the next device...")

    # If local model loading failed on all devices, fall back to Google Translate
    if translator is None:
        print("Falling back to Google Translate.")
        try:
            g_translator = GoogleTranslator(source=src_lang, target=tgt_lang)
            translated_texts = []
            batch_size = 50
            for i in tqdm(range(0, len(texts), batch_size), desc="Translating with Google"):
                batch = texts[i:i + batch_size]
                try:
                    translated_batch = g_translator.translate_batch(batch)
                    # The output of translate_batch can be None if a batch fails
                    if translated_batch:
                        translated_texts.extend(translated_batch)
                    else:
                        print(f"Warning: A batch translation returned None. Skipping batch.")
                        translated_texts.extend(["" for _ in batch])
                except Exception as batch_e:
                    print(f"Warning: Could not translate a batch with Google Translate. Error: {batch_e}")
                    translated_texts.extend(["" for _ in batch])
            return translated_texts
        except Exception as e:
            print(f"FATAL: Google Translate also failed. Error: {e}")
            # Return empty strings for all texts as a last resort
            return ["" for _ in texts]

    # If local model loaded successfully
    translated_texts = []
    for text in tqdm(texts, desc=f"Translating with local model on {loaded_device}"):
        if text.strip():
            try:
                result = translator(text)
                translated_texts.append(result[0]['translation_text'])
            except Exception as e:
                print(f"Warning: Could not translate text: '{text}'. Error: {e}")
                translated_texts.append("") # Append empty string on error
        else:
            translated_texts.append("")
    return translated_texts

def save_to_txt(pages_content, output_file):
    """Saves the extracted text content to a .txt file, with page separators."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, page_items in enumerate(tqdm(pages_content, desc="Writing to TXT")):
            f.write(f"--- Page {i+1} ---\n\n")
            for item in page_items:
                if item['text'].strip():
                    f.write(item['text'] + '\n')
            f.write('\n')

def count_words(pages_content):
    """Counts the total number of words in the extracted content."""
    total_words = 0
    for page_items in tqdm(pages_content, desc="Counting words"):
        for item in page_items:
            if item['text'].strip():
                total_words += len(item['text'].split())
    return total_words

def download_google_fonts(font_list, fonts_dir="fonts"):
    """
    Downloads font files from Google Fonts by fetching the CSS and parsing font URLs.
    """
    # These are standard system fonts and are not available on Google Fonts.
    unavailable_fonts = ["arial", "times new roman", "courier new", "verdana", "georgia"]

    if not os.path.exists(fonts_dir):
        os.makedirs(fonts_dir)
        print(f"Created font directory at: {os.path.abspath(fonts_dir)}")

    print(f"Starting download of {len(font_list)} font families from Google Fonts...")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    # Regex to find `url(...)` in the CSS
    url_pattern = re.compile(r'url\((https://[^\)]+)\)')

    for font_name in tqdm(font_list, desc="Downloading Fonts"):
        if font_name.lower() in unavailable_fonts:
            print(f"\nInfo: '{font_name}' is a standard system font and is not available on Google Fonts. Skipping.")
            continue
        try:
            # 1. Fetch the CSS
            font_url_name = font_name.replace(" ", "+")
            css_url = f"https://fonts.googleapis.com/css?family={font_url_name}:400,700,400italic,700italic"
            css_response = requests.get(css_url, headers=headers, timeout=15)
            
            # Handle cases where the font is not on Google Fonts (like Arial)
            if css_response.status_code == 400:
                print(f"\nInfo: '{font_name}' is not available on Google Fonts and will be skipped.")
                continue

            css_response.raise_for_status()
            css_content = css_response.text

            # 2. Parse CSS to find font file URLs
            font_urls = url_pattern.findall(css_content)
            if not font_urls:
                print(f"\nWarning: Could not find font URLs for '{font_name}'. It might be unavailable or the API changed.")
                continue

            # 3. Download each font file
            for font_url in set(font_urls): # Use set to avoid downloading duplicates
                try:
                    font_response = requests.get(font_url, headers=headers, timeout=30)
                    font_response.raise_for_status()
                    
                    # Extract filename from URL or headers
                    filename = font_url.split('/')[-1]
                    
                    # Save the font file
                    with open(os.path.join(fonts_dir, filename), 'wb') as f:
                        f.write(font_response.content)

                except requests.exceptions.RequestException as e:
                    print(f"\nError downloading a font file for '{font_name}' from {font_url}: {e}")

        except requests.exceptions.RequestException as e:
            print(f"\nError fetching CSS for '{font_name}': {e}")
        except Exception as e:
            print(f"\nAn unexpected error occurred for font '{font_name}': {e}")

def get_builtin_font(fontname, for_length_calc=False):
    # Map common PDF font names to PyMuPDF built-in fonts
    fontname_lower = fontname.lower()

    is_bold = "bold" in fontname_lower
    is_italic = "italic" in fontname_lower or "oblique" in fontname_lower

    base_font_short = "helv"  # Default short name
    base_font_full = "helvetica" # Default full name

    if "times" in fontname_lower:
        base_font_short = "ti"
        base_font_full = "times"
    elif "courier" in fontname_lower:
        base_font_short = "cour"
        base_font_full = "courier"

    # For get_text_length, we need more explicit names.
    if for_length_calc:
        if base_font_full == "times":
            if is_bold and is_italic: return "times-bolditalic"
            if is_bold: return "times-bold"
            if is_italic: return "times-italic"
            return "times-roman"
        else: # covers helvetica and courier
            if is_bold and is_italic: return f"{base_font_full}-boldoblique"
            if is_bold: return f"{base_font_full}-bold"
            if is_italic: return f"{base_font_full}-oblique"
            return base_font_full

    # For insert_text, short names are fine.
    if is_bold and is_italic:
        return f"{base_font_short}bi"  # e.g., helvbi, tibi, courbi
    elif is_bold:
        return f"{base_font_short}b"  # e.g., helvb, tib, courb
    elif is_italic:
        return f"{base_font_short}i"  # e.g., helvi, tii, couri
    else:
        return base_font_short  # e.g., helv, tiro, cour

def int_to_rgb01(color_int):
    # Convert integer color to (r, g, b) tuple in 0-1 range
    r = ((color_int >> 16) & 255) / 255.0
    g = ((color_int >> 8) & 255) / 255.0
    b = (color_int & 255) / 255.0
    return (r, g, b)

def find_best_font_match(font_family, is_bold, is_italic, local_fonts_map):
    """
    Finds the best matching font file from the local font map.
    
    Args:
        font_family (str): The base font family name (e.g., 'roboto').
        is_bold (bool): Whether a bold style is requested.
        is_italic (bool): Whether an italic style is requested.
        local_fonts_map (dict): A map of font families to their style variations.

    Returns:
        str: The path to the best matching font file, or None if no suitable match is found.
    """
    family_styles = local_fonts_map.get(font_family)
    if not family_styles:
        return None

    # 1. Perfect match (e.g., Bold Italic -> Bold Italic)
    if is_bold and is_italic and family_styles.get('bolditalic'):
        return family_styles['bolditalic']
    if is_bold and not is_italic and family_styles.get('bold'):
        return family_styles['bold']
    if is_italic and not is_bold and family_styles.get('italic'):
        return family_styles['italic']
    if not is_bold and not is_italic and family_styles.get('regular'):
        return family_styles['regular']

    # 2. Partial match (e.g., Bold Italic -> Bold or Italic if not available)
    if is_bold and family_styles.get('bold'):
        return family_styles['bold']
    if is_italic and family_styles.get('italic'):
        return family_styles['italic']

    # 3. Fallback to regular style of the same family
    if family_styles.get('regular'):
        return family_styles['regular']
    
    # 4. Fallback to any available style in the family
    return next(iter(family_styles.values()), None)

def rebuild_pdf(input_pdf, pages_content, translated_texts, font_buffers, output_pdf):
    # Create a new PDF with only the translated text overlays
    src_doc = fitz.open(input_pdf)
    new_doc = fitz.open()

    # Ensure the local fonts directory exists and load local fonts
    fonts_dir = "fonts"
    if not os.path.exists(fonts_dir):
        os.makedirs(fonts_dir)
        print(f"INFO: Created a '{fonts_dir}' directory.")
        print(f"      To improve font quality, place any required .ttf or .otf font files here.")
    
    # Create a structured map of local fonts: {family: {style: path}}
    local_fonts_map = {}
    if os.path.exists(fonts_dir):
        for f in os.listdir(fonts_dir):
            if f.lower().endswith(('.ttf', '.otf')):
                path = os.path.join(fonts_dir, f)
                filename = os.path.splitext(f)[0].lower()
                
                # Very basic style parsing from filename
                if '-' in filename:
                    family, *style_parts = filename.split('-')
                    style_str = "".join(style_parts)
                else:
                    family = filename
                    style_str = "regular"

                family = family.replace(" ", "")
                
                style = 'regular' # Default
                if 'bold' in style_str and 'italic' in style_str: style = 'bolditalic'
                elif 'bold' in style_str: style = 'bold'
                elif 'italic' in style_str or 'oblique' in style_str: style = 'italic'
                elif 'regular' in style_str or style_str == "": style = 'regular'

                if family not in local_fonts_map:
                    local_fonts_map[family] = {}
                local_fonts_map[family][style] = path

    # A counter to keep track of which translated text to use
    text_index = 0
    
    # Regex to detect various bullet points (hyphens, dashes, bullets, asterisks, numbered lists)
    bullet_regex = re.compile(r'^\s*([-\–\—\•\*\d]+\.?\s*)\s*')

    for page_num, page_items in enumerate(tqdm(pages_content, desc="Rebuilding PDF")):
        if page_num >= len(src_doc):
            break
        src_page = src_doc[page_num]
        # Create a new blank page with the same size
        page_rect = src_page.rect
        new_page = new_doc.new_page(width=page_rect.width, height=page_rect.height)
        
        # Keep track of fonts already inserted on this page
        inserted_fonts = set()

        for item in page_items:
            if not item['text'].strip():
                continue

            # Get the correct translated text.
            try:
                translated_text = translated_texts[text_index]
                text_index += 1
            except IndexError:
                print("Warning: Ran out of translated texts.")
                break
            
            if not translated_text.strip():
                continue

            rect = fitz.Rect(item['bbox'])
            original_font_name = item.get('font', 'helv')
            fontsize = item['size']
            color = int_to_rgb01(item['color'])
            font_to_use = None

            # --- Font Selection Logic ---
            font_name_lower = original_font_name.lower()
            is_bold = 'bold' in font_name_lower or 'black' in font_name_lower or 'heavy' in font_name_lower
            is_italic = 'italic' in font_name_lower or 'oblique' in font_name_lower
            base_font_family = re.split(r'[,_-]', font_name_lower)[0].replace(" ", "")
            font_path = find_best_font_match(base_font_family, is_bold, is_italic, local_fonts_map)
            font_alias = f"local_{base_font_family}_{is_bold}_{is_italic}"

            # 1. Try local font file
            if font_path:
                if font_alias not in inserted_fonts:
                    try:
                        new_page.insert_font(fontname=font_alias, fontfile=font_path)
                        inserted_fonts.add(font_alias)
                        font_to_use = font_alias
                    except Exception: pass
                else:
                    font_to_use = font_alias
            
            # 2. Try embedded font buffer
            if not font_to_use:
                font_buffer = font_buffers.get(original_font_name)
                if font_buffer:
                    buffer_font_alias = original_font_name
                    if buffer_font_alias not in inserted_fonts:
                        try:
                            new_page.insert_font(fontname=buffer_font_alias, fontbuffer=font_buffer)
                            inserted_fonts.add(buffer_font_alias)
                            font_to_use = buffer_font_alias
                        except Exception: pass
                    else:
                        font_to_use = buffer_font_alias

            # 3. Fallback to built-in font
            if not font_to_use:
                font_to_use = get_builtin_font(original_font_name)

            # --- Bullet Point Handling ---
            bullet_match = bullet_regex.match(item['text'])
            bullet_text = ""
            if bullet_match:
                bullet_text = bullet_match.group(1)
                # Redraw the original bullet
                bullet_font = get_builtin_font(original_font_name) # Use a reliable font for bullet
                bullet_width = fitz.get_text_length(bullet_text, fontname=bullet_font, fontsize=fontsize)
                new_page.insert_text(rect.bottom_left, bullet_text, fontname=bullet_font, fontsize=fontsize, color=color)
                # Adjust rect for the translated text
                rect.x0 += bullet_width

            # --- Dynamic Font Sizing ---
            text_len = len(translated_text)
            if text_len > 0:
                available_width = rect.width
                # This can fail if font is not found, so wrap it
                try:
                    text_width = fitz.get_text_length(translated_text, fontname=font_to_use, fontsize=fontsize)
                    if text_width > available_width and available_width > 0:
                        fontsize = max(1, fontsize * (available_width / text_width))
                except Exception:
                    # If length calculation fails, use a fallback font for it
                    try:
                        fallback_calc_font = get_builtin_font(font_to_use, for_length_calc=True)
                        text_width = fitz.get_text_length(translated_text, fontname=fallback_calc_font, fontsize=fontsize)
                        if text_width > available_width and available_width > 0:
                            fontsize = max(1, fontsize * (available_width / text_width))
                    except Exception as calc_e:
                        print(f"Warning: Could not calculate text width even with fallback. Fontsize may be incorrect. Error: {calc_e}")


            # --- Insert Text ---
            try:
                new_page.insert_text(rect.bottom_left, translated_text,
                                     fontname=font_to_use,
                                     fontsize=fontsize,
                                     color=color)
            except Exception as e:
                print(f"Warning: Could not insert text with font '{font_to_use}'. Error: {e}")
                # As a last resort, try inserting with the most basic font
                try:
                    fallback_font = "helvetica"
                    new_page.insert_text(rect.bottom_left, translated_text,
                                         fontname=fallback_font,
                                         fontsize=fontsize,
                                         color=color)
                except Exception as final_e:
                    print(f"FATAL: Ultimate fallback failed for text block. Skipping. Error: {final_e}")


    # Save the new PDF
    new_doc.save(output_pdf, garbage=4, deflate=True, clean=True)

def main():
    parser = argparse.ArgumentParser(description="Translate a PDF file.")
    parser.add_argument("input_pdf", help="The path to the input PDF file.")
    parser.add_argument("-o", "--output_pdf", default=None, help="The path to the output PDF file. If not provided, it defaults to 'filename+US.pdf' for English translation.")
    parser.add_argument("--target_lang", default="en", help="The target language for translation.")
    parser.add_argument("--source_lang", default=None, help="The source language of the text. If not provided, it will be auto-detected.")
    parser.add_argument("--fast", action="store_true", help="Enable fast mode: extract text to a .txt file without translation.")
    parser.add_argument("--count", action="store_true", help="Count the total words in the document.")
    parser.add_argument("--output_txt", default="extracted_text.txt", help="The path to the output .txt file for fast mode.")
    parser.add_argument("--device", default="GPU,CPU", help="The device(s) to use for translation, in order of preference (e.g., 'GPU,CPU').")
    parser.add_argument("--download-fonts", action="store_true", help="Download a curated list of popular fonts from Google Fonts to the 'fonts' directory.")

    args = parser.parse_args()

    if args.download_fonts:
        popular_fonts = [
            "Roboto", "Open Sans", "Lato", "Montserrat", "Oswald",
            "Source Sans Pro", "Raleway", "PT Sans", "Lora", "Noto Sans",
            "Arial", "Times New Roman", "Courier New", "Verdana", "Georgia"
        ]
        download_google_fonts(popular_fonts)
        print("\nFont download process finished.")
        print(f"Fonts have been saved to the '{os.path.abspath('fonts')}' directory.")
        print("You can now run the script again to perform a translation.")
        return

    output_pdf = args.output_pdf
    if output_pdf is None:
        if args.target_lang == 'en':
            base, ext = os.path.splitext(args.input_pdf)
            output_pdf = f"{base}+US.pdf"
        else:
            # Default for other languages if not specified
            base, ext = os.path.splitext(args.input_pdf)
            output_pdf = f"{base}_{args.target_lang}.pdf"


    print("Extracting content from the PDF...")
    pages_content, font_buffers = extract_pdf_content_with_ocr(args.input_pdf)

    if args.count:
        print("Counting words...")
        total_words = count_words(pages_content)
        print(f"Total word count: {total_words}")
        return

    if args.fast:
        print(f"Fast mode enabled. Saving extracted text to {args.output_txt}...")
        save_to_txt(pages_content, args.output_txt)
        print("Text extraction complete.")
        return

    print("Detecting source language...")
    src_lang = detect_language(pages_content, source_lang=args.source_lang)
    print(f"Source language detected: {src_lang}")

    print("Translating text...")
    all_texts_to_translate = []
    # Handle bullet points before sending for translation
    bullet_regex = re.compile(r'^\s*([-\–\—\•\*\d]+\.?\s*)\s*')
    for page in pages_content:
        for item in page:
            if item['text'].strip():
                text_to_translate = item['text']
                match = bullet_regex.match(text_to_translate)
                if match:
                    # If it's a list item, only translate the text after the bullet
                    text_to_translate = text_to_translate[match.end():]
                all_texts_to_translate.append(text_to_translate)

    translated_texts = translate_texts(all_texts_to_translate, src_lang, args.target_lang, args.device)

    print("Rebuilding the PDF with translated text...")
    rebuild_pdf(args.input_pdf, pages_content, translated_texts, font_buffers, output_pdf)

    print(f"PDF translation complete. Output saved to {output_pdf}")

if __name__ == "__main__":
    main()
