import subprocess
import threading
import aiohttp
import asyncio
import random
import logging
import re
import json
import os
import csv
import sys
import time
from urllib.parse import urljoin, urlparse
from lxml import html
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from plyer import notification
import pandas as pd
from datetime import datetime
import psutil
import aiofiles
from aiohttp import ClientTimeout

# Настройка логирования
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

# Устанавливаем BASE_PATH как директорию скрипта
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Создаем папку для логов
LOG_DIR = os.path.join(BASE_PATH, "output", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

file_handler = logging.FileHandler(os.path.join(LOG_DIR, "scraper_logs.txt"))
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

class TextHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.insert(tk.END, msg + '\n')
        self.text_widget.see(tk.END)
        self.text_widget.config(state=tk.DISABLED)

# Фильтрация текста (ещё ослаблена)
def is_valid_text(text, keywords=None):
    text = text.strip()
    if not text:  # Убрали минимальную длину
        return False
    if re.search(r'https?://\S+|www\.\S+', text):  # Только ссылки отсекаем
        return False
    if keywords and not any(keyword.lower() in text.lower() for keyword in keywords):
        return False
    return True

def is_valid_url(url, allowed_domain):
    parsed_url = urlparse(url)
    # Исключаем служебные страницы и категории
    if any(x in url for x in ["action=edit", "redlink=1", "/w/index.php", "/wiki/Категория:", "/wiki/Шаблон:", "/wiki/Служебная:"]):
        return False
    return bool(parsed_url.netloc) and bool(parsed_url.scheme) and parsed_url.netloc == allowed_domain

# Проверка доступности страницы
async def test_page_availability(session, url, retries=5):
    for attempt in range(retries):
        try:
            async with session.head(url, allow_redirects=True, timeout=ClientTimeout(total=10)) as response:
                if response.status == 429:
                    logger.warning(f"429 для {url}, ждём {attempt + 1} сек")
                    await asyncio.sleep(attempt + 1)
                    continue
                return response.status == 200
        except asyncio.TimeoutError:
            logger.error(f"Таймаут для {url}, попытка {attempt+1}/{retries}")
            if attempt < retries - 1:
                await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Попытка {attempt+1}/{retries} для {url}: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(1)
    return False

# Получение данных со страницы
async def fetch_page_data(session, url, base_url, allowed_domain, keywords=None, xpath_filter=None):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36', 'Accept-Encoding': 'gzip, deflate'}
        async with session.get(url, headers=headers, timeout=ClientTimeout(total=15)) as response:
            logger.info(f"Запрос {url}: статус {response.status}")
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type:
                logger.info(f"Пропущен URL {url}: не HTML ({content_type})")
                return [], []
            text = await response.text()
            data, links = await process_page_data(text, base_url, allowed_domain, keywords, xpath_filter)
            logger.info(f"Собрано {len(data)} записей и {len(links)} ссылок с {url}")
            return data, links
    except asyncio.TimeoutError:
        logger.error(f"Таймаут при загрузке {url}")
        return [], []
    except UnicodeDecodeError as e:
        logger.error(f"Ошибка декодирования {url}: {e}")
        return [], []
    except Exception as e:
        logger.error(f"Ошибка при обработке {url}: {e}")
        return [], []

# Парсинг страницы
async def process_page_data(text, base_url, allowed_domain, keywords=None, xpath_filter=None):
    tree = html.fromstring(text)
    if xpath_filter:
        text_data = set(tree.xpath(xpath_filter))
    else:
        text_data = set(tree.xpath('//text()'))
    text_data = {t.strip() for t in text_data if is_valid_text(t, keywords)}

    links = tree.xpath('//a/@href')
    links = [urljoin(base_url, link) for link in links if is_valid_url(urljoin(base_url, link), allowed_domain)]

    return list(text_data), links

# Асинхронное сохранение чекпойнтов
async def save_checkpoint_async(data, file_path, file_format, chunk_size=5000, total_collected=0):
    if len(data) >= chunk_size or (len(data) > 0 and time.time() - last_activity_time > 30):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(list(data), ensure_ascii=False))
        logger.info(f"Сохранён чекпойнт: {len(data)} записей в {file_path}, всего собрано: {total_collected + len(data)}")
        return set()
    return data

# Сохранение visited_urls
async def save_visited_urls(visited_urls, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
        await f.write(json.dumps(list(visited_urls), ensure_ascii=False))
    logger.info(f"Сохранено {len(visited_urls)} посещённых URL в {file_path}")

# Загрузка visited_urls
async def load_visited_urls(file_path):
    if os.path.exists(file_path):
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            return set(json.loads(content)) if content.strip() else set()
    return set()

# Сохранение urls_to_visit на диск
async def save_urls_to_visit(urls_to_visit, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
        await f.write(json.dumps(urls_to_visit, ensure_ascii=False))
    logger.info(f"Сохранено {len(urls_to_visit)} URL для посещения в {file_path}")

# Загрузка urls_to_visit с диска
async def load_urls_to_visit(file_path):
    if os.path.exists(file_path):
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            return json.loads(content) if content.strip() else []
    return []

# Объединение чекпойнтов
async def combine_checkpoints(checkpoint_base, output_file, file_format, cycle_num=None):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    checkpoint_files = [f for f in os.listdir(os.path.dirname(checkpoint_base)) if f.startswith(os.path.basename(checkpoint_base)[:-5]) and f.endswith('.json')]
    
    hash_file = output_file + '.hash'
    unique_data = set()
    if os.path.exists(hash_file):
        with open(hash_file, 'r', encoding='utf-8') as f:
            unique_data = set(f.read().splitlines())

    new_entries = 0
    with open(output_file, 'a', encoding='utf-8') as f_out, open(hash_file, 'a', encoding='utf-8') as f_hash:
        for file in checkpoint_files:
            async with aiofiles.open(os.path.join(os.path.dirname(checkpoint_base), file), 'r', encoding='utf-8') as infile:
                content = await infile.read()
                if content.strip():
                    chunk = json.loads(content)
                    for item in chunk:
                        item_str = json.dumps(item, ensure_ascii=False)
                        item_hash = str(hash(item_str))
                        if item_hash not in unique_data:
                            unique_data.add(item_hash)
                            f_out.write(item_str + '\n')
                            f_hash.write(item_hash + '\n')
                            new_entries += 1
    
    logger.info(f"Данные объединены в {output_file} (JSONL-формат, добавлено уникальных записей: {new_entries}) для цикла {cycle_num if cycle_num is not None else 'финального'}")

# Основной процесс сбора данных
async def get_data_from_site(start_urls, num_units=10000000, min_delay=0.1, max_delay=0.5,
                            stop_event=None, pause_event=None, max_concurrent=125, max_depth=7,
                            keywords=None, xpath_filter=None, proxies=None, progress_callback=None):
    global last_activity_time
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(BASE_PATH, "output", f"scraper_run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_file = os.path.join(output_dir, f"scraped_checkpoint_{timestamp}.json")
    visited_urls_file = os.path.join(output_dir, f"visited_urls_{timestamp}.json")
    urls_to_visit_file = os.path.join(output_dir, f"urls_to_visit_{timestamp}.json")
    all_data_file = os.path.join(output_dir, f"scraped_all_{timestamp}.json")
    cycle_interval = 100000
    cycle_num = 0
    max_queue_size = 50000

    urls_to_visit = [(url, 0) for url in start_urls]
    allowed_domain = urlparse(start_urls[0]).netloc
    total_stats = {"pages_processed": 0, "total_collected": 0, "start_time": time.time()}
    max_urls_in_memory = 1000
    max_visited_in_memory = 5000
    global_visited_urls = await load_visited_urls(visited_urls_file)
    active_proxies = set(proxies) if proxies else set()

    while total_stats["total_collected"] < num_units and urls_to_visit:
        collected_data = set()
        visited_urls = set()
        semaphore = asyncio.Semaphore(max_concurrent)
        stats = {"pages_processed": 0, "start_time": time.time(), "total_collected": 0}
        checkpoint_counter = 0
        last_activity_time = time.time()

        async def process_url(url, depth, session, proxy_list):
            async with semaphore:
                if url in visited_urls or url in global_visited_urls or depth > max_depth:
                    return [], []
                if stop_event and stop_event.is_set():
                    return [], []
                if pause_event and pause_event.is_set():
                    await pause_event.wait()

                visited_urls.add(url)
                global_visited_urls.add(url)
                proxy = random.choice(list(proxy_list)) if proxy_list else "No proxy"
                logger.info(f"Обрабатываем (глубина {depth}): {url} через {proxy}")
                session._default_proxies = {'http': proxy, 'https': proxy} if proxy_list else {}
                stats["pages_processed"] += 1
                if await test_page_availability(session, url):
                    data, links = await fetch_page_data(session, url, url, allowed_domain, keywords, xpath_filter)
                    if not data and not links:
                        logger.warning(f"Пустой результат для {url}")
                        if proxy_list and proxy in active_proxies and len(active_proxies) > 1:
                            active_proxies.discard(proxy)
                            logger.warning(f"Прокси {proxy} исключён из активных (пустой результат)")
                    return data, links
                logger.warning(f"Страница {url} недоступна")
                if proxy_list and proxy in active_proxies and len(active_proxies) > 1:
                    active_proxies.discard(proxy)
                    logger.warning(f"Прокси {proxy} исключён из активных (недоступна)")
                return [], []

        connector = aiohttp.TCPConnector(ssl=False, limit=200)
        async with aiohttp.ClientSession(connector=connector) as session:
            if proxies:
                logger.info(f"Используем ротацию прокси: {len(active_proxies)} активных")
            else:
                logger.info("Прокси не используются")
            logger.info(f"Начинаем цикл {cycle_num} с {len(urls_to_visit)} URL")
            checkpoint_interval = 5000
            progress_update_interval = 50000
            last_progress_update = 0
            empty_batches = 0  # Счётчик пустых батчей

            while stats["total_collected"] < cycle_interval and urls_to_visit and total_stats["total_collected"] < num_units:
                if stop_event and stop_event.is_set():
                    logger.info("Парсинг остановлен пользователем.")
                    break
                if pause_event and pause_event.is_set():
                    logger.info("Парсинг на паузе...")
                    await pause_event.wait()

                if len(urls_to_visit) > max_urls_in_memory:
                    excess_urls = urls_to_visit[max_urls_in_memory:]
                    urls_to_visit = urls_to_visit[:max_urls_in_memory]
                    if len(excess_urls) + len(urls_to_visit) > max_queue_size:
                        excess_urls = excess_urls[:max_queue_size - len(urls_to_visit)]
                    await save_urls_to_visit(excess_urls, urls_to_visit_file)

                if len(global_visited_urls) > max_visited_in_memory:
                    await save_visited_urls(global_visited_urls, visited_urls_file)
                    global_visited_urls = set()

                batch = urls_to_visit[:max_concurrent]
                urls_to_visit = urls_to_visit[max_concurrent:]
                logger.info(f"Обрабатываем батч из {len(batch)} URL")

                tasks = [process_url(url, depth, session, active_proxies) for url, depth in batch]
                try:
                    results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=600)
                except asyncio.TimeoutError:
                    logger.warning("Таймаут на batch, завершаем задачи.")
                    results = [([], []) for _ in batch]
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                except Exception as e:
                    logger.error(f"Критическая ошибка в batch: {e}")
                    results = [([], []) for _ in batch]

                batch_collected = False
                for (current_url, depth), result in zip(batch, results):
                    if isinstance(result, Exception):
                        logger.error(f"Ошибка обработки {current_url}: {result}")
                        continue
                    if isinstance(result, tuple) and len(result) == 2:
                        page_data, link_urls = result
                        if page_data:
                            collected_data.update(page_data)
                            batch_collected = True
                        urls_to_visit.extend([(link, depth + 1) for link in link_urls if link not in global_visited_urls])
                        logger.info(f"Собрано в текущем наборе: {len(collected_data)}, всего в цикле: {stats['total_collected'] + len(collected_data)}")

                        if len(collected_data) >= checkpoint_interval:
                            checkpoint_file_part = f"{checkpoint_file[:-5]}_part{checkpoint_counter}.json"
                            stats["total_collected"] += len(collected_data)
                            total_stats["total_collected"] += len(collected_data)
                            collected_data = await save_checkpoint_async(collected_data, checkpoint_file_part, 'json', total_collected=total_stats["total_collected"])
                            checkpoint_counter += 1
                            last_activity_time = time.time()

                        if stats["total_collected"] + len(collected_data) - last_progress_update >= progress_update_interval and progress_callback:
                            last_progress_update = stats["total_collected"] + len(collected_data)
                            progress_callback((total_stats["total_collected"] / num_units) * 100, total_stats)

                process = psutil.Process(os.getpid())
                memory_usage = process.memory_info().rss / (1024 * 1024)
                logger.info(f"Память процесса: {memory_usage:.2f} МБ")
                if memory_usage > 1000:
                    logger.warning(f"Память превысила 1000 МБ: {memory_usage:.2f} МБ, выгружаем.")
                    if collected_data:
                        checkpoint_file_part = f"{checkpoint_file[:-5]}_part{checkpoint_counter}.json"
                        stats["total_collected"] += len(collected_data)
                        total_stats["total_collected"] += len(collected_data)
                        collected_data = await save_checkpoint_async(collected_data, checkpoint_file_part, 'json', total_collected=total_stats["total_collected"])
                        checkpoint_counter += 1
                    visited_urls.clear()
                    await save_visited_urls(global_visited_urls, visited_urls_file)
                    last_activity_time = time.time()

                if not batch_collected:
                    empty_batches += 1
                    if empty_batches >= 3:  # Если 3 батча подряд пустые
                        logger.warning("Три пустых батча подряд, очищаем очередь и добавляем начальные URL.")
                        urls_to_visit = [(url, 0) for url in start_urls if url not in global_visited_urls]
                        empty_batches = 0
                        if os.path.exists(urls_to_visit_file):
                            os.remove(urls_to_visit_file)
                else:
                    empty_batches = 0

                if time.time() - last_activity_time > 60 or (not batch_collected and not urls_to_visit):
                    logger.warning("Detected hang or empty collection, saving remainder and restarting cycle.")
                    if collected_data:
                        checkpoint_file_part = f"{checkpoint_file[:-5]}_part{checkpoint_counter}.json"
                        stats["total_collected"] += len(collected_data)
                        total_stats["total_collected"] += len(collected_data)
                        await save_checkpoint_async(collected_data, checkpoint_file_part, 'json', total_collected=total_stats["total_collected"])
                        await save_visited_urls(global_visited_urls, visited_urls_file)
                    break

                await asyncio.sleep(random.uniform(min_delay, max_delay))

            if collected_data:
                checkpoint_file_part = f"{checkpoint_file[:-5]}_part{checkpoint_counter}.json"
                stats["total_collected"] += len(collected_data)
                total_stats["total_collected"] += len(collected_data)
                await save_checkpoint_async(collected_data, checkpoint_file_part, 'json', total_collected=total_stats["total_collected"])
                await save_visited_urls(global_visited_urls, visited_urls_file)

            await combine_checkpoints(checkpoint_file, all_data_file, 'json', cycle_num)

            excess_urls = await load_urls_to_visit(urls_to_visit_file)
            if len(excess_urls) + len(urls_to_visit) > max_queue_size:
                excess_urls = excess_urls[:max_queue_size - len(urls_to_visit)]
            urls_to_visit.extend(excess_urls)
            if os.path.exists(urls_to_visit_file):
                os.remove(urls_to_visit_file)

            total_stats["pages_processed"] += stats["pages_processed"]
            intermediate_file = os.path.join(output_dir, f"scraped_intermediate_{cycle_num}.json")
            await combine_checkpoints(checkpoint_file, intermediate_file, 'json', cycle_num)
            cycle_num += 1
            await asyncio.sleep(5)

    elapsed_time = time.time() - total_stats["start_time"]
    total_stats["start_time"] = elapsed_time
    
    for f in os.listdir(output_dir):
        if f.startswith(f"scraped_checkpoint_{timestamp}"):
            os.remove(os.path.join(output_dir, f))
    
    logger.info(f"Completed. Total collected: {total_stats['total_collected']}. Pages: {total_stats['pages_processed']}. Time: {elapsed_time:.2f} sec")
    return [], total_stats

def save_data_to_file(data, file_path, file_format, indent=False):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if file_format == 'json':
        mode = 'a' if os.path.exists(file_path) else 'w'
        with open(file_path, mode, encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4 if indent else None)
            if mode == 'a':
                f.write('\n')
    elif file_format == 'csv':
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Text"])
            for item in data:
                writer.writerow([item])
    elif file_format == 'xlsx':
        df = pd.DataFrame(data, columns=["Text"])
        df.to_excel(file_path, index=False)
    logger.info(f"Saved to {file_path} ({file_format.upper()})")

def save_logs(log_text, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(log_text.get("1.0", tk.END))
    logger.info(f"Logs saved to {file_path}")

def send_notification(title, message):
    notification.notify(title=title, message=message, timeout=10)

class WebScraperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Super Web Scraper")
        self.folder_path = None
        self.scraping_running = False
        self.paused = False
        self.stop_event = asyncio.Event()
        self.pause_event = asyncio.Event()

        self.url_label = ttk.Label(root, text="Initial URLs (comma-separated):")
        self.url_label.grid(row=0, column=0, padx=10, pady=5)
        self.url_entry = ttk.Entry(root, width=50)
        self.url_entry.grid(row=0, column=1, columnspan=3, padx=10, pady=5)
        self.url_entry.insert(0, "https://ru.wikipedia.org/wiki/Вторая_мировая_война")

        self.num_units_label = ttk.Label(root, text="Data amount:")
        self.num_units_label.grid(row=1, column=0, padx=10, pady=5)
        self.num_units_entry = ttk.Entry(root, width=50)
        self.num_units_entry.grid(row=1, column=1, columnspan=3, padx=10, pady=5)
        self.num_units_entry.insert(0, "10000000")  # 10M records

        self.min_delay_label = ttk.Label(root, text="Min delay (s):")
        self.min_delay_label.grid(row=2, column=0, padx=10, pady=5)
        self.min_delay_entry = ttk.Entry(root, width=15)
        self.min_delay_entry.grid(row=2, column=1, padx=10, pady=5)
        self.min_delay_entry.insert(0, "0.1")

        self.max_delay_label = ttk.Label(root, text="Max delay (s):")
        self.max_delay_label.grid(row=2, column=2, padx=10, pady=5)
        self.max_delay_entry = ttk.Entry(root, width=15)
        self.max_delay_entry.grid(row=2, column=3, padx=10, pady=5)
        self.max_delay_entry.insert(0, "0.5")

        self.max_depth_label = ttk.Label(root, text="Max depth:")
        self.max_depth_label.grid(row=3, column=0, padx=10, pady=5)
        self.max_depth_entry = ttk.Entry(root, width=15)
        self.max_depth_entry.grid(row=3, column=1, padx=10, pady=5)
        self.max_depth_entry.insert(0, "7")  # Увеличено до 7

        self.max_concurrent_label = ttk.Label(root, text="Max threads:")
        self.max_concurrent_label.grid(row=3, column=2, padx=10, pady=5)
        self.max_concurrent_entry = ttk.Entry(root, width=15)
        self.max_concurrent_entry.grid(row=3, column=3, padx=10, pady=5)
        self.max_concurrent_entry.insert(0, "125")

        self.keywords_label = ttk.Label(root, text="Keywords (comma-separated):")
        self.keywords_label.grid(row=4, column=0, padx=10, pady=5)
        self.keywords_entry = ttk.Entry(root, width=50)
        self.keywords_entry.grid(row=4, column=1, columnspan=3, padx=10, pady=5)

        self.xpath_label = ttk.Label(root, text="XPath filter (optional):")
        self.xpath_label.grid(row=5, column=0, padx=10, pady=5)
        self.xpath_entry = ttk.Entry(root, width=50)
        self.xpath_entry.grid(row=5, column=1, columnspan=3, padx=10, pady=5)

        self.proxy_label = ttk.Label(root, text="Proxies (http://... one per line):")
        self.proxy_label.grid(row=6, column=0, padx=10, pady=5)
        self.proxy_text = tk.Text(root, height=15, width=50)
        self.proxy_text.grid(row=6, column=1, columnspan=3, padx=10, pady=5)

        self.file_format_label = ttk.Label(root, text="File format:")
        self.file_format_label.grid(row=7, column=0, padx=10, pady=5)
        self.file_format_combobox = ttk.Combobox(root, values=["json", "csv", "xlsx"], width=15)
        self.file_format_combobox.grid(row=7, column=1, padx=10, pady=5)
        self.file_format_combobox.set("json")

        self.browse_button = ttk.Button(root, text="Select folder", command=self.browse_folder)
        self.browse_button.grid(row=8, column=0, padx=10, pady=5)
        self.start_button = ttk.Button(root, text="Start", command=self.run_scraper)
        self.start_button.grid(row=8, column=1, padx=10, pady=5)
        self.pause_button = ttk.Button(root, text="Pause", command=self.toggle_pause, state=tk.DISABLED)
        self.pause_button.grid(row=8, column=2, padx=10, pady=5)
        self.stop_button = ttk.Button(root, text="Stop", command=self.stop_scraping, state=tk.DISABLED)
        self.stop_button.grid(row=8, column=3, padx=10, pady=5)

        self.save_config_button = ttk.Button(root, text="Save config", command=self.save_config)
        self.save_config_button.grid(row=9, column=0, padx=10, pady=5)
        self.load_config_button = ttk.Button(root, text="Load config", command=self.load_config)
        self.load_config_button.grid(row=9, column=1, padx=10, pady=5)
        self.save_logs_button = ttk.Button(root, text="Save logs", command=self.save_logs)
        self.save_logs_button.grid(row=9, column=2, padx=10, pady=5)

        self.progress_label = ttk.Label(root, text="Progress:")
        self.progress_label.grid(row=10, column=0, padx=10, pady=5)
        self.progress_bar = ttk.Progressbar(root, length=400, mode='determinate')
        self.progress_bar.grid(row=10, column=1, columnspan=3, padx=10, pady=5)

        self.stats_label = ttk.Label(root, text="Stats: pages - 0, time - 0 sec")
        self.stats_label.grid(row=11, column=0, columnspan=4, padx=10, pady=5)

        self.log_text = tk.Text(root, height=15, width=80)
        self.log_text.grid(row=12, column=0, columnspan=4, padx=10, pady=10)
        self.log_text.config(state=tk.DISABLED)

        log_handler = TextHandler(self.log_text)
        log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(log_handler)

    def browse_folder(self):
        self.folder_path = filedialog.askdirectory()
        if self.folder_path:
            logger.info(f"Folder selected: {self.folder_path}")

    def update_progress(self, value, stats):
        self.progress_bar['value'] = value
        elapsed_time = time.time() - stats["start_time"] if self.scraping_running else stats["start_time"]
        self.stats_label.config(text=f"Stats: pages - {stats['pages_processed']}, total collected - {stats['total_collected']}, time - {elapsed_time:.2f} sec")
        if self.scraping_running:
            self.root.after(1000, lambda: self.update_progress(value, stats))

    def validate_inputs(self):
        start_urls = [url.strip() for url in self.url_entry.get().split(',') if url.strip()]
        num_units = self.num_units_entry.get()
        min_delay = self.min_delay_entry.get()
        max_delay = self.max_delay_entry.get()
        max_depth = self.max_depth_entry.get()
        max_concurrent = self.max_concurrent_entry.get()
        file_format = self.file_format_combobox.get()
        keywords = self.keywords_entry.get()
        xpath_filter = self.xpath_entry.get() or None
        proxies = [p.strip() for p in self.proxy_text.get("1.0", tk.END).splitlines() if p.strip()]

        if not all([start_urls, num_units, min_delay, max_delay, max_depth, max_concurrent, file_format]):
            send_notification("Error", "Fill in all required fields.")
            return None

        try:
            num_units = int(num_units)
            max_depth = int(max_depth)
            max_concurrent = int(max_concurrent)
            min_delay = float(min_delay)
            max_delay = float(max_delay)
            if num_units <= 0 or max_depth <= 0 or max_concurrent <= 0 or min_delay < 0 or max_delay < min_delay:
                raise ValueError
        except ValueError:
            send_notification("Error", "Enter valid numerical values.")
            return None

        return {
            "start_urls": start_urls,
            "num_units": num_units,
            "min_delay": min_delay,
            "max_delay": max_delay,
            "max_depth": max_depth,
            "max_concurrent": max_concurrent,
            "file_format": file_format,
            "keywords": [k.strip() for k in keywords.split(',')] if keywords else None,
            "xpath_filter": xpath_filter,
            "proxies": proxies or None
        }

    def run_scraper(self):
        inputs = self.validate_inputs()
        if not inputs:
            return

        self.scraping_running = True
        self.paused = False
        self.stop_event.clear()
        self.pause_event.clear()
        self.progress_bar['value'] = 0

        for widget in [self.url_entry, self.num_units_entry, self.min_delay_entry, 
                       self.max_delay_entry, self.max_depth_entry, self.max_concurrent_entry, 
                       self.keywords_entry, self.xpath_entry, self.proxy_text, 
                       self.file_format_combobox, self.start_button]:
            widget.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL)

        threading.Thread(target=self.start_async_scraping, args=(inputs,)).start()

    def start_async_scraping(self, inputs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.scrape_data(inputs))

    async def scrape_data(self, inputs):
        try:
            data, stats = await get_data_from_site(
                inputs["start_urls"], inputs["num_units"], inputs["min_delay"], inputs["max_delay"],
                stop_event=self.stop_event, pause_event=self.pause_event, 
                max_concurrent=inputs["max_concurrent"], max_depth=inputs["max_depth"], 
                keywords=inputs["keywords"], xpath_filter=inputs["xpath_filter"], 
                proxies=inputs["proxies"], progress_callback=self.update_progress
            )
            send_notification("Success", f"Scraping completed, collected {stats['total_collected']} records in checkpoints.")
        except Exception as e:
            logger.error(f"Scraping error: {e}")
            send_notification("Error", "An error occurred during scraping.")
        finally:
            self.scraping_running = False
            self.paused = False
            for widget in [self.url_entry, self.num_units_entry, self.min_delay_entry, 
                           self.max_delay_entry, self.max_depth_entry, self.max_concurrent_entry, 
                           self.keywords_entry, self.xpath_entry, self.proxy_text, 
                           self.file_format_combobox, self.start_button]:
                widget.config(state=tk.NORMAL)
            self.pause_button.config(text="Pause", state=tk.DISABLED)
            self.stop_button.config(state=tk.DISABLED)

    def toggle_pause(self):
        if self.paused:
            self.pause_event.clear()
            self.pause_button.config(text="Pause")
            logger.info("Scraping resumed.")
        else:
            self.pause_event.set()
            self.pause_button.config(text="Resume")
            logger.info("Scraping paused.")
        self.paused = not self.paused

    def stop_scraping(self):
        logger.info("Stopping scraping.")
        self.stop_event.set()
        self.pause_event.clear()
        self.paused = False
        self.pause_button.config(text="Pause")

    def save_config(self):
        config = self.validate_inputs()
        if not config:
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
            logger.info(f"Configuration saved to {file_path}")

    def load_config(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.url_entry.delete(0, tk.END)
            self.url_entry.insert(0, ','.join(config["start_urls"]))
            self.num_units_entry.delete(0, tk.END)
            self.num_units_entry.insert(0, str(config["num_units"]))
            self.min_delay_entry.delete(0, tk.END)
            self.min_delay_entry.insert(0, str(config["min_delay"]))
            self.max_delay_entry.delete(0, tk.END)
            self.max_delay_entry.insert(0, str(config["max_delay"]))
            self.max_depth_entry.delete(0, tk.END)
            self.max_depth_entry.insert(0, str(config["max_depth"]))
            self.max_concurrent_entry.delete(0, tk.END)
            self.max_concurrent_entry.insert(0, str(config["max_concurrent"]))
            self.keywords_entry.delete(0, tk.END)
            self.keywords_entry.insert(0, ','.join(config["keywords"] or []))
            self.xpath_entry.delete(0, tk.END)
            self.xpath_entry.insert(0, config["xpath_filter"] or '')
            self.proxy_text.delete("1.0", tk.END)
            self.proxy_text.insert("1.0", '\n'.join(config["proxies"] or []))
            self.file_format_combobox.set(config["file_format"])
            logger.info(f"Configuration loaded from {file_path}")

    def save_logs(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            save_logs(self.log_text, file_path)

if __name__ == "__main__":
    root = tk.Tk()
    app = WebScraperApp(root)
    root.mainloop()