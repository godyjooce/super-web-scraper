# Super Web Scraper

A high-performance, asynchronous web scraper built with Python, designed to crawl websites, extract data, and save it in various formats (JSON, CSV, XLSX). It supports proxy rotation, depth-limited crawling, keyword filtering, and a user-friendly GUI built with Tkinter. The scraper is optimized for large-scale data collection (up to 10M+ records) with checkpointing and robust error handling.

## Features
- **Asynchronous Crawling**: Uses `aiohttp` for efficient, concurrent HTTP requests.
- **Proxy Support**: Rotates through a list of proxies to avoid rate-limiting.
- **Customizable Filters**: Supports keyword-based filtering and XPath for targeted data extraction.
- **GUI Interface**: Tkinter-based interface for easy configuration and real-time monitoring.
- **Checkpointing**: Saves intermediate data to prevent loss during long runs.
- **Multiple Output Formats**: Saves data as JSON, CSV, or XLSX.
- **Logging and Notifications**: Detailed logs and desktop notifications for key events.
- **Pause/Resume/Stop**: Control scraping sessions via the GUI.
- **Memory Management**: Monitors and optimizes memory usage for large-scale scraping.

## Requirements
- Python 3.8+
- Operating System: Windows (due to `BASE_PATH` and `plyer` notifications; can be adapted for other OS)
- Dependencies listed in `requirements.txt`

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/godyjooce/super-web-scraper.git
   cd super-web-scraper
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Scraper**:
   ```bash
   python scraper.py
   ```
   This launches the GUI where you can configure and start the scraping process.

2. **GUI Configuration**:
   - **Initial URLs**: Enter comma-separated URLs to start crawling (e.g., `https://ru.wikipedia.org/wiki/Вторая_мировая_война`).
   - **Data Amount**: Specify the target number of records (default: 10M).
   - **Min/Max Delay**: Set request delays (in seconds) to avoid overloading servers (default: 0.1–0.5s).
   - **Max Depth**: Limit crawling depth (default: 7).
   - **Max Threads**: Number of concurrent requests (default: 125).
   - **Keywords**: Optional comma-separated keywords to filter text (e.g., `война, история`).
   - **XPath Filter**: Optional XPath expression to target specific elements (e.g., `//p/text()`).
   - **Proxies**: List HTTP proxies (one per line, e.g., `http://proxy:port`).
   - **File Format**: Choose output format (`json`, `csv`, or `xlsx`).
   - **Folder**: Select an output directory for data and logs.

3. **Control Buttons**:
   - **Start**: Begin scraping with the configured settings.
   - **Pause/Resume**: Temporarily halt or resume scraping.
   - **Stop**: Terminate the scraping process.
   - **Save/Load Config**: Save or load settings as JSON.
   - **Save Logs**: Export logs to a text file.

4. **Output**:
   - Data is saved in the selected format in a timestamped folder (e.g., `C:\Users\scraper_run_20250512_123456`).
   - Logs are saved as `scraper_logs.txt` and can be exported via the GUI.
   - Checkpoints are saved as JSON files and combined into a final output file (e.g., `scraped_all_20250512_123456.json`).

## Example Configuration
To scrape Wikipedia articles about World War II:
- **Initial URLs**: `https://ru.wikipedia.org/wiki/Вторая_мировая_война`
- **Data Amount**: `1000000` (1M records)
- **Min Delay**: `0.1`
- **Max Delay**: `0.5`
- **Max Depth**: `5`
- **Max Threads**: `50`
- **Keywords**: `война, история, сражение`
- **XPath Filter**: `//p/text()`
- **Proxies**: (Optional) List of proxies, one per line
- **File Format**: `json`
- **Folder**: `C:\Users\YourUsername\ScrapedData`

## Project Structure
```
super-web-scraper/
├── scraper.py           # Main script with scraper and GUI
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── output/             # Output directory (created during runtime)
    ├── scraper_logs.txt
    ├── scraped_all_*.json
    ├── visited_urls_*.json
    ├── urls_to_visit_*.json
```

## Notes
- **Proxy Usage**: Ensure proxies are reliable and formatted correctly (`http://proxy:port`). The scraper rotates proxies and removes unresponsive ones.
- **Memory Management**: The scraper monitors memory usage and saves checkpoints if it exceeds 1GB.
- **Error Handling**: Handles timeouts, Unicode errors, and HTTP errors (e.g., 429) with retries.
- **Wikipedia-Specific**: The scraper excludes Wikipedia service pages (e.g., edit pages, categories) by default.
- **Customization**: Modify `BASE_PATH` in `scraper.py` for non-Windows systems or custom output locations.

## Troubleshooting
- **GUI Not Responding**: Ensure Tkinter is installed (`pip install tk`).
- **Proxy Errors**: Verify proxy connectivity and format. Disable proxies by leaving the proxy field empty.
- **Memory Issues**: Reduce `max_concurrent` or `num_units` for lower resource usage.
- **No Data Collected**: Check if URLs are accessible and keywords/XPath are correctly specified.


Inspired by large-scale web scraping challenges.