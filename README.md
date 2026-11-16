# swing-data

每個交易日由 GitHub Actions 向 Yahoo Finance 抓取觀察名單的日 K 與市場序列，放到 `data` 分支，供 tw-us-swing-skill 下載。

- `universe.csv`：觀察名單（symbol, market, sector, name, yahoo）。要增刪股票改這個檔。
- `fetch_data.py`：抓取程式。
- `.github/workflows/fetch.yml`：排程（台北 14:20 與隔日 05:40），也可在 Actions 頁面手動執行。

首次使用：到 Actions 頁面啟用工作流程，手動執行一次 `fetch-market-data`，確認出現 `data` 分支。
