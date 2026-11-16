#!/usr/bin/env python3
"""行情抓取：把個股日 K 與市場序列存成 quant_model.py 需要的 CSV。

來源（--source）：
  yahoo   直接向 Yahoo Finance 抓。需要能連到 Yahoo 的環境：自己的電腦、GitHub Actions。
          Claude 雲端環境連不到 Yahoo。
  github  從 GitHub 資料分支下載已抓好的 CSV（由 GitHub Actions 每日更新）。Claude 雲端環境用這個。
  auto    已設定 GitHub repo 就用 github，否則用 yahoo。（預設）

輸出（--out，預設 ./data）：
  bars.csv           date,symbol,market,sector,open,high,low,close,volume,bar_complete
  market.csv         date,series,value
  fetch_report.json  抓取時間、各市場最新完成日 K 日期、缺漏清單

價格為還原權息後的價格（Yahoo auto_adjust），最新一根等於實際成交價。Yahoo 是非官方來源，請抽查。
台股最新一日若 Yahoo 還沒有，會用證交所／櫃買中心的每日收盤行情補上（OpenAPI）。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FILES = ("bars.csv", "market.csv", "fetch_report.json")
# 市場序列名稱 -> (Yahoo 代號, 用哪個市場的收盤時間判斷當日數值是否完成)
MARKET_SERIES = {
    "TAIEX": ("^TWII", "TW"), "SPX": ("^GSPC", "US"), "SOX": ("^SOX", "US"), "VIX": ("^VIX", "US"),
    "GOLD": ("GC=F", "US"), "OIL": ("CL=F", "US"), "USDTWD": ("TWD=X", "US"), "DXY": ("DX-Y.NYB", "US"),
    "US10Y": ("^TNX", "US"),
}
# 收盤後多久才把當日 K 棒視為完成（當地時間）
CLOSE = {"TW": ("Asia/Taipei", dt.time(14, 0)), "US": ("America/New_York", dt.time(16, 20))}


# ----------------------------------------------------------------- universe
def load_universe(path, add=None) -> pd.DataFrame:
    u = pd.read_csv(path, dtype=str).fillna("")
    u.columns = [c.strip().lower() for c in u.columns]
    for col in ("market", "sector", "name", "yahoo"):
        if col not in u.columns:
            u[col] = ""
    extra = []
    for item in add or []:
        for sym in filter(None, (s.strip() for s in item.split(","))):
            market = "TW" if re.fullmatch(r"\d{4,6}[A-Z]?", sym) else "US"
            extra.append({"symbol": sym, "market": market, "sector": "UNCLASSIFIED", "name": "", "yahoo": ""})
    if extra:
        u = pd.concat([u, pd.DataFrame(extra)], ignore_index=True)
    u["symbol"] = u["symbol"].str.strip()
    u["market"] = u["market"].str.strip().str.upper().replace("", "US")
    u["sector"] = u["sector"].replace("", "UNCLASSIFIED")
    return u[u["symbol"] != ""].drop_duplicates("symbol", keep="first").reset_index(drop=True)


def yahoo_candidates(row) -> list:
    """台股未指定 yahoo 代號時，先試上市 .TW，再試上櫃 .TWO。"""
    if row["yahoo"]:
        return [row["yahoo"]]
    if row["market"] == "TW":
        return [f"{row['symbol']}.TW", f"{row['symbol']}.TWO"]
    return [row["symbol"]]


# ----------------------------------------------------------------- yahoo
def _yf_download(tickers: list, start: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise SystemExit("需要 yfinance：python -m pip install yfinance") from exc
    return yf.download(tickers, start=start, interval="1d", auto_adjust=True, group_by="ticker",
                       threads=True, progress=False)


def normalize_download(raw: pd.DataFrame | None, tickers: list) -> dict:
    """把 yfinance 回傳（多代號為 MultiIndex 欄位，單代號可能是單層）整理成 {代號: 日K DataFrame}。"""
    out = {}
    if raw is None or len(raw) == 0:
        return out
    multi = isinstance(raw.columns, pd.MultiIndex)
    for t in tickers:
        if multi:
            lv0, lv1 = raw.columns.get_level_values(0), raw.columns.get_level_values(1)
            if t in lv0:
                d = raw[t]
            elif t in lv1:
                d = raw.xs(t, axis=1, level=1)
            else:
                continue
        elif len(tickers) == 1:
            d = raw
        else:
            continue
        d = d.rename(columns=lambda c: str(c).strip().lower())
        need = ["open", "high", "low", "close", "volume"]
        if any(c not in d.columns for c in need):
            continue
        d = d[need].apply(pd.to_numeric, errors="coerce")
        d = d.dropna(subset=["open", "high", "low", "close"])
        d = d[(d[["open", "high", "low", "close"]] > 0).all(axis=1)]
        if d.empty:
            continue
        idx = pd.to_datetime(d.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        d = d.set_axis(idx.normalize()).rename_axis("date").reset_index()
        d["volume"] = d["volume"].fillna(0)
        out[t] = d.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    return out


def download_all(tickers: list, start: str, downloader=None, batch: int = 40) -> dict:
    downloader = downloader or _yf_download
    got = {}
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        try:
            got.update(normalize_download(downloader(chunk, start), chunk))
        except SystemExit:
            raise
        except Exception as exc:  # 單一批次失敗不中斷，缺漏會寫進報告
            print(f"[警告] 批次下載失敗（{chunk[0]} 等 {len(chunk)} 檔）：{exc}", file=sys.stderr)
    return got


def last_complete_date(market: str, now_utc: dt.datetime) -> dt.date:
    """該市場「已收盤」的最新日期：當地今天已過收盤緩衝時間就是今天，否則是昨天。"""
    tz, close = CLOSE[market if market in CLOSE else "US"]
    local = now_utc.astimezone(ZoneInfo(tz))
    return local.date() if local.time() >= close else local.date() - dt.timedelta(days=1)


def fetch_yahoo(universe: pd.DataFrame, years: float, out_dir, now_utc=None, downloader=None, http=None,
                use_official: bool = True) -> dict:
    now_utc = now_utc or dt.datetime.now(dt.timezone.utc)
    start = (now_utc.date() - dt.timedelta(days=int(years * 365.25) + 5)).isoformat()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 個股：第一輪試第一個候選代號，缺的再試下一個（上市/上櫃）
    cands = {r["symbol"]: yahoo_candidates(r) for _, r in universe.iterrows()}
    resolved, frames = {}, {}
    for rnd in range(max(len(c) for c in cands.values()) if cands else 0):
        want = {c[rnd]: s for s, c in cands.items() if s not in resolved and len(c) > rnd}
        if not want:
            break
        for tkr, d in download_all(list(want), start, downloader).items():
            resolved[want[tkr]], frames[want[tkr]] = tkr, d
    # 台股最新一日：Yahoo 有時晚一天，用證交所／櫃買的每日收盤行情補上
    official = {"enabled": use_official}
    tw_quotes, taiex_rows = {}, []
    if use_official and (universe["market"] == "TW").any():
        tw_last_before = max((d["date"].iloc[-1] for s, d in frames.items()
                              if s in set(universe.loc[universe["market"] == "TW", "symbol"])), default=None)
        tw_quotes, taiex_rows, diag = fetch_tw_official(http)
        official.update(diag)
        official["yahoo_tw_last_date"] = str(tw_last_before.date()) if tw_last_before is not None else None
        official.update(patch_tw_latest(frames, universe, tw_quotes))
    bars, missing = [], []
    for _, r in universe.iterrows():
        d = frames.get(r["symbol"])
        if d is None:
            missing.append(r["symbol"])
            continue
        d = d.copy()
        d["bar_complete"] = d["date"].dt.date <= last_complete_date(r["market"], now_utc)
        d.insert(1, "symbol", r["symbol"])
        d.insert(2, "market", r["market"])
        d.insert(3, "sector", r["sector"])
        bars.append(d)
    if not bars:
        raise SystemExit("沒有抓到任何個股資料：請確認這個環境連得到 Yahoo Finance，或改用 --source github。")
    bars = pd.concat(bars, ignore_index=True)
    bars["date"] = bars["date"].dt.strftime("%Y-%m-%d")
    bars.to_csv(out_dir / "bars.csv", index=False)

    # 市場序列：未完成的當日數值直接捨棄
    got = download_all([v[0] for v in MARKET_SERIES.values()], start, downloader)
    rows, series_missing = [], []
    for name, (tkr, clock) in MARKET_SERIES.items():
        d = got.get(tkr)
        if d is None:
            series_missing.append(name)
            continue
        if name == "TAIEX" and taiex_rows:      # 官方加權指數收盤補上 Yahoo 缺的日期
            extra = [(pd.Timestamp(x), v) for x, v in taiex_rows if pd.Timestamp(x) > d["date"].iloc[-1]]
            if extra:
                add = pd.DataFrame(extra, columns=["date", "close"])
                d = pd.concat([d[["date", "close"]], add], ignore_index=True).sort_values("date")
                official["taiex_patched_dates"] = [str(x.date()) for x, _ in extra]
        d = d[d["date"].dt.date <= last_complete_date(clock, now_utc)]
        rows.append(pd.DataFrame({"date": d["date"].dt.strftime("%Y-%m-%d"), "series": name, "value": d["close"]}))
    series_missing.append("TWVIX")          # Yahoo 沒有台指 VIX；需要時請自行加入 market.csv
    mkt = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date", "series", "value"])
    mkt.to_csv(out_dir / "market.csv", index=False)

    done = bars[bars["bar_complete"]]
    report = {
        "generated_at_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), "source": "yahoo (yfinance, auto_adjust)",
        "symbols_ok": int(bars["symbol"].nunique()), "symbols_missing": missing,
        "resolved_yahoo_tickers": resolved,
        "last_complete_bar": {m: g["date"].max() for m, g in done.groupby("market")},
        "has_incomplete_bar": {m: bool((~g["bar_complete"]).any()) for m, g in bars.groupby("market")},
        "market_series_ok": sorted(set(mkt["series"])), "market_series_missing": series_missing,
        "tw_official": official,
        "note": "價格已還原權息；非官方來源，請抽查。bar_complete=False 的 K 棒不會被模型使用。",
    }
    (out_dir / "fetch_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


# ----------------------------------------------------------------- 台股官方資料（補 Yahoo 缺的最新一日）
OFFICIAL = {
    "twse": "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL",
    "tpex": "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes",
    "taiex": "https://openapi.twse.com.tw/v1/exchangeReport/FMTQIK",
}
KEYS = {  # 欄位名稱可能調整，逐一嘗試
    "code": ("Code", "SecuritiesCompanyCode", "證券代號", "代號"),
    "date": ("Date", "date", "日期", "資料日期"),
    "open": ("OpeningPrice", "Open", "開盤價", "開盤"),
    "high": ("HighestPrice", "High", "最高價", "最高"),
    "low": ("LowestPrice", "Low", "最低價", "最低"),
    "close": ("ClosingPrice", "Close", "收盤價", "收盤"),
    "volume": ("TradeVolume", "TradingShares", "成交股數"),
    "taiex": ("TAIEX", "發行量加權股價指數"),
}


def _http_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=40) as f:
        return json.loads(f.read().decode("utf-8-sig"))


def _pick(rec: dict, field: str):
    for k in KEYS[field]:
        if k in rec and str(rec[k]).strip() != "":
            return rec[k]
    return None


def _num(x) -> float:
    try:
        return float(str(x).replace(",", "").strip())
    except (TypeError, ValueError):
        return float("nan")


def parse_tw_date(x):
    """民國 1151116、115/11/16 或西元 20261116、2026-11-16 -> date；無法解析回傳 None。"""
    digits = re.sub(r"\D", "", str(x or ""))
    try:
        if len(digits) == 7:
            return dt.date(int(digits[:3]) + 1911, int(digits[3:5]), int(digits[5:7]))
        if len(digits) == 8:
            return dt.date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))
    except ValueError:
        return None
    return None


def fetch_tw_official(http=None) -> tuple[dict, list, dict]:
    """回傳 ({代號: 最新一日 K}, [(日期, 加權指數收盤)], 診斷資訊)。任何一段失敗都只記錄，不中斷。"""
    http = http or _http_json
    quotes, taiex, diag = {}, [], {}
    try:
        rows = http(OFFICIAL["taiex"])
        for r in rows:
            d, v = parse_tw_date(_pick(r, "date")), _num(_pick(r, "taiex"))
            if d and v == v and v > 0:
                taiex.append((d, v))
        diag["taiex"] = {"status": "ok", "records": len(rows), "last_date": str(max(d for d, _ in taiex)) if taiex else None,
                         "keys": sorted(rows[0])[:12] if rows else []}
    except Exception as exc:
        diag["taiex"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:200]}
    fallback_date = max((d for d, _ in taiex), default=None)
    for name in ("twse", "tpex"):
        try:
            rows = http(OFFICIAL[name])
            n, dates = 0, set()
            for r in rows:
                code = str(_pick(r, "code") or "").strip()
                d = parse_tw_date(_pick(r, "date")) or (fallback_date if name == "twse" else None)
                o, h, l, c = (_num(_pick(r, k)) for k in ("open", "high", "low", "close"))
                if not code or d is None or not all(x == x and x > 0 for x in (o, h, l, c)):
                    continue
                vol = _num(_pick(r, "volume"))
                quotes[code] = {"date": d, "open": o, "high": h, "low": l, "close": c, "volume": 0.0 if vol != vol else vol}
                n, dates = n + 1, dates | {d}
            diag[name] = {"status": "ok", "records": len(rows), "usable": n, "dates": sorted(str(x) for x in dates)[-2:],
                          "keys": sorted(rows[0])[:14] if rows else []}
        except Exception as exc:
            diag[name] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:200]}
    return quotes, taiex, diag


def patch_tw_latest(frames: dict, universe: pd.DataFrame, quotes: dict) -> dict:
    """Yahoo 的台股日 K 若少了官方已公布的最新一日，就用官方資料補上；兩邊都有同一天時比對收盤價。"""
    patched, diffs, gaps = [], [], []
    for _, r in universe[universe["market"] == "TW"].iterrows():
        d, q = frames.get(r["symbol"]), quotes.get(r["symbol"])
        if d is None or q is None:
            continue
        last = d["date"].iloc[-1].date()
        if q["date"] > last:
            if len(pd.bdate_range(last, q["date"])) > 2:
                gaps.append(r["symbol"])            # 中間還缺其他交易日，官方每日檔補不了
            row = {"date": pd.Timestamp(q["date"]), **{k: q[k] for k in ("open", "high", "low", "close", "volume")}}
            frames[r["symbol"]] = pd.concat([d, pd.DataFrame([row])], ignore_index=True)
            patched.append(r["symbol"])
        elif q["date"] == last and d["close"].iloc[-1] > 0:
            diffs.append(abs(q["close"] / d["close"].iloc[-1] - 1) * 100)
    return {"patched_symbols": len(patched), "patched_date": str(max((quotes[s]["date"] for s in patched), default="")) or None,
            "symbols_with_older_gap": gaps, "close_check_n": len(diffs),
            "close_check_max_diff_pct": round(max(diffs), 3) if diffs else None}


# ----------------------------------------------------------------- github
def fetch_github(repo: str, branch: str, out_dir, clone_url: str | None = None) -> dict:
    """從 GitHub 資料分支取得 CSV。先用 git clone（只需 github.com），失敗再試 raw 下載。"""
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo or ""):
        raise SystemExit("GitHub repo 需為 owner/name 格式（設定在 assets/data_source.json 或用 --repo）")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    url = clone_url or f"https://github.com/{repo}.git"
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "r"
        r = subprocess.run(["git", "clone", "-q", "--depth", "1", "--single-branch", "--branch", branch, url, str(dst)],
                           capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            for name in FILES:
                if not (dst / name).exists():
                    raise SystemExit(f"資料分支 {branch} 裡找不到 {name}：請確認 GitHub Actions 已成功執行過一次。")
                shutil.copy(dst / name, out_dir / name)
        else:
            try:
                for name in FILES:
                    with urllib.request.urlopen(f"https://raw.githubusercontent.com/{repo}/{branch}/{name}", timeout=60) as f:
                        (out_dir / name).write_bytes(f.read())
            except Exception as exc:
                raise SystemExit(f"無法從 GitHub 取得資料（git: {r.stderr.strip()[:200]}；raw: {exc}）。"
                                 "請確認 repo 是公開的、名稱正確、資料分支已建立。") from exc
    return json.loads((out_dir / "fetch_report.json").read_text(encoding="utf-8"))


# ----------------------------------------------------------------- CLI
def describe(report: dict, now_utc=None) -> list:
    now_utc = now_utc or dt.datetime.now(dt.timezone.utc)
    gen = dt.datetime.strptime(report["generated_at_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    age_h = (now_utc - gen).total_seconds() / 3600
    lines = [f"資料抓取時間：{gen.astimezone(ZoneInfo('Asia/Taipei')):%Y-%m-%d %H:%M} 台北（{age_h:.1f} 小時前），來源 {report['source']}"]
    for m, d in sorted(report["last_complete_bar"].items()):
        tail = "；另有未完成的當日 K 棒（不使用）" if report["has_incomplete_bar"].get(m) else ""
        lines.append(f"{m} 最新完成日 K：{d}{tail}")
    lines.append(f"個股 {report['symbols_ok']} 檔" + (f"；抓不到：{', '.join(report['symbols_missing'])}" if report["symbols_missing"] else ""))
    lines.append(f"市場序列：{', '.join(report['market_series_ok'])}；缺：{', '.join(report['market_series_missing']) or '無'}")
    off = report.get("tw_official") or {}
    if off.get("patched_symbols"):
        lines.append(f"台股 {off['patched_date']} 的日 K 由證交所／櫃買資料補上 {off['patched_symbols']} 檔（Yahoo 只到 {off.get('yahoo_tw_last_date')}）；"
                     "這一根是未還原的原始價。")
    bad = [k for k in ("twse", "tpex", "taiex") if (off.get(k) or {}).get("status") == "error"]
    if bad:
        lines.append(f"[提示] 台股官方資料取得失敗：{', '.join(bad)}（詳見 fetch_report.json），台股最新一日可能缺漏。")
    if off.get("close_check_max_diff_pct") is not None and off["close_check_max_diff_pct"] > 0.5:
        lines.append(f"[警告] Yahoo 與官方收盤價最大差異 {off['close_check_max_diff_pct']}%：請抽查資料。")
    if off.get("symbols_with_older_gap"):
        lines.append(f"[警告] 這些台股中間缺了交易日：{', '.join(off['symbols_with_older_gap'][:10])}")
    if age_h > 96:
        lines.append("[警告] 資料已超過 4 天未更新：請檢查 GitHub Actions 是否仍在執行，分析時必須註明資料日期。")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("auto", "yahoo", "github"), default="auto")
    ap.add_argument("--universe", default=str(ROOT / "assets" / "universe.csv"))
    ap.add_argument("--add", action="append", default=[], help="臨時加入的代號，逗號分隔（僅 yahoo 來源有效）")
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--out", default="data")
    ap.add_argument("--repo", default=None, help="GitHub repo（owner/name）；預設讀 assets/data_source.json")
    ap.add_argument("--branch", default=None)
    ap.add_argument("--no-official", action="store_true", help="不使用證交所／櫃買資料補台股最新一日")
    a = ap.parse_args(argv)
    ds = {}
    ds_path = ROOT / "assets" / "data_source.json"
    if ds_path.exists():
        ds = json.loads(ds_path.read_text(encoding="utf-8"))
    repo, branch = a.repo or ds.get("github_repo", ""), a.branch or ds.get("branch", "data")
    source = a.source if a.source != "auto" else ("github" if repo else "yahoo")
    if source == "github":
        if a.add:
            print("[提示] --add 對 github 來源無效：要新增股票請修改 repo 裡的 universe.csv。", file=sys.stderr)
        report = fetch_github(repo, branch, a.out)
    else:
        report = fetch_yahoo(load_universe(a.universe, a.add), a.years, a.out, use_official=not a.no_official)
    for line in describe(report):
        print(line)
    print(f"已寫入 {Path(a.out).resolve()}/bars.csv、market.csv、fetch_report.json")


if __name__ == "__main__":
    main()
