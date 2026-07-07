from __future__ import annotations

from pathlib import Path
from typing import Iterable
import base64
import json
import math

import numpy as np
import pandas as pd
import streamlit as st


APP_TITLE = "就活向け 企業ランキング"
APP_DIR = Path(__file__).resolve().parent
DATA_PATH_CANDIDATES = [
    APP_DIR / "data" / "company_data.csv.gz",
    APP_DIR / "company_data.csv.gz",
    APP_DIR / "streamlit_cloud_app" / "data" / "company_data.csv.gz",
]

COL_COMPANY = "企業名"
COL_INDUSTRY = "しょくばらぼ_業種"
COL_DETAIL_1 = "詳細業種1"
COL_DETAIL_2 = "詳細業種2"
COL_DETAIL_3 = "詳細業種3"
COL_SALARY = "平均年間給与"
COL_AGE = "平均年齢"
COL_TENURE = "平均勤続年数"
COL_ROE = "ROE"
COL_EQUITY = "自己資本比率"
COL_SALES = "売上高・営業収益"
COL_OPERATING = "営業利益"
COL_EMPLOYEES = "従業員数"
COL_PAID_LEAVE_DAYS = "有給休暇取得日数"
COL_60H_WORKERS = "60時間超残業者数"
COL_REGULAR_TENURE = "正社員平均勤続年数"
COL_EMPLOYEE_AVG_AGE = "従業員平均年齢"
COL_FEMALE_MANAGER = "女性管理職比率_評価用"
COL_FEMALE_EXECUTIVE = "女性役員比率"
COL_MALE_CHILDCARE = "男性育休取得率_評価用"
COL_FEMALE_CHILDCARE = "女性育休取得率_評価用"
COL_WAGE_GAP_ALL = "男女賃金差異_全労働者_評価用"
COL_WAGE_GAP_REGULAR = "男女賃金差異_正規_評価用"
COL_WAGE_GAP_NON_REGULAR = "男女賃金差異_非正規_評価用"
COL_HEALTH_MANAGEMENT = "健康経営銘柄"

COL_PAID_LEAVE_RATE = "有給休暇取得率"
COL_OVERTIME = "月平均残業時間(評価用)"
COL_OVERTIME_SCORE = "残業スコア"
COL_ADJ_SALARY = "補正後年収(参考)"
COL_NO_OT_SALARY = "残業なし換算年収"
COL_PRESENCE_BONUS = "制度加点"
COL_TOTAL = "総合スコア"
COL_DEVIATION = "ランキング内偏差値"


METRICS = {
    "待遇・働きやすさ": {
        COL_SALARY: "平均年収",
        COL_PAID_LEAVE_RATE: "有給取得率",
        COL_PAID_LEAVE_DAYS: "有給取得日数",
        COL_OVERTIME: "残業時間の少なさ",
        COL_ROE: "ROE",
        COL_TENURE: "平均勤続年数",
        COL_REGULAR_TENURE: "正社員平均勤続年数",
    },
    "規模・安定性": {
        COL_EQUITY: "自己資本比率",
        COL_SALES: "売上高",
        COL_OPERATING: "営業利益",
        COL_EMPLOYEES: "従業員数",
    },
    "多様性・WLB": {
        COL_FEMALE_MANAGER: "女性管理職比率",
        COL_FEMALE_EXECUTIVE: "女性役員比率",
        COL_MALE_CHILDCARE: "男性育休取得率",
        COL_FEMALE_CHILDCARE: "女性育休取得率",
        COL_WAGE_GAP_ALL: "男女賃金差異(全労働者)",
        COL_WAGE_GAP_REGULAR: "男女賃金差異(正規)",
        COL_WAGE_GAP_NON_REGULAR: "男女賃金差異(非正規)",
        COL_HEALTH_MANAGEMENT: "健康経営",
    },
    "制度・機会": {
        "副業・兼業制度": "副業・兼業制度",
        "職務限定正社員": "職務限定正社員",
        "勤務地限定正社員": "勤務地限定正社員",
        "短時間正社員": "短時間正社員",
        "研修制度": "研修制度",
        "メンター制度": "メンター制度",
        "自己啓発支援": "自己啓発支援",
        "キャリア相談制度": "キャリア相談制度",
        "社内検定制度": "社内検定制度",
        "インターン受入": "インターン受入",
        "職場見学・体験受入": "職場見学・体験受入",
        "えるぼし認定": "えるぼし認定",
        "プラチナえるぼし認定": "プラチナえるぼし認定",
        "くるみん認定": "くるみん認定",
        "プラチナくるみん認定": "プラチナくるみん認定",
    },
}

METRIC_LABELS = {col: label for group in METRICS.values() for col, label in group.items()}

DEFAULT_WEIGHTS = {
    COL_SALARY: 10,
    COL_PAID_LEAVE_RATE: 10,
    COL_PAID_LEAVE_DAYS: 0,
    COL_OVERTIME: 8,
    COL_ROE: 3,
    COL_TENURE: 3,
    COL_REGULAR_TENURE: 0,
    COL_EQUITY: 3,
    COL_SALES: 2,
    COL_OPERATING: 2,
    COL_EMPLOYEES: 2,
    COL_FEMALE_MANAGER: 0,
    COL_FEMALE_EXECUTIVE: 0,
    COL_MALE_CHILDCARE: 2,
    COL_FEMALE_CHILDCARE: 0,
    COL_WAGE_GAP_ALL: 0,
    COL_WAGE_GAP_REGULAR: 0,
    COL_WAGE_GAP_NON_REGULAR: 0,
    COL_HEALTH_MANAGEMENT: 2,
}

MAIN_WEIGHT_COLS = [
    COL_SALARY,
    COL_PAID_LEAVE_RATE,
    COL_OVERTIME,
    COL_ROE,
    COL_TENURE,
]

REVERSE_SCORE_COLS = {COL_OVERTIME, COL_60H_WORKERS}
BINARY_SCORE_COLS = {
    COL_HEALTH_MANAGEMENT, "副業・兼業制度", "職務限定正社員", "勤務地限定正社員",
    "短時間正社員", "研修制度", "メンター制度", "自己啓発支援", "キャリア相談制度",
    "社内検定制度", "インターン受入", "職場見学・体験受入", "えるぼし認定",
    "プラチナえるぼし認定", "くるみん認定", "プラチナくるみん認定",
}

PRESENCE_BONUS_COLS = {
    "テレワーク制度": ["しょくばらぼ_テレワーク制度-可否", "女性活躍DB_在宅勤務・テレワーク"],
    "時間単位有給制度": ["女性活躍DB_年次有給休暇時間単位取得制度"],
    "フレックスタイム制度": ["女性活躍DB_フレックスタイム制度"],
    "短時間勤務制度": ["女性活躍DB_短時間勤務制度"],
    "病気・不妊治療休暇": ["女性活躍DB_病気・不妊治療休暇"],
}

DEFAULT_PRESENCE_BONUS = {
    "テレワーク制度": 0.0,
    "時間単位有給制度": 1.0,
    "フレックスタイム制度": 0.0,
    "短時間勤務制度": 0.0,
    "病気・不妊治療休暇": 0.0,
}

DEFAULT_MANUAL_FILTERS = {
    COL_SALARY: ("1500000", "50000000"),
    COL_PAID_LEAVE_RATE: ("0", "100"),
    COL_PAID_LEAVE_DAYS: ("0", "30"),
    COL_OVERTIME: ("0", "200"),
    COL_60H_WORKERS: ("0", "100000"),
    COL_ROE: ("-5000", "5000"),
    COL_TENURE: ("0", "50"),
    COL_REGULAR_TENURE: ("0", "60"),
    COL_AGE: ("18", "80"),
    COL_EMPLOYEE_AVG_AGE: ("15", "80"),
    COL_EQUITY: ("-1000", "100"),
    COL_SALES: ("0", "100000000000000"),
    COL_OPERATING: ("-5000000000000", "10000000000000"),
    COL_EMPLOYEES: ("1", "1000000"),
    COL_FEMALE_MANAGER: ("0", "100"),
    COL_FEMALE_EXECUTIVE: ("0", "100"),
    COL_MALE_CHILDCARE: ("0", "500"),
    COL_FEMALE_CHILDCARE: ("0", "500"),
    COL_WAGE_GAP_ALL: ("0", "200"),
    COL_WAGE_GAP_REGULAR: ("0", "200"),
    COL_WAGE_GAP_NON_REGULAR: ("0", "200"),
    COL_HEALTH_MANAGEMENT: ("0", "1"),
}
for _binary_col in BINARY_SCORE_COLS:
    DEFAULT_MANUAL_FILTERS.setdefault(_binary_col, ("0", "1"))

EXTERNAL_INDUSTRY_OVERTIME_HOURS = {
    "A": 12.0, "B": 14.0, "C": 18.0, "D": 24.0, "E": 18.0, "F": 16.0,
    "G": 20.0, "H": 25.0, "I": 17.0, "J": 18.0, "K": 18.0, "L": 22.0,
    "M": 20.0, "N": 18.0, "O": 12.0, "P": 10.0, "Q": 14.0, "R": 20.0,
}


st.set_page_config(page_title=APP_TITLE, page_icon="🎯", layout="wide")


def load_settings_from_url() -> dict:
    try:
        raw = st.query_params.get("settings", "")
        if isinstance(raw, list):
            raw = raw[0] if raw else ""
        if not raw:
            return {}
        padding = "=" * (-len(raw) % 4)
        decoded = base64.urlsafe_b64decode((raw + padding).encode("ascii")).decode("utf-8")
        data = json.loads(decoded)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def encode_settings_for_url(settings: dict) -> str:
    payload = json.dumps(settings, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def setting_value(settings: dict, section: str, key: str, default):
    section_data = settings.get(section, {})
    if isinstance(section_data, dict) and key in section_data:
        return section_data[key]
    return default


def setting_int(settings: dict, section: str, key: str, default: int, min_value: int | None = None, max_value: int | None = None) -> int:
    try:
        value = int(setting_value(settings, section, key, default))
    except (TypeError, ValueError):
        value = int(default)
    if min_value is not None:
        value = max(value, min_value)
    if max_value is not None:
        value = min(value, max_value)
    return value


def setting_float(settings: dict, section: str, key: str, default: float, min_value: float | None = None, max_value: float | None = None) -> float:
    try:
        value = float(setting_value(settings, section, key, default))
    except (TypeError, ValueError):
        value = float(default)
    if min_value is not None:
        value = max(value, min_value)
    if max_value is not None:
        value = min(value, max_value)
    return value


def setting_bool(settings: dict, section: str, key: str, default: bool = False) -> bool:
    value = setting_value(settings, section, key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def setting_text(settings: dict, section: str, key: str, default: str = "") -> str:
    value = setting_value(settings, section, key, default)
    return "" if value is None else str(value)


def setting_list(settings: dict, section: str, key: str) -> list:
    value = setting_value(settings, section, key, [])
    return value if isinstance(value, list) else []


def parse_number(value) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if text in {"", "-", "None", "nan", "NaN"}:
        return np.nan
    multiplier = 1.0
    if "兆" in text:
        multiplier = 1_000_000_000_000
    elif "億" in text:
        multiplier = 100_000_000
    elif "万円" in text or text.endswith("万"):
        multiplier = 10_000
    cleaned = (
        text.replace(",", "")
        .replace("円", "")
        .replace("万円", "")
        .replace("万", "")
        .replace("億円", "")
        .replace("億", "")
        .replace("兆円", "")
        .replace("兆", "")
        .replace("%", "")
        .replace("歳", "")
        .replace("年", "")
        .replace("時間", "")
        .strip()
    )
    try:
        return float(cleaned) * multiplier
    except ValueError:
        return np.nan


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return df[col].map(parse_number).astype("float64")


def is_present(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) > 0
    text = str(value).strip().lower()
    if text in {"", "-", "0", "0.0", "false", "no", "none", "nan", "なし", "無", "無し", "ない"}:
        return False
    if text in {"1", "1.0", "true", "yes", "on", "あり", "有", "○", "〇", "可"}:
        return True
    try:
        return float(text) > 0
    except ValueError:
        return True


def presence_series(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    result = pd.Series(False, index=df.index)
    for col in cols:
        if col in df.columns:
            result |= df[col].map(is_present).fillna(False)
    return result.astype(float)


def mean_available(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    existing = [numeric_series(df, col) for col in cols if col in df.columns]
    if not existing:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.concat(existing, axis=1).replace([np.inf, -np.inf], np.nan).mean(axis=1)


def percentile_score(series: pd.Series) -> pd.Series:
    valid = series.replace([np.inf, -np.inf], np.nan)
    valid_series = valid.dropna()
    if valid_series.empty:
        return pd.Series(np.nan, index=series.index, dtype="float64")
    mean = valid_series.mean()
    std = valid_series.std()
    if std == 0 or pd.isna(std):
        result = pd.Series(np.full(len(series), 50.0), index=series.index, dtype="float64")
        result[valid.isna()] = np.nan
        return result
    z_scores = (valid - mean) / std
    result = pd.Series(np.nan, index=series.index, dtype="float64")
    valid_mask = z_scores.notna()
    if valid_mask.any():
        vec_erf = np.vectorize(math.erf)
        result[valid_mask] = (1.0 + vec_erf(z_scores[valid_mask] / np.sqrt(2.0))) / 2.0 * 100
    return result


def deviation_score(series: pd.Series) -> pd.Series:
    valid = series.replace([np.inf, -np.inf], np.nan)
    valid_series = valid.dropna()
    if valid_series.empty:
        return pd.Series(np.nan, index=series.index, dtype="float64")
    std = valid_series.std()
    if pd.isna(std) or std == 0:
        result = pd.Series(np.full(len(series), 50.0), index=series.index, dtype="float64")
        result[valid.isna()] = np.nan
        return result
    return 50 + 10 * (valid - valid_series.mean()) / std


def format_yen(value) -> str:
    if pd.isna(value):
        return "-"
    value = float(value)
    if abs(value) >= 100_000_000:
        return f"{value / 100_000_000:.1f}億円"
    if abs(value) >= 10_000:
        return f"{value / 10_000:.0f}万円"
    return f"{value:,.0f}円"


def format_salary(value) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value) / 10_000:.0f}万円"


def format_percent(value) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.1f}%"


def format_yes_no(value) -> str:
    if pd.isna(value):
        return "-"
    return "あり" if is_present(value) else "-"


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    data_path = next((path for path in DATA_PATH_CANDIDATES if path.exists()), None)
    if data_path is None:
        searched = "\n".join(f"- `{path.as_posix()}`" for path in DATA_PATH_CANDIDATES)
        st.error(
            "内蔵データファイルが見つかりません。\n\n"
            "GitHubリポジトリに `data/company_data.csv.gz` をアップロードしてください。\n\n"
            f"探した場所:\n{searched}"
        )
        st.stop()
    df = pd.read_csv(data_path, compression="gzip", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    return prepare_data(df)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [COL_SALARY, COL_AGE, COL_TENURE, COL_ROE, COL_EQUITY, COL_SALES, COL_OPERATING, COL_EMPLOYEES]:
        out[col] = numeric_series(out, col)

    out[COL_PAID_LEAVE_DAYS] = numeric_series(out, "しょくばらぼ_正社員の有給休暇取得日数_数値")
    out[COL_PAID_LEAVE_RATE] = mean_available(out, [
        "しょくばらぼ_年次有給休暇取得率（全体）-取得率_数値",
        "しょくばらぼ_年次有給休暇取得率（雇用管理区分）-取得率（一覧）_数値",
        "女性活躍DB_8.(1)年次有給休暇の取得率-対象労働者(%)_数値",
        "女性活躍DB_8.(2)年次有給休暇の取得率(区)-有給休暇取得率(%)_数値",
    ])
    out[COL_OVERTIME] = mean_available(out, [
        "しょくばらぼ_対象労働者全体の月平均の法定時間外労働時間と法定休日労働時間の合計-平均残業時間（詳細）_数値",
        "しょくばらぼ_月平均所定外労働時間_数値",
        "女性活躍DB_6.一月当たりの労働者の平均残業時間-平均残業時間(時間)_数値",
    ])
    out[COL_60H_WORKERS] = numeric_series(out, "しょくばらぼ_平均の法定時間外労働60時間以上の労働者の数_数値")
    out[COL_REGULAR_TENURE] = numeric_series(out, "しょくばらぼ_正社員の平均継続勤務年数_数値")
    out[COL_EMPLOYEE_AVG_AGE] = numeric_series(out, "しょくばらぼ_従業員の平均年齢_数値")
    out[COL_FEMALE_MANAGER] = mean_available(out, [
        "女性管理職比率",
        "しょくばらぼ_管理職に占める女性の割合_数値",
        "女性活躍DB_10.管理職に占める女性労働者の割合-割合(%)_数値",
    ])
    out[COL_FEMALE_EXECUTIVE] = mean_available(out, [
        "しょくばらぼ_役員に占める女性の割合_数値",
        "女性活躍DB_11.役員に占める女性の割合-割合(%)_数値",
    ])
    out[COL_MALE_CHILDCARE] = mean_available(out, [
        "男性育休取得率",
        "しょくばらぼ_育児休業取得率（男性）-男性取得率（一覧）_数値",
        "女性活躍DB_5.男女別の育児休業取得率-男性(%)_数値",
    ])
    out[COL_FEMALE_CHILDCARE] = mean_available(out, [
        "しょくばらぼ_育児休業取得率（女性）-女性取得率（一覧）_数値",
        "女性活躍DB_5.男女別の育児休業取得率-女性(%)_数値",
    ])
    out[COL_WAGE_GAP_ALL] = mean_available(out, [
        "男女賃金差異_全労働者",
        "しょくばらぼ_男女の賃金の差異-全労働者_数値",
        "女性活躍DB_14.男女の賃金の差異1-全労働者(%)_数値",
    ])
    out[COL_WAGE_GAP_REGULAR] = mean_available(out, [
        "男女賃金差異_正規",
        "しょくばらぼ_男女の賃金の差異-うち正規雇用労働者_数値",
        "女性活躍DB_14.男女の賃金の差異2-うち正規雇用労働者(%)_数値",
    ])
    out[COL_WAGE_GAP_NON_REGULAR] = mean_available(out, [
        "男女賃金差異_非正規",
        "しょくばらぼ_男女の賃金の差異-うち非正規雇用労働者_数値",
        "女性活躍DB_14.男女の賃金の差異3-うち非正規雇用労働者(%)_数値",
    ])
    out[COL_HEALTH_MANAGEMENT] = presence_series(out, ["しょくばらぼ_健康経営銘柄-認定有無"])
    if COL_SALES in out.columns and COL_OPERATING in out.columns:
        invalid_margin = out[COL_OPERATING] > out[COL_SALES]
        out.loc[invalid_margin, [COL_SALES, COL_OPERATING]] = np.nan
    if COL_EQUITY in out.columns:
        out.loc[out[COL_EQUITY] > 100, COL_EQUITY] = np.nan
    if COL_SALES in out.columns:
        out.loc[out[COL_SALES] < 0, COL_SALES] = np.nan
    presence_map = {
        "副業・兼業制度": ["しょくばらぼ_副業・兼業-可否"],
        "職務限定正社員": ["しょくばらぼ_多様な正社員制度-制度-職務限定正社員"],
        "勤務地限定正社員": ["しょくばらぼ_多様な正社員制度-制度-勤務地限定正社員"],
        "短時間正社員": ["しょくばらぼ_多様な正社員制度-制度-短時間正社員"],
        "研修制度": ["しょくばらぼ_研修制度-有無"],
        "メンター制度": ["しょくばらぼ_メンター制度-有無"],
        "自己啓発支援": ["しょくばらぼ_自己啓発支援制度-有無"],
        "キャリア相談制度": ["しょくばらぼ_キャリアコンサルティング制度-有無"],
        "社内検定制度": ["しょくばらぼ_社内検定制度-有無"],
        "インターン受入": ["しょくばらぼ_インターンシップの受入-可否"],
        "職場見学・体験受入": ["しょくばらぼ_職場見学・職場体験の受入-可否"],
        "えるぼし認定": ["しょくばらぼ_えるぼし認定-認定有無"],
        "プラチナえるぼし認定": ["しょくばらぼ_プラチナえるぼし認定-認定有無"],
        "くるみん認定": ["しょくばらぼ_くるみん-認定有無"],
        "プラチナくるみん認定": ["しょくばらぼ_プラチナくるみん-認定有無"],
    }
    for label, cols in presence_map.items():
        out[label] = presence_series(out, cols)
    for label, cols in PRESENCE_BONUS_COLS.items():
        out[label] = presence_series(out, cols)

    if COL_INDUSTRY not in out.columns:
        out[COL_INDUSTRY] = ""
    for col in [COL_DETAIL_1, COL_DETAIL_2, COL_DETAIL_3]:
        if col not in out.columns:
            out[col] = ""
    return out


def apply_salary_corrections(df: pd.DataFrame, enable_age: bool, enable_overtime: bool) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    salary_col = COL_SALARY
    if enable_age:
        mask = out[COL_SALARY].notna() & out[COL_AGE].notna() & (out[COL_SALARY] > 0) & (out[COL_AGE] > 0)
        if mask.sum() > 2:
            slope, intercept = np.polyfit(out.loc[mask, COL_AGE], out.loc[mask, COL_SALARY], 1)
            avg_age = out.loc[mask, COL_AGE].mean()
            safe_age = out[COL_AGE].fillna(avg_age).replace(0, avg_age)
            out[COL_ADJ_SALARY] = out[COL_SALARY] - slope * (safe_age - avg_age)
            salary_col = COL_ADJ_SALARY
        else:
            out[COL_ADJ_SALARY] = out[COL_SALARY]
            salary_col = COL_ADJ_SALARY

    if enable_overtime:
        assumed = out[COL_OVERTIME].copy()
        industry_code = out[COL_INDUSTRY].astype(str).str.extract(r"^([A-R])", expand=False)
        industry_fallback = industry_code.map(EXTERNAL_INDUSTRY_OVERTIME_HOURS).fillna(18.0)
        assumed = assumed.fillna(industry_fallback)
        annual_overtime_ratio = (assumed.clip(lower=0) * 12 * 1.25) / 1920
        out[COL_NO_OT_SALARY] = out[salary_col] / (1 + annual_overtime_ratio * 0.65)
        salary_col = COL_NO_OT_SALARY

    return out, salary_col


def compute_ranking(df: pd.DataFrame, weights: dict[str, int], bonus_points: dict[str, float], salary_score_col: str) -> pd.DataFrame:
    out = df.copy()
    out[COL_TOTAL] = 0.0
    out["_valid_weight"] = 0.0
    out[COL_OVERTIME_SCORE] = percentile_score(-out[COL_OVERTIME])

    for col, weight in weights.items():
        if weight <= 0 or col not in out.columns:
            continue
        if col == COL_SALARY:
            score = percentile_score(out[salary_score_col])
        elif col in REVERSE_SCORE_COLS:
            score = percentile_score(-out[col])
        elif col in BINARY_SCORE_COLS:
            score = out[col].clip(lower=0, upper=1) * 100
        else:
            score = percentile_score(out[col])
        valid = score.notna()
        out.loc[valid, COL_TOTAL] += score[valid] * (weight / 10)
        out.loc[valid, "_valid_weight"] += 100 * (weight / 10)

    valid_total = out["_valid_weight"] > 0
    out.loc[valid_total, COL_TOTAL] = out.loc[valid_total, COL_TOTAL] / out.loc[valid_total, "_valid_weight"] * 100
    out.loc[~valid_total, COL_TOTAL] = np.nan

    out[COL_PRESENCE_BONUS] = 0.0
    for label, points in bonus_points.items():
        if points > 0 and label in out.columns:
            out.loc[out[label] > 0, COL_PRESENCE_BONUS] += float(points)
    out.loc[out[COL_TOTAL].notna(), COL_TOTAL] += out.loc[out[COL_TOTAL].notna(), COL_PRESENCE_BONUS]
    return out


def display_dataframe(df: pd.DataFrame, cols: list[str], limit: int):
    view = df[cols].head(limit).copy()
    formatters = {
        COL_TOTAL: "{:.1f}点",
        COL_DEVIATION: "{:.1f}",
        COL_OVERTIME_SCORE: "{:.1f}点",
        COL_PRESENCE_BONUS: "+{:.1f}点",
        COL_SALARY: format_salary,
        COL_ADJ_SALARY: format_salary,
        COL_NO_OT_SALARY: format_salary,
        COL_SALES: format_yen,
        COL_OPERATING: format_yen,
        COL_PAID_LEAVE_RATE: format_percent,
        COL_OVERTIME: lambda v: "-" if pd.isna(v) else f"{float(v):.1f}時間",
        COL_AGE: lambda v: "-" if pd.isna(v) else f"{float(v):.1f}歳",
        COL_TENURE: lambda v: "-" if pd.isna(v) else f"{float(v):.1f}年",
        COL_PAID_LEAVE_DAYS: lambda v: "-" if pd.isna(v) else f"{float(v):.1f}日",
        COL_REGULAR_TENURE: lambda v: "-" if pd.isna(v) else f"{float(v):.1f}年",
        COL_EMPLOYEE_AVG_AGE: lambda v: "-" if pd.isna(v) else f"{float(v):.1f}歳",
        COL_60H_WORKERS: lambda v: "-" if pd.isna(v) else f"{float(v):.0f}人",
        COL_ROE: format_percent,
        COL_EQUITY: format_percent,
        COL_FEMALE_MANAGER: format_percent,
        COL_FEMALE_EXECUTIVE: format_percent,
        COL_MALE_CHILDCARE: format_percent,
        COL_FEMALE_CHILDCARE: format_percent,
        COL_WAGE_GAP_ALL: format_percent,
        COL_WAGE_GAP_REGULAR: format_percent,
        COL_WAGE_GAP_NON_REGULAR: format_percent,
        COL_HEALTH_MANAGEMENT: format_yes_no,
        "副業・兼業制度": format_yes_no,
        "職務限定正社員": format_yes_no,
        "勤務地限定正社員": format_yes_no,
        "短時間正社員": format_yes_no,
        "研修制度": format_yes_no,
        "メンター制度": format_yes_no,
        "自己啓発支援": format_yes_no,
        "キャリア相談制度": format_yes_no,
        "社内検定制度": format_yes_no,
        "インターン受入": format_yes_no,
        "職場見学・体験受入": format_yes_no,
        "えるぼし認定": format_yes_no,
        "プラチナえるぼし認定": format_yes_no,
        "くるみん認定": format_yes_no,
        "プラチナくるみん認定": format_yes_no,
        "テレワーク制度": format_yes_no,
        "時間単位有給制度": format_yes_no,
        "フレックスタイム制度": format_yes_no,
        "短時間勤務制度": format_yes_no,
        "病気・不妊治療休暇": format_yes_no,
    }
    for col, formatter in formatters.items():
        if col in view.columns:
            if isinstance(formatter, str):
                view[col] = view[col].map(lambda v, fmt=formatter: "-" if pd.isna(v) else fmt.format(v))
            else:
                view[col] = view[col].map(formatter)
    st.dataframe(view, use_container_width=True)


st.markdown("### 企業ランキング・マッチングツール")
st.info(
    "左側のサイドバーで重み、フィルタ、制度加点を調整すると、"
    "あなたの就活軸に合わせて企業ランキングが自動更新されます。"
)

with st.expander("使い方とデータについて", expanded=True):
    st.markdown(
        """
        1. **重み**で、年収・有給取得率・残業・ROE・勤続年数などの重要度を調整します。
        2. **制度加点**で、テレワーク制度、時間単位有給制度、フレックスタイム制度などを固定点として加算できます。
        3. **業種フィルタ**や**詳細業種フィルタ**で、見たい業界だけに絞り込めます。
        4. **有給取得率がある企業だけで評価**、**残業時間がある企業だけで評価**を使うと、スコア計算の母集団もその条件に合わせて変わります。
        5. 設定を残したい場合は、サイドバー下部の**現在の設定をURLに保存**を押して、そのURLをブックマークしてください。

        データは、EDINET・しょくばらぼ・女性活躍推進データベース由来の情報を統合した固定データです。
        年収データが取得できていない企業は、ランキングの母集団から除外します。
        この公開版にはデータ収集機能やCSVアップロード機能は含めていません。
        """
    )

with st.expander("スコア計算方法を詳しく見る", expanded=False):
    st.markdown(
        """
        **基本方針**

        各指標は、ランキング対象になっている企業だけを母集団として、まず0〜100点の指標スコアに変換します。
        その後、左側で設定した重みを掛けて平均し、最後に制度加点を足します。

        **指標スコアの作り方**

        - 年収、有給取得率、ROE、勤続年数などは、値が高いほど高スコアになります。
        - 残業時間、60時間以上労働者数、男女賃金差異などは、値が低いほど高スコアになります。
        - 健康経営、男性育休、テレワーク制度、時間単位有給制度などの有無指標は、あり=100点、なし=0点として扱います。
        - 年収データがない企業は、ランキングの母集団から除外します。
        - 年収以外で欠損している指標は、その企業のスコア計算から外します。欠損を0点扱いにはしません。

        **総合スコア**

        総合スコアは、各指標スコアに `重み / 10` を掛けた加重平均です。
        例えば年収の重みを10、有給取得率の重みを5にすると、年収は有給取得率の2倍強く効きます。
        重みが0の指標はスコア計算に使いません。

        **制度加点**

        制度加点は、総合スコアの加重平均を出した後に直接足します。
        そのため、テレワーク制度や時間単位有給制度などに大きな加点を設定すると、総合スコアが100点を超える場合があります。

        **年収補正**

        - 「平均年齢で年収補正」をオンにすると、平均年齢の高低による年収差をならした参考年収を使います。
        - 「残業なし換算年収で評価」をオンにすると、残業時間を考慮して、残業代込みの見かけの年収を少し割り引いた年収で評価します。
        - 残業時間が欠損している企業は、業種ごとの仮定残業時間を使って補正します。

        **ランキング内偏差値**

        ランキング内偏差値は、最終的な総合スコアをもとに、現在のフィルタ後の企業群の中で計算しています。
        業種フィルタや「有給取得率がある企業だけで評価」などを変えると、偏差値の母集団も変わります。
        """
    )

df_base = load_data()
saved_settings = load_settings_from_url()
if saved_settings:
    st.sidebar.caption("URL内の保存設定を読み込みました")

with st.sidebar:
    st.header("ランキング設定")
    ranking_count = st.number_input("表示企業数", min_value=10, max_value=500, value=setting_int(saved_settings, "display", "ranking_count", 100, 10, 500), step=10)
    enable_age_correction = st.toggle("平均年齢で年収補正", value=setting_bool(saved_settings, "flags", "enable_age_correction", True))
    enable_overtime_salary = st.toggle("残業なし換算年収で評価", value=setting_bool(saved_settings, "flags", "enable_overtime_salary", True))
    require_paid_leave = st.checkbox("有給取得率がある企業だけで評価", value=setting_bool(saved_settings, "population", "require_paid_leave", False))
    require_overtime = st.checkbox("残業時間がある企業だけで評価", value=setting_bool(saved_settings, "population", "require_overtime", False))

    industries = sorted([x for x in df_base[COL_INDUSTRY].dropna().astype(str).unique() if x.strip()])
    selected_industries = st.multiselect(
        "業種フィルタ",
        industries,
        default=[value for value in setting_list(saved_settings, "industry_filter", "selected") if value in industries],
    )
    detailed_values = sorted(set(
        pd.concat([df_base[COL_DETAIL_1], df_base[COL_DETAIL_2], df_base[COL_DETAIL_3]], ignore_index=True)
        .dropna().astype(str).str.strip()
    ) - {""})
    selected_details = st.multiselect(
        "詳細業種フィルタ",
        detailed_values,
        default=[value for value in setting_list(saved_settings, "detailed_industry_filter", "selected") if value in detailed_values],
    )

    st.subheader("主要な重み")
    weights: dict[str, int] = {}
    for col in MAIN_WEIGHT_COLS:
        if col in df_base.columns:
            weights[col] = st.slider(
                METRIC_LABELS.get(col, col),
                0,
                10,
                setting_int(saved_settings, "weights", col, DEFAULT_WEIGHTS.get(col, 0), 0, 10),
                1,
                key=f"w_{col}",
            )

    with st.expander("詳細な重み設定", expanded=False):
        for group_name, group_metrics in METRICS.items():
            st.markdown(f"**{group_name}**")
            for col, label in group_metrics.items():
                if col in df_base.columns and col not in MAIN_WEIGHT_COLS:
                    weights[col] = st.slider(
                        label,
                        0,
                        10,
                        setting_int(saved_settings, "weights", col, DEFAULT_WEIGHTS.get(col, 0), 0, 10),
                        1,
                        key=f"w_{col}",
                    )

    with st.expander("詳細設定：制度加点", expanded=False):
        bonus_points = {
            label: st.number_input(f"{label}: ありなら加点", min_value=0.0, max_value=50.0,
                                   value=setting_float(saved_settings, "presence_bonus_points", label, DEFAULT_PRESENCE_BONUS.get(label, 0.0), 0.0, 50.0), step=0.5, key=f"bonus_{label}")
            for label in PRESENCE_BONUS_COLS
        }

    with st.expander("詳細設定：ランキングに表示する列", expanded=False):
        base_display = {
        COL_DEVIATION: st.checkbox("ランキング内偏差値", value=setting_bool(saved_settings, "display_columns", COL_DEVIATION, True)),
        COL_PRESENCE_BONUS: st.checkbox("制度加点", value=setting_bool(saved_settings, "display_columns", COL_PRESENCE_BONUS, True)),
        COL_OVERTIME_SCORE: st.checkbox("残業スコア", value=setting_bool(saved_settings, "display_columns", COL_OVERTIME_SCORE, True)),
        COL_INDUSTRY: st.checkbox("業種", value=setting_bool(saved_settings, "display_columns", COL_INDUSTRY, True)),
        COL_DETAIL_1: st.checkbox("詳細業種1", value=setting_bool(saved_settings, "display_columns", COL_DETAIL_1, True)),
        COL_SALARY: st.checkbox("平均年収", value=setting_bool(saved_settings, "display_columns", COL_SALARY, True)),
        COL_ADJ_SALARY: st.checkbox("補正後年収", value=setting_bool(saved_settings, "display_columns", COL_ADJ_SALARY, True)),
        COL_NO_OT_SALARY: st.checkbox("残業なし換算年収", value=setting_bool(saved_settings, "display_columns", COL_NO_OT_SALARY, True)),
        COL_AGE: st.checkbox("平均年齢", value=setting_bool(saved_settings, "display_columns", COL_AGE, True)),
        COL_TENURE: st.checkbox("平均勤続年数", value=setting_bool(saved_settings, "display_columns", COL_TENURE, True)),
        COL_REGULAR_TENURE: st.checkbox("正社員平均勤続年数", value=setting_bool(saved_settings, "display_columns", COL_REGULAR_TENURE, False)),
        COL_PAID_LEAVE_RATE: st.checkbox("有給取得率", value=setting_bool(saved_settings, "display_columns", COL_PAID_LEAVE_RATE, True)),
        COL_PAID_LEAVE_DAYS: st.checkbox("有給取得日数", value=setting_bool(saved_settings, "display_columns", COL_PAID_LEAVE_DAYS, False)),
        COL_OVERTIME: st.checkbox("月平均残業時間", value=setting_bool(saved_settings, "display_columns", COL_OVERTIME, True)),
        COL_60H_WORKERS: st.checkbox("60時間超残業者数", value=setting_bool(saved_settings, "display_columns", COL_60H_WORKERS, False)),
        COL_FEMALE_MANAGER: st.checkbox("女性管理職比率", value=setting_bool(saved_settings, "display_columns", COL_FEMALE_MANAGER, False)),
        COL_FEMALE_EXECUTIVE: st.checkbox("女性役員比率", value=setting_bool(saved_settings, "display_columns", COL_FEMALE_EXECUTIVE, False)),
        COL_MALE_CHILDCARE: st.checkbox("男性育休取得率", value=setting_bool(saved_settings, "display_columns", COL_MALE_CHILDCARE, False)),
        COL_FEMALE_CHILDCARE: st.checkbox("女性育休取得率", value=setting_bool(saved_settings, "display_columns", COL_FEMALE_CHILDCARE, False)),
        COL_WAGE_GAP_ALL: st.checkbox("男女賃金差異(全労働者)", value=setting_bool(saved_settings, "display_columns", COL_WAGE_GAP_ALL, False)),
        COL_HEALTH_MANAGEMENT: st.checkbox("健康経営", value=setting_bool(saved_settings, "display_columns", COL_HEALTH_MANAGEMENT, True)),
        "テレワーク制度": st.checkbox("テレワーク制度", value=setting_bool(saved_settings, "display_columns", "テレワーク制度", True)),
        "時間単位有給制度": st.checkbox("時間単位有給制度", value=setting_bool(saved_settings, "display_columns", "時間単位有給制度", True)),
        "フレックスタイム制度": st.checkbox("フレックスタイム制度", value=setting_bool(saved_settings, "display_columns", "フレックスタイム制度", True)),
        "短時間勤務制度": st.checkbox("短時間勤務制度", value=setting_bool(saved_settings, "display_columns", "短時間勤務制度", True)),
        "病気・不妊治療休暇": st.checkbox("病気・不妊治療休暇", value=setting_bool(saved_settings, "display_columns", "病気・不妊治療休暇", True)),
        COL_SALES: st.checkbox("売上高", value=setting_bool(saved_settings, "display_columns", COL_SALES, False)),
        COL_OPERATING: st.checkbox("営業利益", value=setting_bool(saved_settings, "display_columns", COL_OPERATING, False)),
        }

    filter_targets = {col: label for col, label in METRIC_LABELS.items() if col in df_base.columns}

    with st.expander("外れ値フィルタ（スコア計算前に無効化）", expanded=False):
        st.caption("範囲外の値を欠損扱いにします。企業自体は除外せず、その指標だけスコア計算から外します。空欄は制限なしです。")
        manual_filters = {}
        for col, label in filter_targets.items():
            default_min, default_max = DEFAULT_MANUAL_FILTERS.get(col, ("", ""))
            c1, c2 = st.columns(2)
            manual_filters[col] = (
                c1.text_input(
                    f"{label} 最小値",
                    value=setting_text(saved_settings, "manual_filter_min", col, default_min),
                    key=f"manual_min_{col}",
                ),
                c2.text_input(
                    f"{label} 最大値",
                    value=setting_text(saved_settings, "manual_filter_max", col, default_max),
                    key=f"manual_max_{col}",
                ),
            )

    with st.expander("ランキング表示フィルタ（足切り条件）", expanded=False):
        st.caption("指定した条件を満たさない企業はランキングから除外します。偏差値などの母集団も、この条件を通った企業だけになります。空欄は制限なしです。")
        display_filters = {}
        for col, label in filter_targets.items():
            c1, c2 = st.columns(2)
            display_filters[col] = (
                c1.text_input(
                    f"{label} 下限",
                    value=setting_text(saved_settings, "display_filter_min", col, ""),
                    key=f"disp_min_{col}",
                ),
                c2.text_input(
                    f"{label} 上限",
                    value=setting_text(saved_settings, "display_filter_max", col, ""),
                    key=f"disp_max_{col}",
                ),
            )

    if st.button("現在の設定をURLに保存", use_container_width=True):
        settings_payload = {
            "version": 1,
            "weights": weights,
            "presence_bonus_points": bonus_points,
            "flags": {
                "enable_age_correction": bool(enable_age_correction),
                "enable_overtime_salary": bool(enable_overtime_salary),
            },
            "population": {
                "require_paid_leave": bool(require_paid_leave),
                "require_overtime": bool(require_overtime),
            },
            "display": {
                "ranking_count": int(ranking_count),
            },
            "display_columns": base_display,
            "manual_filter_min": {col: values[0] for col, values in manual_filters.items()},
            "manual_filter_max": {col: values[1] for col, values in manual_filters.items()},
            "display_filter_min": {col: values[0] for col, values in display_filters.items()},
            "display_filter_max": {col: values[1] for col, values in display_filters.items()},
            "industry_filter": {"selected": selected_industries},
            "detailed_industry_filter": {"selected": selected_details},
        }
        st.query_params["settings"] = encode_settings_for_url(settings_payload)
        st.success("設定をURLに保存しました。このページをブックマークすると次回も同じ設定で開けます。")

df, salary_score_col = apply_salary_corrections(df_base, enable_age_correction, enable_overtime_salary)

for col, (min_text, max_text) in manual_filters.items():
    if col in df.columns:
        min_text = "" if min_text is None else str(min_text)
        max_text = "" if max_text is None else str(max_text)
        if min_text.strip():
            try:
                df.loc[df[col] < float(min_text), col] = np.nan
            except ValueError:
                pass
        if max_text.strip():
            try:
                df.loc[df[col] > float(max_text), col] = np.nan
            except ValueError:
                pass

mask = pd.Series(True, index=df.index)
if salary_score_col in df.columns:
    mask &= df[salary_score_col].notna() & (df[salary_score_col] > 0)
if require_paid_leave:
    mask &= df[COL_PAID_LEAVE_RATE].notna()
if require_overtime:
    mask &= df[COL_OVERTIME].notna()
if selected_industries:
    mask &= df[COL_INDUSTRY].astype(str).isin(selected_industries)
if selected_details:
    detail_mask = (
        df[COL_DETAIL_1].astype(str).isin(selected_details)
        | df[COL_DETAIL_2].astype(str).isin(selected_details)
        | df[COL_DETAIL_3].astype(str).isin(selected_details)
    )
    mask &= detail_mask
for col, (min_text, max_text) in display_filters.items():
    if col in df.columns:
        min_text = "" if min_text is None else str(min_text)
        max_text = "" if max_text is None else str(max_text)
        if min_text.strip():
            try:
                mask &= df[col] >= float(min_text)
            except ValueError:
                pass
        if max_text.strip():
            try:
                mask &= df[col] <= float(max_text)
            except ValueError:
                pass

df_population = df[mask].copy()
ranked = compute_ranking(df_population, weights, bonus_points, salary_score_col)
ranked = ranked.sort_values(COL_TOTAL, ascending=False).reset_index(drop=True)
ranked.index = ranked.index + 1
ranked[COL_DEVIATION] = deviation_score(ranked[COL_TOTAL])

cols = [COL_COMPANY, COL_TOTAL]
for col, enabled in base_display.items():
    if enabled and col in ranked.columns and col not in cols:
        cols.append(col)

left, right = st.columns([2, 1])
left.subheader(f"企業ランキング トップ{int(ranking_count)}")
right.metric("評価対象企業数", f"{len(ranked):,}社")

if ranked.empty:
    st.warning("条件に合う企業がありません。フィルタを緩めてください。")
else:
    display_dataframe(ranked, cols, int(ranking_count))
    csv = ranked[cols].to_csv(index=True, encoding="utf-8-sig")
    st.download_button("表示中ランキングをCSVでダウンロード", csv, "company_ranking.csv", "text/csv")

with st.expander("スコア計算について"):
    st.write(
        "各指標を評価対象企業内で0〜100点に偏差化し、設定した重みで加重平均します。"
        "その後、制度加点に設定した固定点を総合スコアへ直接加算します。"
        "そのため、制度加点を大きくすると総合スコアが100点を超える場合があります。"
    )
