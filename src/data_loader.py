import json
import re
from pathlib import Path
from typing import Tuple
import pandas as pd


# Directory containing the example JSON files with applicants, prospects and
# job vacancies. These files are used to build a small training dataset when
# available.
DATA_DIR = Path(__file__).resolve().parents[1] / "data_source"


def parse_remuneracao(raw: str) -> float:
    """Convert salary string to float handling different formats."""
    clean = re.sub(r"[^0-9.,-]", "", (raw or "")).lstrip(".,")
    if not clean:
        return 0.0
    if "," in clean and "." in clean:
        clean = clean.replace(".", "").replace(",", ".")
    elif "," in clean:
        if clean.count(",") > 1:
            last = clean.rfind(",")
            clean = clean[:last].replace(",", "") + "." + clean[last + 1 :]
        else:
            clean = clean.replace(",", ".")
    else:
        if clean.count(".") > 1:
            last = clean.rfind(".")
            clean = clean[:last].replace(".", "") + clean[last:]
        elif clean.count(".") == 1 and len(clean.split(".")[-1]) not in (1, 2):
            clean = clean.replace(".", "")
    try:
        return float(clean)
    except ValueError:
        return 0.0


def load_json(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def load_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load applicants and job postings.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Applicants and jobs dataframes.
    """
    applicants_file = DATA_DIR / "applicants.json"
    jobs_file = DATA_DIR / "vagas.json"

    applicants = load_json(applicants_file)
    jobs = load_json(jobs_file)

    return applicants, jobs


def load_match_dataset() -> pd.DataFrame:
    """Build training dataframe from applicants and prospects JSON files.

    The resulting dataframe contains simple engineered features and a
    ``match`` column indicating if the candidate was hired for any position.
    """
    applicants_file = DATA_DIR / "applicants.json"
    prospects_file = DATA_DIR / "prospects.json"

    if not applicants_file.exists() or not prospects_file.exists():
        raise FileNotFoundError("Training JSON files not found")

    with open(applicants_file, "r", encoding="utf-8") as f:
        applicants_raw = json.load(f)

    # Build dataframe with basic numeric features
    feats = []
    for code, data in applicants_raw.items():
        basic = data.get("infos_basicas", {})
        prof = data.get("informacoes_profissionais", {}) or {}

        objective = basic.get("objetivo_profissional") or ""
        title = prof.get("titulo_profissional") or ""
        remun = (prof.get("remuneracao") or "").strip()
        remun_val = parse_remuneracao(remun)

        feats.append(
            {
                "codigo": str(code),
                "objective_len": len(objective),
                "title_len": len(title),
                "remuneracao": remun_val,
            }
        )

    feat_df = pd.DataFrame(feats)

    with open(prospects_file, "r", encoding="utf-8") as f:
        prospects_raw = json.load(f)

    labels = {}
    for job in prospects_raw.values():
        for cand in job.get("prospects", []):
            code = str(cand.get("codigo"))
            status = cand.get("situacao_candidado", "")
            hired = "Contratado" in status
            labels[code] = labels.get(code, False) or hired

    label_df = pd.DataFrame(
        {"codigo": list(labels.keys()), "match": [int(v) for v in labels.values()]}
    )

    df = feat_df.merge(label_df, on="codigo", how="inner")
    return df


if __name__ == "__main__":
    applicants, jobs = load_dataset()
    print(applicants.head())
    print(jobs.head())
