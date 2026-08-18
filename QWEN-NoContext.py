import os
import re
import json
import pandas as pd
from tqdm.auto import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# CONFIG
# =========================
COMBINED_FILE = "questions_5000.csv"
OUTPUT_FILE = "qwen_none5.csv"

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MAX_PATENTS = 5000
MAX_NEW_TOKENS = 220

TEMPERATURE = 0.0
TOP_P = 0.9
DO_SAMPLE = TEMPERATURE > 0

# =========================
# HELPERS
# =========================
def clean_text(s: str) -> str:
    s = "" if pd.isna(s) else str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_option_label(x: str):
    s = clean_text(x).lower()
    s = s.replace("statement_", "").replace("option_", "").replace("choice_", "")
    s = re.sub(r"[^a-d]", "", s)
    return s if s in {"a", "b", "c", "d"} else None


def extract_json_block(text: str):
    if not text:
        return None

    text = text.strip()

    try:
        return json.loads(text)
    except:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None

    try:
        return json.loads(m.group(0))
    except:
        return None


def extract_single_letter(text: str):
    if not text:
        return None

    text = text.strip().lower()

    if text in {"a", "b", "c", "d"}:
        return text

    patterns = [
        r'"answer"\s*:\s*"([a-d])"',
        r"\boption\s*([a-d])\b",
        r"\banswer\s*[:\-]?\s*([a-d])\b",
        r"^\s*([a-d])[)\].:\- ]",
        r"\b([a-d])\b",
    ]

    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1)

    return None


def judge_prompt_without_evidence(question, a, b, c, d):
    return (
        "You are answering a patent-related multiple-choice question.\n"
        "Use your own general knowledge and reasoning only.\n"
        "Choose the ONE best answer from the options.\n\n"
        "Return valid JSON only with exactly these keys:\n"
        "{\n"
        '  "answer": "a",\n'
        '  "explanation": "2-3 sentence explanation here."\n'
        "}\n\n"
        "Rules:\n"
        "- answer must be exactly one lowercase letter: a, b, c, or d\n"
        "- explanation must be 2-3 sentences\n"
        "- do not output markdown fences\n\n"
        f"QUESTION:\n{clean_text(question)}\n\n"
        "OPTIONS:\n"
        f"a) {clean_text(a)}\n"
        f"b) {clean_text(b)}\n"
        f"c) {clean_text(c)}\n"
        f"d) {clean_text(d)}\n"
    )

# =========================
# CHECK FILE
# =========================
if not os.path.exists(COMBINED_FILE):
    raise FileNotFoundError(f"❌ File not found: {COMBINED_FILE}")

# =========================
# LOAD DATA
# =========================
combined_df = pd.read_csv(COMBINED_FILE, dtype=str)

for col in combined_df.columns:
    combined_df[col] = combined_df[col].fillna("").astype(str).str.strip()

combined_required = [
    "patent_id",
    "row_id",
    "question",
    "option_a",
    "option_b",
    "option_c",
    "option_d",
]

missing_combined = [c for c in combined_required if c not in combined_df.columns]

if missing_combined:
    raise ValueError(f"❌ Missing columns in combined file: {missing_combined}")

merged = combined_df.copy()

# Optional: limit by unique patent_id
if MAX_PATENTS is not None:
    keep_ids = merged["patent_id"].drop_duplicates().head(MAX_PATENTS)
    merged = merged[merged["patent_id"].isin(keep_ids)].copy()

merged = merged.sort_values(["patent_id", "row_id"]).reset_index(drop=True)

print(f"🔎 combined_unique_patents rows: {len(combined_df)}")
print(f"🔎 rows being answered: {len(merged)}")
print(f"🔎 patent_ids being answered: {merged['patent_id'].nunique()}")

# Add blank columns so output looks similar
if "cpc_subclass" not in merged.columns:
    merged["cpc_subclass"] = ""

if "first_claim" not in merged.columns:
    merged["first_claim"] = ""

if "patent_abstract" not in merged.columns:
    merged["patent_abstract"] = ""

# =========================
# LOAD MODEL
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

print(f"🧠 Loading model: {MODEL_ID}")
print(f"🖥️ Device: {device} | dtype: {dtype}")
print(f"🌡️ Temperature: {TEMPERATURE} | top_p: {TOP_P} | do_sample: {DO_SAMPLE}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map="auto" if torch.cuda.is_available() else None,
)

if not torch.cuda.is_available():
    model = model.to("cpu")

model.eval()

# =========================
# MODEL CALLS
# =========================
@torch.no_grad()
def run_single_prompt(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    else:
        inputs = {k: v.to("cpu") for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[-1]

    generation_kwargs = {
        **inputs,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": DO_SAMPLE,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if DO_SAMPLE:
        generation_kwargs["temperature"] = TEMPERATURE
        generation_kwargs["top_p"] = TOP_P

    outputs = model.generate(**generation_kwargs)

    gen_tokens = outputs[0][input_len:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


@torch.no_grad()
def local_judge_without_evidence(question, a, b, c, d):
    prompt = judge_prompt_without_evidence(question, a, b, c, d)

    raw_output = run_single_prompt(prompt)
    parsed = extract_json_block(raw_output)

    if parsed is not None:
        answer = normalize_option_label(parsed.get("answer", ""))
        explanation = clean_text(parsed.get("explanation", ""))
    else:
        answer = extract_single_letter(raw_output)
        explanation = ""

    if answer in {"a", "b", "c", "d"}:
        return {
            "answer_letter": answer,
            "explanation": explanation,
            "model_error": "",
            "raw_output": raw_output,
            "used_cached_answer": False,
            "cache_source_row_id": "",
        }

    return {
        "answer_letter": None,
        "explanation": explanation,
        "model_error": "could not parse model output",
        "raw_output": raw_output,
        "used_cached_answer": False,
        "cache_source_row_id": "",
    }

# =========================
# RUN
# =========================
results = []

for _, row in tqdm(merged.iterrows(), total=len(merged), desc="Answering questions without evidence"):
    patent_id = row["patent_id"]
    row_id = row["row_id"]

    question = row["question"]
    a = row["option_a"]
    b = row["option_b"]
    c = row["option_c"]
    d = row["option_d"]

    if not question or not all([a, b, c, d]):
        results.append({
            "patent_id": patent_id,
            "row_id": row_id,
            "answer_letter": None,
            "explanation": "",
            "model_error": "missing question/options",
            "raw_output": "",
            "used_cached_answer": False,
            "cache_source_row_id": "",
        })
        continue

    try:
        judged = local_judge_without_evidence(question, a, b, c, d)

        results.append({
            "patent_id": patent_id,
            "row_id": row_id,
            **judged
        })

    except Exception as e:
        results.append({
            "patent_id": patent_id,
            "row_id": row_id,
            "answer_letter": None,
            "explanation": "",
            "model_error": f"hard_failure: {repr(e)}",
            "raw_output": "",
            "used_cached_answer": False,
            "cache_source_row_id": "",
        })

results_df = pd.DataFrame(results)

# =========================
# OPTIONAL COMPARISON TO GOLD LABEL
# =========================
final_df = merged.merge(results_df, on=["patent_id", "row_id"], how="left")

if "correct_option" in final_df.columns:
    final_df["correct_option_norm"] = (
        final_df["correct_option"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    final_df["model_matches_correct_option"] = (
        final_df["answer_letter"].fillna("").str.lower()
        ==
        final_df["correct_option_norm"].fillna("").str.lower()
    )
else:
    final_df["correct_option_norm"] = ""
    final_df["model_matches_correct_option"] = None

# =========================
# SAVE
# =========================
final_cols = [
    "patent_id",
    "row_id",
    "cpc_subclass",
    "question",
    "option_a",
    "option_b",
    "option_c",
    "option_d",
    "correct_option",
    "answer_letter",
    "explanation",
    "model_matches_correct_option",
    "used_cached_answer",
    "cache_source_row_id",
    "model_error",
    "raw_output",
    "first_claim",
    "patent_abstract",
]

final_cols = [c for c in final_cols if c in final_df.columns]

final_df = (
    final_df[final_cols]
    .sort_values(["patent_id", "row_id"])
    .reset_index(drop=True)
)

final_df.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ Done. Saved → {OUTPUT_FILE}")

if "model_matches_correct_option" in final_df.columns:
    print("\n📊 Accuracy summary:")
    print(final_df["model_matches_correct_option"].value_counts(dropna=False).to_string())

# Download file
from google.colab import files
files.download(OUTPUT_FILE)
