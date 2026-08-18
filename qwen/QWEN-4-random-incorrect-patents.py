import os
import re
import json
import random
import pandas as pd
from tqdm.auto import tqdm
from google.colab import drive
drive.mount('/content/drive')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# CONFIG
# =========================
PATENT_FILE = "filtered_A61 - filtered_A61 - filtered_A61 - filtered_A61.csv"
QUESTIONS_FILE = "questions_5000.csv"
OUTPUT_FILE = "/content/drive/MyDrive/4random_incorrect_patents.csv"

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
# MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

MAX_PATENTS = 5000

MAX_NEW_TOKENS = 220

TEMPERATURE = 1.0
TOP_P = 0.9
DO_SAMPLE = TEMPERATURE > 0

VALID_ANSWERS = {"a", "b", "c", "d"}

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# =========================
# HELPERS
# =========================
def clean_text(s: str) -> str:
    s = "" if pd.isna(s) else str(s)
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def truncate_text(s: str, max_chars: int) -> str:
    s = clean_text(s)
    return s[:max_chars]


def normalize_option_label(x: str):
    s = clean_text(x).lower()
    s = s.replace("statement_", "").replace("option_", "").replace("choice_", "")
    s = re.sub(r"[^a-d]", "", s)
    return s if s in VALID_ANSWERS else None


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

    if text in VALID_ANSWERS:
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


def build_patent_text(first_claim: str, patent_abstract: str) -> str:
    return (
        f"FIRST CLAIM:\n{truncate_text(first_claim, 3000)}\n\n"
        f"PATENT ABSTRACT:\n{truncate_text(patent_abstract, 1500)}"
    )


def build_four_random_patent_prompt(
    patent_blocks,
    question,
    a,
    b,
    c,
    d
):
    patents_text = ""

    for i, p in enumerate(patent_blocks, start=1):
        patents_text += (
            f"\n====================\n"
            f"PATENT {i}\n"
            f"patent_id: {p['patent_id']}\n\n"
            f"{p['evidence_text']}\n"
        )

    return (
        "You are a strict patent multiple-choice judge.\n"
        "You will be given FOUR random patent evidence blocks.\n"
        "These patent evidence blocks may NOT contain the correct patent for the question.\n"
        "Use only the provided patent evidence blocks when deciding your answer.\n"
        "Do NOT use outside knowledge.\n"
        "Choose the ONE answer choice best supported by the provided evidence.\n\n"

        "Return valid JSON only with exactly these keys:\n"
        "{\n"
        '  "selected_patent_number": 1,\n'
        '  "selected_patent_id": "patent id here",\n'
        '  "answer": "a",\n'
        '  "explanation": "2-3 sentence explanation here."\n'
        "}\n\n"

        "Rules:\n"
        "- selected_patent_number must be 1, 2, 3, or 4\n"
        "- selected_patent_id must be one of the patent_ids shown\n"
        "- answer must be exactly one lowercase letter: a, b, c, or d\n"
        "- explanation must be 2-3 sentences\n"
        "- explanation must briefly explain which patent evidence supports the answer\n"
        "- do not output markdown fences\n\n"

        f"PATENT EVIDENCE BLOCKS:\n{patents_text}\n\n"

        f"QUESTION:\n{clean_text(question)}\n\n"
        "OPTIONS:\n"
        f"a) {clean_text(a)}\n"
        f"b) {clean_text(b)}\n"
        f"c) {clean_text(c)}\n"
        f"d) {clean_text(d)}\n"
    )

# =========================
# CHECK FILES
# =========================
for fp in [PATENT_FILE, QUESTIONS_FILE]:
    if not os.path.exists(fp):
        raise FileNotFoundError(f"❌ File not found: {fp}")

# =========================
# LOAD DATA
# =========================
patents_df = pd.read_csv(PATENT_FILE, dtype=str)
questions_df = pd.read_csv(QUESTIONS_FILE, dtype=str)

for df in [patents_df, questions_df]:
    for col in df.columns:
        df[col] = df[col].fillna("").astype(str).str.strip()

patent_required = [
    "patent_id",
    "first_claim",
    "patent_abstract"
]

question_required = [
    "patent_id",
    "row_id",
    "question",
    "option_a",
    "option_b",
    "option_c",
    "option_d",
]

missing_patent = [c for c in patent_required if c not in patents_df.columns]
missing_question = [c for c in question_required if c not in questions_df.columns]

if missing_patent:
    raise ValueError(f"❌ Missing columns in {PATENT_FILE}: {missing_patent}")

if missing_question:
    raise ValueError(f"❌ Missing columns in {QUESTIONS_FILE}: {missing_question}")

# Keep one evidence row per patent_id
patents_unique = (
    patents_df[["patent_id", "first_claim", "patent_abstract"]]
    .drop_duplicates(subset=["patent_id"])
    .copy()
)

patents_unique["evidence_text"] = patents_unique.apply(
    lambda r: build_patent_text(r["first_claim"], r["patent_abstract"]),
    axis=1
)

# Match questions to patents only so every question has a real correct patent_id
merged = questions_df.merge(
    patents_unique,
    on="patent_id",
    how="inner",
    suffixes=("", "_correct_patent")
).copy()

if MAX_PATENTS is not None:
    keep_ids = merged["patent_id"].drop_duplicates().head(MAX_PATENTS)
    merged = merged[merged["patent_id"].isin(keep_ids)].copy()

merged = merged.sort_values(["patent_id", "row_id"]).reset_index(drop=True)

print(f"🔎 patent file unique patents: {patents_unique['patent_id'].nunique()}")
print(f"🔎 questions file rows: {len(questions_df)}")
print(f"🔎 matched rows after join: {len(merged)}")
print(f"🔎 matched patent_ids: {merged['patent_id'].nunique()}")

# =========================
# RANDOM PATENT LOOKUPS
# =========================
patent_id_to_row = {
    row["patent_id"]: row
    for _, row in patents_unique.iterrows()
}

all_patent_ids = patents_unique["patent_id"].tolist()

if len(all_patent_ids) < 5:
    raise ValueError("❌ Need at least 5 unique patents so we can exclude the correct one and sample 4 random incorrect patents.")


def make_four_random_incorrect_patent_blocks(correct_patent_id: str):
    wrong_patent_ids = [
        pid for pid in all_patent_ids
        if clean_text(pid) != clean_text(correct_patent_id)
    ]

    if len(wrong_patent_ids) < 4:
        raise ValueError(
            f"❌ Not enough incorrect patents to sample 4 for correct patent_id={correct_patent_id}"
        )

    sampled_ids = random.sample(wrong_patent_ids, 4)

    blocks = []

    for pid in sampled_ids:
        row = patent_id_to_row[pid]

        blocks.append({
            "patent_id": pid,
            "evidence_text": row["evidence_text"],
            "is_correct_patent": False,
            "sampling_method": "random_incorrect",
        })

    for i, block in enumerate(blocks, start=1):
        block["shown_patent_number"] = i

    return blocks

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
    out_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    return out_text


def judge_with_four_random_incorrect_patents(row):
    correct_patent_id = row["patent_id"]

    patent_blocks = make_four_random_incorrect_patent_blocks(correct_patent_id)

    prompt = build_four_random_patent_prompt(
        patent_blocks=patent_blocks,
        question=row["question"],
        a=row["option_a"],
        b=row["option_b"],
        c=row["option_c"],
        d=row["option_d"]
    )

    raw_output = run_single_prompt(prompt)
    parsed = extract_json_block(raw_output)

    answer = None
    explanation = ""
    selected_patent_number = None
    selected_patent_id = ""

    if parsed is not None:
        answer = normalize_option_label(parsed.get("answer", ""))
        explanation = clean_text(parsed.get("explanation", ""))

        selected_patent_number = parsed.get("selected_patent_number", None)
        selected_patent_id = clean_text(parsed.get("selected_patent_id", ""))

    else:
        answer = extract_single_letter(raw_output)

    all_shown_patent_ids = []
    random_incorrect_patent_ids = []

    for block in patent_blocks:
        all_shown_patent_ids.append(block["patent_id"])
        random_incorrect_patent_ids.append(block["patent_id"])

    correct_patent_was_shown = clean_text(correct_patent_id) in [
        clean_text(pid) for pid in all_shown_patent_ids
    ]

    selected_correct_patent = (
        clean_text(selected_patent_id) == clean_text(correct_patent_id)
    )

    if answer not in VALID_ANSWERS:
        return {
            "answer_letter": None,
            "explanation": explanation,
            "selected_patent_number": selected_patent_number,
            "selected_patent_id": selected_patent_id,
            "correct_patent_was_shown": correct_patent_was_shown,
            "selected_correct_patent": selected_correct_patent,
            "random_incorrect_patent_ids": ";".join(random_incorrect_patent_ids),
            "all_shown_patent_ids": ";".join(all_shown_patent_ids),
            "model_error": "could not parse model output",
            "raw_output": raw_output,
        }

    return {
        "answer_letter": answer,
        "explanation": explanation,
        "selected_patent_number": selected_patent_number,
        "selected_patent_id": selected_patent_id,
        "correct_patent_was_shown": correct_patent_was_shown,
        "selected_correct_patent": selected_correct_patent,
        "random_incorrect_patent_ids": ";".join(random_incorrect_patent_ids),
        "all_shown_patent_ids": ";".join(all_shown_patent_ids),
        "model_error": "",
        "raw_output": raw_output,
    }

# =========================
# RUN
# =========================
results = []

for _, row in tqdm(merged.iterrows(), total=len(merged), desc="Answering with 4 random incorrect patents"):
    try:
        judged = judge_with_four_random_incorrect_patents(row)

        results.append({
            "patent_id": row["patent_id"],
            "row_id": row["row_id"],
            **judged
        })

    except Exception as e:
        results.append({
            "patent_id": row["patent_id"],
            "row_id": row["row_id"],
            "answer_letter": None,
            "explanation": "",
            "selected_patent_number": None,
            "selected_patent_id": "",
            "correct_patent_was_shown": None,
            "selected_correct_patent": None,
            "random_incorrect_patent_ids": "",
            "all_shown_patent_ids": "",
            "model_error": f"hard_failure: {repr(e)}",
            "raw_output": "",
        })

results_df = pd.DataFrame(results)

# =========================
# MERGE RESULTS BACK
# =========================
final_df = merged.merge(
    results_df,
    on=["patent_id", "row_id"],
    how="left"
)

# =========================
# ACCURACY
# =========================
if "correct_option" in final_df.columns:
    final_df["correct_option_norm"] = (
        final_df["correct_option"]
        .astype(str)
        .str.strip()
        .str.lower()
        .apply(normalize_option_label)
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
    "question",
    "option_a",
    "option_b",
    "option_c",
    "option_d",
    "correct_option",
    "correct_option_norm",
    "answer_letter",
    "model_matches_correct_option",
    "explanation",
    "selected_patent_number",
    "selected_patent_id",
    "correct_patent_was_shown",
    "selected_correct_patent",
    "random_incorrect_patent_ids",
    "all_shown_patent_ids",
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

print("\n📊 Answer distribution:")
print(final_df["answer_letter"].value_counts(dropna=False).to_string())

if "model_matches_correct_option" in final_df.columns:
    print("\n📊 Accuracy summary:")
    print(final_df["model_matches_correct_option"].value_counts(dropna=False).to_string())

print("\n📊 Was the correct patent accidentally shown?")
print(final_df["correct_patent_was_shown"].value_counts(dropna=False).to_string())

print("\n📊 Did model select the correct patent?")
print(final_df["selected_correct_patent"].value_counts(dropna=False).to_string())
