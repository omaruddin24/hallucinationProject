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
FILTERED_A61_FILE = "filtered_A61 - filtered_A61.csv"
COMBINED_FILE = "questions_5000.csv"
OUTPUT_FILE = "531qwen_answers_with_idk_NO_CACHE.csv"

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
# MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

MAX_PATENTS = 5000
MAX_NEW_TOKENS = 220

TEMPERATURE = 0.0
TOP_P = 0.9
DO_SAMPLE = TEMPERATURE > 0

VALID_ANSWERS = {"a", "b", "c", "d", "e"}

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
    s = re.sub(r"[^a-e]", "", s)
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
        r'"answer"\s*:\s*"([a-e])"',
        r"\boption\s*([a-e])\b",
        r"\banswer\s*[:\-]?\s*([a-e])\b",
        r"^\s*([a-e])[)\].:\- ]",
        r"\b([a-e])\b",
    ]

    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1)

    if "idk" in text or "i don't know" in text or "insufficient evidence" in text:
        return "e"

    return None


def truncate_text(s: str, max_chars: int = 4000) -> str:
    s = clean_text(s)
    return s[:max_chars]


def build_evidence(first_claim: str, patent_abstract: str) -> str:
    fc = truncate_text(first_claim, 3500)
    pa = truncate_text(patent_abstract, 1500)

    return f"FIRST CLAIM:\n{fc}\n\nPATENT ABSTRACT:\n{pa}"


def judge_prompt_with_explanation(evidence, question, a, b, c, d) -> str:
    return (
        "You are a strict patent multiple-choice judge.\n"
        "Use ONLY the provided patent evidence.\n"
        "Do NOT use outside knowledge.\n"
        "Choose the ONE option best supported by the evidence.\n"
        "If the evidence is missing, unclear, contradictory, or insufficient, choose option e.\n\n"
        "Return valid JSON only with exactly these keys:\n"
        "{\n"
        '  "answer": "e",\n'
        '  "explanation": "2-3 sentence explanation here."\n'
        "}\n\n"
        "Rules:\n"
        "- answer must be exactly one lowercase letter: a, b, c, d, or e\n"
        "- a, b, c, and d are answer choices\n"
        "- e means IDK / insufficient evidence\n"
        "- choose e if the evidence does not clearly support one of a, b, c, or d\n"
        "If you really do not know the answer, put e (IDK) do not try and guess randomly unless u are sure"
        "plz kind of put e somewhat often if ur not sure"
        "- do not guess between a, b, c, and d if the evidence is not enough\n"
        "- explanation must be 2-3 sentences\n"
        "- explanation must explain why the chosen answer is supported, or why IDK is appropriate\n"
        "- do not mention any information not in the evidence\n"
        "- do not output markdown fences\n\n"
        f"EVIDENCE:\n{evidence}\n\n"
        f"QUESTION:\n{clean_text(question)}\n\n"
        "OPTIONS:\n"
        f"a) {clean_text(a)}\n"
        f"b) {clean_text(b)}\n"
        f"c) {clean_text(c)}\n"
        f"d) {clean_text(d)}\n"
        "e) IDK / insufficient evidence\n"
    )

# =========================
# CHECK FILES
# =========================
for fp in [FILTERED_A61_FILE, COMBINED_FILE]:
    if not os.path.exists(fp):
        raise FileNotFoundError(f"File not found: {fp}")

# =========================
# LOAD DATA — ROBUST CSV READER
# =========================
import csv

filtered_df = pd.read_csv(
    FILTERED_A61_FILE,
    dtype=str,
    engine="python",
    quoting=csv.QUOTE_MINIMAL,
    on_bad_lines="skip"
)

combined_df = pd.read_csv(
    COMBINED_FILE,
    dtype=str,
    engine="python",
    quoting=csv.QUOTE_MINIMAL,
    on_bad_lines="skip"
)

# =========================
# PREPARE DATA
# =========================
optional_filtered_cols = ["cpc_subclass"]
available_filtered_cols = [
    c for c in ["patent_id", "first_claim", "patent_abstract"] + optional_filtered_cols
    if c in filtered_df.columns
]

filtered_unique = (
    filtered_df[available_filtered_cols]
    .drop_duplicates(subset=["patent_id"])
    .copy()
)

merged = combined_df.merge(
    filtered_unique,
    on="patent_id",
    how="inner"
).copy()

if MAX_PATENTS is not None:
    keep_ids = merged["patent_id"].drop_duplicates().head(MAX_PATENTS)
    merged = merged[merged["patent_id"].isin(keep_ids)].copy()

merged = merged.sort_values(["patent_id", "row_id"]).reset_index(drop=True)

print(f"filtered_A61 unique patents: {filtered_unique['patent_id'].nunique()}")
print(f"combined rows: {len(combined_df)}")
print(f"matched rows after join: {len(merged)}")
print(f"matched patent_ids: {merged['patent_id'].nunique()}")

# =========================
# LOAD MODEL
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

print(f"Loading model: {MODEL_ID}")
print(f"Device: {device} | dtype: {dtype}")
print(f"Temperature: {TEMPERATURE} | top_p: {TOP_P} | do_sample: {DO_SAMPLE}")

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

    target_device = model.device if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

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
def local_judge_with_explanation(first_claim, patent_abstract, question, a, b, c, d):
    evidence = build_evidence(first_claim, patent_abstract)

    prompt = judge_prompt_with_explanation(
        evidence=evidence,
        question=question,
        a=a,
        b=b,
        c=c,
        d=d
    )

    raw_output = run_single_prompt(prompt)
    parsed = extract_json_block(raw_output)

    if parsed is not None:
        answer = normalize_option_label(parsed.get("answer", ""))
        explanation = clean_text(parsed.get("explanation", ""))
    else:
        answer = extract_single_letter(raw_output)
        explanation = ""

    if answer in VALID_ANSWERS:
        return {
            "answer_letter": answer,
            "explanation": explanation,
            "model_error": "",
            "raw_output": raw_output,
        }

    return {
        "answer_letter": None,
        "explanation": explanation,
        "model_error": "could not parse model output",
        "raw_output": raw_output,
    }

# =========================
# RUN — NO CACHE
# =========================
results = []

for _, row in tqdm(merged.iterrows(), total=len(merged), desc="Answering patent questions"):
    patent_id = row["patent_id"]
    row_id = row["row_id"]

    question = row["question"]
    a = row["option_a"]
    b = row["option_b"]
    c = row["option_c"]
    d = row["option_d"]

    first_claim = row["first_claim"]
    patent_abstract = row["patent_abstract"]

    if not first_claim or not patent_abstract or not question or not all([a, b, c, d]):
        results.append({
            "patent_id": patent_id,
            "row_id": row_id,
            "answer_letter": "e",
            "explanation": "The required evidence, question, or answer choices are missing. Therefore, the safest answer is IDK / insufficient evidence.",
            "model_error": "missing claim/abstract/question/options",
            "raw_output": "",
        })
        continue

    try:
        judged = local_judge_with_explanation(
            first_claim=first_claim,
            patent_abstract=patent_abstract,
            question=question,
            a=a,
            b=b,
            c=c,
            d=d
        )

        results.append({
            "patent_id": patent_id,
            "row_id": row_id,
            **judged
        })

    except Exception as e:
        results.append({
            "patent_id": patent_id,
            "row_id": row_id,
            "answer_letter": "e",
            "explanation": "The model call failed, so there is insufficient usable output to choose a supported option. Therefore, the answer is IDK.",
            "model_error": f"hard_failure: {repr(e)}",
            "raw_output": "",
        })

results_df = pd.DataFrame(results)

# =========================
# COMPARE TO GOLD LABEL
# =========================
final_df = merged.merge(results_df, on=["patent_id", "row_id"], how="left")

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
    "cpc_subclass",
    "question",
    "option_a",
    "option_b",
    "option_c",
    "option_d",
    "correct_option",
    "correct_option_norm",
    "answer_letter",
    "explanation",
    "model_matches_correct_option",
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

print(f"\nDone. Saved → {OUTPUT_FILE}")

print("\nAnswer distribution:")
print(final_df["answer_letter"].value_counts(dropna=False).to_string())

if "model_matches_correct_option" in final_df.columns:
    print("\nAccuracy summary:")
    print(final_df["model_matches_correct_option"].value_counts(dropna=False).to_string())
