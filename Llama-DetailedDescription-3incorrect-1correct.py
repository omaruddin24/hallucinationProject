import os
import re
import json
import random
import pandas as pd
from tqdm.auto import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# CONFIG
# =========================
QUESTIONS_FILE = "questions_5000.csv"
DESCRIPTION_FILE = "patent_id_description_text.csv"
OUTPUT_FILE = "/content/drive/MyDrive/4patents_description_WITH_IDK.csv"

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

MAX_ROWS = 5000
MAX_DESCRIPTION_CHARS = 5000
MAX_NEW_TOKENS = 250

TEMPERATURE = 1.0
TOP_P = 0.9
DO_SAMPLE = TEMPERATURE > 0

VALID_ANSWERS = {"a", "b", "c", "d", "e"}

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# =========================
# HELPERS
# =========================
def clean_text(s):
    s = "" if pd.isna(s) else str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def clean_pid(x):
    return str(x).strip()


def truncate_text(s, max_chars):
    return clean_text(s)[:max_chars]


def normalize_option_label(x):
    s = clean_text(x).lower()
    s = s.replace("statement_", "").replace("option_", "").replace("choice_", "")
    s = re.sub(r"[^a-e]", "", s)
    return s if s in VALID_ANSWERS else None


def extract_json_block(text):
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


def extract_single_letter(text):
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

    return None


def build_description_evidence(description_text):
    return "DETAILED DESCRIPTION:\n" + truncate_text(description_text, MAX_DESCRIPTION_CHARS)


def build_four_patent_prompt(patent_blocks, question, a, b, c, d):
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
        "You will be given FOUR patent evidence blocks.\n"
        "Only ONE of the four patents is the correct patent for the question.\n"
        "Use ONLY the patent evidence to answer the question.\n"
        "Do NOT use outside knowledge.\n\n"

        "Choose one answer:\n"
        "- a, b, c, or d if one answer choice is clearly supported by the relevant patent evidence\n"
        "- e if the evidence does not provide enough support to confidently choose a, b, c, or d\n\n"

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
        "- answer must be exactly one lowercase letter: a, b, c, d, or e\n"
        "- e means IDK / not enough evidence\n"
        "- use e when the evidence is missing, vague, contradictory, or does not directly support any option\n"
        "- explanation must briefly explain which patent evidence supports the answer, or why e was chosen\n"
        "- do not output markdown fences\n\n"

        f"PATENT EVIDENCE BLOCKS:\n{patents_text}\n\n"

        f"QUESTION:\n{clean_text(question)}\n\n"
        "OPTIONS:\n"
        f"a) {clean_text(a)}\n"
        f"b) {clean_text(b)}\n"
        f"c) {clean_text(c)}\n"
        f"d) {clean_text(d)}\n"
        "e) IDK / not enough evidence\n"
    )

# =========================
# CHECK FILES
# =========================
for fp in [QUESTIONS_FILE, DESCRIPTION_FILE]:
    if not os.path.exists(fp):
        raise FileNotFoundError(f"File not found: {fp}")

# =========================
# LOAD QUESTIONS
# =========================
questions_df = pd.read_csv(QUESTIONS_FILE, dtype=str)

for col in questions_df.columns:
    questions_df[col] = questions_df[col].fillna("").astype(str).str.strip()

required_question_cols = [
    "patent_id",
    "row_id",
    "question",
    "option_a",
    "option_b",
    "option_c",
    "option_d",
]

missing_q = [c for c in required_question_cols if c not in questions_df.columns]
if missing_q:
    raise ValueError(f"Missing columns in questions file: {missing_q}")

# =========================
# REPAIR DESCRIPTION FILE
# same method that gave 5,000 matches
# =========================
records = []
current_pid = None
current_desc = []

with open(DESCRIPTION_FILE, "r", encoding="utf-8", errors="ignore") as f:
    header = f.readline()

    for line in f:
        line = line.rstrip("\n")

        m = re.match(r'^"?(\d{7,12})"?[,|\t](.*)$', line)

        if m:
            if current_pid is not None:
                records.append({
                    "patent_id": current_pid,
                    "description_text": " ".join(current_desc)
                })

            current_pid = m.group(1).strip()
            current_desc = [m.group(2).strip()]
        else:
            if current_pid is not None:
                current_desc.append(line.strip())

if current_pid is not None:
    records.append({
        "patent_id": current_pid,
        "description_text": " ".join(current_desc)
    })

description_df = pd.DataFrame(records)

if "patent_id" not in description_df.columns or "description_text" not in description_df.columns:
    raise ValueError("Description repair failed.")

# =========================
# CLEAN IDS
# =========================
questions_df["patent_id_clean"] = questions_df["patent_id"].apply(clean_pid)
description_df["patent_id_clean"] = description_df["patent_id"].apply(clean_pid)

description_unique = (
    description_df[["patent_id_clean", "patent_id", "description_text"]]
    .drop_duplicates(subset=["patent_id_clean"])
    .copy()
)

description_unique["evidence_text"] = description_unique["description_text"].apply(
    build_description_evidence
)

description_unique["similarity_text"] = description_unique["description_text"].map(clean_text)

# =========================
# MERGE QUESTIONS WITH DESCRIPTIONS
# =========================
merged = questions_df.merge(
    description_unique[["patent_id_clean", "description_text", "evidence_text"]],
    on="patent_id_clean",
    how="inner"
).copy()

merged = merged.sort_values("row_id").reset_index(drop=True)
merged = merged.head(MAX_ROWS).copy()

print("=" * 50)
print("MATCH SUMMARY")
print("=" * 50)
print(f"Question rows loaded:        {len(questions_df):,}")
print(f"Description unique patents:  {description_unique['patent_id_clean'].nunique():,}")
print(f"Rows after merge:            {len(merged):,}")
print(f"Unique patents after merge:  {merged['patent_id_clean'].nunique():,}")

if len(merged) < MAX_ROWS:
    raise ValueError(f"Only {len(merged)} matched rows found, expected {MAX_ROWS}.")

# =========================
# BUILD TF-IDF SIMILARITY INDEX FROM DESCRIPTIONS
# =========================
print("\nBuilding TF-IDF similarity index from detailed descriptions...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000
)

tfidf_matrix = vectorizer.fit_transform(description_unique["similarity_text"])

patent_id_to_index = {
    pid: idx for idx, pid in enumerate(description_unique["patent_id_clean"].tolist())
}

patent_id_to_row = {
    row["patent_id_clean"]: row
    for _, row in description_unique.iterrows()
}

def get_three_similar_patents(correct_patent_id_clean):
    correct_idx = patent_id_to_index[correct_patent_id_clean]

    sims = cosine_similarity(
        tfidf_matrix[correct_idx],
        tfidf_matrix
    ).flatten()

    candidates = []

    for idx, score in enumerate(sims):
        candidate_id = description_unique.iloc[idx]["patent_id_clean"]

        if candidate_id == correct_patent_id_clean:
            continue

        candidates.append((candidate_id, score))

    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

    return candidates[:3]


def make_four_patent_blocks(correct_patent_id_clean):
    similar_patents = get_three_similar_patents(correct_patent_id_clean)

    blocks = []

    for pid, sim_score in similar_patents:
        row = patent_id_to_row[pid]
        blocks.append({
            "patent_id": row["patent_id"],
            "patent_id_clean": pid,
            "evidence_text": row["evidence_text"],
            "is_correct_patent": False,
            "similarity_to_correct": sim_score,
        })

    correct_row = patent_id_to_row[correct_patent_id_clean]
    blocks.append({
        "patent_id": correct_row["patent_id"],
        "patent_id_clean": correct_patent_id_clean,
        "evidence_text": correct_row["evidence_text"],
        "is_correct_patent": True,
        "similarity_to_correct": 1.0,
    })

    random.shuffle(blocks)

    for i, block in enumerate(blocks, start=1):
        block["shown_patent_number"] = i

    return blocks

# =========================
# LOAD MODEL
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

print(f"\nLoading model: {MODEL_ID}")
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
def run_single_prompt(prompt):
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


def judge_with_four_patents(row):
    correct_patent_id_clean = row["patent_id_clean"]

    patent_blocks = make_four_patent_blocks(correct_patent_id_clean)

    prompt = build_four_patent_prompt(
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

    if answer not in VALID_ANSWERS:
        answer = "e"
        explanation = explanation or "IDK because the model output could not be parsed."

    correct_patent_shown_number = None
    all_shown_patent_ids = []
    similar_patent_ids = []

    for block in patent_blocks:
        all_shown_patent_ids.append(block["patent_id"])

        if block["is_correct_patent"]:
            correct_patent_shown_number = block["shown_patent_number"]
        else:
            similar_patent_ids.append(block["patent_id"])

    selected_correct_patent = (
        clean_text(selected_patent_id) == clean_text(row["patent_id"])
    )

    return {
        "answer_letter": answer,
        "explanation": explanation,
        "selected_patent_number": selected_patent_number,
        "selected_patent_id": selected_patent_id,
        "correct_patent_shown_number": correct_patent_shown_number,
        "selected_correct_patent": selected_correct_patent,
        "similar_patent_ids": ";".join(similar_patent_ids),
        "all_shown_patent_ids": ";".join(all_shown_patent_ids),
        "model_error": "",
        "raw_output": raw_output,
    }

# =========================
# RUN
# =========================
results = []

for _, row in tqdm(merged.iterrows(), total=len(merged), desc="Answering with 4 description patents + IDK"):
    try:
        judged = judge_with_four_patents(row)

        results.append({
            "patent_id": row["patent_id"],
            "row_id": row["row_id"],
            **judged
        })

    except Exception as e:
        results.append({
            "patent_id": row["patent_id"],
            "row_id": row["row_id"],
            "answer_letter": "e",
            "explanation": "IDK because the model call failed.",
            "selected_patent_number": None,
            "selected_patent_id": "",
            "correct_patent_shown_number": "",
            "selected_correct_patent": None,
            "similar_patent_ids": "",
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
# e will be counted wrong unless correct_option is also e
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
    "correct_patent_shown_number",
    "selected_correct_patent",
    "similar_patent_ids",
    "all_shown_patent_ids",
    "model_error",
    "raw_output",
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

print("\nDid model select the correct patent?")
print(final_df["selected_correct_patent"].value_counts(dropna=False).to_string())
