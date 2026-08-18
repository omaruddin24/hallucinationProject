# ============================ 
# 📘 Patent MCQ Generator — Local LLaMA (TACC-ready)
# Uses meta-llama/Llama-3.1-8B-Instruct via transformers
# ============================

import os, json, time, random, re, io, csv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm.auto import tqdm
from jsonschema import validate

# ---------- Local LLaMA Setup ----------

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

print("===== CUDA Check =====")
print("CUDA:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("======================")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",  # Automatically pushes layers to GPU(s)
)

print("✅ LLaMA model ready.\n")

# ---------- Config ----------

CANDIDATE_PATHS = ["filtered_A61.csv"]  # You can scp this file into your working directory

for p in CANDIDATE_PATHS:
    if os.path.exists(p):
        INPUT_CSV = p
        break
else:
    raise FileNotFoundError("Could not find filtered_A61.csv in current directory.")

# WRITE OUTPUT INTO CURRENT DIR (you can change to $SCRATCH if you want)
# OUTPUT_CSV = f"{os.environ['SCRATCH']}/patent_mcq_output.csv"
OUTPUT_CSV = "patent_mcq_TEST25output.csv"

CHECKPOINT_EVERY = 25
RESUME = False
MAX_PATENTS = 10000
MAX_CHARS_PER_TEXT = 4000
TIMEOUT_RETRY = 2
REFINE_RETRY  = 1  # (kept for compatibility, not heavily used)
SLEEP_MIN, SLEEP_MAX = 1.5, 3.0
TEMPERATURE = 0.01

# ---------- CSV Loader (robust) ----------

def load_csv_best_effort(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, dtype=str)
        print("CSV load: default engine ✓")
        return df
    except Exception:
        pass
    try:
        df = pd.read_csv(path, dtype=str, engine="python", on_bad_lines="skip")
        print("CSV load: python engine, on_bad_lines=skip ✓")
        return df
    except Exception:
        pass
    try:
        df = pd.read_csv(
            path,
            dtype=str,
            engine="python",
            on_bad_lines="skip",
            sep=",",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
        )
        print("CSV load: python QUOTE_NONE ✓")
        return df
    except Exception:
        pass
    try:
        df = pd.read_csv(path, dtype=str, engine="python", on_bad_lines="skip", sep=";")
        print("CSV load: python sep=';' ✓")
        return df
    except Exception:
        pass

    # Last resort: strip null bytes and reload
    with open(path, "rb") as f:
        raw = f.read()
    repaired = raw.replace(b"\x00", b"")
    text = repaired.decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(text), dtype=str, engine="python", on_bad_lines="skip")
    print("CSV load: repaired text ✓")
    return df

# ---------- Column helpers ----------

PREFERRED_TEXT_ORDER = [
    "first_claim", "claims", "description", "abstract",
    "full_text", "text", "body", "content"
]

def pick_best_text_for_row(row) -> str:
    # Prefer canonical text columns
    for col in PREFERRED_TEXT_ORDER:
        if col in row.index:
            val = str(row[col] or "").strip()
            if val:
                return val
    # Otherwise, pick the longest non-empty string field
    candidates = []
    for _, val in row.items():
        if isinstance(val, str) and val.strip():
            candidates.append((len(val), val))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    return ""

def guess_id_column(df):
    for c in ["patent_id","id","publication_number","document_id","application_id"]:
        for col in df.columns:
            if col.lower() == c:
                return col
    return df.columns[0]

def trim_text(s, max_chars=MAX_CHARS_PER_TEXT):
    s = (s or "").strip();    return s if len(s) <= max_chars else s[:max_chars]

# ---------- Prompts ----------

def build_system_prompt():
    return (
        "You MUST follow this exact workflow:\n"
	"\n"
	"STEP 1 (Quote-first):\n"
	"- Pick evidence_quote: a contiguous verbatim span from the patent text, max 12 words.\n"
	"- evidence_quote MUST appear exactly in the patent text.\n"
	"\n"
	"STEP 2 (Answer from quote):\n"
	"- Set correct_option to be an EXACT contiguous substring of evidence_quote (case-insensitive).\n"
	"\n"
	"STEP 3 (Question):\n"
	"- Write a multiple-choice question that starts with \"What\" or \"Why\" or \"How\"\n"
	"- The hypothesis must be EXACTLY: hypothesis = question + \" \" + option.\n"
	"- Question must read grammatically with ANY option.\n"
	"\n"
	"STEP 4 (Distractors):\n"
	"- Provide 3 wrong options that do not have much to do with the patent, but are not ocmpletely random, like it is somewhat related to the patent, but not completely\n"
	"- Each wrong option MUST NOT appear anywhere inside evidence_quote (case-insensitive substring).\n"
	"-After lowercasing and removing punctuation, NONE of the WRONG option’s content words may appear in evidence_quote.\n"
	"- Content words EXCLUDE stopwords: {the,a,an,of,to,in,on,for,with,and,or,is,are,was,were,be,been,being,that,this,these,those,it,as,by,from}.\n"
	"- Wrong options must be NOT-ENTAILED by evidence_quote (contradiction).\n"
	"\n"
	"-- HARD LEXICAL BAN (NO EXCEPTIONS):\n"
	"1) Normalize evidence_quote and each option by: lowercase + remove all punctuation.\n"
	"2) Split into words by whitespace.\n"
	"3) Remove stopwords: {the,a,an,of,to,in,on,for,with,and,or,is,are,was,were,be,been,being,that,this,these,those,it,as,by,from}.\n"
	"4) For EACH wrong option: NONE of its remaining words may appear in the normalized evidence_quote word set.\n"
	"5) Additionally: the entire wrong option string MUST NOT be a substring of evidence_quote (case-insensitive).\n"
	"\n"
	"STEP 5 (Self-check):\n"
	"- Verify: correct_option is a substring of evidence_quote.\n"
	"- Verify: no wrong option is a substring of evidence_quote.\n"
	"- Verify: evidence_quote has 12 words or fewer.\n"
	"- Verify: evidence_quote is verbatim from patent text.\n"
	"- Then output JSON only.\n"

    )

def build_user_prompt(patent_text):
    schema_hint = {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 4,
                "maxItems": 4
            },
            "correct_index": {"type": "integer", "minimum": 0, "maximum": 3},
            "evidence_quote": {"type": "string"}
        },
        "required": ["question", "options", "correct_index", "evidence_quote"]
    }
    return (
        "Create ONE MCQ based only on this patent text.\n"
        "Return a SINGLE JSON object with only the required keys.\n"
        f"Schema: {json.dumps(schema_hint, ensure_ascii=False)}\n\n"
        f"PATENT TEXT START\n{patent_text}\nPATENT TEXT END"
    )

def build_refine_prompt(previous_json, patent_text):
    return (
        "Fix the JSON. Return STRICT JSON only.\n\n"
        f"Patent text:\n{patent_text}\n\n"
        f"JSON to fix:\n{previous_json}"
    )

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "options": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {"type": "string"}
        },
        "correct_index": {"type": "integer", "minimum": 0, "maximum": 3},
        "evidence_quote": {"type": "string"}
    },
    "required": ["question", "options", "correct_index", "evidence_quote"]
}

# ---------- Robust JSON extraction ----------

def extract_all_balanced_json(text: str):
    blocks = []
    inside_str = False
    escape = False
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if inside_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                inside_str = False
        else:
            if ch == '"':
                inside_str = True
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start != -1:
                        blocks.append(text[start:i+1])
    return blocks

def parse_and_validate(raw_text: str):
    blocks = extract_all_balanced_json(raw_text)
    if not blocks:
        raise ValueError("No JSON object found.")

    def looks_like_answer(d):
        return (
            isinstance(d, dict)
            and "question" in d
            and "options" in d
            and "correct_index" in d
            and "evidence_quote" in d
        )

    last_err = None

    for b in blocks:
        try:
            data = json.loads(b)
            if looks_like_answer(data) and "properties" not in data:
                validate(instance=data, schema=RESPONSE_SCHEMA)
                return data, b
        except Exception as e:
            last_err = e

    # Regex fallback
    m = re.search(
        r'\{[^{}]*"question"[^{}]*"options"[^{}]*"correct_index"[^{}]*"evidence_quote"[^{}]*\}',
        raw_text,
        re.DOTALL,
    )
    if m:
        try:
            data = json.loads(m.group(0))
            validate(instance=data, schema=RESPONSE_SCHEMA)
            return data, m.group(0)
        except Exception as e:
            last_err = e

    raise ValueError(f"Could not validate any JSON. Last error: {last_err}")

# ---------- Local model call helpers ----------

def call_model_local(system_prompt, user_prompt,
                     temperature=TEMPERATURE, max_new_tokens=600):
    """
    Call local LLaMA model using an instruction-style prompt.
    """

    # LLaMA-3.1 Instruct-style prompt formatting
    full_prompt = (
        f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
        f"{user_prompt} [/INST]"
    )

    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Optional: small sleep to avoid hammering GPU if you want a tiny delay
    time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))
    return text

def ask_model(patent_text):
    system = build_system_prompt()
    user   = build_user_prompt(patent_text)
    return call_model_local(system, user)

def refine_model(previous_json_block, patent_text):
    system = build_system_prompt()
    user   = build_refine_prompt(previous_json_block, patent_text)
    return call_model_local(system, user, temperature=0.0)

# ---------- Load input ----------

df = load_csv_best_effort(INPUT_CSV)
print(f"Loaded rows: {len(df)}")

df = df.reset_index(drop=True)
df = df.iloc[1000:].reset_index(drop=True)
df.insert(0, "row_id", df.index.astype(int))
id_col = guess_id_column(df)

if MAX_PATENTS is not None:
    df = df.head(MAX_PATENTS).copy()

print(f"Processing exactly {len(df)} rows.")

processed_row_ids = set()

if RESUME and os.path.exists(OUTPUT_CSV):
    try:
        prev = pd.read_csv(OUTPUT_CSV, dtype=str)
        if "row_id" in prev.columns:
            prev = prev[prev["row_id"].astype(str).str.isdigit()]
            processed_row_ids = set(prev["row_id"].astype(str).tolist())
            print(f"Resuming: skipping {len(processed_row_ids)} rows.")
    except Exception:
        print("Resume failed; starting fresh.")

rows_buffer = []
since_ckpt = 0

def ensure_columns(df_in):
    cols = [
        "row_id", "patent_id", "status", "question",
        "option_a", "option_b", "option_c", "option_d",
        "correct_option", "evidence_quote", "error", "last_model_json"
    ]
    for c in cols:
        if c not in df_in.columns:
            df_in[c] = ""
    return df_in[cols]

def flush_checkpoint():
    global rows_buffer, since_ckpt
    if not rows_buffer:
        return

    new_df = pd.DataFrame(rows_buffer)
    # Keep only successful ones
    new_df = new_df[new_df["status"].eq("ok")].copy()
    rows_buffer = []
    since_ckpt = 0

    if os.path.exists(OUTPUT_CSV):
        try:
            existing = pd.read_csv(OUTPUT_CSV, dtype=str)
        except Exception:
            existing = pd.DataFrame(columns=[])
        if "status" in existing.columns:
            existing = existing[existing["status"].eq("ok")].copy()
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged.drop_duplicates(subset=["row_id"], keep="last", inplace=True)
    else:
        merged = new_df

    if not merged.empty:
        merged = ensure_columns(merged)
        merged["row_id_num"] = pd.to_numeric(merged["row_id"], errors="coerce")
        merged = merged[merged["row_id_num"].notna()].sort_values("row_id_num")
        merged = merged.drop(columns=["row_id_num"])

    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"💾 Saved → {OUTPUT_CSV} (+{len(new_df)} rows)")

# ---------- MCQ Generation Loop ----------

try:
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating MCQs"):
        row_id = str(row["row_id"])
        if RESUME and row_id in processed_row_ids:
            continue

        pid = str(row.get(id_col, ""))
        base_text = pick_best_text_for_row(row)
        text = trim_text(base_text)

        if not text.strip():
            continue

        success = False
        json_block_used = None

        # First attempt(s): direct model call
        for attempt in range(1, TIMEOUT_RETRY + 1):
            try:
                model_out = ask_model(text)
                data, json_block = parse_and_validate(model_out)
                json_block_used = json_block
                success = True
                break
            except Exception as e:
                # Could add logging here if desired
                time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

        # If failed but we have some JSON-ish block, try refine once
        if not success and json_block_used:
            try:
                refined_out = refine_model(json_block_used, text)
                data, _ = parse_and_validate(refined_out)
                success = True
            except Exception:
                success = False

        if success:
            opts = data.get("options", [])
            correct_idx = data.get("correct_index", "")
            record = {
                "row_id": row_id,
                "patent_id": pid,
                "status": "ok",
                "question": data.get("question", "").strip(),
                "option_a": opts[0] if len(opts) > 0 else "",
                "option_b": opts[1] if len(opts) > 1 else "",
                "option_c": opts[2] if len(opts) > 2 else "",
                "option_d": opts[3] if len(opts) > 3 else "",
                "correct_option": (
                    ["A", "B", "C", "D"][correct_idx]
                    if isinstance(correct_idx, int) and 0 <= correct_idx <= 3
                    else ""
                ),
                "evidence_quote": data.get("evidence_quote", "").strip(),
                "error": "",
                "last_model_json": ""  # could optionally store json_block_used
            }
            rows_buffer.append(record)

        since_ckpt += 1
        if since_ckpt >= CHECKPOINT_EVERY:
            flush_checkpoint()

except KeyboardInterrupt:
    print("\n⛔ Interrupted. Saving progress...")
finally:
    flush_checkpoint()
    print("✔️ Finished.")
