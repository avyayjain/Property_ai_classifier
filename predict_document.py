"""
Property AI — V7 Document Classification

Uses:
    - TF-IDF + Logistic Regression, trained on v7 training templates
    - Fine-tuned BERT-Tiny v7 augmented candidate
    - 512-token input limit
    - Optional overlapping-window diagnostics

Compatible with:
    property_ai_review.py
    sale_deed_extractor.py

Supported labels:
    sale_deed
    property_tax_receipt
    encumbrance_certificate
    other_or_unknown

IMPORTANT:
    Model scores are not calibrated probabilities.
    Predictions do not establish ownership, authenticity, title,
    encumbrance status, or legal validity.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_DIR = (
    PROJECT_DIR
    / "property_bert_tiny_candidate_v7_augmented"
)

SPLIT_METADATA_PATH = (
    PROJECT_DIR
    / "model_comparison_v7"
    / "split_metadata.json"
)

DATASET_CANDIDATES = [
    PROJECT_DIR / "synthetic_property_documents_v7_long.csv",
    PROJECT_DIR / "synthetic_property_documents_v7.csv",
]

MAX_LENGTH = 512

# Diagnostics are enabled for the next test.
# The diagnostic prints document text to the terminal.
# Set False before using private property documents.
DEBUG_WINDOW_DIAGNOSTICS = True

WINDOW_OVERLAP = 128

MIN_WORDS = 4

# Provisional review thresholds, not calibrated probabilities.
MIN_BASELINE_SCORE = 0.60
MIN_BERT_SCORE = 0.60

SUPPORTED_LABELS = {
    "sale_deed",
    "property_tax_receipt",
    "encumbrance_certificate",
}

ALL_LABELS = sorted(
    SUPPORTED_LABELS | {"other_or_unknown"}
)

DOCUMENT_MARKERS = {
    "sale_deed": [
        "sale deed",
        "deed of sale",
        "conveyance deed",
        "deed of conveyance",
        "sale consideration",
        "seller",
        "buyer",
        "transferor",
        "transferee",
        "vendor",
        "purchaser",
    ],
    "property_tax_receipt": [
        "property tax receipt",
        "property tax payment",
        "municipal property tax",
        "tax payment receipt",
        "tax amount",
        "property tax assessment",
    ],
    "encumbrance_certificate": [
        "encumbrance certificate",
        "certificate of encumbrance",
        "encumbrance search",
        "registration search certificate",
        "registered interests certificate",
        "search period",
    ],
}


# ============================================================
# 2. SELECT THE CORRECT V7 DATASET
# ============================================================

def get_dataset_path() -> Path:
    """
    Prefer the exact dataset recorded in the v7 split metadata.
    Otherwise use an available v7 dataset candidate.
    """

    if SPLIT_METADATA_PATH.exists():
        metadata = json.loads(
            SPLIT_METADATA_PATH.read_text(
                encoding="utf-8"
            )
        )

        recorded_name = metadata.get("dataset_file")

        if recorded_name:
            recorded_path = PROJECT_DIR / recorded_name

            if recorded_path.exists():
                return recorded_path

            raise FileNotFoundError(
                "The dataset named in the v7 split metadata "
                f"was not found:\n{recorded_path}"
            )

    for candidate in DATASET_CANDIDATES:
        if candidate.exists():
            raise FileNotFoundError(
                "The v7 split metadata is missing:\n"
                f"{SPLIT_METADATA_PATH}\n\n"
                "A template-grouped split is required to train the "
                "TF-IDF baseline without accidentally including "
                "held-out test templates. Restore the v7 experiment "
                "metadata instead of training on the full dataset."
            )

    raise FileNotFoundError(
        "No v7 dataset was found. Expected one of:\n"
        + "\n".join(str(path) for path in DATASET_CANDIDATES)
    )


# ============================================================
# 3. LOAD AND VALIDATE V7 TRAINING DATA
# ============================================================

def load_v7_training_data() -> pd.DataFrame:
    """
    Load only rows belonging to templates assigned to the v7 training
    split. All variants of those training templates may be used.

    This avoids training the TF-IDF baseline on held-out test templates.
    """

    if not SPLIT_METADATA_PATH.exists():
        raise FileNotFoundError(
            "V7 split metadata not found:\n"
            f"{SPLIT_METADATA_PATH}"
        )

    metadata = json.loads(
        SPLIT_METADATA_PATH.read_text(
            encoding="utf-8"
        )
    )

    dataset_path = get_dataset_path()

    df = pd.read_csv(dataset_path)

    required_columns = {
        "text",
        "label",
        "template_id",
        "variant",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "V7 dataset is missing required columns: "
            + ", ".join(sorted(missing))
        )

    split_details = metadata.get("classes", {})

    if not split_details:
        raise ValueError(
            "The v7 split metadata does not contain class template lists."
        )

    training_templates = set()

    for label, details in split_details.items():
        training_templates.update(
            details.get("train_templates", [])
        )

    if not training_templates:
        raise ValueError(
            "No training templates were found in the v7 metadata."
        )

    df = df.dropna(
        subset=[
            "text",
            "label",
            "template_id",
        ]
    ).copy()

    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df["template_id"] = (
        df["template_id"].astype(str).str.strip()
    )

    df = df[
        df["template_id"].isin(training_templates)
        & df["text"].ne("")
        & df["label"].ne("")
    ].copy()

    if df.empty:
        raise ValueError(
            "No rows from the v7 training templates were found."
        )

    found_templates = set(df["template_id"].unique())

    missing_templates = training_templates - found_templates

    if missing_templates:
        raise ValueError(
            "Some training templates are missing from the dataset: "
            + ", ".join(sorted(missing_templates)[:10])
        )

    unexpected_labels = (
        set(df["label"].unique()) - set(ALL_LABELS)
    )

    if unexpected_labels:
        raise ValueError(
            "Unexpected class labels found: "
            + ", ".join(sorted(unexpected_labels))
        )

    missing_labels = (
        set(ALL_LABELS) - set(df["label"].unique())
    )

    if missing_labels:
        raise ValueError(
            "The v7 training split is missing classes: "
            + ", ".join(sorted(missing_labels))
        )

    print("TF-IDF dataset:", dataset_path.name)
    print("TF-IDF training rows:", len(df))
    print("TF-IDF training templates:", len(found_templates))
    print(
        "TF-IDF training variants:",
        sorted(df["variant"].astype(str).unique()),
    )

    return df.reset_index(drop=True)


# ============================================================
# 4. LOAD TF-IDF BASELINE
# ============================================================

def load_baseline() -> Pipeline:
    """
    Train TF-IDF + Logistic Regression using only the v7 training
    templates, including their available variants.
    """

    df = load_v7_training_data()

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(
        df["text"].tolist(),
        df["label"].tolist(),
    )

    return model


# ============================================================
# 5. LOAD V7 BERT-TINY MODEL
# ============================================================

def load_transformer():
    """
    Load the saved v7 augmented model and its matching tokenizer.

    Returns:
        tokenizer, model, device, label_mapping
    """

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            "V7 augmented BERT model was not found:\n"
            f"{MODEL_DIR}\n\n"
            "Check that train_compare_v7.py created the model directory."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
    ).to(device)

    model.eval()

    raw_mapping = getattr(
        model.config,
        "id2label",
        {},
    ) or {}

    label_mapping = {}

    for index in range(model.config.num_labels):
        label = raw_mapping.get(
            index,
            raw_mapping.get(str(index)),
        )

        if label is None:
            raise ValueError(
                "The v7 BERT model has no complete id2label mapping. "
                "Check config.json and the saved training model."
            )

        label = str(label).strip()

        if label.upper() == f"LABEL_{index}":
            raise ValueError(
                "The model contains generic output labels. "
                "Refusing to guess the training label order."
            )

        if label not in ALL_LABELS:
            raise ValueError(
                f"Unexpected label in BERT configuration: {label}"
            )

        label_mapping[index] = label

    if set(label_mapping.values()) != set(ALL_LABELS):
        raise ValueError(
            "The saved v7 model does not contain the expected "
            "four unique document labels."
        )

    print("BERT model directory:", MODEL_DIR.name)
    print("Loaded BERT label mapping:", label_mapping)
    print("Maximum input length:", MAX_LENGTH)
    print("Device:", device)

    return tokenizer, model, device, label_mapping


# ============================================================
# 6. LOAD TEXT OR OCR JSON
# ============================================================

def load_document(
    path: str | Path,
) -> list[dict[str, Any]]:
    """
    Support:
        - UTF-8 TXT files
        - JSON lists of page records
        - JSON objects with a 'pages' list
        - OCR pages using 'page', 'method', and 'text' fields

    Output:
        [
            {
                "page_number": 1,
                "text": "...",
                "extraction_method": "..."
            }
        ]
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input document not found:\n{path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Input path is not a file:\n{path}"
        )

    if path.suffix.lower() == ".json":
        with path.open(
            "r",
            encoding="utf-8-sig",
        ) as file:
            data = json.load(file)

        if isinstance(data, dict):
            if isinstance(data.get("pages"), list):
                raw_pages = data["pages"]

            elif isinstance(data.get("text"), str):
                raw_pages = [data]

            else:
                raise ValueError(
                    "JSON must contain a 'pages' list or a 'text' field."
                )

        elif isinstance(data, list):
            raw_pages = data

        else:
            raise ValueError(
                "Unsupported JSON structure."
            )

        pages = []

        for index, item in enumerate(raw_pages, start=1):
            if not isinstance(item, dict):
                continue

            page_text = str(
                item.get("text") or ""
            ).strip()

            if not page_text:
                continue

            pages.append(
                {
                    "page_number": item.get(
                        "page_number",
                        item.get("page", index),
                    ),
                    "text": page_text,
                    "extraction_method": str(
                        item.get(
                            "extraction_method",
                            item.get("method", "unknown"),
                        )
                    ),
                }
            )

        if not pages:
            raise ValueError(
                "No non-empty page text was found in the JSON file."
            )

        return pages

    text = path.read_text(
        encoding="utf-8-sig"
    ).strip()

    if not text:
        raise ValueError(
            "The input text file is empty."
        )

    return [
        {
            "page_number": 1,
            "text": text,
            "extraction_method": "text_file",
        }
    ]


# ============================================================
# 7. TF-IDF PREDICTION
# ============================================================

def predict_baseline(
    model: Pipeline,
    text: str,
) -> tuple[str, float]:
    """Return the top TF-IDF class and its score."""

    probabilities = model.predict_proba([text])[0]

    index = int(probabilities.argmax())

    return (
        str(model.classes_[index]),
        float(probabilities[index]),
    )


# ============================================================
# 8. RESOLVE BERT OUTPUT LABELS
# ============================================================

def get_label_for_index(
    index: int,
    model,
    label_mapping: dict[int, str] | None,
) -> str:
    if label_mapping is not None:
        label = label_mapping.get(index)

        if label is not None:
            return str(label)

    raw_mapping = getattr(
        model.config,
        "id2label",
        {},
    ) or {}

    label = raw_mapping.get(
        index,
        raw_mapping.get(str(index)),
    )

    if label is None:
        raise ValueError(
            f"Cannot resolve BERT output index {index} to a class label."
        )

    return str(label)


# ============================================================
# 9. BERT SINGLE-WINDOW PREDICTION
# ============================================================

def predict_transformer(
    tokenizer,
    model,
    device,
    text: str,
    label_mapping: dict[int, str] | None = None,
) -> tuple[str, float]:
    """
    Predict from the full text, up to 512 tokens.

    For a document longer than 512 tokens, the main prediction
    still uses right-side truncation. The optional window diagnostic
    below examines other sections separately.
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    model.eval()

    with torch.inference_mode():
        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )[0]

    index = int(
        probabilities.argmax().item()
    )

    label = get_label_for_index(
        index,
        model,
        label_mapping,
    )

    score = float(
        probabilities[index].item()
    )

    return label, score


# ============================================================
# 10. OPTIONAL BERT OVERLAPPING-WINDOW DIAGNOSTICS
# ============================================================

def diagnose_transformer_windows(
    tokenizer,
    model,
    device,
    text: str,
    label_mapping: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """
    Predict each overlapping window independently.

    These results are diagnostic only. They are not aggregated
    and do not replace the main document-level prediction.
    """

    if not DEBUG_WINDOW_DIAGNOSTICS:
        return []

    full_tokens = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False,
    )

    print("\n" + "=" * 72)
    print("V7 BERT INPUT AND WINDOW DIAGNOSTICS")
    print("=" * 72)

    print("Document characters:", len(text))
    print("Full document tokens:", len(full_tokens))
    print("Maximum window length:", MAX_LENGTH)
    print("Window overlap:", WINDOW_OVERLAP)

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        stride=WINDOW_OVERLAP,
        return_overflowing_tokens=True,
        padding="max_length",
        return_tensors="pt",
    )

    model_inputs = {
        key: value.to(device)
        for key, value in encoded.items()
        if key != "overflow_to_sample_mapping"
    }

    model.eval()

    with torch.inference_mode():
        outputs = model(**model_inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )

    window_count = int(
        model_inputs["input_ids"].shape[0]
    )

    print("Number of windows:", window_count)

    results = []

    markers = [
        "sale deed",
        "conveyance deed",
        "seller",
        "buyer",
        "property details",
        "sale consideration",
        "registration details",
        "encumbrance certificate",
        "property tax receipt",
    ]

    for index in range(window_count):
        window_probabilities = probabilities[index]

        predicted_index = int(
            window_probabilities.argmax().item()
        )

        label = get_label_for_index(
            predicted_index,
            model,
            label_mapping,
        )

        score = float(
            window_probabilities[predicted_index].item()
        )

        token_ids = (
            encoded["input_ids"][index]
            .detach()
            .cpu()
            .tolist()
        )

        attention_mask = (
            encoded["attention_mask"][index]
            .detach()
            .cpu()
        )

        active_tokens = int(
            attention_mask.sum().item()
        )

        visible_token_ids = token_ids[:active_tokens]

        visible_text = tokenizer.decode(
            visible_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        print("\n" + "-" * 72)
        print(f"WINDOW {index + 1}/{window_count}")
        print("-" * 72)

        print("Tokens:", active_tokens)
        print("Prediction:", label)
        print("Score:", round(score, 4))

        print("\nText visible to BERT:")
        print(visible_text)

        visible_lower = visible_text.lower()

        print("\nMarker check:")

        for marker in markers:
            print(
                f"  {marker}: "
                f"{'VISIBLE' if marker in visible_lower else 'NOT VISIBLE'}"
            )

        results.append(
            {
                "window_number": index + 1,
                "token_count": active_tokens,
                "bert_label": label,
                "bert_score": round(score, 4),
                "text_preview": visible_text,
            }
        )

    print("\nWindow predictions have not been combined.")
    print("=" * 72)

    return results


# ============================================================
# 11. DOCUMENT MARKER CHECK
# ============================================================

def has_document_markers(
    text: str,
    label: str,
) -> bool:
    markers = DOCUMENT_MARKERS.get(label, [])

    normalized_text = text.casefold()

    return any(
        marker.casefold() in normalized_text
        for marker in markers
    )


# ============================================================
# 12. CLASSIFY ONE PAGE
# ============================================================

def classify_page(
    text,
    baseline,
    tokenizer,
    transformer,
    device,
    label_mapping=None,
) -> dict[str, Any]:
    """
    Classify one page using TF-IDF and the V7 BERT candidate.

    Compatible with property_ai_review.py.

    A successful preliminary classification still requires manual review.
    """

    text = str(text or "").strip()

    word_count = len(text.split())

    result = {
        "word_count": word_count,
        "baseline_label": None,
        "baseline_score": None,
        "bert_label": None,
        "bert_score": None,
        "models_agree": False,
        "suggested_type": None,
        "status": "needs_review",
        "reason": "",
        "acceptance_rules_passed": False,
    }

    if word_count < MIN_WORDS:
        result["reason"] = (
            "Too little text was extracted. "
            "Check the OCR output and original page."
        )
        return result

    # --------------------------------------------------------
    # A. Main document-level predictions
    # --------------------------------------------------------

    baseline_label, baseline_score = predict_baseline(
        baseline,
        text,
    )

    bert_label, bert_score = predict_transformer(
        tokenizer,
        transformer,
        device,
        text,
        label_mapping,
    )

    result.update(
        {
            "baseline_label": baseline_label,
            "baseline_score": round(baseline_score, 4),
            "bert_label": bert_label,
            "bert_score": round(bert_score, 4),
            "models_agree": baseline_label == bert_label,
        }
    )

    # --------------------------------------------------------
    # B. Diagnostic only
    # --------------------------------------------------------

    diagnose_transformer_windows(
        tokenizer,
        transformer,
        device,
        text,
        label_mapping,
    )

    # --------------------------------------------------------
    # C. Classification safeguards
    # --------------------------------------------------------

    if baseline_label != bert_label:
        result["status"] = "needs_review"
        result["reason"] = (
            "The TF-IDF and BERT-Tiny models disagree. "
            "The document type cannot be resolved automatically."
        )
        return result

    if baseline_label == "other_or_unknown":
        result["status"] = "needs_review"
        result["reason"] = (
            "Both models predict other_or_unknown. "
            "The document may be outside the supported categories "
            "or the extracted text may not contain sufficient evidence."
        )
        return result

    if (
        baseline_label not in SUPPORTED_LABELS
        or bert_label not in SUPPORTED_LABELS
    ):
        result["status"] = "needs_review"
        result["reason"] = (
            "A model returned a label outside the supported categories."
        )
        return result

    result["suggested_type"] = baseline_label

    if (
        baseline_score < MIN_BASELINE_SCORE
        or bert_score < MIN_BERT_SCORE
    ):
        result["status"] = "needs_review"
        result["reason"] = (
            "Both models agree, but at least one score is below "
            "the configured provisional threshold."
        )
        return result

    if not has_document_markers(text, baseline_label):
        result["suggested_type"] = None
        result["status"] = "needs_review"
        result["reason"] = (
            "Expected document-type markers were not found. "
            "Manual review is required."
        )
        return result

    result["status"] = "candidate_for_manual_review"
    result["acceptance_rules_passed"] = True
    result["reason"] = (
        "Both models agree on a supported class and the preliminary "
        "rules passed. The classification is still unverified and "
        "must be checked against the original document."
    )

    return result


# ============================================================
# 13. STANDALONE TEST RUNNER
# ============================================================

def main() -> None:
    """
    Optional standalone usage:

        python predict_document.py .\\synthetic_sale_deed.txt

    The main application remains property_ai_review.py.
    """

    try:
        if len(sys.argv) > 1:
            input_path = Path(
                sys.argv[1].strip().strip('"')
            )
        else:
            raw_path = input(
                "Enter an OCR JSON or TXT file path: "
            ).strip().strip('"')

            if not raw_path:
                print("No input file provided.")
                return

            input_path = Path(raw_path)

        pages = load_document(input_path)

        print("\nLoading v7 TF-IDF baseline...")
        baseline = load_baseline()

        print("\nLoading v7 BERT-Tiny...")
        tokenizer, transformer, device, label_mapping = (
            load_transformer()
        )

        print("\nDevice:", device)

        results = []

        for page in pages:
            print("\n" + "=" * 72)
            print(f"PROCESSING PAGE {page['page_number']}")
            print("=" * 72)

            prediction = classify_page(
                page["text"],
                baseline,
                tokenizer,
                transformer,
                device,
                label_mapping,
            )

            results.append(
                {
                    "page_number": page["page_number"],
                    "extraction_method": page["extraction_method"],
                    "prediction": prediction,
                }
            )

            print("\nMAIN DOCUMENT PREDICTIONS")

            print(
                "TF-IDF:",
                prediction["baseline_label"],
                "| score:",
                prediction["baseline_score"],
            )

            print(
                "BERT-Tiny V7:",
                prediction["bert_label"],
                "| score:",
                prediction["bert_score"],
            )

            print("Models agree:", prediction["models_agree"])
            print("Status:", prediction["status"])
            print("Reason:", prediction["reason"])

        output_path = input_path.with_name(
            f"{input_path.stem}_classification_v7.json"
        )

        report = {
            "source_file": str(input_path.resolve()),
            "model_directory": str(MODEL_DIR),
            "max_length": MAX_LENGTH,
            "pages": results,
            "limitations": [
                "The classifier was evaluated on synthetic documents.",
                "Model scores are not calibrated probabilities.",
                "OCR errors may change classification results.",
                "Window predictions are diagnostic only and are not aggregated.",
                "Extracted fields require manual verification.",
                "The model does not certify ownership, title, authenticity, "
                "encumbrance status, or legal validity.",
            ],
        }

        output_path.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print("\nClassification report saved:")
        print(output_path.resolve())

    except Exception as exc:
        print(f"\nERROR: {exc}")
        raise


if __name__ == "__main__":
    main()