# Property AI — Property Document Screening Assistant

Property AI is a Python-based document-screening prototype designed to process residential property documents, extract candidate information, classify document types, and flag potential inconsistencies across multiple documents.

The initial target market is Delhi, India.

**Important:** Property AI is a screening assistant, not a legal verification system. It does not establish property ownership, document authenticity, title validity, registration validity, or the absence of encumbrances.

## Features

- **PDF text extraction:** Extracts embedded text using PyMuPDF.
- **OCR fallback:** Uses Tesseract OCR for scanned documents when configured.
- **Document classification:** Uses a TF-IDF baseline and a fine-tuned BERT-Tiny classifier.
- **Candidate-field extraction:** Identifies potentially relevant information in property documents.
- **Cross-document comparison:** Flags possible inconsistencies in extracted candidate values.
- **Human review:** Supports review of uncertain predictions and potential discrepancies.
- **Exports:** Supports JSON and CSV outputs.
- **Testing:** Includes automated tests for document processing, comparisons, and safety safeguards.

Feature availability depends on the current implementation and its configured dependencies.

## Technology Stack

- Python 3.10
- Streamlit
- PyMuPDF
- Tesseract OCR
- scikit-learn
- Hugging Face Transformers
- PyTorch
- pytest


This is the intended core structure. Additional modules, model assets, training scripts, and tests may exist in the repository.

## Prerequisites

- Windows 10 or Windows 11 for local development.
- Python 3.10.
- Git, if you intend to use GitHub.
- Tesseract OCR installed and accessible when scanned-document OCR is required.
- The dependencies and model assets required by the current implementation.

Create a virtual environment if one does not already exist:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install Python dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Check the Python and package versions if installation fails:

```powershell
python --version
python -m pip check
```

### Configure Tesseract OCR

Install Tesseract OCR separately if required. Confirm that Windows can locate it:

```powershell
tesseract --version
```

If the command is not recognized, add the Tesseract installation directory to your PATH or configure the executable path in the application.

OCR availability and language support depend on the local installation.

## Run the Application

From the project root, with the virtual environment activated:

```powershell
python -m streamlit run main.py
```

Open the local URL printed by Streamlit in your terminal.

Use synthetic or otherwise authorized test documents while validating the application.

## Run Tests

Run the focused document-bundle tests:

```powershell
python -m pytest .\test_document_bundle.py -v
```

Run the full test suite:

```powershell
python -m pytest -q
```

Review any failures before deployment. Do not assume that previous test results still apply after code changes.

## Model Evaluation

The project has previously been evaluated using synthetic documents. Historical results were reported at approximately 98.68% for precision, recall, and F1 in a particular evaluation.

These results are preliminary and must not be interpreted as real-world property-document accuracy. Synthetic data may be easier to classify than genuine documents and may not represent real-world variation.

A rigorous evaluation should include:

- A clearly defined training, validation, and test split.
- Comparison against a TF-IDF baseline.
- Precision, recall, macro-F1, and a confusion matrix.
- Error analysis for OCR, classification, and field extraction.
- Independent ground-truth annotations for genuine documents before making real-world performance claims.

## Limitations and Safety

- OCR errors may propagate into downstream classification and extraction.
- Extracted candidate fields may be incomplete, incorrect, or ambiguous.
- Similarity-based comparison may flag legitimate variations or miss meaningful discrepancies.
- A flagged discrepancy is a reason for human review, not proof of fraud.
- Synthetic evaluation does not establish generalization to real property documents.
- The prototype does not replace a qualified property lawyer, official land records, or professional due diligence.

Before processing real customer documents, implement appropriate privacy controls, secure storage, access restrictions, and data-retention policies.

## GitHub and Deployment

Before publishing the repository:

1. Verify `.gitignore` excludes `.venv/`, caches, secrets, temporary files, and private documents.
2. Audit model assets and confirm which files are needed for inference.
3. Verify `requirements.txt` and any required system packages.
4. Test the application in a fresh environment.
5. Test OCR, classification, document comparison, and exports.
6. Deploy a synthetic-data demo only after checking hosting compatibility, model download requirements, memory limits, and startup time.

Streamlit Community Cloud is a potential hosting option, subject to its current resource limits and the application's dependencies.

## Future Work

- Evaluate on independently annotated, authorized real-world documents.
- Improve OCR robustness and extraction accuracy.
- Assess model calibration and uncertainty handling.
- Improve normalization of property identifiers, addresses, and dates.
- Add secure document handling and configurable retention policies.
- Document deployment reproducibility and model versions.
- Conduct systematic error analysis and compare alternative classifiers.

## Disclaimer

Property AI is an experimental document-screening prototype for research and educational purposes. It does not provide legal advice or certify property ownership, title, authenticity, registration, or encumbrance status.
