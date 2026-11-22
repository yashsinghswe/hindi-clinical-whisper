# Hindi Clinical Whisper: fine-tuning Whisper Small for doctor–patient speech

Fine-tuning OpenAI **Whisper Small** for Hindi/Hinglish clinical consultations: medicine names, doses, symptoms and code-switched speech that off-the-shelf ASR gets badly wrong.

| | Word error rate (held-out set) |
|---|---|
| Whisper Small (base) | ~80% |
| **Fine-tuned (this pipeline)** | **6–7%** |

Trained on **2,000+ synthetic recordings** of clinical conversations over several incremental rounds, each fine-tuning from the previous best model. No real patient audio is included in this repo.

> Built by me as an AI Engineer at DiligentMinds Consulting for a clinical speech AI product (2025–2026).

**Stack:** PyTorch · Hugging Face Transformers / Datasets · Whisper · jiwer · librosa · Google Colab (GPU)

---

## The notebook

[`train_whisper_small_hindi_clinical.ipynb`](train_whisper_small_hindi_clinical.ipynb) is one **incremental** training round, included as a representative example. It starts from the previous fine-tuned checkpoint, trains on a new set of full-length conversations, and exports a separate model, so the earlier one is never overwritten. The other rounds used the same pipeline on their own datasets.

### Pipeline

1. **Match data**: strictly pairs `CONV_###` CSV rows with `CONV_###.wav` files, and fails loudly on any mismatch.
2. **Split before chunking**: train/validation/test splits are made per *recording*, so chunks of one conversation can't leak across splits.
3. **Sentence-aware chunking**: long conversations are cut into complete-sentence segments under Whisper's 30 s / token limits. Cut points are snapped to quiet regions in the audio, and only an oversized single sentence falls back to a finer boundary.
4. **Hard completeness gate**: training stops if any recording failed, is missing, or didn't fully cover its text or audio. Nothing is silently dropped.
5. **Training**: `Seq2SeqTrainer`, fp16, lr 1e-5 with linear warmup, eval/save every 100 steps, best checkpoint chosen by WER, fully resumable.
6. **Evaluation & export**: WER on the held-out test set, per-sample predictions CSV, and a JSON training report.

### Key decisions
- **Synthetic data**: real clinical audio is scarce and sensitive, so scripted, varied consultations were generated and recorded to cover vocabulary such as drug names, dosages and negations.
- **Incremental rounds**: each new dataset fine-tunes from the last good model, and every version is kept for comparison.
- **WER isn't the whole story**: errors in medications, doses and negations matter far more than filler words, so candidates are also reviewed on those before deployment.

## Running it
Open the notebook in Google Colab with a GPU runtime. Put your dataset in Google Drive under `MyDrive/medical-whisper/` (a CSV with `recording_id, transcript` columns plus matching WAV files), set the paths in **Step 3: Configuration**, and run the cells in order.

## Related
- [realtime-clinical-scribe](https://github.com/yashsinghswe/realtime-clinical-scribe): the live consultation app that serves this model.

## Author
**Yash Singh** · GenAI Engineer · [LinkedIn](https://linkedin.com/in/yashsinghswe)
