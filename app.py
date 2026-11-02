"""Streamlit demo: no accounts, no external services, no visitor text logging."""
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from supportlens.predict import ArtifactError, load_active_bundle, predict  # noqa: E402

st.set_page_config(page_title="SupportLens", page_icon="🔎", layout="centered")
st.markdown("""
<style>
    .block-container {max-width: 820px; padding-top: 2.5rem; padding-bottom: 3rem;}
    h1, h2, h3 {letter-spacing: -0.025em;}
    .eyebrow {color:#397965; font-size:.78rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase;}
    .intro {color:#4a5c58; font-size:1.08rem; line-height:1.55;}
    .stButton button {border-radius: 8px; font-weight: 650;}
    [data-testid="stMetric"] {background:#f2f7f4; border:1px solid #dfebe3; border-radius:10px; padding:14px;}
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] > div {
        white-space: normal !important; overflow-wrap: anywhere; overflow: visible !important;
        text-overflow: clip !important; line-height: 1.15;
    }
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="eyebrow">An intent classification demo</div>', unsafe_allow_html=True)
st.title("SupportLens")
st.markdown('<p class="intro">Explore how a model categorizes English banking support questions and flags uncertain predictions for review.</p>', unsafe_allow_html=True)
st.info("Please don’t enter account numbers, passwords, or other sensitive personal banking information.")

samples = {
    "Write my own question": "",
    "Card delivery": "Where is the card I ordered?",
    "Cash withdrawal": "Why was I charged a fee for withdrawing cash?",
    "Card security": "I think my card has been stolen. What should I do?",
}
sample = st.selectbox("Try a sample or write your own", list(samples))
question = st.text_area("Support question", value=samples[sample], height=125,
                        placeholder="For example: Where is the card I ordered?")


@st.cache_resource
def cached_bundle(release_signature):
    """Changing the release pointer invalidates the cached model."""
    return load_active_bundle(ROOT / "artifacts")


def current_bundle():
    directory = ROOT / "artifacts"
    pointer = directory / "active.json"
    signature_path = pointer if pointer.exists() else directory / "metadata.json"
    return cached_bundle((signature_path.name, signature_path.read_bytes()))


def readable(label):
    return label.replace("_", " ").capitalize()


if st.button("Predict intent", type="primary"):
    try:
        bundle = current_bundle()
        result = predict(question, bundle)
    except ArtifactError:
        st.error("The model files are missing or incompatible. Follow the setup instructions in the README.")
    except ValueError as exc:
        st.warning(str(exc))
    except (OSError, KeyError, ImportError) as exc:
        st.error("The model files are missing or incompatible. Follow the setup instructions in the README.")
    else:
        st.subheader("Prediction")
        left, right = st.columns(2)
        left.metric("Suggested category", readable(result["label"]))
        right.metric("Model score", f"{result['score']:.1%}")
        if result["review"]:
            reasons = result["review_reasons"]
            if "review_all" in reasons:
                st.warning("Review recommended: this release flags every prediction for review.")
            elif "low_score" in reasons and "weak_word_support" in reasons:
                st.warning(f"Review recommended: the main score is below {bundle[2]['review_threshold']:.0%}, "
                           "and a second model gives limited support to this category.")
            elif "weak_word_support" in reasons:
                st.warning("Review recommended: a second model gives limited support to this category, "
                           "even though the displayed model score is high.")
            else:
                st.warning(f"Review recommended: the top score is below the {bundle[2]['review_threshold']:.0%} review threshold.")
        else:
            st.success("No review flag from the available checks. A human can still review this suggestion.")
        if result["alternatives"]:
            st.caption("Other suggestions: " + " · ".join(
                f"{readable(item['label'])} ({item['score']:.1%})" for item in result["alternatives"]))
        st.caption("A high score can still be wrong, especially for an unfamiliar or ambiguous question.")

with st.expander("About the model and evaluation"):
    try:
        meta = current_bundle()[2]
        if meta.get("artifact_version") in (2, 3):
            locked = json.loads((ROOT / "reports" / "experiment_v2" / "final_lock.json").read_text(encoding="utf-8"))
            if locked["selected_release"] != meta["candidate_id"]:
                raise ValueError("The active model and evaluation report do not match")
            metrics = locked["assessment_results"][meta["candidate_id"]][meta["score_kind"]]
            st.write(f"Word and character TF-IDF with logistic regression. On the development assessment "
                     f"({metrics['examples']:,} examples; {metrics['supported_labels']} labels), accuracy was "
                     f"**{metrics['accuracy']:.1%}** and macro F1 was **{metrics['macro_f1']:.3f}**. "
                     "This assessment also informed the release decision.")
            st.write("Scores use temperature scaling fitted on development predictions. Calibration can help, "
                     "but scores are not guaranteed probabilities.")
        else:
            measured = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
            metrics = measured["models"][measured["selected_before_test"]]
            st.write(f"V1 logistic regression: official test accuracy **{metrics['accuracy']:.1%}** "
                     f"and macro F1 **{metrics['macro_f1']:.3f}** on {metrics['examples']:,} examples.")
        if meta.get("review_all", False):
            st.write("All predictions are flagged for review.")
        else:
            st.write(f"Main score review threshold: {meta['review_threshold']:.0%}. A high score does not reliably detect an unfamiliar question.")
        if meta.get("review_guard"):
            st.write("A second word-only model can also flag a suggestion when it gives the category limited support. "
                     "The models share training data and can both be wrong; this check does not reliably detect every ambiguity.")
    except (OSError, KeyError, ValueError):
        st.write("Evaluation evidence is unavailable. See the project README for setup.")

st.divider()
st.caption("Educational demo using public research data. This is not a bank service, does not access accounts, "
           "and does not answer or route real support requests.")
st.caption("Data: [BANKING77 by PolyAI](https://huggingface.co/datasets/PolyAI/banking77), "
           "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).")
