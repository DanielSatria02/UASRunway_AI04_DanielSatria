"""
Runway Boutique - AI Inventory Analyzer
Main UI application using Streamlit
"""
import streamlit as st
import requests
import logging

from config import (
    DEFAULT_MODEL, PAGE_TITLE, PAGE_ICON, LAYOUT, 
    SAMPLE_ROWS_DISPLAY, PRODUCT_INFO_HEIGHT
)
from services import (
    load_data, get_product_options, get_product_details, 
    format_product_details, ollama_service, build_rules_based_output_table
)
from utils import (
    get_guardrail_prompt_types,
    get_guardrail_question_list,
    get_system_prompt,
    get_task_list,
)
from utils.metrics_loader import load_training_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_guardrail_v2_prompt(chat_history, product_context=""):
    """Build a single prompt from chat history with strict guardrail behavior."""
    system_instruction = """
You are Guardrail V.2 for Runway Boutique.
You are a safe, professional AI assistant focused on fashion inventory analysis.

Rules:
- Refuse requests for passwords, private data, secrets, or personal credentials.
- Refuse unsupported competitor intelligence or invented business facts.
- Reject misleading categorization requests and explain safer alternatives.
- When asked about recategorization, provide a concrete, retail-friendly recommendation.
- For recategorization answers, include: Current Category, Suggested Category, Reason, and Display Guidance.
- If a request is unclear, ask one clarifying question.
- Keep responses concise, clear, and business-oriented.
""".strip()

    transcript = []
    for item in chat_history[-12:]:
        role = item.get("role", "user").upper()
        content = item.get("content", "")
        transcript.append(f"{role}: {content}")

    context_block = ""
    if product_context.strip():
        context_block = f"Selected product context:\n{product_context.strip()}\n\n"

    return (
        f"{system_instruction}\n\n"
        f"{context_block}"
        "Conversation:\n"
        f"{'\n'.join(transcript)}\n\n"
        "ASSISTANT:"
    )

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
)

st.title(f"{PAGE_ICON} Runway Boutique - AI Inventory Analyzer")
st.caption("AI application for evaluating and recategorizing clothing collections for climate and market conditions.")

# ============================================================
# SIDEBAR - SETTINGS
# ============================================================
with st.sidebar:
    st.header("⚙️ Settings")
    st.caption("Model and guardrail configuration")

    # Build model options from local Ollama, with sensible fallbacks.
    available_models = ollama_service.list_models()
    fallback_models = ["qwen2.5:1.5b", "qwen3.5:0.8b", DEFAULT_MODEL]
    model_options = sorted(set(available_models + fallback_models))

    default_index = model_options.index(DEFAULT_MODEL) if DEFAULT_MODEL in model_options else 0
    model = st.selectbox("Ollama model", options=model_options, index=default_index)

    guardrail_version = st.selectbox(
        "Guardrail version",
        options=["Guardrail V.1", "Guardrail V.2"],
        index=0,
    )

    if not available_models:
        st.caption("Could not fetch model list from Ollama. Showing fallback options.")

    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)

    if guardrail_version == "Guardrail V.2":
        if st.button("Clear chat history"):
            st.session_state["guardrail_v2_messages"] = [
                {
                    "role": "assistant",
                    "content": "Guardrail V.2 is ready. Ask anything related to inventory analysis.",
                }
            ]
    
    # Check Ollama connection
    if st.button("Test Ollama Connection"):
        if ollama_service.check_connection():
            st.success("✅ Ollama is running!")
        else:
            st.error("❌ Cannot connect to Ollama. Make sure it's running: `ollama serve`")
    
    st.info("Make sure the Ollama model is running (`ollama serve`).")


if guardrail_version == "Guardrail V.2":
    st.subheader("Guardrail V.2 Chat")
    st.caption("Clean chat mode with built-in safety boundaries for prompt testing.")

    try:
        df_v2 = load_data()
    except FileNotFoundError:
        st.error("File 'styles.csv' not found. Make sure the file is in the same folder as RunwayML.py")
        st.stop()

    product_options_v2 = get_product_options(df_v2)
    selected_option_v2 = st.selectbox(
        "Product context for chat (recommended):",
        product_options_v2,
        key="v2_product_selector",
    )
    selected_id_v2 = int(selected_option_v2.split(" - ")[0])
    product_details_v2 = get_product_details(df_v2, selected_id_v2)
    product_details_text_v2 = format_product_details(product_details_v2)
    with st.expander("Selected product details", expanded=False):
        st.text_area(
            "V.2 Product Information",
            product_details_text_v2,
            height=150,
            disabled=True,
            key="v2_product_details",
        )

    if "guardrail_v2_messages" not in st.session_state:
        st.session_state["guardrail_v2_messages"] = [
            {
                "role": "assistant",
                "content": "Guardrail V.2 is ready. Ask anything related to inventory analysis.",
            }
        ]

    for message in st.session_state["guardrail_v2_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_chat_input = st.chat_input("Message Guardrail V.2")

    if user_chat_input:
        st.session_state["guardrail_v2_messages"].append(
            {"role": "user", "content": user_chat_input}
        )
        with st.chat_message("user"):
            st.markdown(user_chat_input)

        with st.chat_message("assistant"):
            with st.spinner(f"Thinking with {model}..."):
                try:
                    full_prompt = build_guardrail_v2_prompt(
                        st.session_state["guardrail_v2_messages"],
                        product_context=product_details_text_v2,
                    )
                    assistant_reply = ollama_service.call_model(full_prompt, model, temperature)
                    if not assistant_reply.strip():
                        assistant_reply = (
                            "I could not generate a response this turn. "
                            "Please retry your message."
                        )
                except requests.exceptions.ConnectionError:
                    assistant_reply = "Cannot connect to Ollama. Start it with: ollama serve"
                except requests.exceptions.HTTPError as e:
                    assistant_reply = f"Ollama HTTP error: {e}"
                except Exception as e:
                    assistant_reply = f"Error: {e}"

                st.markdown(assistant_reply)

        st.session_state["guardrail_v2_messages"].append(
            {"role": "assistant", "content": assistant_reply}
        )

    st.stop()


# ============================================================
# LOAD DATA (V.1)
# ============================================================
try:
    df = load_data()
except FileNotFoundError:
    st.error("File 'styles.csv' not found. Make sure the file is in the same folder as RunwayML.py")
    st.stop()

# ============================================================
# MAIN DASHBOARD
# ============================================================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Dataset Inventory")
    
    # Load and display training metrics
    metrics = load_training_metrics()
    if metrics:
        st.subheader("**Machine Learning Model Performance**")
        
        # Create metric columns with custom styling
        metric_cols = st.columns(5)
        
        with metric_cols[0]:
            accuracy_val = metrics.get('accuracy', 0)
            st.markdown(f"<div style='text-align: center'><small>Accuracy</small><br><span style='font-size: 18px; font-weight: bold'>{accuracy_val*100:.1f}%</span></div>", unsafe_allow_html=True)
        
        with metric_cols[1]:
            st.markdown(f"<div style='text-align: center'><small>Missing</small><br><span style='font-size: 18px; font-weight: bold'>{metrics.get('missing_deleted', 0)}</span></div>", unsafe_allow_html=True)
        
        with metric_cols[2]:
            rainy_f1 = metrics.get('rainy_f1', 0)
            st.markdown(f"<div style='text-align: center'><small>Rainy F1</small><br><span style='font-size: 18px; font-weight: bold'>{rainy_f1*100:.1f}%</span></div>", unsafe_allow_html=True)
        
        with metric_cols[3]:
            rainy_recall = metrics.get('rainy_recall', 0)
            st.markdown(f"<div style='text-align: center'><small>Rainy Recall</small><br><span style='font-size: 18px; font-weight: bold'>{rainy_recall*100:.1f}%</span></div>", unsafe_allow_html=True)
        
        with metric_cols[4]:
            rainy_prec = metrics.get('rainy_precision', 0)
            st.markdown(f"<div style='text-align: center'><small>Rainy Prec</small><br><span style='font-size: 18px; font-weight: bold'>{rainy_prec*100:.1f}%</span></div>", unsafe_allow_html=True)
        
        # Additional metrics
        metric_cols2 = st.columns(5)
        
        with metric_cols2[0]:
            sunny_f1 = metrics.get('sunny_f1', 0)
            st.markdown(f"<div style='text-align: center'><small>Sunny F1</small><br><span style='font-size: 18px; font-weight: bold'>{sunny_f1*100:.1f}%</span></div>", unsafe_allow_html=True)
        
        with metric_cols2[1]:
            sunny_recall = metrics.get('sunny_recall', 0)
            st.markdown(f"<div style='text-align: center'><small>Sunny Recall</small><br><span style='font-size: 18px; font-weight: bold'>{sunny_recall*100:.1f}%</span></div>", unsafe_allow_html=True)
        
        with metric_cols2[2]:
            sunny_prec = metrics.get('sunny_precision', 0)
            st.markdown(f"<div style='text-align: center'><small>Sunny Prec</small><br><span style='font-size: 18px; font-weight: bold'>{sunny_prec*100:.1f}%</span></div>", unsafe_allow_html=True)
        
        with metric_cols2[3]:
            st.markdown(f"<div style='text-align: center'><small>Initial</small><br><span style='font-size: 18px; font-weight: bold'>{metrics.get('initial_rows', 0):,}</span></div>", unsafe_allow_html=True)
        
        with metric_cols2[4]:
            st.markdown(f"<div style='text-align: center'><small>Train Samples</small><br><span style='font-size: 18px; font-weight: bold'>{metrics.get('train_size', 0):,}</span></div>", unsafe_allow_html=True)
        
        st.divider()
    
    # Display sample data from dataset
    st.dataframe(
        df[['id', 'productDisplayName', 'season', 'articleType', 'usage']].head(SAMPLE_ROWS_DISPLAY),
        use_container_width=True
    )

    st.subheader("rules-based output")
    rules_source_df = df[['productDisplayName', 'season', 'masterCategory']].head(SAMPLE_ROWS_DISPLAY).copy()
    rules_output_df = build_rules_based_output_table(rules_source_df)
    st.dataframe(rules_output_df, use_container_width=True)
    
    st.subheader("2. Select Product for Analysis")
    
    # Get product options
    product_options = get_product_options(df)
    selected_option = st.selectbox("Search and select product:", product_options)
    
    # Extract and get product details
    selected_id = int(selected_option.split(" - ")[0])
    product_details = get_product_details(df, selected_id)
    product_details_text = format_product_details(product_details)
    
    st.text_area(
        "Product Information:",
        product_details_text,
        height=PRODUCT_INFO_HEIGHT,
        disabled=True
    )

    st.subheader("3. Guardrail V.1 Prompt Testing")
    prompt_mode_options = ["Use Case Specific"] + get_guardrail_prompt_types()
    prompt_mode = st.selectbox("Choose which prompt set to run:", prompt_mode_options)

    selected_question = ""
    task = ""
    if prompt_mode == "Use Case Specific":
        task_list = get_task_list()
        task = st.selectbox("Choose a use-case specific task:", task_list)
    elif prompt_mode == "Concept Question":
        selected_question = st.selectbox(
            "Concept question examples:",
            get_guardrail_question_list("Concept Question")
        )
    elif prompt_mode == "Misleading Question":
        selected_question = st.selectbox(
            "Misleading question examples:",
            get_guardrail_question_list("Misleading Question")
        )
    elif prompt_mode == "Privacy Request":
        selected_question = st.selectbox(
            "Privacy request examples:",
            get_guardrail_question_list("Privacy Request")
        )
    elif prompt_mode == "Cheating Request":
        selected_question = st.selectbox(
            "Cheating request examples:",
            get_guardrail_question_list("Cheating Request")
        )

    run = st.button("Analyze with AI")

# ============================================================
# ANALYSIS OUTPUT
# ============================================================
with col2:
    st.subheader("AI Analysis Output")
    
    if run:
        # Generate prompt and call Ollama
        if prompt_mode == "Use Case Specific":
            full_prompt = get_system_prompt(product_details_text, task_name=task)
        else:
            full_prompt = get_system_prompt(
                product_details_text,
                question_type=prompt_mode,
                question_text=selected_question,
            )
        
        with st.spinner(f"Analyzing inventory using {model}..."):
            try:
                result = ollama_service.call_model(full_prompt, model, temperature)
                if result.strip():
                    st.success("Complete!")
                    st.markdown(result)
                else:
                    st.warning(
                        f"The model '{model}' returned an empty response. "
                        "Try running `ollama run <model>` once in terminal to warm it up, then retry."
                    )
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to Ollama. Make sure Ollama is running: `ollama serve`")
            except requests.exceptions.HTTPError as e:
                st.error(f"❌ Ollama HTTP error: {e}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    else:
        st.info("Select a product on the left, choose an AI task, then click the analyze button to get suitability analysis.")