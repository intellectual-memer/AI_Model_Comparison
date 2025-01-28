import streamlit as st
from pipelines import get_rag_pipelines
from retriever import get_retriever
from evaluation import evaluate_answers

# Backend API keys (set securely)

HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

st.title("OpenAI-o1 vs Deepseek-R1")

# File uploader for documents
uploaded_files = st.file_uploader(
    "Upload documents (txt, pdf, docx)", 
    type=["txt", "pdf", "docx"], 
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Processing uploaded documents..."):
        retriever = get_retriever(uploaded_files)
        st.success("Documents processed successfully!")

        # Model selection dropdown
        model_choice = st.selectbox(
            "Choose the model to use:",
            options=["DeepSeek-R1", "OpenAI o-1"],
        )

        # Load RAG pipeline dynamically based on the model choice
        if model_choice == "DeepSeek-R1":
            process_query = get_rag_pipelines(retriever, HUGGINGFACE_API_KEY, None)
        elif model_choice == "OpenAI o-1":
            process_query = get_rag_pipelines(retriever, None, OPENAI_API_KEY)
        else:
            process_query = None

        # Single query evaluation with dynamic ground truth
        if process_query:
            st.subheader("Ask a Question")
            query = st.text_input("Enter your question:")

            if query:
                with st.spinner("Retrieving ground truth..."):
                    # Retrieve the most relevant document as ground truth
                    retrieved_docs = retriever.get_relevant_documents(query)
                    ground_truth_context = retrieved_docs[0].page_content if retrieved_docs else "No relevant document found."

                with st.spinner("Fetching answers..."):
                    # Generate answers from the selected model
                    results = process_query(query)

                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**DeepSeek-R1**" if model_choice.startswith("DeepSeek") else "**OpenAI o-1**")
                    st.write(results[f"{model_choice.split()[0]} Answer"])
                with col2:
                    st.write("**Retrieved Context**")
                    st.write(ground_truth_context)

                # Display ground truth and evaluate answers
                st.write("### Ground Truth Context")
                st.write(ground_truth_context)

                with st.spinner("Evaluating answers..."):
                    evaluation_results = evaluate_answers(
                        ground_truth_context,
                        results["DeepSeek-R1 Answer"] if model_choice.startswith("DeepSeek") else None,
                        results["OpenAI-o-1 Answer"] if model_choice.startswith("OpenAI") else None,
                    )
                    st.write("### Evaluation Results")
                    st.json(evaluation_results)
else:
    st.warning("Please upload documents to start.")
