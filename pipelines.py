from retriever import get_retriever
from api_calls import call_huggingface_inference_api, call_openai_api

def get_rag_pipelines(retriever, hf_api_key=None, openai_api_key=None):
    """
    Create a RAG pipeline using Hugging Face and OpenAI APIs.

    Args:
        retriever: LangChain retriever for fetching relevant documents.
        hf_api_key: Hugging Face API key for DeepSeek-R1.
        openai_api_key: OpenAI API key for OpenAI o-1.

    Returns:
        A function for processing queries using both models.
    """
    def process_query(query):
        # Retrieve relevant documents
        relevant_docs = retriever.get_relevant_documents(query)
        context = " ".join([doc.page_content for doc in relevant_docs])

        # Call APIs for answers
        deepseek_answer = None
        openai_answer = None

        if hf_api_key:
            deepseek_answer = call_huggingface_inference_api("deepseek-ai/DeepSeek-R1", context, query, hf_api_key)

        if openai_api_key:
            openai_answer = call_openai_api(context, query, openai_api_key)

        return {
            "context": context,
            "DeepSeek-R1 Answer": deepseek_answer,
            "OpenAI-o-1 Answer": openai_answer,
        }

    return process_query
