import requests
import openai

def call_huggingface_inference_api(model_name, context, query, api_key):
    """
    Call the Hugging Face Inference API for question answering.

    Args:
        model_name (str): The Hugging Face model name (e.g., "deepseek/deepseek-r1").
        context (str): The retrieved context for the query.
        query (str): The user's query.
        api_key (str): Your Hugging Face API key.

    Returns:
        str: The model's answer.
    """
    url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": {"question": query, "context": context}}

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json().get("answer", "No answer found.")
    else:
        raise Exception(f"Hugging Face API error: {response.text}")

def call_openai_api(context, query, api_key, model="text-davinci-003"):
    """
    Call the OpenAI API for question answering.

    Args:
        context (str): The retrieved context for the query.
        query (str): The user's query.
        api_key (str): Your OpenAI API key.
        model (str): The OpenAI model to use (e.g., "text-davinci-003").

    Returns:
        str: The model's answer.
    """
    openai.api_key = api_key

    prompt = f"Answer the question based on the context below:\n\nContext: {context}\n\nQuestion: {query}\nAnswer:"

    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=200,
        temperature=0.7,
    )

    return response.choices[0].text.strip()
