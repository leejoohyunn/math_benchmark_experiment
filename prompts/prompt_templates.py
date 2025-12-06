"""
Prompt Templates for Different Model Types
"""

def get_prompt(question: str, model_name: str, config: dict) -> str:
    """
    Get formatted prompt based on model type

    Args:
        question: Math problem to solve
        model_name: Name of the model
        config: Configuration dict with prompt templates

    Returns:
        Formatted prompt string
    """

    # Determine model category
    if 'base' in model_name.lower() and 'instruct' not in model_name.lower():
        # Base models (no instruction tuning)
        template = config['prompts']['base_models']['template']
    elif 'thinking' in model_name.lower():
        # Thinking models (special reasoning format)
        template = config['prompts']['thinking_models']['template']
    else:
        # Instruct models (instruction-tuned)
        template = config['prompts']['instruct_models']['template']

    # Format the template with the question
    prompt = template.format(question=question)

    return prompt


# Example usage and template definitions
BASE_TEMPLATE = "Question: {question}\n\nAnswer:"

INSTRUCT_TEMPLATE = """You are a helpful math tutor. Solve the following math problem step by step.

Problem: {question}

Solution:"""

THINKING_TEMPLATE = """Solve the following math problem. Show your reasoning process.

Problem: {question}

Let's think step by step:"""


def get_default_templates():
    """Return default prompt templates"""
    return {
        'base_models': {'template': BASE_TEMPLATE},
        'instruct_models': {'template': INSTRUCT_TEMPLATE},
        'thinking_models': {'template': THINKING_TEMPLATE}
    }
