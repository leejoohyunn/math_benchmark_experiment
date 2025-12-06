"""
Model Loading and Inference Module
Supports GPT-4o-mini (API) and HuggingFace models
"""

import os
import torch
from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ModelWrapper:
    """Wrapper class for different model types"""

    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        self.model_name = model_name
        self.model_config = model_config
        self.model_type = model_config['type']

        if self.model_type == 'api':
            self.model = None
            self.tokenizer = None
            self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        elif self.model_type == 'huggingface':
            self.model, self.tokenizer = self._load_huggingface_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _load_huggingface_model(self):
        """Load HuggingFace model and tokenizer"""
        model_id = self.model_config['model_id']
        dtype = getattr(torch, self.model_config.get('dtype', 'float16'))
        device_map = self.model_config.get('device_map', 'auto')
        load_in_8bit = self.model_config.get('load_in_8bit', False)

        print(f"Loading model: {model_id}")

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            load_in_8bit=load_in_8bit,
            trust_remote_code=True
        )

        # Set pad token if not exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.5) -> str:
        """Generate response from the model"""

        if self.model_type == 'api':
            return self._generate_api(prompt, max_new_tokens, temperature)
        elif self.model_type == 'huggingface':
            return self._generate_huggingface(prompt, max_new_tokens, temperature)

    def _generate_api(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        """Generate using OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_config['model_name'],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=self.model_config.get('top_p', 1.0)
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API Error: {e}")
            return ""

    def _generate_huggingface(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        """Generate using HuggingFace model"""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode only the generated part (excluding input)
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return generated_text.strip()

    def unload(self):
        """Unload model from memory"""
        if self.model is not None:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
            print(f"Model {self.model_name} unloaded")


def load_model(model_name: str, model_config: Dict[str, Any]) -> ModelWrapper:
    """
    Load a model based on configuration

    Args:
        model_name: Name of the model
        model_config: Configuration dictionary for the model

    Returns:
        ModelWrapper instance
    """
    print(f"Initializing model: {model_name}")
    return ModelWrapper(model_name, model_config)


def get_model_response(
    model: ModelWrapper,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.5
) -> str:
    """
    Get response from a model

    Args:
        model: ModelWrapper instance
        prompt: Input prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated text
    """
    return model.generate(prompt, max_new_tokens, temperature)
