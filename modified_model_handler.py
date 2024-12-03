import os
from typing import Dict, List, Optional
import yaml
import re
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from dotenv import load_dotenv
from peft import PeftModel
from ..graph.graph_processor import GraphProcessor
from ..preference.response_ranker import ResponseRanker
from ..utils.prompt_manager import PromptManager
from ..utils.conversation_manager import ConversationManager


class ModelHandler:
    """Handles interactions with the fine-tuned LLM and coordinates with graph processor"""

    def __init__(self, config_path: str = "configs/config.yaml", hf_token: Optional[str] = None):
        # Load environment variables
        load_dotenv()

        # Load configuration
        self.config = self._load_config(config_path)

        # Setup logging
        logging.basicConfig(level=self.config['system']['log_level'])
        self.logger = logging.getLogger(__name__)

        # Get HuggingFace token
        self.hf_token = hf_token or os.getenv('HUGGINGFACE_TOKEN')
        if self.config['model']['use_auth_token'] and not self.hf_token:
            raise ValueError("HuggingFace token is required but not provided.")

        # Set device
        if torch.cuda.is_available() and self.config['model']['device'] == 'cuda':
            self.device = torch.device('cuda')
            self.logger.info("Using CUDA device")
        else:
            self.device = torch.device('cpu')
            self.logger.info("Using CPU device")

        # Initialize components
        self.model, self.tokenizer = self._load_model()
        self.graph_processor = GraphProcessor(config_path)
        self.prompt_manager = PromptManager()
        self.conversation_manager = ConversationManager()
        self.response_ranker = ResponseRanker()

    def _load_model(self):
        """Load model and tokenizer from HuggingFace"""
        try:
            base_model_id = "unsloth/llama-3.2-3b-instruct-bnb-4bit"  # Changed to Unsloth base model
            adapter_model_id = self.config['model']['model_id']
            self.logger.info(f"Loading model from {adapter_model_id}")

            tokenizer = AutoTokenizer.from_pretrained(
                base_model_id,
                token=self.hf_token if self.config['model']['use_auth_token'] else None
            )

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                token=self.hf_token if self.config['model']['use_auth_token'] else None,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
                low_cpu_mem_usage=True
            )

            model = PeftModel.from_pretrained(
                base_model,
                adapter_model_id,
                token=self.hf_token if self.config['model']['use_auth_token'] else None
            )

            if self.device.type == "cpu":
                model = model.to(self.device)

            return model, tokenizer

        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def detect_emotion(self, text: str, conversation_history: str = "") -> Dict:
        """Task 1: Emotion Detection"""
        try:
            # Create prompt using prompt manager
            prompt = self.prompt_manager.create_emotion_detection_prompt(
                    text=text,
                    conversation_history=conversation_history,
            )

            # Tokenize and generate
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.config['model']['max_length'],
                    temperature=self.config['model']['temperature'],
                    top_p=self.config['model']['top_p'],
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode and parse response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("=======================================================")
            print("response: ",response) ##debugging
            parsed_response = self._parse_emotion_response(response)
            print("=======================================================")
            print("Final Parsed Response for emotion and its cause detection: ", parsed_response)

            if parsed_response['emotion'] == 'unknown':
                raise ValueError("Could not detect emotion from model response")

            return parsed_response

        except Exception as e:
            self.logger.error(f"Error in emotion detection: {str(e)}")
            return {
                'emotion': 'neutral',  # Default fallback
                'cause': 'unclear',
                'confidence': 0.0
            }