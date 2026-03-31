"""
LLM Client Module for Task Specification Agent

This module provides a factory pattern for creating LLM clients with support for
Azure OpenAI and other providers. It includes connection testing, retry logic,
and structured output support.
"""

import os
import logging
import time
import json
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pathlib import Path

import yaml
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Optional imports for HuggingFace (only needed if using local models)
try:
    import torch
    from langchain_huggingface import HuggingFacePipeline
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    torch = None

# Optional imports for AWS Bedrock (only needed if using Bedrock models)
try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None
    ClientError = Exception


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLMConnectionError(Exception):
    """Custom exception for LLM connection and API errors."""
    pass


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.

    All LLM client implementations must inherit from this class and
    implement the required abstract methods.
    """

    @abstractmethod
    def invoke(self, prompt: str) -> str:
        """
        Send a prompt to the LLM and get a text response.

        Args:
            prompt: The text prompt to send to the LLM

        Returns:
            The LLM's text response

        Raises:
            LLMConnectionError: If the LLM call fails
        """
        pass

    @abstractmethod
    def invoke_with_schema(self, prompt: str, output_schema: dict) -> dict:
        """
        Send a prompt to the LLM and get a structured response matching the schema.

        Args:
            prompt: The text prompt to send to the LLM
            output_schema: JSON schema defining the expected output structure

        Returns:
            Dictionary matching the provided schema

        Raises:
            LLMConnectionError: If the LLM call fails or output doesn't match schema
        """
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test the connection to the LLM service.

        Returns:
            True if connection is successful, False otherwise
        """
        pass


class AzureLLMClient(BaseLLMClient):
    """
    Azure OpenAI LLM client implementation.

    This client uses LangChain's AzureChatOpenAI to interact with Azure OpenAI services.
    It includes automatic retry logic and connection testing capabilities.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Azure LLM client.

        Args:
            config: Configuration dictionary from llm_config.yaml

        Raises:
            LLMConnectionError: If required environment variables are missing
        """
        self.config = config
        self.max_retries = 3
        self.retry_delay = 1  # seconds

        # Get required environment variables
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

        # Validate environment variables
        if not self.api_key:
            raise LLMConnectionError("AZURE_OPENAI_API_KEY environment variable is not set")
        if not self.endpoint:
            raise LLMConnectionError("AZURE_OPENAI_ENDPOINT environment variable is not set")
        if not self.deployment_name:
            raise LLMConnectionError("AZURE_OPENAI_DEPLOYMENT_NAME environment variable is not set")

        # Initialize Azure ChatOpenAI client
        try:
            self.client = AzureChatOpenAI(
                azure_endpoint=self.endpoint,
                azure_deployment=self.deployment_name,
                api_key=self.api_key,
                api_version=config.get("api_version", "2024-02-15-preview"),
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 2000),
            )
            logger.info(f"Azure LLM client initialized with deployment: {self.deployment_name}")
        except Exception as e:
            raise LLMConnectionError(f"Failed to initialize Azure LLM client: {str(e)}")

    def test_connection(self) -> bool:
        """
        Test the connection to Azure OpenAI service.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            logger.info("Testing Azure OpenAI connection...")
            test_prompt = "Hello, this is a connection test. Please respond with 'OK'."
            response = self.client.invoke([HumanMessage(content=test_prompt)])
            logger.info(f"Connection test successful. Response: {response.content[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False

    def _invoke_with_retry(self, messages: list, retry_count: int = 0) -> str:
        """
        Internal method to invoke the LLM with retry logic.

        Args:
            messages: List of message objects to send
            retry_count: Current retry attempt number

        Returns:
            The LLM's response content

        Raises:
            LLMConnectionError: If all retry attempts fail
        """
        try:
            response = self.client.invoke(messages)
            try:
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    from src.common.token_tracker import tracker
                    tracker.record(
                        step="task_spec_agent",
                        model=getattr(self, "model_name", "azure"),
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                    )
            except Exception:
                pass
            return response.content
        except Exception as e:
            if retry_count < self.max_retries:
                logger.warning(
                    f"LLM invocation failed (attempt {retry_count + 1}/{self.max_retries}): {str(e)}"
                )
                time.sleep(self.retry_delay * (retry_count + 1))  # Exponential backoff
                return self._invoke_with_retry(messages, retry_count + 1)
            else:
                error_msg = f"LLM invocation failed after {self.max_retries} attempts: {str(e)}"
                logger.error(error_msg)
                raise LLMConnectionError(error_msg)

    def invoke(self, prompt: str) -> str:
        """
        Send a prompt to the Azure LLM and get a text response.

        Args:
            prompt: The text prompt to send to the LLM

        Returns:
            The LLM's text response

        Raises:
            LLMConnectionError: If the LLM call fails after all retries
        """
        logger.debug(f"Invoking Azure LLM with prompt (length: {len(prompt)} chars)")
        messages = [HumanMessage(content=prompt)]
        response = self._invoke_with_retry(messages)
        logger.debug(f"Received response (length: {len(response)} chars)")
        return response

    def invoke_with_schema(self, prompt: str, output_schema: dict) -> dict:
        """
        Send a prompt to the Azure LLM and get a structured response matching the schema.

        Args:
            prompt: The text prompt to send to the LLM
            output_schema: JSON schema defining the expected output structure

        Returns:
            Dictionary matching the provided schema

        Raises:
            LLMConnectionError: If the LLM call fails or output doesn't match schema
        """
        logger.debug(f"Invoking Azure LLM with schema (prompt length: {len(prompt)} chars)")

        try:
            # Create a parser for JSON output
            parser = JsonOutputParser()

            # Add schema instructions to the prompt
            schema_instruction = f"\n\nPlease provide your response as a JSON object matching this schema:\n{yaml.dump(output_schema, default_flow_style=False)}"
            schema_instruction += "\n\nIMPORTANT: Return ONLY the JSON object, without any additional text or markdown formatting."

            full_prompt = prompt + schema_instruction

            # Invoke the LLM
            messages = [HumanMessage(content=full_prompt)]
            response_text = self._invoke_with_retry(messages)

            # Parse the response
            # Remove markdown code blocks if present
            cleaned_response = response_text.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]  # Remove ```json
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]  # Remove ```
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]  # Remove closing ```
            cleaned_response = cleaned_response.strip()

            # Parse JSON
            import json
            try:
                result = json.loads(cleaned_response)
                logger.debug("Successfully parsed structured response")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
                logger.error(f"Response text: {cleaned_response[:200]}...")
                raise LLMConnectionError(f"Failed to parse LLM response as JSON: {str(e)}")

        except LLMConnectionError:
            raise
        except Exception as e:
            error_msg = f"Failed to invoke LLM with schema: {str(e)}"
            logger.error(error_msg)
            raise LLMConnectionError(error_msg)


class HuggingFaceLLMClient(BaseLLMClient):
    """
    HuggingFace LLM client implementation.

    This client uses HuggingFace Transformers to run local models like EXAONE, LLaMA, etc.
    It supports both CPU and GPU execution with automatic device detection.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the HuggingFace LLM client.

        Args:
            config: Configuration dictionary from llm_config.yaml

        Raises:
            LLMConnectionError: If model initialization fails or HuggingFace not installed
        """
        if not HUGGINGFACE_AVAILABLE:
            raise LLMConnectionError(
                "HuggingFace dependencies not installed. "
                "Install with: pip install torch transformers langchain-huggingface"
            )

        self.config = config
        self.model_name = config.get("model_name", "LGAI-EXAONE/EXAONE-3.5-32B-Instruct")

        # Device configuration with automatic detection
        device_config = config.get("device", "cuda")
        if device_config == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"
        else:
            self.device = device_config

        # Determine torch dtype
        dtype_str = config.get("torch_dtype", "float16")
        if dtype_str == "float16" and self.device == "cpu":
            logger.warning("float16 not supported on CPU. Using float32 instead.")
            self.torch_dtype = torch.float32
        elif dtype_str == "float16":
            self.torch_dtype = torch.float16
        elif dtype_str == "bfloat16":
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float32

        # Generation parameters
        self.max_new_tokens = config.get("max_new_tokens", 2000)
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 0.9)
        self.do_sample = config.get("do_sample", True)

        try:
            # Initialize model and tokenizer
            logger.info(f"Loading HuggingFace model: {self.model_name}")
            logger.info(f"Device: {self.device}, Dtype: {self.torch_dtype}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )

            if self.device == "cpu":
                self.model = self.model.to(self.device)

            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
                device=0 if self.device == "cuda" and torch.cuda.is_available() else -1,
            )

            # Create LangChain wrapper
            self.llm = HuggingFacePipeline(pipeline=self.pipeline)

            logger.info("HuggingFace model loaded successfully!")

        except Exception as e:
            error_msg = f"Failed to initialize HuggingFace model: {str(e)}"
            logger.error(error_msg)
            raise LLMConnectionError(error_msg)

    def test_connection(self) -> bool:
        """
        Test if the HuggingFace model can generate responses.

        Returns:
            True if model is working correctly, False otherwise
        """
        try:
            logger.info("Testing HuggingFace model...")
            test_prompt = "Hello, this is a connection test. Please respond with 'OK'."
            response = self.invoke(test_prompt)
            logger.info(f"Model test successful. Response: {response[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Model test failed: {str(e)}")
            return False

    def invoke(self, prompt: str) -> str:
        """
        Send a prompt to the HuggingFace model and get a text response.

        Args:
            prompt: The text prompt to send to the model

        Returns:
            The model's text response

        Raises:
            LLMConnectionError: If the model call fails
        """
        logger.debug(f"Invoking HuggingFace model with prompt (length: {len(prompt)} chars)")

        try:
            # Generate response using the pipeline
            result = self.llm.invoke(prompt)

            # Extract the generated text
            if isinstance(result, str):
                generated_text = result
            else:
                generated_text = str(result)

            # Some models repeat the prompt in the output, so we try to remove it
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()

            logger.debug(f"Received response (length: {len(generated_text)} chars)")
            return generated_text

        except Exception as e:
            error_msg = f"HuggingFace model invocation failed: {str(e)}"
            logger.error(error_msg)
            raise LLMConnectionError(error_msg)

    def invoke_with_schema(self, prompt: str, output_schema: dict) -> dict:
        """
        Send a prompt to the HuggingFace model and get a structured response matching the schema.

        Args:
            prompt: The text prompt to send to the model
            output_schema: JSON schema defining the expected output structure

        Returns:
            Dictionary matching the provided schema

        Raises:
            LLMConnectionError: If the model call fails or output doesn't match schema
        """
        logger.debug(f"Invoking HuggingFace model with schema (prompt length: {len(prompt)} chars)")

        try:
            # Add schema instructions to the prompt
            schema_instruction = f"\n\nPlease provide your response as a JSON object matching this schema:\n{yaml.dump(output_schema, default_flow_style=False)}"
            schema_instruction += "\n\nIMPORTANT: Return ONLY the JSON object, without any additional text or markdown formatting."

            full_prompt = prompt + schema_instruction

            # Invoke the model
            response_text = self.invoke(full_prompt)

            # Parse the response
            # Remove markdown code blocks if present
            cleaned_response = response_text.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]  # Remove ```json
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]  # Remove ```
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]  # Remove closing ```
            cleaned_response = cleaned_response.strip()

            # Try to find JSON boundaries if the model added extra text
            start_idx = cleaned_response.find('{')
            end_idx = cleaned_response.rfind('}')

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                cleaned_response = cleaned_response[start_idx:end_idx + 1]

            # Parse JSON
            try:
                result = json.loads(cleaned_response)
                logger.debug("Successfully parsed structured response")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse model response as JSON: {str(e)}")
                logger.error(f"Response text: {cleaned_response[:200]}...")
                raise LLMConnectionError(f"Failed to parse model response as JSON: {str(e)}")

        except LLMConnectionError:
            raise
        except Exception as e:
            error_msg = f"Failed to invoke HuggingFace model with schema: {str(e)}"
            logger.error(error_msg)
            raise LLMConnectionError(error_msg)


class BedrockLLMClient(BaseLLMClient):
    """
    AWS Bedrock LLM client implementation.

    This client uses boto3 to interact with AWS Bedrock's converse API,
    supporting Claude Opus 4.6 and other Bedrock-hosted models.
    It includes automatic retry logic and connection testing capabilities.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AWS Bedrock LLM client.

        Args:
            config: Configuration dictionary from llm_config.yaml

        Raises:
            LLMConnectionError: If boto3 is not installed or required config is missing
        """
        if not BOTO3_AVAILABLE:
            raise LLMConnectionError(
                "boto3 not installed. Install with: pip install boto3>=1.28.0"
            )

        self.config = config
        self.max_retries = 3
        self.retry_delay = 1  # seconds

        # Model and region configuration
        self.model_id = config.get("model", "us.anthropic.claude-opus-4-6-v1")
        self.region = config.get("region", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 2000)

        # Set bearer token from environment variable if provided
        bearer_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
        if bearer_token:
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = bearer_token

        # Initialize boto3 Bedrock client
        try:
            self.client = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.region,
            )
            logger.info(
                f"AWS Bedrock LLM client initialized with model: {self.model_id}, "
                f"region: {self.region}"
            )
        except Exception as e:
            raise LLMConnectionError(f"Failed to initialize AWS Bedrock client: {str(e)}")

    def test_connection(self) -> bool:
        """
        Test the connection to the AWS Bedrock service.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            logger.info("Testing AWS Bedrock connection...")
            test_prompt = "Hello, this is a connection test. Please respond with 'OK'."
            response = self.invoke(test_prompt)
            logger.info(f"Connection test successful. Response: {response[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False

    def _invoke_with_retry(self, messages: list, retry_count: int = 0) -> str:
        """
        Internal method to invoke the Bedrock converse API with retry logic.

        Args:
            messages: List of message dicts in Bedrock converse format
            retry_count: Current retry attempt number

        Returns:
            The LLM's response text

        Raises:
            LLMConnectionError: If all retry attempts fail
        """
        try:
            response = self.client.converse(
                modelId=self.model_id,
                messages=messages,
                inferenceConfig={
                    "temperature": self.temperature,
                    "maxTokens": self.max_tokens,
                },
            )
            return response["output"]["message"]["content"][0]["text"]
        except Exception as e:
            # Retry on throttling errors
            error_name = type(e).__name__
            is_throttling = (
                "ThrottlingException" in error_name
                or "TooManyRequestsException" in error_name
                or (hasattr(e, "response") and
                    e.response.get("Error", {}).get("Code") in
                    ("ThrottlingException", "TooManyRequestsException"))
            )

            if retry_count < self.max_retries and is_throttling:
                wait_time = self.retry_delay * (2 ** retry_count)  # Exponential backoff
                logger.warning(
                    f"Bedrock throttled (attempt {retry_count + 1}/{self.max_retries}), "
                    f"retrying in {wait_time}s: {str(e)}"
                )
                time.sleep(wait_time)
                return self._invoke_with_retry(messages, retry_count + 1)
            elif retry_count < self.max_retries:
                logger.warning(
                    f"Bedrock invocation failed (attempt {retry_count + 1}/{self.max_retries}): {str(e)}"
                )
                time.sleep(self.retry_delay * (retry_count + 1))
                return self._invoke_with_retry(messages, retry_count + 1)
            else:
                error_msg = f"Bedrock invocation failed after {self.max_retries} attempts: {str(e)}"
                logger.error(error_msg)
                raise LLMConnectionError(error_msg)

    def invoke(self, prompt: str) -> str:
        """
        Send a prompt to AWS Bedrock and get a text response.

        Args:
            prompt: The text prompt to send to the LLM

        Returns:
            The LLM's text response

        Raises:
            LLMConnectionError: If the LLM call fails after all retries
        """
        logger.debug(f"Invoking Bedrock model with prompt (length: {len(prompt)} chars)")
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        response = self._invoke_with_retry(messages)
        logger.debug(f"Received response (length: {len(response)} chars)")
        return response

    def invoke_with_schema(self, prompt: str, output_schema: dict) -> dict:
        """
        Send a prompt to AWS Bedrock and get a structured response matching the schema.

        Args:
            prompt: The text prompt to send to the LLM
            output_schema: JSON schema defining the expected output structure

        Returns:
            Dictionary matching the provided schema

        Raises:
            LLMConnectionError: If the LLM call fails or output doesn't match schema
        """
        logger.debug(f"Invoking Bedrock model with schema (prompt length: {len(prompt)} chars)")

        try:
            # Add schema instructions to the prompt
            schema_instruction = (
                f"\n\nPlease provide your response as a JSON object matching this schema:\n"
                f"{yaml.dump(output_schema, default_flow_style=False)}"
            )
            schema_instruction += (
                "\n\nIMPORTANT: Return ONLY the JSON object, without any additional text "
                "or markdown formatting."
            )

            full_prompt = prompt + schema_instruction

            # Invoke the model
            response_text = self.invoke(full_prompt)

            # Parse the response — remove markdown code blocks if present
            cleaned_response = response_text.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()

            # Try to find JSON boundaries if the model added extra text
            start_idx = cleaned_response.find('{')
            end_idx = cleaned_response.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                cleaned_response = cleaned_response[start_idx:end_idx + 1]

            try:
                result = json.loads(cleaned_response)
                logger.debug("Successfully parsed structured response")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Bedrock response as JSON: {str(e)}")
                logger.error(f"Response text: {cleaned_response[:200]}...")
                raise LLMConnectionError(f"Failed to parse Bedrock response as JSON: {str(e)}")

        except LLMConnectionError:
            raise
        except Exception as e:
            error_msg = f"Failed to invoke Bedrock model with schema: {str(e)}"
            logger.error(error_msg)
            raise LLMConnectionError(error_msg)


class LLMClientFactory:
    """
    Factory class for creating LLM clients.

    This factory loads configuration from llm_config.yaml and creates
    the appropriate LLM client based on the provider.
    """

    _config = None
    _config_path = None

    @classmethod
    def _load_config(cls) -> Dict[str, Any]:
        """
        Load the LLM configuration from llm_config.yaml.

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        if cls._config is not None:
            return cls._config

        # Find config file
        config_path = Path(__file__).parent.parent.parent / "configs" / "llm_config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"LLM config file not found: {config_path}")

        # Load YAML
        try:
            with open(config_path, 'r') as f:
                cls._config = yaml.safe_load(f)
            cls._config_path = config_path
            logger.info(f"Loaded LLM configuration from: {config_path}")
            return cls._config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse LLM config file: {str(e)}")

    @classmethod
    def get_client(cls, provider: Optional[str] = None) -> BaseLLMClient:
        """
        Create and return an LLM client for the specified provider.

        Args:
            provider: The LLM provider to use ('azure', 'huggingface', etc.).
                     If None, uses the default_provider from config.

        Returns:
            An instance of BaseLLMClient for the specified provider

        Raises:
            ValueError: If provider is not supported
            LLMConnectionError: If client initialization fails
        """
        # Load configuration
        config = cls._load_config()

        # Determine provider
        if provider is None:
            provider = config.get("default_provider", "azure")
            logger.info(f"Using default provider: {provider}")
        else:
            logger.info(f"Using specified provider: {provider}")

        # Create client based on provider
        if provider == "azure":
            azure_config = config.get("azure_openai")
            if not azure_config:
                raise ValueError("Azure OpenAI configuration not found in llm_config.yaml")
            return AzureLLMClient(azure_config)

        elif provider == "huggingface":
            hf_config = config.get("huggingface")
            if not hf_config:
                raise ValueError("HuggingFace configuration not found in llm_config.yaml")
            return HuggingFaceLLMClient(hf_config)

        elif provider == "bedrock":
            bedrock_config = config.get("bedrock")
            if not bedrock_config:
                raise ValueError("Bedrock configuration not found in llm_config.yaml")
            return BedrockLLMClient(bedrock_config)

        else:
            raise ValueError(
                f"Unsupported LLM provider: {provider}. "
                "Supported providers: 'azure', 'huggingface', 'bedrock'"
            )


# Convenience function for quick client access
def get_llm_client(provider: Optional[str] = None) -> BaseLLMClient:
    """
    Convenience function to get an LLM client.

    Args:
        provider: The LLM provider to use. If None, uses default from config.

    Returns:
        An instance of BaseLLMClient
    """
    return LLMClientFactory.get_client(provider)


if __name__ == "__main__":
    """
    Example usage and testing of the LLM client.
    """
    import sys

    # Determine which provider to use (default or from command line)
    provider = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        print(f"Creating LLM client (provider: {provider or 'default'})...")
        client = get_llm_client(provider=provider)

        print("\nTesting connection...")
        if client.test_connection():
            print("Connection test passed!")
        else:
            print("Connection test failed!")
            sys.exit(1)

        # Example 1: Simple text invocation
        print("\n--- Example 1: Simple Text Invocation ---")
        prompt = "What are the three primary colors? Answer in one sentence."
        response = client.invoke(prompt)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")

        # Example 2: Structured output with schema
        print("\n--- Example 2: Structured Output with Schema ---")
        schema = {
            "type": "object",
            "properties": {
                "colors": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "count": {"type": "integer"}
            },
            "required": ["colors", "count"]
        }

        prompt = "List the three primary colors."
        structured_response = client.invoke_with_schema(prompt, schema)
        print(f"Prompt: {prompt}")
        print(f"Structured Response: {structured_response}")

        print("\nAll tests completed successfully!")

    except LLMConnectionError as e:
        print(f"LLM Connection Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
