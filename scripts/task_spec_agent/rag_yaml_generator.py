"""
RAG-based YAML Generator for Isaac Sim Task Specifications

Uses LangChain to:
1. Load existing task YAML examples into vector store
2. Retrieve similar examples based on user query
3. Generate new YAML using few-shot prompting with GPT
"""

import os
import yaml
import logging
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGYAMLGenerator:
    """RAG-based YAML generator using LangChain."""

    def __init__(self, tasks_dir: str = None, robot_type: str = "franka"):
        """
        Initialize the RAG YAML generator.

        Args:
            tasks_dir: Path to tasks directory containing example YAMLs
            robot_type: Target robot type (franka, ur10, openarm, so101)
        """
        self.robot_type = robot_type

        # Set paths
        project_root = Path(__file__).parent.parent.parent
        self.tasks_dir = Path(tasks_dir) if tasks_dir else project_root / "tasks"
        self.vector_store_path = project_root / "data" / "vector_store"

        # Load config
        config_path = project_root / "configs" / "llm_config.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Initialize LLM
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
            api_version=self.config["azure_openai"]["api_version"],
            temperature=1.0,
        )

        # Initialize embeddings (using local HuggingFace model)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Load or create vector store
        self.vector_store = self._load_or_create_vector_store()

        logger.info(f"RAGYAMLGenerator initialized for robot: {robot_type}")

    def _load_yaml_examples(self) -> List[Document]:
        """Load all YAML examples from tasks directory."""
        documents = []

        for yaml_path in self.tasks_dir.rglob("*.yaml"):
            try:
                with open(yaml_path) as f:
                    content = f.read()
                    data = yaml.safe_load(content)

                # Extract metadata for better retrieval
                task_name = data.get("task", {}).get("name", "")
                description = data.get("task", {}).get("description", "")
                robot = yaml_path.parts[-3] if len(yaml_path.parts) >= 3 else ""
                task_type = yaml_path.parts[-2] if len(yaml_path.parts) >= 2 else ""

                # Create searchable text
                search_text = f"Task: {task_name}\nDescription: {description}\nRobot: {robot}\nType: {task_type}"

                doc = Document(
                    page_content=search_text,
                    metadata={
                        "path": str(yaml_path),
                        "robot": robot,
                        "task_type": task_type,
                        "task_name": task_name,
                        "full_yaml": content,
                    }
                )
                documents.append(doc)

            except Exception as e:
                logger.warning(f"Failed to load {yaml_path}: {e}")

        logger.info(f"Loaded {len(documents)} YAML examples")
        return documents

    def _load_or_create_vector_store(self) -> FAISS:
        """Load existing vector store or create new one."""
        if self.vector_store_path.exists():
            try:
                logger.info("Loading existing vector store...")
                return FAISS.load_local(
                    str(self.vector_store_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                logger.warning(f"Failed to load vector store: {e}")

        # Create new vector store
        logger.info("Creating new vector store...")
        documents = self._load_yaml_examples()
        vector_store = FAISS.from_documents(documents, self.embeddings)

        # Save for future use
        self.vector_store_path.parent.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(self.vector_store_path))

        return vector_store

    def retrieve_similar_examples(self, query: str, k: int = 3) -> List[dict]:
        """
        Retrieve similar YAML examples based on query.

        Args:
            query: Natural language task description
            k: Number of examples to retrieve

        Returns:
            List of similar examples with metadata
        """
        # Filter by robot type if specified
        results = self.vector_store.similarity_search(
            query,
            k=k * 2,  # Get more, then filter
        )

        # Filter by robot type
        filtered = []
        for doc in results:
            if self.robot_type in doc.metadata.get("robot", "").lower():
                filtered.append({
                    "task_name": doc.metadata.get("task_name"),
                    "task_type": doc.metadata.get("task_type"),
                    "yaml_content": doc.metadata.get("full_yaml"),
                    "path": doc.metadata.get("path"),
                })
                if len(filtered) >= k:
                    break

        # If not enough, add from other robots
        if len(filtered) < k:
            for doc in results:
                if doc.metadata.get("task_name") not in [f["task_name"] for f in filtered]:
                    filtered.append({
                        "task_name": doc.metadata.get("task_name"),
                        "task_type": doc.metadata.get("task_type"),
                        "yaml_content": doc.metadata.get("full_yaml"),
                        "path": doc.metadata.get("path"),
                    })
                    if len(filtered) >= k:
                        break

        logger.info(f"Retrieved {len(filtered)} similar examples for: {query[:50]}...")
        return filtered

    def generate(self, task_description: str) -> str:
        """
        Generate YAML task specification from natural language description.

        Args:
            task_description: Natural language description of the task

        Returns:
            Generated YAML string
        """
        # Retrieve similar examples
        examples = self.retrieve_similar_examples(task_description, k=2)

        # Build few-shot prompt
        examples_text = ""
        for i, ex in enumerate(examples, 1):
            examples_text += f"\n--- Example {i}: {ex['task_name']} ---\n"
            examples_text += ex['yaml_content']
            examples_text += "\n"

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Isaac Sim task specification generator.
Generate a valid YAML task specification based on the user's description.

Follow these rules:
1. Use the exact same YAML structure as the examples
2. Use appropriate asset paths from the examples
3. Set realistic physics parameters
4. Define clear success criteria in the goal section
5. Use the target robot type: {robot_type}

Here are similar task examples for reference:
{examples}
"""),
            ("user", """Generate a YAML task specification for:
{task_description}

Output ONLY the YAML content, no explanations or markdown."""),
        ])

        chain = prompt | self.llm | StrOutputParser()

        result = chain.invoke({
            "robot_type": self.robot_type,
            "examples": examples_text,
            "task_description": task_description,
        })

        # Clean up result
        result = result.strip()
        if result.startswith("```yaml"):
            result = result[7:]
        if result.startswith("```"):
            result = result[3:]
        if result.endswith("```"):
            result = result[:-3]

        logger.info("YAML generation complete")
        return result.strip()

    def save(self, yaml_content: str, output_path: str) -> None:
        """Save generated YAML to file."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w') as f:
            f.write(yaml_content)
        logger.info(f"Saved to: {output}")


def main():
    """Test the RAG YAML generator."""
    import sys

    # Test query
    query = sys.argv[1] if len(sys.argv) > 1 else "Pick up the red cube and stack it on top of the blue cube"
    robot = sys.argv[2] if len(sys.argv) > 2 else "franka"

    print(f"\nQuery: {query}")
    print(f"Robot: {robot}")
    print("=" * 60)

    # Generate
    generator = RAGYAMLGenerator(robot_type=robot)
    yaml_content = generator.generate(query)

    print("\nGenerated YAML:")
    print("-" * 60)
    print(yaml_content)

    # Save
    output_path = Path(__file__).parent.parent.parent / "outputs" / "generated_task.yaml"
    generator.save(yaml_content, str(output_path))
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
