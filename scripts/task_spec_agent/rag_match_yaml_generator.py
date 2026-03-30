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

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

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

                # Extract object names for richer search text
                object_names = [
                    a.get("name", "") for a in data.get("assets", [])
                    if a.get("type") in ("rigid",) and a.get("name")
                ]
                objects_str = ", ".join(object_names) if object_names else ""

                # Extract goal description
                goal_desc = data.get("goal", {}).get("description", "")

                # Create searchable text with strong keyword signals
                search_text = (
                    f"{task_type} {task_type} {task_type}\n"  # boost task type
                    f"Task: {task_name}\n"
                    f"Type: {task_type}\n"
                    f"Description: {description}\n"
                    f"Goal: {goal_desc}\n"
                    f"Objects: {objects_str}\n"
                    f"Robot: {robot}"
                )

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
            k=max(k * 5, 10),  # Get more candidates for robot filtering
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

    def generate(self, task_description: str, parsed_task=None, task_plan=None,
                 validation_result=None) -> str:
        """
        Find the most similar existing YAML task specification.

        Uses vector search to find the closest matching task YAML from
        the tasks/ directory and returns it directly (no LLM generation).

        Args:
            task_description: Natural language description of the task
            parsed_task: Optional ParsedTask (used to enrich search query)
            task_plan: Optional TaskPlan (used to enrich search query)
            validation_result: Optional ValidationResult (unused)

        Returns:
            Content of the most similar existing YAML file
        """
        # Build search query as natural language sentence
        # Structured formats ("Objects: x | Actions: y") match poorly against
        # the DB's natural-language search text, so we reconstruct a sentence.
        query = None
        if task_plan and hasattr(task_plan, 'description') and task_plan.description:
            # task_plan.description is already an English sentence from the LLM
            query = task_plan.description
        elif parsed_task:
            # Build a natural sentence from parsed components
            parts = []
            actions = getattr(parsed_task, 'actions', []) or []
            objects = getattr(parsed_task, 'objects', []) or []
            if actions:
                parts.append(" ".join(actions))
            if objects:
                parts.append("the " + " and ".join(objects))
            query = " ".join(parts) if parts else None

        # Fall back to raw task_description (works fine for English input)
        if not query:
            query = task_description

        # Retrieve the single most similar example
        examples = self.retrieve_similar_examples(query, k=1)

        if not examples:
            raise ValueError("No similar task YAML found in vector store")

        best_match = examples[0]
        logger.info(f"Best match: {best_match['task_name']} (from {best_match['path']})")

        return best_match['yaml_content']

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
