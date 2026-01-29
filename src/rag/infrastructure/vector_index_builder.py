"""
Vector Index Builder for Dense Retrieval.

Builds and manages vector indices for Korean-optimized dense retrieval.
Supports multiple embedding models and efficient indexing strategies.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .dense_retriever import DenseRetriever, DenseRetrieverConfig

logger = logging.getLogger(__name__)


class VectorIndexBuilder:
    """
    Builder for creating and managing vector indices.

    Features:
    - Automatic model download from HuggingFace
    - Batch processing for large document collections
    - Index persistence with pickle serialization
    - Progress tracking and logging
    - Support for multiple embedding models
    """

    def __init__(
        self,
        model_name: str = "jhgan/ko-sbert-multinli",
        index_dir: Optional[str] = None,
        batch_size: int = 64,
    ):
        """
        Initialize Vector Index Builder.

        Args:
            model_name: HuggingFace model name for embeddings.
            index_dir: Directory to store vector indices.
            batch_size: Batch size for embedding generation.
        """
        self.model_name = model_name
        self.index_dir = Path(index_dir or "data/vector_indices")
        self.batch_size = batch_size

        # Create index directory
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Initialize retriever
        config = DenseRetrieverConfig(batch_size=batch_size)
        self.retriever = DenseRetriever(model_name=model_name, config=config)

    def build_index_from_json(
        self,
        json_path: str,
        content_field: str = "content",
        id_field: str = "id",
        metadata_fields: Optional[List[str]] = None,
    ) -> str:
        """
        Build vector index from JSON file.

        Args:
            json_path: Path to JSON file containing documents.
            content_field: Field name containing document text.
            id_field: Field name containing document ID.
            metadata_fields: List of fields to include in metadata.

        Returns:
            Path to saved index file.
        """
        import json

        logger.info(f"Building index from JSON: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract documents
        documents = []
        for item in data:
            doc_id = item.get(id_field, str(len(documents)))
            content = item.get(content_field, "")

            # Extract metadata
            metadata = {}
            if metadata_fields:
                for field in metadata_fields:
                    if field in item:
                        metadata[field] = item[field]
            else:
                # Include all fields except content and id
                metadata = {
                    k: v for k, v in item.items() if k not in [content_field, id_field]
                }

            documents.append((doc_id, content, metadata))

        return self.build_index(documents)

    def build_index(
        self,
        documents: List[Tuple[str, str, Dict]],
        index_name: Optional[str] = None,
    ) -> str:
        """
        Build vector index from document list.

        Args:
            documents: List of (doc_id, content, metadata) tuples.
            index_name: Optional name for the index file.

        Returns:
            Path to saved index file.
        """
        logger.info(f"Building vector index for {len(documents)} documents")

        # Clear existing index
        self.retriever.clear()

        # Add documents
        self.retriever.add_documents(documents)

        # Generate index name
        if index_name is None:
            model_short = self.model_name.split("/")[-1]
            index_name = f"{model_short}_index"

        # Save index
        index_path = self.index_dir / f"{index_name}.pkl"
        self.retriever.save_index(str(index_path))

        # Print cache stats
        cache_stats = self.retriever.get_cache_stats()
        logger.info(f"Cache stats: {cache_stats}")

        logger.info(f"Index saved to: {index_path}")
        return str(index_path)

    def load_index(self, index_path: str) -> bool:
        """
        Load existing vector index.

        Args:
            index_path: Path to index file.

        Returns:
            True if loaded successfully.
        """
        return self.retriever.load_index(index_path)

    def search(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[str, float, str, Dict]]:
        """
        Search using the loaded index.

        Args:
            query: Search query text.
            top_k: Maximum number of results.
            score_threshold: Optional minimum similarity score.

        Returns:
            List of (doc_id, score, content, metadata) tuples.
        """
        return self.retriever.search(
            query, top_k=top_k, score_threshold=score_threshold
        )


def build_all_indices(
    data_dir: str = "data/processed",
    index_dir: str = "data/vector_indices",
    models: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Build vector indices for all available JSON files.

    Args:
        data_dir: Directory containing processed JSON files.
        index_dir: Directory to save vector indices.
        models: List of model names to use. Defaults to Korean-optimized models.

    Returns:
        Dictionary mapping model names to index paths.
    """
    data_path = Path(data_dir)
    index_path = Path(index_dir)

    if not data_path.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return {}

    # Default to Korean-optimized models
    if models is None:
        models = ["jhgan/ko-sbert-multinli", "BAAI/bge-m3", "jhgan/ko-sbert-sts"]

    # Find all JSON files
    json_files = list(data_path.glob("*.json"))

    if not json_files:
        logger.warning(f"No JSON files found in {data_dir}")
        return {}

    logger.info(f"Found {len(json_files)} JSON files")

    # Build indices for each model
    index_paths = {}
    for model_name in models:
        logger.info(f"Building index with model: {model_name}")

        try:
            builder = VectorIndexBuilder(
                model_name=model_name,
                index_dir=str(index_path),
                batch_size=64,
            )

            # Build index from each JSON file
            for json_file in json_files:
                logger.info(f"Processing {json_file.name}")

                try:
                    index_path_str = builder.build_index_from_json(
                        str(json_file),
                        content_field="content",
                        id_field="id",
                    )

                    # Store index path
                    key = f"{model_name}_{json_file.stem}"
                    index_paths[key] = index_path_str

                except Exception as e:
                    logger.error(f"Failed to build index for {json_file.name}: {e}")

        except Exception as e:
            logger.error(f"Failed to initialize builder for {model_name}: {e}")

    logger.info(f"Built {len(index_paths)} vector indices")
    return index_paths


def download_model(model_name: str) -> bool:
    """
    Download embedding model from HuggingFace.

    Args:
        model_name: HuggingFace model name.

    Returns:
        True if downloaded successfully.
    """
    logger.info(f"Downloading model: {model_name}")

    try:
        from sentence_transformers import SentenceTransformer

        # Download model
        model = SentenceTransformer(model_name)
        dims = model.get_sentence_embedding_dimension()

        logger.info(f"Model downloaded successfully: {model_name} (dims={dims})")
        return True

    except Exception as e:
        logger.error(f"Failed to download model {model_name}: {e}")
        return False


def list_available_models() -> Dict[str, Dict]:
    """
    List all available Korean embedding models.

    Returns:
        Dictionary mapping model names to model info.
    """
    return DenseRetriever.list_models()


if __name__ == "__main__":
    # Example usage
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python vector_index_builder.py <command> [args]")
        print("Commands:")
        print("  build <json_path> [model_name]")
        print("  build-all [data_dir] [index_dir]")
        print("  download <model_name>")
        print("  list-models")
        sys.exit(1)

    command = sys.argv[1]

    if command == "build":
        json_path = (
            sys.argv[2] if len(sys.argv) > 2 else "data/processed/regulations.json"
        )
        model_name = sys.argv[3] if len(sys.argv) > 3 else "jhgan/ko-sbert-multinli"

        builder = VectorIndexBuilder(model_name=model_name)
        index_path = builder.build_index_from_json(json_path)
        print(f"Index built: {index_path}")

    elif command == "build-all":
        data_dir = sys.argv[2] if len(sys.argv) > 2 else "data/processed"
        index_dir = sys.argv[3] if len(sys.argv) > 3 else "data/vector_indices"

        index_paths = build_all_indices(data_dir, index_dir)
        print(f"Built {len(index_paths)} indices:")
        for key, path in index_paths.items():
            print(f"  {key}: {path}")

    elif command == "download":
        model_name = sys.argv[2] if len(sys.argv) > 2 else "jhgan/ko-sbert-multinli"
        success = download_model(model_name)
        if success:
            print(f"Model downloaded: {model_name}")
        else:
            print(f"Failed to download: {model_name}")
            sys.exit(1)

    elif command == "list-models":
        models = list_available_models()
        print("Available Korean embedding models:")
        for model_name, info in models.items():
            print(f"\n{model_name}:")
            for key, value in info.items():
                print(f"  {key}: {value}")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
