
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .examples import AGENT_EXAMPLES

class SemanticRouter:
    """
    A fast, deterministic router that classifies queries by embedding similarity.
    Uses the same embedding model as the RAG pipeline to avoid extra dependencies.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print("🔄 Loading embedding model for router...")
        self.model = SentenceTransformer(model_name)

        # Build a flat list of (query, agent) pairs and embed them all at once
        self.routes = []
        self.agent_embeddings = {}

        for agent_name, examples in AGENT_EXAMPLES.items():
            embeddings = self.model.encode(examples, normalize_embeddings=True)
            self.agent_embeddings[agent_name] = embeddings
            self.routes.append(agent_name)

        print(f"✅ Router ready with {len(self.routes)} agents")

    def classify(self, query: str) -> tuple[str, float]:
        """
        Classify a query into an agent name.

        Returns:
            (agent_name, confidence_score) where confidence is the max cosine similarity.
        """
        query_embedding = self.model.encode([query], normalize_embeddings=True)

        best_agent = None
        best_score = -1.0

        for agent_name, embeddings in self.agent_embeddings.items():
            # Compare query to every example for this agent
            similarities = cosine_similarity(query_embedding, embeddings)[0]
            max_sim = float(np.max(similarities))

            if max_sim > best_score:
                best_score = max_sim
                best_agent = agent_name

        return best_agent, best_score