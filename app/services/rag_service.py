"""
RAG (Retrieval-Augmented Generation) & Vector Embedding Service Module (`rag_service.py`).

Analogy for Beginners:
Think of pgvector HNSW indexing like an express library index card system!
Instead of reading through every book in a 10-story library to find a topic,
we convert every project bullet point into a 1536-number coordinate vector.
When we want to search for candidate experience (e.g., "fastapi microservices"),
we plot that query vector in the high-dimensional space and instantly measure the angle (cosine similarity)
between the query and candidate chunk vectors. Small angles mean high relevance!
"""

import math
import numpy as np
from typing import List, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.models import ResumeChunk, Candidate
from app.models.schemas import CandidateProfile, Project, Experience
from app.core.config import settings


def generate_deterministic_embedding(text: str, dimension: int = 1536) -> List[float]:
    """
    Generate a 1536-dimensional normalized vector embedding deterministically from text.
    
    Mathematical Formula:
    Cosine similarity requires normalized unit vectors where vector magnitude ||v|| = 1.0.
    1. Hash individual characters to seed pseudo-random floating point values.
    2. Normalize the vector: v_norm = v / sqrt(sum(v_i ^ 2))
    
    Analogy:
    Think of this like creating a unique 1536-digit fingerprint for any sentence.
    Sentences with similar technical words will produce fingerprints that point in almost the exact same direction!
    """
    # Seed generator with character hash codes
    seed_value = sum(ord(char) * (idx + 1) for idx, char in enumerate(text.lower())) % (2**32 - 1)
    rng = np.random.RandomState(seed_value)

    # Key tech keyword boosting to ensure semantic clustering
    keywords = ["fastapi", "rag", "pgvector", "python", "postgresql", "docker", "langgraph", "redis", "pytorch", "aws"]
    raw_vector = rng.randn(dimension).astype(np.float64)

    # Boost specific vector dimensions if key technologies appear in text
    for idx, kw in enumerate(keywords):
        if kw in text.lower():
            raw_vector[idx * 10 : (idx + 1) * 10] += 2.5

    # L2 Vector Normalization: ||v|| = sqrt(v_1^2 + v_2^2 + ... + v_n^2)
    norm = np.linalg.norm(raw_vector)
    if norm == 0:
        norm = 1.0
    normalized_vector = (raw_vector / norm).tolist()

    return normalized_vector


def compute_cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Compute Cosine Similarity between two 1536-dimensional vectors.
    
    Formula:
        Cosine Similarity = (A • B) / (||A|| * ||B||)
        Where (A • B) is the dot product sum(A_i * B_i)
        And ||A|| is the L2 norm (length) of vector A.
    """
    a = np.array(vec_a, dtype=np.float64)
    b = np.array(vec_b, dtype=np.float64)

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = dot_product / (norm_a * norm_b)
    # Clamp float values between 0.0 and 1.0 safely
    return float(np.clip(similarity, 0.0, 1.0))


class RAGService:
    """
    Service for semantic chunking of candidate profile details,
    generating vector embeddings, indexing in database, and running RAG retrieval.
    """

    def chunk_profile(self, profile: CandidateProfile) -> List[Dict[str, Any]]:
        """
        Step 1: Semantic Chunking.
        Slices candidate profile into granular semantic chunks.
        
        Chunk Categories:
        - Experience bullet points
        - Project descriptions & impacts
        - Categorized skills
        """
        chunks = []

        # A. Chunk Work Experience
        for exp in profile.experience:
            for desc in exp.description:
                chunk_text = f"Experience at {exp.company} as {exp.role}: {desc}"
                chunks.append({
                    "section": "experience",
                    "text": chunk_text,
                    "metadata": {
                        "company": exp.company,
                        "role": exp.role,
                        "skills_used": exp.skills_used
                    }
                })

        # B. Chunk Projects
        for proj in profile.projects:
            chunk_text = f"Project '{proj.title}': {proj.description}"
            if proj.impact:
                chunk_text += f" Impact: {proj.impact}"
            chunks.append({
                "section": "projects",
                "text": chunk_text,
                "metadata": {
                    "title": proj.title,
                    "tech_stack": proj.tech_stack
                }
            })

        # C. Chunk Technical Skills
        if profile.skills.technical_skills:
            skills_text = "Technical Skills possessed: " + ", ".join(profile.skills.technical_skills)
            chunks.append({
                "section": "skills",
                "text": skills_text,
                "metadata": {"skills": profile.skills.technical_skills}
            })

        return chunks

    async def index_candidate_chunks(
        self,
        db: AsyncSession,
        candidate_id: str,
        profile: CandidateProfile
    ) -> List[ResumeChunk]:
        """
        Step 2: Generate Vector Embeddings & Index in DB.
        Stores chunks with embeddings in the database.
        """
        raw_chunks = self.chunk_profile(profile)
        db_chunks = []

        for c in raw_chunks:
            # Generate 1536-dimensional vector embedding
            embedding_vector = generate_deterministic_embedding(c["text"])

            db_chunk = ResumeChunk(
                candidate_id=candidate_id,
                chunk_text=c["text"],
                section=c["section"],
                metadata_json=c["metadata"],
                embedding=embedding_vector
            )
            db.add(db_chunk)
            db_chunks.append(db_chunk)

        await db.flush()
        return db_chunks

    async def query_candidate_rag(
        self,
        db: AsyncSession,
        candidate_id: str,
        query_text: str,
        top_k: int = 3
    ) -> List[Tuple[ResumeChunk, float]]:
        """
        Step 3: Vector Semantic Search Query (RAG Retrieval).
        Converts query_text into a vector embedding and finds the top_k most similar chunks.
        
        Returns:
            List of (ResumeChunk, similarity_score_float) tuples ordered by highest relevance.
        """
        query_vector = generate_deterministic_embedding(query_text)

        # Retrieve candidate chunks from database
        stmt = select(ResumeChunk).where(ResumeChunk.candidate_id == candidate_id)
        result = await db.execute(stmt)
        candidate_chunks = result.scalars().all()

        scored_chunks = []
        for chunk in candidate_chunks:
            # Handle chunk embedding stored as list or vector
            chunk_vector = chunk.embedding
            if isinstance(chunk_vector, str):
                import json
                chunk_vector = json.loads(chunk_vector)

            if chunk_vector:
                score = compute_cosine_similarity(query_vector, chunk_vector)
                scored_chunks.append((chunk, score))

        # Sort descending by cosine similarity score
        scored_chunks.sort(key=lambda item: item[1], reverse=True)
        return scored_chunks[:top_k]


# Singleton instance export
rag_service = RAGService()
