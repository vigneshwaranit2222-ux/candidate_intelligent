"""
Intelligent Role Recommender Engine (`recommender.py`).

Analogy for Beginners:
Think of our Role Matcher like a smart career counselor sitting down with a candidate resume.
To figure out which tech role fits best, the counselor uses two scoring tools:
1. Vector Semantic Match (60% weight): Checks the candidate's overall experience story against role descriptions (like comparing book summaries).
2. Skill Check (40% weight): Counts how many required tools (like FastAPI, SQL, Docker) the candidate already knows vs. what they still need to learn.
Combining both creates a transparent, fair 100-point Hybrid Match Score!
"""

from typing import List, Dict, Any
from app.models.schemas import CandidateProfile, RoleRecommendation
from app.services.rag_service import generate_deterministic_embedding, compute_cosine_similarity


# Standard Target Role Taxonomies with Descriptions & Required Core Skill Sets
TARGET_ROLES_TAXONOMY = {
    "Backend Engineer": {
        "description": "Architecting high-throughput REST APIs, database schemas, microservices, async task queues, and server logic.",
        "required_skills": ["Python", "FastAPI", "PostgreSQL", "SQL", "Docker", "Redis", "REST APIs"]
    },
    "Data / AI Engineer": {
        "description": "Building RAG vector search pipelines, LLM agent orchestrations, data pipelines, PyTorch models, and pgvector storage.",
        "required_skills": ["Python", "RAG", "pgvector", "LangGraph", "Vector Search", "PyTorch", "LLM", "PostgreSQL"]
    },
    "Full Stack Engineer": {
        "description": "Building end-to-end web applications combining React/TypeScript frontends with FastAPI/Node backends and SQL databases.",
        "required_skills": ["Python", "JavaScript", "TypeScript", "FastAPI", "React", "HTML/CSS", "PostgreSQL"]
    },
    "DevOps Engineer": {
        "description": "Managing cloud infrastructure, Docker containers, CI/CD pipelines, Kubernetes, monitoring, and AWS deployments.",
        "required_skills": ["Docker", "AWS", "Git", "Kubernetes", "Linux", "CI/CD", "Python"]
    },
    "Frontend Engineer": {
        "description": "Creating responsive user interfaces, dynamic modern web apps, CSS design systems, and frontend state management.",
        "required_skills": ["JavaScript", "TypeScript", "React", "HTML/CSS", "UI/UX", "State Management"]
    }
}


class RecommenderService:
    """
    Hybrid Role Recommendation Engine combining:
    - 60% Vector Semantic Similarity
    - 40% Required Skill Match Percentage
    """

    def compute_role_match(self, profile: CandidateProfile, role_name: str, role_info: Dict[str, Any], completeness: float = 0.0) -> RoleRecommendation:
        """
        Calculates hybrid match score for a single target role.
        
        Formula:
            Hybrid Score = (0.60 * Vector Similarity Score) + (0.40 * Skill Match Score)
        """
        # 1. Prepare Candidate Text Representation
        tech_skills = profile.skills.technical_skills
        exp_texts = [f"{e.company} {e.role} " + " ".join(e.description) for e in profile.experience]
        proj_texts = [f"{p.title} {p.description} {p.impact}" for p in profile.projects]
        candidate_summary = f"Skills: {', '.join(tech_skills)}. Experience: {' '.join(exp_texts)}. Projects: {' '.join(proj_texts)}"

        # 2. Vector Semantic Similarity (60% Weight)
        candidate_vec = generate_deterministic_embedding(candidate_summary)
        role_vec = generate_deterministic_embedding(f"{role_name}: {role_info['description']} Skills: {', '.join(role_info['required_skills'])}")
        
        raw_cosine = compute_cosine_similarity(candidate_vec, role_vec)
        vector_similarity_score = round(raw_cosine * 100.0, 2)

        # 3. Direct Required Skill Match (40% Weight)
        required_skills = role_info["required_skills"]
        candidate_skills_upper = [s.strip().upper() for s in tech_skills]

        matched_skills = []
        missing_skills = []

        for skill in required_skills:
            if any(skill.upper() in cs for cs in candidate_skills_upper):
                matched_skills.append(skill)
            else:
                missing_skills.append(skill)

        skill_match_ratio = len(matched_skills) / len(required_skills) if required_skills else 1.0
        skill_match_score = round(skill_match_ratio * 100.0, 2)

        project_relevance_score = round(100 * sum(1 for p in profile.projects if any(s.lower() in (p.description + ' ' + ' '.join(p.tech_stack)).lower() for s in required_skills)) / max(1, len(profile.projects)), 2)
        experience_relevance_score = round(100 * sum(1 for e in profile.experience if any(s.lower() in ' '.join(e.description + e.skills_used).lower() for s in required_skills)) / max(1, len(profile.experience)), 2)
        # Configurable production default: semantic 35%, skills 30%, projects 20%, experience 10%, completeness 5%.
        hybrid_score = round(.35*vector_similarity_score + .30*skill_match_score + .20*project_relevance_score + .10*experience_relevance_score + .05*completeness, 2)

        # 5. Build Rationale Explanation
        rationale = (
            f"Strong match for {role_name}! Possesses {len(matched_skills)} of {len(required_skills)} required skills "
            f"({', '.join(matched_skills[:3])}). High semantic similarity ({vector_similarity_score}%) based on project experience."
        )

        return RoleRecommendation(
            role_name=role_name,
            hybrid_score=hybrid_score,
            vector_similarity_score=vector_similarity_score,
            skill_match_score=skill_match_score,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            match_rationale=rationale
            , project_relevance_score=project_relevance_score, experience_relevance_score=experience_relevance_score,
            profile_completeness_score=completeness, supporting_projects=[p.title for p in profile.projects if any(s.lower() in (p.description + ' '.join(p.tech_stack)).lower() for s in matched_skills)]
        )

    def get_top_recommendations(self, profile: CandidateProfile, top_k: int = 5, completeness: float = 0.0) -> List[RoleRecommendation]:
        """
        Evaluates profile against all target roles in taxonomy and returns top_k recommendations ordered by hybrid score.
        """
        recommendations = []

        for role_name, role_info in TARGET_ROLES_TAXONOMY.items():
            rec = self.compute_role_match(profile, role_name, role_info, completeness)
            recommendations.append(rec)

        # Sort descending by hybrid_score
        recommendations.sort(key=lambda r: r.hybrid_score, reverse=True)
        return recommendations[:top_k]


# Singleton instance export
recommender_service = RecommenderService()
