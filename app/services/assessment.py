"""
Adaptive Assessment Engine with Dynamic Difficulty Scaling (`assessment.py`).

Analogy for Beginners:
Think of this adaptive assessment engine like a smart video game that adjusts difficulty in real-time!
- If you answer a question with deep architectural mastery (Correct/Deep), you rank up (+1 Level, up to Level 5).
- If your answer is missing a small detail (Partial), you stay at your current level while the interviewer probes the gap.
- If your answer demonstrates fundamental misunderstandings (Weak/Incorrect), you drop down (-1 Level, down to Level 1) to build back up.
Every question is dynamically customized using the candidate's actual resume projects!
"""

from typing import Dict, Any, List, Tuple
from app.models.schemas import (
    CandidateProfile, AssessmentTurnInput, AssessmentTurnOutput, DimensionEvaluation
)


# Difficulty Level Taxonomy Descriptions
LEVEL_TAXONOMY = {
    1: "Level 1 (Core Fundamentals): Syntax, basic data structures, and foundational definitions.",
    2: "Level 2 (Applied Implementation): Writing clean async loops, error handling, and framework usage.",
    3: "Level 3 (Intermediate Optimization): Database indexing, query optimization, and edge case handling.",
    4: "Level 4 (Advanced Systems): Async queues, distributed caching, memory management, and microservice scaling.",
    5: "Level 5 (Principal System Architecture): High-availability trade-off design, pgvector HNSW parameter tuning, and fault-tolerant event streams."
}


class AssessmentService:
    """
    Adaptive Technical Assessment State Machine Service.
    """

    def evaluate_candidate_answer(
        self,
        candidate_answer: str,
        current_level: int
    ) -> DimensionEvaluation:
        """
        Evaluates candidate response across 4 core dimensions and determines state transition classification.
        
        Evaluated Dimensions:
        1. Technical Fundamentals (0-100)
        2. Role Depth (0-100)
        3. Problem Solving (0-100)
        4. Communication Clarity (0-100)
        """
        answer_length = len(candidate_answer.strip())
        keywords_deep = ["rag", "pgvector", "async", "cache", "redis", "indexing", "trade-off", "architecture", "latency", "scale", "hnsw", "concurrency"]
        keywords_basic = ["python", "code", "use", "make", "function", "data", "database", "api"]

        deep_count = sum(1 for kw in keywords_deep if kw in candidate_answer.lower())
        basic_count = sum(1 for kw in keywords_basic if kw in candidate_answer.lower())

        # Classification Heuristics
        if deep_count >= 2 and answer_length >= 50:
            classification = "Correct / Deep"
            tech_score = 90.0 + min(deep_count * 2, 10)
            role_score = 88.0 + min(deep_count * 2, 10)
            problem_score = 85.0 + min(deep_count * 2, 10)
            comm_score = 90.0
        elif basic_count >= 1 or answer_length >= 20:
            classification = "Partial"
            tech_score = 70.0
            role_score = 65.0
            problem_score = 68.0
            comm_score = 75.0
        else:
            classification = "Weak / Incorrect"
            tech_score = 45.0
            role_score = 40.0
            problem_score = 42.0
            comm_score = 50.0

        return DimensionEvaluation(
            technical_fundamentals=round(tech_score, 1),
            role_depth=round(role_score, 1),
            problem_solving=round(problem_score, 1),
            communication_clarity=round(comm_score, 1),
            turn_classification=classification
        )

    def calculate_level_transition(self, current_level: int, classification: str) -> Tuple[int, str]:
        """
        Calculates next difficulty level based on answer classification.
        
        Rules:
        - "Correct / Deep": Level up (+1, Max 5)
        - "Partial": Retain level
        - "Weak / Incorrect": Level down (-1, Min 1)
        """
        if classification == "Correct / Deep":
            new_level = min(current_level + 1, 5)
            level_change = "Increased (+1)" if new_level > current_level else "Retained (Max Level 5)"
        elif classification == "Partial":
            new_level = current_level
            level_change = "Retained"
        else:
            new_level = max(current_level - 1, 1)
            level_change = "Decreased (-1)" if new_level < current_level else "Retained (Min Level 1)"

        return new_level, level_change

    def generate_grounded_question(
        self,
        target_role: str,
        level: int,
        profile: CandidateProfile
    ) -> str:
        """
        Generates next assessment question dynamically grounded on candidate's parsed projects and tech stack.
        """
        top_project = profile.projects[0].title if profile.projects else "RAG Search Pipeline"
        tech = profile.skills.technical_skills[0] if profile.skills.technical_skills else "FastAPI"

        question_bank = {
            1: f"In your experience with {tech}, how do you manage synchronous vs. asynchronous function execution to avoid blocking event loops?",
            2: f"Building on your '{top_project}' project, how did you structure exception handling and request validation using Pydantic schemas?",
            3: f"When querying vector embeddings in PostgreSQL with pgvector for your '{top_project}', what index parameters (e.g., HNSW m & ef_construction) would you tune to optimize search recall vs. speed?",
            4: f"For high-throughput {target_role} microservices, how would you design a multi-layer caching strategy with Redis to mitigate cache stampedes under heavy load?",
            5: f"Principal Architecture: Design a fault-tolerant, horizontally scalable vector search and streaming pipeline handling 50,000 requests per second with sub-50ms SLA."
        }

        return question_bank.get(level, question_bank[1])

    def process_assessment_turn(
        self,
        session_history: List[Dict[str, Any]],
        current_level: int,
        target_role: str,
        candidate_answer: str,
        profile: CandidateProfile
    ) -> Tuple[AssessmentTurnOutput, int, List[Dict[str, Any]]]:
        """
        Executes one full adaptive interview turn.
        
        Steps:
        1. Evaluate candidate answer.
        2. Calculate level transition.
        3. Generate grounded next question.
        4. Append to session turn history.
        """
        # Step 1: Evaluate
        evaluation = self.evaluate_candidate_answer(candidate_answer, current_level)

        # Step 2: Scale Difficulty Level
        new_level, level_change = self.calculate_level_transition(current_level, evaluation.turn_classification)

        # Step 3: Actionable Feedback
        if evaluation.turn_classification == "Correct / Deep":
            feedback = f"Excellent response! Demonstrated strong mastery of {target_role} concepts. Escalating to Level {new_level}."
        elif evaluation.turn_classification == "Partial":
            feedback = f"Good start. You covered key basics, but let's probe deeper into the underlying system trade-offs."
        else:
            feedback = f"Noticeable gaps in fundamentals. Scaling back to Level {new_level} to consolidate core mechanics."

        # Step 4: Generate Next Grounded Question
        next_question = self.generate_grounded_question(target_role, new_level, profile)

        # Determine if session should end (e.g., after 3-5 turns)
        is_completed = len(session_history) >= 4

        # Record Turn
        turn_data = {
            "turn_index": len(session_history) + 1,
            "level": current_level,
            "answer": candidate_answer,
            "evaluation": evaluation.model_dump(),
            "level_change": level_change,
            "next_level": new_level
        }
        updated_history = list(session_history) + [turn_data]

        output = AssessmentTurnOutput(
            session_id="session_123",
            current_level=new_level,
            level_change=level_change,
            previous_question=session_history[-1].get("question", "Initial Question") if session_history else "Tell me about your technical background.",
            evaluation=evaluation,
            feedback=feedback,
            next_question=next_question,
            is_completed=is_completed
        )

        return output, new_level, updated_history


# Singleton instance export
assessment_service = AssessmentService()
