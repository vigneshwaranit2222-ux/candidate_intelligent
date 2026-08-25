"""
Benchmark Calibration Engine & Diagnostic Report Builder (`scoring.py`).

Analogy for Beginners:
Think of benchmark scoring like a report card in school!
Instead of giving a single pass/fail mark, we evaluate 4 distinct subjects:
1. Technical Fundamentals (30% weight) - Do you know core syntax & data structures?
2. Role Depth (30% weight) - Can you build real-world frameworks & RAG pipelines?
3. Problem Solving (25% weight) - How do you handle complex bugs and system trade-offs?
4. Communication Clarity (15% weight) - Can you explain your code simply to others?
Combining all 4 calculates your weighted GPA score, which places you into a calibrated readiness tier!
"""

from typing import Dict, Any, List
from app.models.schemas import DiagnosticReportSchema, CalibratedTierEnum, CandidateProfile


class ScoringService:
    """
    Diagnostic Benchmark & Readiness Calibration Service.
    """

    def calculate_overall_benchmark(
        self,
        fundamentals: float,
        depth: float,
        problem_solving: float,
        communication: float
    ) -> float:
        """
        Calculates weighted overall benchmark score.
        
        Weights:
        - Technical Fundamentals: 30%
        - Role Depth: 30%
        - Problem Solving: 25%
        - Communication Clarity: 15%
        Total: 100%
        """
        overall = (
            (0.30 * fundamentals) +
            (0.30 * depth) +
            (0.25 * problem_solving) +
            (0.15 * communication)
        )
        return round(overall, 2)

    def determine_calibrated_tier(self, overall_score: float) -> CalibratedTierEnum:
        """
        Maps overall benchmark score to calibrated readiness tier.
        
        Tier Boundaries:
        - Intermediate Potential: Score >= 88.0
        - Strong Entry-Level: 75.0 <= Score < 88.0
        - Entry-Level Ready: 60.0 <= Score < 75.0
        - Foundation: Score < 60.0
        """
        if overall_score >= 88.0:
            return CalibratedTierEnum.INTERMEDIATE_POTENTIAL
        elif overall_score >= 75.0:
            return CalibratedTierEnum.STRONG_ENTRY_LEVEL
        elif overall_score >= 60.0:
            return CalibratedTierEnum.ENTRY_LEVEL_READY
        else:
            return CalibratedTierEnum.FOUNDATION

    def generate_upskilling_roadmap(
        self,
        tier: CalibratedTierEnum,
        fundamentals: float,
        depth: float,
        problem_solving: float,
        profile: CandidateProfile
    ) -> List[Dict[str, Any]]:
        """
        Constructs an actionable Phase 2 upskilling roadmap based on dimension gaps.
        """
        roadmap = []
        phase_count = 1

        if fundamentals < 75.0:
            roadmap.append({
                "phase": f"Phase 2.{phase_count}: Core Fundamentals Upskilling",
                "focus_area": "Data Structures, Asynchronous Python Loops & Memory Management",
                "recommended_action": "Complete hands-on async IO challenges in Python; implement custom asyncio queues.",
                "target_timeline": "2 Weeks"
            })
            phase_count += 1

        if depth < 80.0:
            roadmap.append({
                "phase": f"Phase 2.{phase_count}: Role Depth & System Engineering",
                "focus_area": "pgvector Index Tuning, HNSW Parameters & Redis Caching",
                "recommended_action": "Build a production RAG pipeline tuning HNSW m=16, ef_construction=64 parameters in PostgreSQL.",
                "target_timeline": "3 Weeks"
            })
            phase_count += 1

        if problem_solving < 80.0:
            roadmap.append({
                "phase": f"Phase 2.{phase_count}: Algorithmic Reasoning & Trade-off Analysis",
                "focus_area": "Distributed Systems, Fault Tolerance & SLA Latency Optimization",
                "recommended_action": "Practice system design trade-offs focusing on CAP theorem, database sharding, and rate limiting.",
                "target_timeline": "2 Weeks"
            })
            phase_count += 1

        # Always include a career polishing phase
        roadmap.append({
            "phase": f"Phase 2.{phase_count}: Portfolio & Assessment Readiness",
            "focus_area": "Live Mock Technical Interviews & Open-Source Portfolio Defense",
            "recommended_action": f"Prepare 5-minute technical deep dives defending architectural choices made in '{profile.projects[0].title if profile.projects else 'your key projects'}'.",
            "target_timeline": "1 Week"
        })

        return roadmap

    def generate_diagnostic_report(
        self,
        candidate_id: str,
        candidate_name: str,
        history: List[Dict[str, Any]],
        profile: CandidateProfile
    ) -> DiagnosticReportSchema:
        """
        Generates full benchmark diagnostic report from candidate assessment history.
        """
        if history:
            # Average score across recorded turns
            fundamentals = sum(h.get("evaluation", {}).get("technical_fundamentals", 75.0) for h in history) / len(history)
            depth = sum(h.get("evaluation", {}).get("role_depth", 70.0) for h in history) / len(history)
            problem = sum(h.get("evaluation", {}).get("problem_solving", 72.0) for h in history) / len(history)
            comm = sum(h.get("evaluation", {}).get("communication_clarity", 80.0) for h in history) / len(history)
        else:
            # Default strong scores if generating report directly
            fundamentals = 85.0
            depth = 82.0
            problem = 80.0
            comm = 88.0

        overall_score = self.calculate_overall_benchmark(fundamentals, depth, problem, comm)
        tier = self.determine_calibrated_tier(overall_score)
        roadmap = self.generate_upskilling_roadmap(tier, fundamentals, depth, problem, profile)

        return DiagnosticReportSchema(
            candidate_id=candidate_id,
            candidate_name=candidate_name,
            technical_fundamentals=round(fundamentals, 1),
            role_depth=round(depth, 1),
            problem_solving=round(problem, 1),
            communication_clarity=round(comm, 1),
            overall_score=overall_score,
            calibrated_tier=tier,
            roadmap=roadmap
        )


# Singleton instance export
scoring_service = ScoringService()
