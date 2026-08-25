from typing import Dict, List
from app.models.schemas import CandidateProfile
from app.core.config import settings


class ReadinessBenchmarkService:
    def build(self, history: List[Dict], profile: CandidateProfile, role: str, verification_confidence: float) -> Dict:
        if history:
            avg = lambda key, fallback: sum(t.get("evaluation", {}).get(key, fallback) for t in history) / len(history)
            fundamentals, depth, problem, communication = avg("technical_fundamentals", 50), avg("role_depth", 50), avg("problem_solving", 50), avg("communication_clarity", 50)
        else:
            fundamentals = depth = problem = communication = 0.0
        project_understanding = min(100.0, 45 + 15 * len(profile.projects))
        overall = round(.25*fundamentals + .22*depth + .18*problem + .15*project_understanding + .10*communication + .10*verification_confidence, 2)
        tiers = settings.READINESS_TIERS
        tier = next((item["label"] for item in reversed(tiers) if overall >= item["minimum"]), tiers[0]["label"])
        dimensions = {"Technical Fundamentals": fundamentals, "Role Knowledge": depth, "Applied Problem Solving": problem, "Project Understanding": project_understanding, "Communication": communication}
        improvements = [{"competency": name, "current_score": round(score, 1), "target_score": 75, "priority": "High" if score < 60 else "Medium", "reason": f"Important for {role} readiness"} for name, score in sorted(dimensions.items(), key=lambda x: x[1])[:5] if score < 75]
        roadmap = [{"week": idx + 1, "focus": item["competency"], "topics": [item["competency"]], "practical_tasks": [f"Complete a hands-on {item['competency']} exercise"], "mini_project": f"Apply {item['competency']} in a {role} mini project", "mock_interview_topics": [item["competency"]]} for idx, item in enumerate(improvements[:4])]
        return {"overall_readiness": overall, "tier": tier, "role_readiness": {role: overall}, "confidence": round(verification_confidence / 100, 2), "dimensions": dimensions, "improvement_areas": improvements, "roadmap": roadmap}


readiness_benchmark_service = ReadinessBenchmarkService()
