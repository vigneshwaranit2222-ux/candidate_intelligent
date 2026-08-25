from typing import Dict, List
from app.models.schemas import CandidateProfile
from app.services.recommender import TARGET_ROLES_TAXONOMY


class AssessmentBlueprintService:
    def generate(self, profile: CandidateProfile, target_role: str, verified_skills: List[str] = None) -> Dict:
        required = TARGET_ROLES_TAXONOMY.get(target_role, {}).get("required_skills", [])
        candidate_skills = profile.skills.technical_skills
        domains = list(dict.fromkeys((verified_skills or []) + [s for s in candidate_skills if s in required] + required))[:5]
        return {"target_role": target_role, "domains": domains or ["Problem Solving"], "difficulty": "dynamic",
                "competency_objectives": [{"competency": d, "role": target_role, "learning_objective": f"Demonstrate applied {d}"} for d in domains]}


assessment_blueprint_service = AssessmentBlueprintService()
