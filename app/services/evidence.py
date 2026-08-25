"""Evidence extraction: resume statements are claims, never automatic verification."""
from typing import List, Dict
from app.models.schemas import CandidateProfile


def extract_evidence(profile: CandidateProfile) -> List[Dict]:
    evidence = []
    for skill in profile.skills.technical_skills:
        supporting = next((f"{p.title}: {p.description}" for p in profile.projects
                           if skill.lower() in (p.description + ' ' + ' '.join(p.tech_stack)).lower()), skill)
        evidence.append({"value": skill, "category": "skill", "source": "resume",
                         "confidence": 0.75 if supporting != skill else 0.55,
                         "evidence_text": supporting, "resume_section": "projects" if supporting != skill else "skills"})
    for project in profile.projects:
        evidence.append({"value": project.title, "category": "project", "source": "resume", "confidence": 0.8,
                         "evidence_text": project.description + (f" Impact: {project.impact}" if project.impact else ""),
                         "resume_section": "projects"})
    for experience in profile.experience:
        evidence.append({"value": experience.role, "category": "experience", "source": "resume", "confidence": 0.8,
                         "evidence_text": " ".join(experience.description), "resume_section": "experience"})
    return evidence
