"""
Pydantic v2 Domain Data Validation Schemas.

Analogy for Beginners:
Think of Pydantic Schemas like strict bouncers at a venue entrance.
Before any raw data (like uploaded resume JSON or HTTP request payloads) is allowed into our application logic,
these schemas inspect every single field, checking types, verifying non-empty values, and cleaning up formatting.
If something is wrong or missing, Pydantic immediately flags a precise error message!
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, EmailStr, field_validator


# ==========================================
# 1. CANDIDATE PROFILE EXTRACTION SCHEMAS
# ==========================================

class Identity(BaseModel):
    """Candidate personal identity and contact information."""
    name: str = Field(..., description="Full legal name of candidate", json_schema_extra={"example": "Vigneshwaran M"})
    email: Optional[str] = Field(None, description="Primary contact email address", json_schema_extra={"example": "vignesh@example.com"})
    phone: Optional[str] = Field(None, description="Primary contact telephone number", json_schema_extra={"example": "+1-555-0199"})
    location: Optional[str] = Field(None, description="Current city, state or country", json_schema_extra={"example": "San Francisco, CA"})
    linkedin: Optional[str] = Field(None, description="LinkedIn profile URL or handle", json_schema_extra={"example": "linkedin.com/in/vigneshwaran"})
    github: Optional[str] = Field(None, description="GitHub profile URL or handle", json_schema_extra={"example": "github.com/vigneshwaran"})


class Education(BaseModel):
    """Academic qualifications and degrees."""
    degree: str = Field(..., description="Degree earned (e.g. B.S., M.S., Ph.D.)", json_schema_extra={"example": "Master of Science in Computer Science"})
    institution: str = Field(..., description="University or college name", json_schema_extra={"example": "Stanford University"})
    year: Optional[str] = Field(None, description="Graduation year or date range", json_schema_extra={"example": "2022"})
    gpa: Optional[str] = Field(None, description="Grade point average if provided", json_schema_extra={"example": "3.9/4.0"})


class Experience(BaseModel):
    """Professional work history."""
    company: str = Field(..., description="Employer company name", json_schema_extra={"example": "TechCorp Innovations"})
    role: str = Field(..., description="Job title held", json_schema_extra={"example": "Senior AI / Software Engineer"})
    duration: Optional[str] = Field(None, description="Employment date range", json_schema_extra={"example": "Jan 2023 - Present"})
    description: List[str] = Field(default_factory=list, description="Key bullet points and operational responsibilities")
    skills_used: List[str] = Field(default_factory=list, description="Technologies applied during role")


class Project(BaseModel):
    """Technical projects, portfolios, or key open-source contributions."""
    title: str = Field(..., description="Name of the project", json_schema_extra={"example": "AI Resume Evaluation Platform"})
    description: str = Field(..., description="High-level description of what the project accomplishes")
    tech_stack: List[str] = Field(default_factory=list, description="Libraries, frameworks, and databases used")
    impact: Optional[str] = Field(None, description="Quantifiable metrics or user impact achieved", json_schema_extra={"example": "Reduced search latency by 40%"})


class Skills(BaseModel):
    """Categorized technical and professional skills."""
    technical_skills: List[str] = Field(default_factory=list, description="Programming languages, frameworks, DBs")
    soft_skills: List[str] = Field(default_factory=list, description="Leadership, communication, problem solving")
    tools: List[str] = Field(default_factory=list, description="DevOps, cloud platforms, IDEs, version control")


class Certification(BaseModel):
    """Professional certifications or credentials."""
    name: str = Field(..., description="Certification title", json_schema_extra={"example": "AWS Certified Solutions Architect"})
    issuer: Optional[str] = Field(None, description="Issuing organization", json_schema_extra={"example": "Amazon Web Services"})
    year: Optional[str] = Field(None, description="Year earned", json_schema_extra={"example": "2023"})


class CandidateProfile(BaseModel):
    """
    Master Candidate Profile Pydantic Schema.
    Combines identity, education, experience, projects, skills, and certifications
    into a standardized JSON structure.
    """
    identity: Identity
    education: List[Education] = Field(default_factory=list)
    experience: List[Experience] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    skills: Skills = Field(default_factory=lambda: Skills())
    certifications: List[Certification] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)


# ==========================================
# 2. COMPLETENESS & GAP-FILLING SCHEMAS
# ==========================================

class CompletenessBreakdown(BaseModel):
    """
    Mathematical breakdown of profile completeness across 5 weighted categories.
    Projects: 25% | Experience: 25% | Skills & Evidence: 20% | Education: 15% | Identity: 15%
    """
    identity_score: float = Field(..., description="Identity category score (0-15)")
    education_score: float = Field(..., description="Education category score (0-15)")
    skills_score: float = Field(..., description="Skills category score (0-20)")
    experience_score: float = Field(..., description="Experience category score (0-25)")
    projects_score: float = Field(..., description="Projects category score (0-25)")
    total_completeness: float = Field(..., description="Total combined completeness percentage (0-100)")
    missing_fields: List[str] = Field(default_factory=list, description="List of detected missing profile areas")
    weak_fields: List[str] = Field(default_factory=list)
    confidence: float = 0.0


class FollowUpQuestion(BaseModel):
    """Structured follow-up question generated by the gap-filling agent."""
    question_id: str = Field(..., description="Unique question identifier")
    target_field: str = Field(..., description="Profile field being addressed (e.g., 'projects.impact')")
    question_text: str = Field(..., description="Clear single-turn question text")
    gap_id: Optional[str] = None
    reason: Optional[str] = None
    expected_answer_type: str = "text"


class GapFillRequest(BaseModel):
    """Payload containing candidate answers to follow-up gap-filling questions."""
    answers: Dict[str, str] = Field(..., description="Mapping of question_id or target_field to candidate answer string")


class GapFillResponse(BaseModel):
    """Response returned after processing gap fill answers."""
    candidate_id: str
    updated_completeness: CompletenessBreakdown
    patched_fields: List[str]
    follow_up_questions: List[FollowUpQuestion] = Field(default_factory=list)


# ==========================================
# 3. ROLE RECOMMENDATION SCHEMAS
# ==========================================

class RoleRecommendation(BaseModel):
    """Individual job role recommendation with breakdown rationale."""
    role_name: str = Field(..., description="Target job title (e.g. 'Backend Engineer')")
    hybrid_score: float = Field(..., description="Combined recommendation score (0.0 - 100.0)")
    vector_similarity_score: float = Field(..., description="60% weighted vector semantic similarity score")
    skill_match_score: float = Field(..., description="40% weighted direct required skill match percentage")
    matched_skills: List[str] = Field(default_factory=list, description="Required skills candidate already possesses")
    missing_skills: List[str] = Field(default_factory=list, description="Required skills candidate needs to acquire")
    match_rationale: str = Field(..., description="Plain-English explanation of why this role matches")
    project_relevance_score: float = 0.0
    experience_relevance_score: float = 0.0
    profile_completeness_score: float = 0.0
    supporting_projects: List[str] = Field(default_factory=list)


class RoleMatchResponse(BaseModel):
    """API payload containing top role recommendations."""
    candidate_id: str
    recommendations: List[RoleRecommendation]


class RoleSelectionRequest(BaseModel):
    """Payload for candidate selecting 2 to 3 target roles."""
    selected_roles: List[str] = Field(..., min_length=1, max_length=3, description="List of 1 to 3 chosen job role names")


class AssessmentStartResponse(BaseModel):
    candidate_id: str
    target_role: str
    blueprint: Dict[str, Any]
    first_question: str


class AssessmentStatusResponse(BaseModel):
    candidate_id: str
    target_role: Optional[str] = None
    current_level: int = 1
    turns_completed: int = 0
    is_completed: bool = False
    competencies_covered: List[str] = Field(default_factory=list)


# ==========================================
# 4. ADAPTIVE ASSESSMENT SCHEMAS
# ==========================================

class AssessmentTurnInput(BaseModel):
    """HTTP input payload for taking a turn in the adaptive technical interview."""
    target_role: str = Field(default="Backend Engineer", description="Selected role for question context")
    candidate_answer: str = Field(..., description="Candidate's technical response to the current question")


class DimensionEvaluation(BaseModel):
    """Detailed score evaluation breakdown for a single answer turn."""
    technical_fundamentals: float = Field(..., description="Score 0-100 for fundamental correctness")
    role_depth: float = Field(..., description="Score 0-100 for architectural / deep system knowledge")
    problem_solving: float = Field(..., description="Score 0-100 for analytical reasoning and edge-case handling")
    communication_clarity: float = Field(..., description="Score 0-100 for clear structure and explanation")
    turn_classification: str = Field(..., description="'Correct / Deep', 'Partial', or 'Weak / Incorrect'")


class AssessmentTurnOutput(BaseModel):
    """HTTP output response after evaluating candidate turn."""
    session_id: str
    current_level: int = Field(..., description="Current difficulty level from 1 to 5")
    level_change: str = Field(..., description="'Increased (+1)', 'Retained', or 'Decreased (-1)'")
    previous_question: str
    evaluation: DimensionEvaluation
    feedback: str = Field(..., description="Actionable mentor feedback on candidate's answer")
    next_question: str = Field(..., description="Next adaptive question dynamically tailored to level and tech stack")
    is_completed: bool = Field(default=False)


# ==========================================
# 5. DIAGNOSTIC REPORT & BENCHMARK SCHEMAS
# ==========================================

class CalibratedTierEnum(str, Enum):
    """Standardized candidate readiness tiers."""
    FOUNDATION = "Foundation"
    ENTRY_LEVEL_READY = "Entry-Level Ready"
    STRONG_ENTRY_LEVEL = "Strong Entry-Level"
    INTERMEDIATE_POTENTIAL = "Intermediate Potential"


class DiagnosticReportSchema(BaseModel):
    """Final candidate benchmark diagnostic report."""
    candidate_id: str
    candidate_name: str
    technical_fundamentals: float = Field(..., description="Score 0-100 across core computer science & syntax")
    role_depth: float = Field(..., description="Score 0-100 across domain-specific frameworks & patterns")
    problem_solving: float = Field(..., description="Score 0-100 across trade-off analysis & algorithmic thinking")
    communication_clarity: float = Field(..., description="Score 0-100 across structural clarity & tech terminology")
    overall_score: float = Field(..., description="Weighted average overall benchmark score")
    calibrated_tier: CalibratedTierEnum = Field(..., description="Standardized placement readiness tier")
    roadmap: List[Dict[str, Any]] = Field(default_factory=list, description="Actionable gap analysis upskilling steps for Phase 2")
    overall_readiness: Optional[float] = None
    readiness_confidence: Optional[float] = None
    role_readiness: Dict[str, float] = Field(default_factory=dict)
    improvement_areas: List[Dict[str, Any]] = Field(default_factory=list)
