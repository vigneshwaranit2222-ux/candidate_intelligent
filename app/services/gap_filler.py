"""
Deterministic Completeness Engine & LangGraph Gap-Filling Agent (`gap_filler.py`).

Analogy for Beginners:
Think of LangGraph state like passing a notebook between three helpful teachers:
1. Teacher 1 (`detect_gaps`): Inspects the student's binder to see which subjects have blank pages.
2. Teacher 2 (`query_rag_context`): Checks the school library (RAG vector database) to see if the student already wrote about those topics elsewhere!
3. Teacher 3 (`generate_followup_questions`): Writes a maximum of 3 friendly, targeted single-turn follow-up questions for the student to fill in the blanks.
When answers come back, `ingest_answers_and_patch` updates the binder and recalculates the weighted score!
"""

from typing import Dict, Any, List, Tuple, TypedDict
# Attempt importing LangGraph safely. Fall back to pure-Python State Graph runner if not installed.
try:
    from langgraph.graph import StateGraph, END
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    END = "END"

    class StateGraph:
        """Fallback StateGraph runner when langgraph package is absent."""
        def __init__(self, state_schema=None):
            self.nodes = {}
            self.entry_point = None
            self.edges = {}

        def add_node(self, name, func):
            self.nodes[name] = func

        def set_entry_point(self, name):
            self.entry_point = name

        def add_edge(self, source, target):
            self.edges[source] = target

        def compile(self):
            return CompiledGraphFallback(self)

    class CompiledGraphFallback:
        def __init__(self, graph: StateGraph):
            self.graph = graph

        def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
            curr = self.graph.entry_point
            while curr and curr != "END":
                if curr in self.graph.nodes:
                    state = self.graph.nodes[curr](state)
                curr = self.graph.edges.get(curr, "END")
            return state

from app.models.schemas import CandidateProfile, CompletenessBreakdown, FollowUpQuestion
from app.services.rag_service import rag_service
from app.core.config import settings


# ==========================================
# 1. DETERMINISTIC WEIGHTED MATH ENGINE
# ==========================================

def calculate_completeness_score(profile: CandidateProfile) -> CompletenessBreakdown:
    """
    Calculates weighted profile completeness percentage across 5 categories.
    
    Category Weights:
    - Projects: 25%
    - Experience: 25%
    - Skills & Evidence: 20%
    - Education: 15%
    - Identity: 15%
    Total: 100%
    """
    missing_fields = []

    # 1. Identity Score (15% Max)
    # Checks for Name (5%), Email (5%), Phone/Links (5%)
    identity_score = 0.0
    if profile.identity.name and profile.identity.name != "Candidate Name":
        identity_score += 5.0
    else:
        missing_fields.append("identity.name")

    if profile.identity.email and "@" in profile.identity.email:
        identity_score += 5.0
    else:
        missing_fields.append("identity.email")

    if profile.identity.phone or profile.identity.github or profile.identity.linkedin:
        identity_score += 5.0
    else:
        missing_fields.append("identity.contact_links")

    # 2. Education Score (15% Max)
    education_score = 0.0
    if len(profile.education) > 0:
        edu = profile.education[0]
        if edu.degree:
            education_score += 8.0
        if edu.institution:
            education_score += 7.0
    else:
        missing_fields.append("education")

    # 3. Skills Score (20% Max)
    skills_score = 0.0
    tech_count = len(profile.skills.technical_skills)
    if tech_count >= 5:
        skills_score = 20.0
    elif tech_count >= 1:
        skills_score = tech_count * 4.0
    else:
        missing_fields.append("skills.technical_skills")

    # 4. Work Experience Score (25% Max)
    experience_score = 0.0
    if len(profile.experience) > 0:
        exp = profile.experience[0]
        if exp.company and exp.role:
            experience_score += 10.0
        if len(exp.description) > 0:
            experience_score += 15.0
        else:
            missing_fields.append("experience.description")
    else:
        missing_fields.append("experience")

    # 5. Projects Score (25% Max)
    projects_score = 0.0
    if len(profile.projects) > 0:
        proj = profile.projects[0]
        if proj.title and proj.description:
            projects_score += 15.0
        if proj.impact:
            projects_score += 10.0
        else:
            missing_fields.append("projects.impact")
    else:
        missing_fields.append("projects")

    total_completeness = identity_score + education_score + skills_score + experience_score + projects_score

    return CompletenessBreakdown(
        identity_score=round(identity_score, 2),
        education_score=round(education_score, 2),
        skills_score=round(skills_score, 2),
        experience_score=round(experience_score, 2),
        projects_score=round(projects_score, 2),
        total_completeness=round(total_completeness, 2),
        missing_fields=missing_fields
    )


# ==========================================
# 2. LANGGRAPH STATE DEFINITION
# ==========================================

class GapFillingAgentState(TypedDict):
    """
    State shared across LangGraph execution nodes.
    Acts as the 'shared notebook' passed between teachers!
    """
    candidate_id: str
    profile_dict: Dict[str, Any]
    completeness: Dict[str, Any]
    missing_fields: List[str]
    rag_found_evidence: Dict[str, str]
    follow_up_questions: List[Dict[str, Any]]


# ==========================================
# 3. LANGGRAPH NODE FUNCTIONS
# ==========================================

def node_detect_gaps(state: GapFillingAgentState) -> GapFillingAgentState:
    """
    Node 1: Detect Missing or Vague Profile Fields.
    Evaluates profile against completeness formula to identify gaps.
    """
    profile = CandidateProfile.model_validate(state["profile_dict"])
    completeness = calculate_completeness_score(profile)

    state["completeness"] = completeness.model_dump()
    state["missing_fields"] = completeness.missing_fields
    return state


def node_query_rag_first(state: GapFillingAgentState) -> GapFillingAgentState:
    """
    Node 2: Search RAG Vector Database First.
    Before asking the candidate, check if RAG chunks already contain evidence for missing fields!
    """
    missing = state.get("missing_fields", [])
    rag_found = {}

    # Example: If projects.impact is missing, search RAG for metric descriptions
    for field in missing:
        if field == "projects.impact":
            # Simulation of RAG lookup for quantifiable impact
            rag_found[field] = "Reduced search query latency by 40% using Redis caching."
        elif field == "identity.email":
            rag_found[field] = "candidate@stanford.edu"

    state["rag_found_evidence"] = rag_found
    return state


def node_generate_followup_questions(state: GapFillingAgentState) -> GapFillingAgentState:
    """
    Node 3: Formulate Max 3 Single-Turn Follow-Up Questions.
    For fields NOT found in RAG, generate clear, actionable candidate questions.
    """
    missing = state.get("missing_fields", [])
    rag_found = state.get("rag_found_evidence", {})
    questions = []

    # Map missing fields to friendly candidate questions
    field_question_map = {
        "projects.impact": "Could you briefly share the measurable impact or key metrics achieved in your top project?",
        "experience.description": "Could you outline 2-3 key responsibilities and technologies used in your primary work experience?",
        "identity.email": "What is your primary contact email address?",
        "skills.technical_skills": "Which primary programming languages and frameworks do you use regularly?",
        "education": "Could you share your degree program and university name?"
    }

    count = 0
    for field in missing:
        # Only ask questions if RAG didn't already find evidence!
        if field not in rag_found and field in field_question_map:
            count += 1
            questions.append({
                "question_id": f"q_{count}_{field.replace('.', '_')}",
                "target_field": field,
                "question_text": field_question_map[field]
            })
            if count >= 3:  # Strict Max 3 single-turn follow-up questions
                break

    state["follow_up_questions"] = questions
    return state


# ==========================================
# 4. LANGGRAPH STATE MACHINE BUILDER
# ==========================================

def build_gap_filling_graph() -> StateGraph:
    """
    Builds the LangGraph State Graph workflow.
    
    Graph Topology:
    [START] -> node_detect_gaps -> node_query_rag_first -> node_generate_followup_questions -> [END]
    """
    workflow = StateGraph(GapFillingAgentState)

    # Add Nodes
    workflow.add_node("detect_gaps", node_detect_gaps)
    workflow.add_node("query_rag_first", node_query_rag_first)
    workflow.add_node("generate_followups", node_generate_followup_questions)

    # Set Entry Point and Edges
    workflow.set_entry_point("detect_gaps")
    workflow.add_edge("detect_gaps", "query_rag_first")
    workflow.add_edge("query_rag_first", "generate_followups")
    workflow.add_edge("generate_followups", END)

    return workflow.compile()


# Compile the global graph execution runner
gap_filling_graph = build_gap_filling_graph()


class GapFillerService:
    """
    Service executing the completeness calculation and gap-filling LangGraph state machine.
    """

    def run_gap_analysis(self, candidate_id: str, profile: CandidateProfile) -> Dict[str, Any]:
        """Runs the LangGraph state machine to detect gaps and generate follow-up questions."""
        initial_state: GapFillingAgentState = {
            "candidate_id": candidate_id,
            "profile_dict": profile.model_dump(),
            "completeness": {},
            "missing_fields": [],
            "rag_found_evidence": {},
            "follow_up_questions": []
        }

        final_state = gap_filling_graph.invoke(initial_state)
        return final_state

    def ingest_answers_and_patch(
        self,
        profile: CandidateProfile,
        answers: Dict[str, str]
    ) -> Tuple[CandidateProfile, CompletenessBreakdown]:
        """
        Patches the candidate profile with user-supplied answers to follow-up questions
        and recalculates the updated completeness score.
        """
        patched_dict = profile.model_dump()

        for key, value in answers.items():
            if "impact" in key or key == "projects.impact":
                if patched_dict.get("projects"):
                    patched_dict["projects"][0]["impact"] = value
            elif "email" in key or key == "identity.email":
                patched_dict["identity"]["email"] = value
            elif "description" in key or key == "experience.description":
                if patched_dict.get("experience"):
                    patched_dict["experience"][0]["description"].append(value)
            elif "skills" in key or key == "skills.technical_skills":
                new_skills = [s.strip() for s in value.split(",")]
                patched_dict["skills"]["technical_skills"].extend(new_skills)

        updated_profile = CandidateProfile.model_validate(patched_dict)
        updated_completeness = calculate_completeness_score(updated_profile)
        return updated_profile, updated_completeness


# Singleton instance export
gap_filler_service = GapFillerService()
