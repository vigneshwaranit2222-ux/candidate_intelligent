"""
API Routes Module (`routes.py`).

Analogy for Beginners:
Think of these API routes like the counter menus in a bank or restaurant.
Each endpoint (`/upload`, `/gap-fill`, `/recommendations`, `/select-roles`, `/turn`, `/report`) is a specific window.
Customers (frontend apps or test clients) send HTTP requests with data, our routes delegate the work to specialized services,
and they return clean, strictly validated JSON responses!
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_async_db
from app.db.models import Candidate, AssessmentSession, DiagnosticReport, CandidateEvidence, SkillVerification, AssessmentBlueprint
from app.models.schemas import (
    CandidateProfile, CompletenessBreakdown, FollowUpQuestion, GapFillRequest, GapFillResponse,
    RoleMatchResponse, RoleSelectionRequest, AssessmentTurnInput, AssessmentTurnOutput,
    DiagnosticReportSchema
    , AssessmentStartResponse, AssessmentStatusResponse
)
from app.services.parser import parser_service
from app.services.rag_service import rag_service
from app.services.gap_filler import gap_filler_service, calculate_completeness_score
from app.services.recommender import recommender_service
from app.services.assessment import assessment_service
from app.services.scoring import scoring_service
from app.services.evidence import extract_evidence
from app.services.assessment_blueprint_service import assessment_blueprint_service
from app.services.readiness import readiness_benchmark_service

router = APIRouter(prefix="/api", tags=["Candidate Intelligence Platform"])


async def generate_short_candidate_id(db: AsyncSession) -> str:
    """
    Generates clean sequential candidate IDs like 'c001', 'c002', 'c003'.
    
    Analogy:
    Think of this like taking a ticket number at a counter!
    Instead of a long 36-character random UUID, candidates receive clean ticket numbers in order: c001, c002, c003...
    """
    stmt = select(Candidate.id)
    result = await db.execute(stmt)
    all_ids = result.scalars().all()

    max_num = 0
    for cid in all_ids:
        if cid and cid.startswith("c") and cid[1:].isdigit():
            num = int(cid[1:])
            if num > max_num:
                max_num = num

    return f"c{max_num + 1:03d}"


# ==========================================
# 1. RESUME UPLOAD & PARSING ENDPOINT
# ==========================================

@router.post("/candidate/upload", status_code=status.HTTP_201_CREATED)
async def upload_candidate_resume(
    file: Optional[UploadFile] = File(None),
    resume_text: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_async_db)
) -> Dict[str, Any]:
    """
    Module 1 Entry Point: Upload candidate resume PDF or text payload.
    
    Workflow:
    1. Extract text from PDF file or form string.
    2. Parse into structured `CandidateProfile` schema.
    3. Calculate initial weighted completeness score.
    4. Save Candidate model to database with clean short ID (e.g., 'c001').
    5. Chunk semantic text and store vector embeddings in pgvector.
    """
    if file:
        file_bytes = await file.read()
        if (file.filename or "").lower().endswith(".docx"):
            raw_resume_text = parser_service.extract_text_from_docx_bytes(file_bytes)
            profile = parser_service.parse_resume(raw_resume_text)
        else:
            raw_resume_text = parser_service.extract_text_from_pdf_bytes(file_bytes)
            profile = parser_service.parse_resume(raw_resume_text)
    elif resume_text:
        profile = parser_service.parse_resume(resume_text)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please provide either a PDF file upload or raw resume_text."
        )

    # Calculate Completeness Score
    completeness = calculate_completeness_score(profile)

    # Generate Short Candidate ID (e.g., 'c001', 'c002')
    short_candidate_id = await generate_short_candidate_id(db)

    # Save Candidate to DB
    db_candidate = Candidate(
        id=short_candidate_id,
        name=profile.identity.name,
        email=profile.identity.email,
        phone=profile.identity.phone,
        parsed_profile=profile.model_dump(), original_resume_text=raw_resume_text if file else resume_text,
        completeness_score=completeness.total_completeness,
        target_roles=[]
    )
    db.add(db_candidate)
    await db.commit()
    await db.refresh(db_candidate)

    # Persist claimed evidence separately. Resume claims are deliberately not verified.
    for item in extract_evidence(profile):
        db.add(CandidateEvidence(candidate_id=db_candidate.id, **item))
        if item["category"] == "skill":
            db.add(SkillVerification(candidate_id=db_candidate.id, skill=item["value"], claimed=True,
                                     evidenced=item["resume_section"] != "skills", verified=False, confidence=item["confidence"]))

    # Index chunks in RAG Vector Store
    await rag_service.index_candidate_chunks(db, db_candidate.id, profile)
    await db.commit()

    # Trigger LangGraph Gap-Filling Analysis
    gap_analysis = gap_filler_service.run_gap_analysis(db_candidate.id, profile)
    follow_ups = gap_analysis.get("follow_up_questions", [])

    return {
        "candidate_id": db_candidate.id,
        "profile": profile.model_dump(),
        "completeness": completeness.model_dump(),
        "follow_up_questions": follow_ups
    }


@router.get("/candidate/{candidate_id}/profile")
async def get_candidate_profile(candidate_id: str, db: AsyncSession = Depends(get_async_db)) -> Dict[str, Any]:
    candidate = (await db.execute(select(Candidate).where(Candidate.id == candidate_id))).scalar_one_or_none()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found.")
    evidence = (await db.execute(select(CandidateEvidence).where(CandidateEvidence.candidate_id == candidate_id))).scalars().all()
    verification = (await db.execute(select(SkillVerification).where(SkillVerification.candidate_id == candidate_id))).scalars().all()
    return {"candidate_id": candidate_id, "profile": candidate.parsed_profile, "completeness_score": candidate.completeness_score,
            "evidence": [{"value": e.value, "category": e.category, "source": e.source, "confidence": e.confidence, "evidence_text": e.evidence_text, "resume_section": e.resume_section} for e in evidence],
            "skill_verification": [{"skill": s.skill, "claimed": s.claimed, "evidenced": s.evidenced, "verified": s.verified, "confidence": s.confidence} for s in verification]}


# ==========================================
# 2. GAP FILLING ENDPOINT
# ==========================================

@router.post("/candidate/{candidate_id}/gap-fill", response_model=GapFillResponse)
async def process_gap_fill_answers(
    candidate_id: str,
    payload: GapFillRequest,
    db: AsyncSession = Depends(get_async_db)
) -> GapFillResponse:
    """
    Module 2 Entry Point: Submit answers to follow-up gap-filling questions.
    Patches candidate profile schema and recalculates completeness score.
    """
    stmt = select(Candidate).where(Candidate.id == candidate_id)
    result = await db.execute(stmt)
    candidate = result.scalar_one_or_none()

    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found.")

    existing_profile = CandidateProfile.model_validate(candidate.parsed_profile)
    updated_profile, updated_completeness = gap_filler_service.ingest_answers_and_patch(
        existing_profile, payload.answers
    )

    # Update Candidate DB model
    candidate.parsed_profile = updated_profile.model_dump()
    candidate.completeness_score = updated_completeness.total_completeness
    await db.commit()

    return GapFillResponse(
        candidate_id=candidate_id,
        updated_completeness=updated_completeness,
        patched_fields=list(payload.answers.keys()),
        follow_up_questions=[]
    )


# ==========================================
# 3. ROLE RECOMMENDATION ENDPOINT
# ==========================================

@router.get("/candidate/{candidate_id}/recommendations", response_model=RoleMatchResponse)
async def get_role_recommendations(
    candidate_id: str,
    db: AsyncSession = Depends(get_async_db)
) -> RoleMatchResponse:
    """
    Module 3 Entry Point: Get top 5 hybrid role recommendations.
    Uses formula: 60% Vector Semantic Similarity + 40% Required Skill Match.
    """
    stmt = select(Candidate).where(Candidate.id == candidate_id)
    result = await db.execute(stmt)
    candidate = result.scalar_one_or_none()

    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found.")

    profile = CandidateProfile.model_validate(candidate.parsed_profile)
    recommendations = recommender_service.get_top_recommendations(profile, top_k=5, completeness=candidate.completeness_score)

    return RoleMatchResponse(
        candidate_id=candidate_id,
        recommendations=recommendations
    )


# ==========================================
# 4. ROLE SELECTION ENDPOINT
# ==========================================

@router.post("/candidate/{candidate_id}/select-roles")
async def select_target_roles(
    candidate_id: str,
    payload: RoleSelectionRequest,
    db: AsyncSession = Depends(get_async_db)
) -> Dict[str, Any]:
    """
    Allows candidate to select 1 to 3 target roles for technical assessment.
    """
    stmt = select(Candidate).where(Candidate.id == candidate_id)
    result = await db.execute(stmt)
    candidate = result.scalar_one_or_none()

    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found.")

    allowed = {r.role_name for r in recommender_service.get_top_recommendations(CandidateProfile.model_validate(candidate.parsed_profile), top_k=5, completeness=candidate.completeness_score)}
    if any(role not in allowed for role in payload.selected_roles):
        raise HTTPException(status_code=422, detail="Selected roles must be from this candidate's recommendations.")
    candidate.target_roles = payload.selected_roles
    await db.commit()

    return {
        "candidate_id": candidate_id,
        "selected_roles": candidate.target_roles,
        "status": "Target roles updated successfully."
    }


# ==========================================
# 5. ADAPTIVE ASSESSMENT TURN ENDPOINT
# ==========================================

@router.post("/assessment/{candidate_id}/start", response_model=AssessmentStartResponse)
async def start_assessment(candidate_id: str, db: AsyncSession = Depends(get_async_db)) -> AssessmentStartResponse:
    candidate = (await db.execute(select(Candidate).where(Candidate.id == candidate_id))).scalar_one_or_none()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found.")
    if not candidate.target_roles:
        raise HTTPException(status_code=409, detail="Select at least one recommended role before starting assessment.")
    profile = CandidateProfile.model_validate(candidate.parsed_profile)
    verified = (await db.execute(select(SkillVerification).where(SkillVerification.candidate_id == candidate_id, SkillVerification.verified == True))).scalars().all()
    role = candidate.target_roles[0]
    blueprint = assessment_blueprint_service.generate(profile, role, [item.skill for item in verified])
    db.add(AssessmentBlueprint(candidate_id=candidate_id, target_role=role, blueprint=blueprint))
    first_question = assessment_service.generate_grounded_question(role, 1, profile)
    return AssessmentStartResponse(candidate_id=candidate_id, target_role=role, blueprint=blueprint, first_question=first_question)


@router.get("/assessment/{candidate_id}/status", response_model=AssessmentStatusResponse)
async def assessment_status(candidate_id: str, db: AsyncSession = Depends(get_async_db)) -> AssessmentStatusResponse:
    session = (await db.execute(select(AssessmentSession).where(AssessmentSession.candidate_id == candidate_id).order_by(AssessmentSession.created_at.desc()))).scalars().first()
    if not session:
        return AssessmentStatusResponse(candidate_id=candidate_id)
    covered = [turn.get("competency") for turn in session.history if turn.get("competency")]
    return AssessmentStatusResponse(candidate_id=candidate_id, target_role=session.target_role, current_level=session.current_level, turns_completed=len(session.history), is_completed=session.is_completed, competencies_covered=covered)

@router.post("/assessment/{candidate_id}/turn", response_model=AssessmentTurnOutput)
async def process_assessment_turn(
    candidate_id: str,
    payload: AssessmentTurnInput,
    db: AsyncSession = Depends(get_async_db)
) -> AssessmentTurnOutput:
    """
    Module 4 Entry Point: Execute a single turn in the adaptive technical interview.
    Evaluates candidate response and scales difficulty from Level 1 to 5.
    """
    stmt = select(Candidate).where(Candidate.id == candidate_id)
    result = await db.execute(stmt)
    candidate = result.scalar_one_or_none()

    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found.")

    profile = CandidateProfile.model_validate(candidate.parsed_profile)

    # Retrieve or create active assessment session
    session_stmt = select(AssessmentSession).where(
        AssessmentSession.candidate_id == candidate_id,
        AssessmentSession.is_completed == False
    )
    session_res = await db.execute(session_stmt)
    assessment_session = session_res.scalar_one_or_none()

    if not assessment_session:
        assessment_session = AssessmentSession(
            candidate_id=candidate_id,
            target_role=payload.target_role,
            current_level=1,
            history=[]
        )
        db.add(assessment_session)
        await db.commit()
        await db.refresh(assessment_session)

    # Process turn via assessment state machine
    output, new_level, updated_history = assessment_service.process_assessment_turn(
        session_history=assessment_session.history or [],
        current_level=assessment_session.current_level,
        target_role=payload.target_role,
        candidate_answer=payload.candidate_answer,
        profile=profile
    )

    # An assessment is the only automatic route from evidence to verification.
    if output.evaluation.turn_classification == "Correct / Deep":
        answer_lower = payload.candidate_answer.lower()
        skills = (await db.execute(select(SkillVerification).where(SkillVerification.candidate_id == candidate_id))).scalars().all()
        for skill in skills:
            if skill.skill.lower() in answer_lower:
                skill.verified = True
                skill.confidence = min(1.0, skill.confidence + 0.2)

    # Update session in DB
    assessment_session.current_level = new_level
    assessment_session.history = updated_history
    assessment_session.is_completed = output.is_completed
    await db.commit()

    return output


# ==========================================
# 6. BENCHMARK & DIAGNOSTIC REPORT ENDPOINT
# ==========================================

@router.get("/candidate/{candidate_id}/report", response_model=DiagnosticReportSchema)
async def get_diagnostic_report(
    candidate_id: str,
    db: AsyncSession = Depends(get_async_db)
) -> DiagnosticReportSchema:
    """
    Module 5 Entry Point: Generate benchmark diagnostic report.
    Returns scores across 4 dimensions, calibrated readiness tier, and Phase 2 upskilling roadmap.
    """
    stmt = select(Candidate).where(Candidate.id == candidate_id)
    result = await db.execute(stmt)
    candidate = result.scalar_one_or_none()

    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found.")

    profile = CandidateProfile.model_validate(candidate.parsed_profile)

    # Retrieve session history if available
    session_stmt = select(AssessmentSession).where(AssessmentSession.candidate_id == candidate_id)
    session_res = await db.execute(session_stmt)
    session = session_res.scalar_one_or_none()

    history = session.history if session else []

    report_schema = scoring_service.generate_diagnostic_report(
        candidate_id=candidate_id,
        candidate_name=candidate.name,
        history=history,
        profile=profile
    )
    verification_rows = (await db.execute(select(SkillVerification).where(SkillVerification.candidate_id == candidate_id))).scalars().all()
    verification_confidence = (sum(item.confidence for item in verification_rows) / len(verification_rows) * 100) if verification_rows else 0.0
    target_role = candidate.target_roles[0] if candidate.target_roles else "General Software Engineer"
    readiness = readiness_benchmark_service.build(history, profile, target_role, verification_confidence)
    report_schema = report_schema.model_copy(update={"overall_readiness": readiness["overall_readiness"], "readiness_confidence": readiness["confidence"], "role_readiness": readiness["role_readiness"], "improvement_areas": readiness["improvement_areas"], "roadmap": readiness["roadmap"] or report_schema.roadmap})

    # Store or update DiagnosticReport DB record
    report_stmt = select(DiagnosticReport).where(DiagnosticReport.candidate_id == candidate_id)
    report_res = await db.execute(report_stmt)
    db_report = report_res.scalar_one_or_none()

    if not db_report:
        db_report = DiagnosticReport(
            candidate_id=candidate_id,
            technical_fundamentals=report_schema.technical_fundamentals,
            role_depth=report_schema.role_depth,
            problem_solving=report_schema.problem_solving,
            communication_clarity=report_schema.communication_clarity,
            overall_score=report_schema.overall_score,
            calibrated_tier=report_schema.calibrated_tier.value,
            roadmap=report_schema.roadmap
        )
        db.add(db_report)
    else:
        db_report.technical_fundamentals = report_schema.technical_fundamentals
        db_report.role_depth = report_schema.role_depth
        db_report.problem_solving = report_schema.problem_solving
        db_report.communication_clarity = report_schema.communication_clarity
        db_report.overall_score = report_schema.overall_score
        db_report.calibrated_tier = report_schema.calibrated_tier.value
        db_report.roadmap = report_schema.roadmap

    await db.commit()

    return report_schema
