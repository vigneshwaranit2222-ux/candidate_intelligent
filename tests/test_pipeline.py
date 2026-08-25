"""
Automated End-to-End Integration Test Suite (`test_pipeline.py`).

Analogy for Beginners:
Think of this test suite like a full rehearsal of a play!
It simulates a real candidate walking through every single stage of our platform:
1. Uploading a resume PDF/Text (`Module 1`).
2. Answering gap-filling follow-up questions (`Module 2`).
3. Receiving top 5 hybrid job recommendations and picking target roles (`Module 3`).
4. Taking an adaptive technical interview where difficulty scales dynamically (`Module 4`).
5. Generating a final benchmark report with a Phase 2 upskilling roadmap (`Module 5`).
Every single assertion verifies that our system operates cleanly with zero bugs!
"""

import os
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_01_health_check(client: AsyncClient):
    """
    Step 1: Test system health check endpoint.
    """
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "AI Candidate Intelligence Platform" in data["app_name"]


@pytest.mark.asyncio
async def test_02_candidate_upload_and_parsing(client: AsyncClient):
    """
    Step 2 (Module 1): Test resume upload, profile parsing, completeness score calculation,
    and vector chunk indexing.
    """
    # Load sample mock resume text
    mock_resume_path = os.path.join(os.path.dirname(__file__), "..", "sample_resumes", "mock_resume.txt")
    with open(mock_resume_path, "r", encoding="utf-8") as f:
        resume_text = f.read()

    response = await client.post(
        "/api/candidate/upload",
        data={"resume_text": resume_text}
    )
    assert response.status_code == 201
    data = response.json()

    assert "candidate_id" in data
    assert data["profile"]["identity"]["name"] is not None
    assert "completeness" in data
    assert data["completeness"]["total_completeness"] > 0.0
    assert len(data["profile"]["skills"]["technical_skills"]) > 0


@pytest.mark.asyncio
async def test_03_gap_filling_flow(client: AsyncClient):
    """
    Step 3 (Module 2): Test submitting follow-up answers, profile patching,
    and recalculating weighted completeness score.
    """
    # 1. Upload Resume
    mock_resume_path = os.path.join(os.path.dirname(__file__), "..", "sample_resumes", "mock_resume.txt")
    with open(mock_resume_path, "r", encoding="utf-8") as f:
        resume_text = f.read()

    upload_res = await client.post("/api/candidate/upload", data={"resume_text": resume_text})
    candidate_id = upload_res.json()["candidate_id"]

    # 2. Submit Gap Fill Answers
    gap_fill_payload = {
        "answers": {
            "projects.impact": "Achieved sub-10ms query latency across 100,000 vector records using pgvector.",
            "identity.email": "vignesh.m@stanford.edu"
        }
    }

    gap_res = await client.post(f"/api/candidate/{candidate_id}/gap-fill", json=gap_fill_payload)
    assert gap_res.status_code == 200
    gap_data = gap_res.json()

    assert gap_data["candidate_id"] == candidate_id
    assert gap_data["updated_completeness"]["total_completeness"] >= upload_res.json()["completeness"]["total_completeness"]


@pytest.mark.asyncio
async def test_04_role_recommendations_and_selection(client: AsyncClient):
    """
    Step 4 (Module 3): Test fetching top 5 hybrid role recommendations (60% vector + 40% skills)
    and selecting target candidate roles.
    """
    # 1. Upload Resume
    mock_resume_path = os.path.join(os.path.dirname(__file__), "..", "sample_resumes", "mock_resume.txt")
    with open(mock_resume_path, "r", encoding="utf-8") as f:
        resume_text = f.read()

    upload_res = await client.post("/api/candidate/upload", data={"resume_text": resume_text})
    candidate_id = upload_res.json()["candidate_id"]

    # 2. Fetch Recommendations
    rec_res = await client.get(f"/api/candidate/{candidate_id}/recommendations")
    assert rec_res.status_code == 200
    rec_data = rec_res.json()

    assert rec_data["candidate_id"] == candidate_id
    recommendations = rec_data["recommendations"]
    assert len(recommendations) == 5
    
    # Verify Hybrid Score formula ordering
    top_rec = recommendations[0]
    assert top_rec["hybrid_score"] > 0.0
    assert "vector_similarity_score" in top_rec
    assert "skill_match_score" in top_rec

    # 3. Select Target Roles
    select_payload = {"selected_roles": ["Backend Engineer", "Data / AI Engineer"]}
    select_res = await client.post(f"/api/candidate/{candidate_id}/select-roles", json=select_payload)
    assert select_res.status_code == 200
    assert select_res.json()["selected_roles"] == ["Backend Engineer", "Data / AI Engineer"]


@pytest.mark.asyncio
async def test_05_adaptive_assessment_turn_execution(client: AsyncClient):
    """
    Step 5 (Module 4): Test adaptive technical assessment turn, verifying difficulty scaling
    (Level 1 to 5) based on answer quality and grounded question generation.
    """
    # 1. Upload Resume
    mock_resume_path = os.path.join(os.path.dirname(__file__), "..", "sample_resumes", "mock_resume.txt")
    with open(mock_resume_path, "r", encoding="utf-8") as f:
        resume_text = f.read()

    upload_res = await client.post("/api/candidate/upload", data={"resume_text": resume_text})
    candidate_id = upload_res.json()["candidate_id"]

    # 2. Take Turn 1 with a Deep/Correct Answer
    deep_answer_payload = {
        "target_role": "Backend Engineer",
        "candidate_answer": "I built an async RAG search pipeline using FastAPI and PostgreSQL pgvector with Redis caching to minimize query latency under heavy concurrency."
    }

    turn1_res = await client.post(f"/api/assessment/{candidate_id}/turn", json=deep_answer_payload)
    assert turn1_res.status_code == 200
    turn1_data = turn1_res.json()

    assert turn1_data["evaluation"]["turn_classification"] == "Correct / Deep"
    assert turn1_data["current_level"] == 2  # Scaled up from Level 1 to Level 2
    assert "Increased" in turn1_data["level_change"]
    assert len(turn1_data["next_question"]) > 0


@pytest.mark.asyncio
async def test_06_diagnostic_report_generation(client: AsyncClient):
    """
    Step 6 (Module 5): Test benchmark diagnostic report generation, verifying 4 dimension scores,
    calibrated readiness tier, and Phase 2 upskilling roadmap.
    """
    # 1. Upload Resume
    mock_resume_path = os.path.join(os.path.dirname(__file__), "..", "sample_resumes", "mock_resume.txt")
    with open(mock_resume_path, "r", encoding="utf-8") as f:
        resume_text = f.read()

    upload_res = await client.post("/api/candidate/upload", data={"resume_text": resume_text})
    candidate_id = upload_res.json()["candidate_id"]

    # 2. Take an Assessment Turn
    turn_payload = {
        "target_role": "Backend Engineer",
        "candidate_answer": "I optimized pgvector HNSW indexing parameters for high recall vector similarity queries."
    }
    await client.post(f"/api/assessment/{candidate_id}/turn", json=turn_payload)

    # 3. Fetch Diagnostic Report
    report_res = await client.get(f"/api/candidate/{candidate_id}/report")
    assert report_res.status_code == 200
    report_data = report_res.json()

    assert report_data["candidate_id"] == candidate_id
    assert report_data["technical_fundamentals"] > 0.0
    assert report_data["role_depth"] > 0.0
    assert report_data["problem_solving"] > 0.0
    assert report_data["communication_clarity"] > 0.0
    assert report_data["overall_score"] > 0.0
    assert report_data["calibrated_tier"] in [
        "Foundation", "Entry-Level Ready", "Strong Entry-Level", "Intermediate Potential"
    ]
    assert len(report_data["roadmap"]) > 0
