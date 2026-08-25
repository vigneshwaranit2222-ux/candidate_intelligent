"""
Database ORM Models Module.

Analogy for Beginners:
Think of these ORM models like blueprint blueprints for filing cabinets in a corporate archive.
Each class represents a single drawer (Database Table), and each variable inside is a specific labeled folder (Column).
For example:
- `Candidate` is the main candidate profile folder holding contact details and parsed resume data.
- `ResumeChunk` holds snippet cards of candidate experience with vector embeddings (like index numbers for high-speed AI lookup).
- `AssessmentSession` records candidate interview answers and state machine difficulty level step by step.
- `DiagnosticReport` stores final calculated evaluation benchmark scores and readiness tiers.
"""

import uuid
from datetime import datetime
from typing import Any, List, Optional
from sqlalchemy import String, Float, Integer, Boolean, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Attempt pgvector import safely. Fall back to JSON storage on non-Postgres engines (e.g., SQLite testing).
try:
    from pgvector.sqlalchemy import Vector
    HAS_PGVECTOR = True
except ImportError:
    HAS_PGVECTOR = False


class Base(DeclarativeBase):
    """Base ORM class providing shared configuration for all database models."""
    pass


class Candidate(Base):
    """
    Candidate Database Table.
    Stores extracted contact details, parsed profile schema structure, completeness scores,
    and target selected roles.
    """
    __tablename__ = "candidates"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, default="Unknown Candidate")
    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    phone: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Store complete validated CandidateProfile dictionary in JSON format
    parsed_profile: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    original_resume_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Completeness score from 0.0 to 100.0
    completeness_score: Mapped[float] = mapped_column(Float, default=0.0)
    
    # List of selected target job roles (e.g., ["Backend Engineer", "Full Stack Engineer"])
    target_roles: Mapped[List[str]] = mapped_column(JSON, default=list)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    chunks: Mapped[List["ResumeChunk"]] = relationship("ResumeChunk", back_populates="candidate", cascade="all, delete-orphan")
    assessment_sessions: Mapped[List["AssessmentSession"]] = relationship("AssessmentSession", back_populates="candidate", cascade="all, delete-orphan")
    diagnostic_reports: Mapped[List["DiagnosticReport"]] = relationship("DiagnosticReport", back_populates="candidate", cascade="all, delete-orphan")
    evidence_items: Mapped[List["CandidateEvidence"]] = relationship("CandidateEvidence", back_populates="candidate", cascade="all, delete-orphan")
    skill_verifications: Mapped[List["SkillVerification"]] = relationship("SkillVerification", back_populates="candidate", cascade="all, delete-orphan")


class ResumeChunk(Base):
    """
    Resume Chunk Database Table for RAG (Retrieval-Augmented Generation).
    Stores individual experience bullet points, project descriptions, and skill snippets
    alongside 1536-dimensional vector embeddings for instant semantic search.
    """
    __tablename__ = "resume_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    candidate_id: Mapped[str] = mapped_column(String(36), ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False)
    
    # Text content of the chunk (e.g., "Architected RAG pipeline with pgvector")
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Section category: "experience", "projects", or "skills"
    section: Mapped[str] = mapped_column(String(100), nullable=False)
    
    # Metadata dictionary for additional context filtering
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    
    # Vector embedding storage: Uses pgvector Vector(1536) if available, otherwise JSON
    if HAS_PGVECTOR:
        embedding = mapped_column(Vector(1536), nullable=True)
    else:
        embedding = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    candidate: Mapped["Candidate"] = relationship("Candidate", back_populates="chunks")


class AssessmentSession(Base):
    """
    Assessment Session Database Table.
    Tracks live adaptive technical assessment turn history, difficulty scaling (Levels 1 to 5),
    and evaluated scores per question turn.
    """
    __tablename__ = "assessment_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    candidate_id: Mapped[str] = mapped_column(String(36), ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False)
    
    # Target role being assessed (e.g. "Backend Engineer")
    target_role: Mapped[str] = mapped_column(String(255), nullable=False, default="General AI/Software Engineer")
    
    # Current difficulty level from 1 (Fundamentals) to 5 (Principal Architectural System Design)
    current_level: Mapped[int] = mapped_column(Integer, default=1)
    
    # History array recording turns: [{question, answer, level, score_breakdown, feedback}, ...]
    history: Mapped[List[dict]] = mapped_column(JSON, default=list)
    
    # Cumulative dimension scores calculated so far
    dimensions: Mapped[dict] = mapped_column(JSON, default=dict)
    
    is_completed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    candidate: Mapped["Candidate"] = relationship("Candidate", back_populates="assessment_sessions")


class DiagnosticReport(Base):
    """
    Benchmark Diagnostic Report Database Table.
    Stores final candidate readiness results across 4 dimensions, overall benchmark score,
    assigned readiness tier, and tailored Phase 2 upskilling roadmap.
    """
    __tablename__ = "diagnostic_reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    candidate_id: Mapped[str] = mapped_column(String(36), ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False)
    
    # 4 Core Assessment Dimension Scores (0.0 to 100.0)
    technical_fundamentals: Mapped[float] = mapped_column(Float, default=0.0)
    role_depth: Mapped[float] = mapped_column(Float, default=0.0)
    problem_solving: Mapped[float] = mapped_column(Float, default=0.0)
    communication_clarity: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Weighted Overall Score (0.0 to 100.0)
    overall_score: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Calibrated Tier: "Foundation", "Entry-Level Ready", "Strong Entry-Level", or "Intermediate Potential"
    calibrated_tier: Mapped[str] = mapped_column(String(100), nullable=False)
    
    # Upskilling Roadmap JSON list of action steps
    roadmap: Mapped[List[dict]] = mapped_column(JSON, default=list)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    candidate: Mapped["Candidate"] = relationship("Candidate", back_populates="diagnostic_reports")


class CandidateEvidence(Base):
    __tablename__ = "candidate_evidence"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    candidate_id: Mapped[str] = mapped_column(String(50), ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False)
    value: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    source: Mapped[str] = mapped_column(String(50), nullable=False, default="resume")
    confidence: Mapped[float] = mapped_column(Float, default=0.7)
    evidence_text: Mapped[str] = mapped_column(Text, nullable=False)
    resume_section: Mapped[str] = mapped_column(String(100), nullable=False)
    candidate: Mapped["Candidate"] = relationship("Candidate", back_populates="evidence_items")


class SkillVerification(Base):
    __tablename__ = "skill_verifications"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    candidate_id: Mapped[str] = mapped_column(String(50), ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False)
    skill: Mapped[str] = mapped_column(String(255), nullable=False)
    claimed: Mapped[bool] = mapped_column(Boolean, default=True)
    evidenced: Mapped[bool] = mapped_column(Boolean, default=False)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    candidate: Mapped["Candidate"] = relationship("Candidate", back_populates="skill_verifications")


class AssessmentBlueprint(Base):
    __tablename__ = "assessment_blueprints"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    candidate_id: Mapped[str] = mapped_column(String(50), ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False)
    target_role: Mapped[str] = mapped_column(String(255), nullable=False)
    blueprint: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
