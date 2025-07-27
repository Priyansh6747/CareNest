from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
from .symptomCore import SymptomAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/medical", tags=["medical"])

# Initialize the symptom analyzer
symptom_analyzer = SymptomAnalyzer()


# Pydantic models for request/response
class HealthCheckResponse(BaseModel):
    status: str
    message: str


class TranscriptionRequest(BaseModel):
    audio_data: str  # Base64 encoded audio data
    model: Optional[str] = "medium"


class TranscriptionResponse(BaseModel):
    transcript: str
    language: Optional[str] = None
    success: bool


class MedicalAnalysisRequest(BaseModel):
    transcript: str
    patient_age_group: Optional[str] = "pregnant women"


class MedicalAnalysisResponse(BaseModel):
    symptom_details: Dict[str, Any]
    recommended_specialty: str
    urgency_level: str
    home_remedies: List[str]
    supportive_care: List[str]
    advice_next_steps: str
    first_aid: Optional[str]
    possible_causes: List[str]
    friendly_summary: str
    success: bool


class RAGQueryRequest(BaseModel):
    query: str
    max_results: Optional[int] = 3


class RAGQueryResponse(BaseModel):
    answer: str
    source_documents: List[Dict[str, Any]]
    success: bool


class AudioToAnalysisRequest(BaseModel):
    audio_data: str  # Base64 encoded audio data
    patient_age_group: Optional[str] = "pregnant women"
    model: Optional[str] = "medium"


class QueryRequest(BaseModel):
    query: str
    use_rag: Optional[bool] = True


class QueryResponse(BaseModel):
    answer: str
    source_documents: Optional[List[Dict[str, Any]]] = None
    success: bool


@router.get("/health-check", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint to verify the service is running"""
    try:
        # Check if the analyzer is properly initialized
        if symptom_analyzer.is_initialized():
            return HealthCheckResponse(
                status="healthy",
                message="Medical analysis service is running properly"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service is not properly initialized"
            )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is currently unavailable"
        )


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(request: TranscriptionRequest):
    """Transcribe audio to text using Whisper"""
    try:
        logger.info("Starting audio transcription")

        transcript, language = await symptom_analyzer.transcribe_audio(
            request.audio_data,
            request.model
        )

        logger.info(f"Transcription completed: {transcript[:100]}...")

        return TranscriptionResponse(
            transcript=transcript,
            language=language,
            success=True
        )

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}"
        )


@router.post("/medical-analysis", response_model=MedicalAnalysisResponse)
async def analyze_medical_symptoms(request: MedicalAnalysisRequest):
    """Analyze medical symptoms from transcript"""
    try:
        logger.info(f"Starting medical analysis for: {request.transcript[:100]}...")

        analysis_result = await symptom_analyzer.analyze_symptoms(
            request.transcript,
            request.patient_age_group
        )

        logger.info("Medical analysis completed successfully")

        return MedicalAnalysisResponse(
            symptom_details=analysis_result.get("symptom_details", {}),
            recommended_specialty=analysis_result.get("recommended_specialty", "General Physician"),
            urgency_level=analysis_result.get("urgency_level", "Routine check-up"),
            home_remedies=analysis_result.get("home_remedies", []),
            supportive_care=analysis_result.get("supportive_care", []),
            advice_next_steps=analysis_result.get("advice_next_steps", ""),
            first_aid=analysis_result.get("first_aid"),
            possible_causes=analysis_result.get("possible_causes", []),
            friendly_summary=analysis_result.get("friendly_summary", ""),
            success=True
        )

    except Exception as e:
        logger.error(f"Medical analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Medical analysis failed: {str(e)}"
        )


@router.post("/rag-query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    """Query the medical knowledge base using RAG"""
    try:
        logger.info(f"Processing RAG query: {request.query}")

        answer, source_docs = await symptom_analyzer.query_knowledge_base(
            request.query,
            request.max_results
        )

        logger.info("RAG query completed successfully")

        return RAGQueryResponse(
            answer=answer,
            source_documents=source_docs,
            success=True
        )

    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG query failed: {str(e)}"
        )


@router.post("/audio-to-analysis", response_model=MedicalAnalysisResponse)
async def audio_to_analysis(request: AudioToAnalysisRequest):
    """Complete pipeline: audio transcription + medical analysis"""
    try:
        logger.info("Starting audio-to-analysis pipeline")

        # First transcribe the audio
        transcript, _ = await symptom_analyzer.transcribe_audio(
            request.audio_data,
            request.model
        )

        logger.info(f"Audio transcribed: {transcript[:100]}...")

        # Then analyze the symptoms
        analysis_result = await symptom_analyzer.analyze_symptoms(
            transcript,
            request.patient_age_group
        )

        logger.info("Audio-to-analysis pipeline completed successfully")

        return MedicalAnalysisResponse(
            symptom_details=analysis_result.get("symptom_details", {}),
            recommended_specialty=analysis_result.get("recommended_specialty", "General Physician"),
            urgency_level=analysis_result.get("urgency_level", "Routine check-up"),
            home_remedies=analysis_result.get("home_remedies", []),
            supportive_care=analysis_result.get("supportive_care", []),
            advice_next_steps=analysis_result.get("advice_next_steps", ""),
            first_aid=analysis_result.get("first_aid"),
            possible_causes=analysis_result.get("possible_causes", []),
            friendly_summary=analysis_result.get("friendly_summary", ""),
            success=True
        )

    except Exception as e:
        logger.error(f"Audio-to-analysis pipeline failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio-to-analysis pipeline failed: {str(e)}"
        )


@router.post("/query", response_model=QueryResponse)
async def general_query(request: QueryRequest):
    """General query endpoint with optional RAG"""
    try:
        logger.info(f"Processing general query: {request.query}")

        if request.use_rag:
            # Use RAG for medical queries
            answer, source_docs = await symptom_analyzer.query_knowledge_base(
                request.query,
                max_results=3
            )

            return QueryResponse(
                answer=answer,
                source_documents=source_docs,
                success=True
            )
        else:
            # Direct LLM query without RAG
            answer = await symptom_analyzer.direct_llm_query(request.query)

            return QueryResponse(
                answer=answer,
                source_documents=None,
                success=True
            )

    except Exception as e:
        logger.error(f"General query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )