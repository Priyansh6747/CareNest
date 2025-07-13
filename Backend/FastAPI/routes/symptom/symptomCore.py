import os
import sys
import logging
import asyncio
import base64
import tempfile
import json
import re
from typing import Dict, List, Tuple, Any, Optional
from io import BytesIO
import warnings

# ML/AI libraries
import whisper
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv

# Configuration
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class SymptomAnalyzer:
    """Core class for medical symptom analysis and RAG functionality"""

    def __init__(self):
        self.groq_model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
        self.embedding_model = None
        self.medical_vector_store = None
        self.symptom_vector_store = None
        self.whisper_model = None
        self.llm = None
        self.qa_chain = None
        self._initialized = False

        # Initialize all components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all ML models and vector stores"""
        try:
            logger.info("Initializing SymptomAnalyzer components...")

            # Initialize embedding model
            self._initialize_embeddings()

            # Initialize vector stores
            self._initialize_vector_stores()

            # Initialize LLM
            self._initialize_llm()

            # Initialize Whisper model (loaded on demand)
            self.whisper_model = None

            self._initialized = True
            logger.info("SymptomAnalyzer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize SymptomAnalyzer: {e}")
            raise

    def _initialize_embeddings(self):
        """Initialize the embedding model"""
        try:
            logger.info("Loading embedding model...")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def _initialize_vector_stores(self):
        """Initialize vector stores for medical and symptom databases"""
        try:
            # Load symptom database vector store (now primary)
            symptom_db_path = "vectorstore/symptom_db_faiss"
            if os.path.exists(f"{symptom_db_path}/index.faiss"):
                logger.info("Loading symptom database vector store...")
                self.symptom_vector_store = FAISS.load_local(
                    symptom_db_path,
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
                logger.info("Symptom database vector store loaded successfully")
            else:
                logger.warning(f"Symptom database not found at {symptom_db_path}")

            # Load medical database vector store (kept as backup)
            medical_db_path = "vectorstore/medical_db_faiss"
            if os.path.exists(f"{medical_db_path}/index.faiss"):
                logger.info("Loading medical database vector store...")
                self.medical_vector_store = FAISS.load_local(
                    medical_db_path,
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
                logger.info("Medical database vector store loaded successfully")
            else:
                logger.warning(f"Medical database not found at {medical_db_path}")

        except Exception as e:
            logger.error(f"Failed to load vector stores: {e}")
            raise

    def _initialize_llm(self):
        """Initialize the HuggingFace LLM"""
        try:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("Missing GROQ_API_KEY in .env")

            logger.info("Connecting to Groq LLM...")
            self.llm = ChatGroq(
                groq_api_key=api_key,
                model_name=self.groq_model_name,
                temperature=0.3,
                max_tokens=1024,
            )

            # Initialize QA chain - prioritize symptom database, fallback to medical database
            primary_db = self.symptom_vector_store or self.medical_vector_store
            if primary_db:
                retriever = MultiQueryRetriever.from_llm(
                    retriever=primary_db.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 3}
                    ),
                    llm=self.llm,
                )

                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    retriever=retriever,
                    chain_type="stuff",
                    return_source_documents=True
                )

            db_type = "symptom" if self.symptom_vector_store else "medical" if self.medical_vector_store else "none"
            logger.info(f"Groq LLM initialized successfully with {db_type} database")

        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

    def is_initialized(self) -> bool:
        """Check if the analyzer is properly initialized"""
        return self._initialized

    def _load_whisper_model(self, model_name: str = "large"):
        """Load Whisper model on demand"""
        if self.whisper_model is None:
            logger.info(f"Loading Whisper model ({model_name})...")
            self.whisper_model = whisper.load_model(model_name)
            logger.info("Whisper model loaded successfully")
        return self.whisper_model

    async def transcribe_audio(self, audio_data: str, model: str = "medium") -> Tuple[str, str]:
        """
        Transcribe base64 encoded audio data to text

        Args:
            audio_data: Base64 encoded audio data
            model: Whisper model size

        Returns:
            Tuple of (transcript, detected_language)
        """
        try:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_data)

            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name

            try:
                # Load Whisper model
                whisper_model = self._load_whisper_model(model)

                # Transcribe audio
                logger.info("Starting transcription...")
                result = whisper_model.transcribe(temp_file_path, task="transcribe")

                transcript = result["text"].strip()
                language = result.get("language", "unknown")

                logger.info(f"Transcription completed. Language: {language}")

                if not transcript or len(transcript.strip()) < 3:
                    raise ValueError("No meaningful speech detected")

                return transcript, language

            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)

        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            raise

    async def analyze_symptoms(self, transcript: str, patient_age_group: str = "newborn") -> Dict[str, Any]:
        """
        Analyze symptoms from transcript using structured prompt

        Args:
            transcript: Patient's symptom description
            patient_age_group: Age group of patient

        Returns:
            Structured analysis results
        """
        try:
            # Create structured prompt based on age group
            if patient_age_group.lower() == "newborn":
                prompt = self._create_newborn_analysis_prompt(transcript)
            else:
                prompt = self._create_general_analysis_prompt(transcript, patient_age_group)

            # Get analysis from LLM
            logger.info("Generating symptom analysis...")

            if self.qa_chain:
                # Use RAG for better medical context
                result = self.qa_chain.invoke({"query": prompt})
                analysis_text = result["result"]
            else:
                # Direct LLM query
                analysis_text = await self._direct_llm_query(prompt)

            # Parse structured response
            parsed_analysis = self._parse_analysis_response(analysis_text)

            logger.info("Symptom analysis completed")
            return parsed_analysis

        except Exception as e:
            logger.error(f"Symptom analysis failed: {e}")
            raise

    def _create_newborn_analysis_prompt(self, transcript: str) -> str:
        """Create structured prompt for newborn symptom analysis"""
        return f"""
    You are a intelligent clinical assistant assessing baby (newborn) based on a caregiver's spoken description. Analyze the transcription carefully and provide a detailed, structured response. This age group is highly vulnerable — treat any concerning symptom with the highest caution. Analyze the transcript and extract the following details clearly and precisely.

    \"\"\"{transcript}\"\"\"

    Your task is to extract and report the following information in a structured format:

    1. 🤒 Symptom Details:
        - For each symptom, list:
            - Symptom name (e.g., cough)
            - Include: duration, severity, frequency, and patterns (e.g., "not feeding for 12 hours", "one-time vomiting", "sleepy for 1 day").
            - Estimate approximate duration if temporal clues like “started today” or “since yesterday” appear.
    2. 🩺 *Recommended Medical Specialty*: Suggest the most appropriate type of specialist (e.g., cardiologist, ENT, general physician).
    3. 🚨 *Urgency Level*:
        - Categorize the urgency of the condition:
            - Emergency (needs help now)
            - Urgent (within 24–48 hours)
            - Non-urgent but important
            - Routine check-up
            - ⚠ If the baby shows danger signs (lethargy, poor feeding, low urine output, cold to touch, abnormal breathing), label this as *Emergency*, not anything else.
    4.  🏠 *Recommended Home Remedies*:
            Suggest simple, safe, and evidence-informed home care measures (e.g., hydration, warm compress, turmeric milk, saltwater gargle).
            Highlight safety notes (e.g., “Avoid if allergic to…” or “Do not exceed recommended use”).
            Mention what to *avoid* doing or consuming during recovery (e.g., caffeine, alcohol, heavy meals, painkillers without advice). 
    5. 💊 *Evidence-Based Supportive Care (If Appropriate)*:
        - Include treatments recommended by WHO/IMCI or pediatric manuals, such as:
            - “Treat the child to prevent low blood sugar” (offer breastfeeding, sugar water)
            - “Give paracetamol for high fever or pain”
            - “Apply tetracycline eye ointment if there is eye discharge”
            - “Give an oral antimalarial if malaria is suspected”
            - “Always note: **“Only if applicable and under medical supervision.”
            - “Mention possible care actions known from neonatal protocols (e.g., Kangaroo care, hydration, warmth).
            - “ Do *not* recommend medications unless standard for neonates.”**        
    6. 💡 *Advice & Next Steps*: 
                - Give a clear, confident recommendation for what to do now.
                - If urgent, emphasize immediate travel to a clinic or hospital.
    7. 🚑 *First-Aid Recommendations* (if urgent): Mention emergency steps to take (e.g., warming the baby, keeping airway clear) *until medical help is available*.
    8. 🧬 *Possible Causes of the Condition*:
                - Based on the symptoms and context, list likely underlying causes (e.g., infection, allergy, lifestyle factors, exposure, environmental conditions, nutritional deficiency, etc.).
                - Mention if there could be multiple causes or if further testing is needed to identify the exact one.
                - List *suspected causes*, using safe language like “may suggest”, “could be”
                - Base suggestions on standard neonatal conditions (e.g., sepsis, hypothermia, dehydration, hypoglycemia).
                - Answer all the possible causes , no matter how many there are.
    9. 💬 *Friendly Summary to the Patient*: One or two-line response directly to the patient.Be firm but supportive. Make it warm, easy to understand, emotionally attached and supportive.

    Important:
    - Be medically cautious: avoid diagnosis, focus on triage and routing.
    - If any section has no info, write “Not specified.”
    - Keep it structured, clear, and avoid jargon unless well-explained.
    """

    def _create_general_analysis_prompt(self, transcript: str, age_group: str) -> str:
        """Create structured prompt for general symptom analysis"""
        return f"""
You are an intelligent clinical assistant assessing a {age_group} patient based on their symptom description. Analyze the transcription and provide a structured response.

Transcript: "{transcript}"

Provide analysis in the following structured format:

1. Symptom Details: List each symptom with duration, severity, frequency
2. Recommended Medical Specialty: Most appropriate specialist
3. Urgency Level: Emergency/Urgent/Non-urgent/Routine
4. Recommended Home Remedies: Safe home care measures
5. Supportive Care: Evidence-based treatments if appropriate
6. Advice & Next Steps: Clear recommendations
7. First-Aid Recommendations: If urgent care needed
8. Possible Causes: Likely underlying causes
9. Friendly Summary: Supportive message to patient

Be medically cautious and focus on appropriate triage.
"""

    def _parse_analysis_response(self, analysis_text: str) -> Dict[str, Any]:
        """Parse the structured analysis response from LLM"""
        try:
            # Initialize result dictionary
            result = {
                "symptom_details": {},
                "recommended_specialty": "General Physician",
                "urgency_level": "Routine check-up",
                "home_remedies": [],
                "supportive_care": [],
                "advice_next_steps": "",
                "first_aid": None,
                "possible_causes": [],
                "friendly_summary": ""
            }

            # Simple parsing based on section headers
            sections = {
                "symptom_details": r"1\.\s*🤒.*?Symptom Details:(.*?)(?=2\.|$)",
                "recommended_specialty": r"2\.\s*🩺.*?Recommended Medical Specialty:(.*?)(?=3\.|$)",
                "urgency_level": r"3\.\s*🚨.*?Urgency Level:(.*?)(?=4\.|$)",
                "home_remedies": r"4\.\s*🏠.*?Home Remedies:(.*?)(?=5\.|$)",
                "supportive_care": r"5\.\s*💊.*?Supportive Care:(.*?)(?=6\.|$)",
                "advice_next_steps": r"6\.\s*💡.*?Advice.*?Next Steps:(.*?)(?=7\.|$)",
                "first_aid": r"7\.\s*🚑.*?First-Aid:(.*?)(?=8\.|$)",
                "possible_causes": r"8\.\s*🧬.*?Possible Causes:(.*?)(?=9\.|$)",
                "friendly_summary": r"9\.\s*💬.*?Friendly Summary:(.*?)(?=10\.|$)"
            }

            for key, pattern in sections.items():
                match = re.search(pattern, analysis_text, re.DOTALL | re.IGNORECASE)
                if match:
                    content = match.group(1).strip()

                    if key in ["home_remedies", "supportive_care", "possible_causes"]:
                        # Parse as list
                        items = [item.strip() for item in re.split(r'[-•]\s*', content) if item.strip()]
                        result[key] = items[:5]  # Limit to 5 items
                    elif key == "symptom_details":
                        # Simple symptom parsing
                        result[key] = {"description": content[:500]}  # Limit length
                    else:
                        # Store as string
                        result[key] = content[:300]  # Limit length

            # If parsing fails, use the full text as friendly summary
            if not any(result.values()):
                result["friendly_summary"] = analysis_text[:200]

            return result

        except Exception as e:
            logger.error(f"Failed to parse analysis response: {e}")
            # Return default structure with original text
            return {
                "symptom_details": {"description": "Analysis parsing failed"},
                "recommended_specialty": "General Physician",
                "urgency_level": "Consult healthcare provider",
                "home_remedies": ["Consult healthcare provider"],
                "supportive_care": [],
                "advice_next_steps": "Please consult a healthcare provider",
                "first_aid": None,
                "possible_causes": ["Unable to determine"],
                "friendly_summary": "Please consult with a healthcare provider for proper assessment."
            }

    async def query_knowledge_base(self, query: str, max_results: int = 3) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Query the symptom knowledge base using RAG

        Args:
            query: User's medical query
            max_results: Maximum number of source documents to return

        Returns:
            Tuple of (answer, source_documents)
        """
        try:
            if not self.qa_chain:
                raise ValueError("QA chain not initialized - symptom database may not be available")

            logger.info(f"Processing RAG query: {query[:100]}...")

            # Run the query through RAG chain
            result = self.qa_chain.invoke({"query": query})

            answer = result["result"]
            source_docs = []

            # Process source documents
            for doc in result.get("source_documents", [])[:max_results]:
                source_docs.append({
                    "source": doc.metadata.get("source", "Unknown"),
                    "content": doc.page_content[:200],  # Limit content length
                    "metadata": doc.metadata
                })

            logger.info("RAG query completed successfully")
            return answer, source_docs

        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            raise

    async def direct_llm_query(self, query: str) -> str:
        """
        Direct query to LLM without RAG

        Args:
            query: User's query

        Returns:
            LLM response
        """
        try:
            if not self.llm:
                raise ValueError("LLM not initialized")

            logger.info("Processing direct LLM query...")

            # Add medical context to the query
            medical_prompt = f"""
As a medical AI assistant, please provide helpful information about the following query. 
Remember to always recommend consulting healthcare professionals for medical advice.

Query: {query}

Response:"""

            response = await self._direct_llm_query(medical_prompt)

            logger.info("Direct LLM query completed")
            return response

        except Exception as e:
            logger.error(f"Direct LLM query failed: {e}")
            raise

    async def _direct_llm_query(self, prompt: str) -> str:
        """Internal method for direct LLM queries"""
        try:
            # Since HuggingFaceEndpoint might not be async, run in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self.llm.invoke, prompt)
            return response
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            raise