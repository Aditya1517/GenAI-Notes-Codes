import streamlit as st
import os
import uuid
import io
import shutil
import re
import json
from collections import Counter
from pypdf import PdfReader
from dotenv import load_dotenv
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Load environment variables (e.g., GOOGLE_API_KEY)
load_dotenv()

# --- Configuration Constants ---
COLLECTION_NAME = "resume_ranker_collection"
TEMP_DOC_DIR = "temp_ranked_resumes"

# Using Google's text-embedding-004 model (768 dimensions)
EMBEDDING_MODEL = "models/text-embedding-004"
DIMENSION = 768

# Common technical skills keywords for better matching
COMMON_SKILLS = [
    'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust', 'php', 'ruby',
    'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring', 'express',
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins', 'ci/cd',
    'machine learning', 'deep learning', 'ai', 'nlp', 'computer vision', 'pytorch', 'tensorflow',
    'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch',
    'git', 'github', 'gitlab', 'agile', 'scrum', 'devops', 'microservices',
    'rest api', 'graphql', 'grpc', 'kafka', 'rabbitmq', 'redis',
    'linux', 'unix', 'bash', 'shell scripting', 'powershell'
] 

# --- API Key Check ---
if "GOOGLE_API_KEY" not in os.environ:
    st.error("🛑 **GOOGLE_API_KEY** environment variable not set. This is required for Google embeddings. Please set it and restart.")
    st.stop()

# Ensure temporary directory exists
os.makedirs(TEMP_DOC_DIR, exist_ok=True)

# --- Initialization Functions ---

def get_milvus_client():
    """Initializes and returns the Milvus client (connecting to standalone Docker server)."""
    with st.spinner("Connecting to Milvus server (http://localhost:19530)..."):
        try:
            # Connect to the Docker server
            client = MilvusClient(uri="http://localhost:19530")
            return client
        except Exception as e:
            st.error(f"Failed to connect to Milvus server. Is Docker running? Error: {e}")
            return None

@st.cache_resource(show_spinner="Loading Google Embedding Model...")
def get_embedding_model():
    """Loads the Google Generative AI embedding model (cached)."""
    try:
        # Uses the GOOGLE_API_KEY from environment
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        return embeddings
    except Exception as e:
        st.error(f"Failed to load embedding model. Ensure GOOGLE_API_KEY is correct. Error: {e}")
        return None


def setup_milvus_collection(client: MilvusClient):
    """Checks and creates the Milvus collection and index if necessary."""
    try:
        if client.has_collection(COLLECTION_NAME):
            return True
    except Exception as e:
        st.error(f"Error checking collection: {e}")
        return False

    st.info("Milvus collection not found. Creating a new collection for resumes...")
    
    try:
        # 1. Define Fields (Schema)
        fields = [
            FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, max_length=100, auto_id=False, description="Document ID"),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=500, description="Original file name"),
            FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=500, description="Local path to file bytes"),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535, description="Extracted text content snippet"),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION, description="Vector embedding (Google text-embedding-004)"),
        ]
        schema = CollectionSchema(fields=fields, description="Collection for Resume Vectors")
        
        # 2. Create Collection
        client.create_collection(
            collection_name=COLLECTION_NAME,
            dimension=DIMENSION,
            metric_type="COSINE",  # Cosine Similarity
            consistency_level="Strong",
            schema=schema
        )

        # 3. ✨ CREATE INDEX (FIX FOR "index not found" ERROR) ✨
        st.info("Creating index on 'embedding' field (AUTOINDEX)...")
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",  # Lets Milvus pick the best index
            metric_type="COSINE"
        )
        client.create_index(
            collection_name=COLLECTION_NAME,
            index_params=index_params
        )

        st.success(f"Milvus Collection '{COLLECTION_NAME}' and index created successfully.")
        return True
    except Exception as e:
        st.error(f"Failed to create collection or index: {e}")
        return False

def get_text_from_pdf(file: UploadedFile) -> str:
    """Extracts text from a Streamlit UploadedFile object (assuming PDF)."""
    try:
        # Use io.BytesIO to treat the uploaded file as a file-like object
        pdf_file = io.BytesIO(file.getvalue())
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.warning(f"Could not read PDF file '{file.name}'. Error: {e}")
        return ""

def save_uploaded_file_locally(uploaded_file, directory: str) -> str:
    """Saves the uploaded file to a local temporary directory with a unique name."""
    unique_filename = f"{uuid.uuid4()}_{uploaded_file.name}"
    file_path = os.path.join(directory, unique_filename)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    return file_path

def cleanup_temp_files():
    """Removes all temporary files from the temp directory."""
    if os.path.exists(TEMP_DOC_DIR):
        shutil.rmtree(TEMP_DOC_DIR)
        os.makedirs(TEMP_DOC_DIR, exist_ok=True)

def get_collection_count(client: MilvusClient) -> int:
    """Gets the number of documents in the collection."""
    try:
        if not client.has_collection(COLLECTION_NAME):
            return 0
        
        result = client.query(
            collection_name=COLLECTION_NAME,
            filter="pk != ''",
            output_fields=["pk"],
            limit=16384 
        )
        return len(result)
    except Exception as e:
        try:
            stats = client.get_collection_stats(COLLECTION_NAME)
            return stats.get('row_count', 0)
        except:
            return 0

def ingest_documents(client: MilvusClient, embeddings_model: GoogleGenerativeAIEmbeddings, uploaded_files: list):
    """Processes uploaded files, generates embeddings, and inserts them into Milvus."""
    if not uploaded_files:
        st.warning("Please upload documents first.")
        return False
    
    try:
        # Clear existing documents and temp files
        cleanup_temp_files()
        
        if client.has_collection(COLLECTION_NAME):
            client.drop_collection(COLLECTION_NAME)
        
        if not setup_milvus_collection(client):
            return False
        
        records_to_insert = []
        
        with st.spinner(f"Processing {len(uploaded_files)} resumes and creating embeddings..."):
            texts_to_embed = []
            file_metadata = []
            
            for file in uploaded_files:
                file_path = save_uploaded_file_locally(file, TEMP_DOC_DIR)
                text_content = get_text_from_pdf(file)
                
                if not text_content or len(text_content.strip()) < 10:
                    st.warning(f"Skipping **{file.name}** - insufficient text content.")
                    try:
                        os.remove(file_path)
                    except:
                        pass
                    continue

                texts_to_embed.append(text_content)
                file_metadata.append({
                    "file_name": file.name,
                    "file_path": file_path,
                    "content_preview": text_content[:]
                })

            if not texts_to_embed:
                st.warning("No valid text content found in uploaded files to index.")
                return False
                
            embeddings_list = embeddings_model.embed_documents(texts_to_embed)
            
            for i, embedding in enumerate(embeddings_list):
                meta = file_metadata[i]
                record = {
                    "pk": str(uuid.uuid4()),
                    "file_name": meta["file_name"],
                    "file_path": meta["file_path"],
                    "content": meta["content_preview"],
                    "embedding": embedding
                }
                records_to_insert.append(record)

            # 4. Insert into Milvus
            if records_to_insert:
                client.insert(
                    collection_name=COLLECTION_NAME,
                    data=records_to_insert
                )
                
                # 5. ✨ LOAD COLLECTION (FIX FOR "collection not loaded" ERROR) ✨
                with st.spinner("Loading collection into memory..."):
                    client.load_collection(collection_name=COLLECTION_NAME)
                
                st.success(f"✅ Successfully processed, indexed, and loaded **{len(records_to_insert)}** documents.")
                return True
            
        return False
        
    except Exception as e:
        st.error(f"Error during document ingestion: {e}")
        return False

def search_documents(client: MilvusClient, embeddings_model: GoogleGenerativeAIEmbeddings, query: str, top_k: int) -> list:
    """Searches Milvus using the query embedding and returns top results."""
    
    try:
        # 1. Embed the query
        with st.spinner("Generating query embedding..."):
            query_vector = embeddings_model.embed_query(query)

        # 2. Perform the Milvus search
        with st.spinner(f"Searching Milvus for the top {top_k} matches..."):
            search_results = client.search(
                collection_name=COLLECTION_NAME,
                data=[query_vector],
                limit=top_k,
                output_fields=["file_name", "file_path", "content"]
            )

        # 3. Extract and format results
        formatted_results = []
        if search_results and len(search_results[0]) > 0:
            for hit in search_results[0]:
                similarity_score = 1 - hit['distance']
                file_path = hit['entity']['file_path']
                file_name = hit['entity']['file_name']
                
                try:
                    with open(file_path, "rb") as file:
                        file_bytes = file.read()
                except FileNotFoundError:
                    st.warning(f"File missing for {file_name}. Cannot provide download.")
                    continue
                except Exception as e:
                    st.warning(f"Error reading file {file_name}: {e}")
                    continue

                formatted_results.append({
                    "score": similarity_score,  # Semantic similarity score
                    "file_name": file_name,
                    "file_bytes": file_bytes,
                    "content_preview": hit['entity'].get('content', '')
                })
        
        # Don't sort here - will be sorted after validation
        return formatted_results
        
    except Exception as e:
        st.error(f"Error during search: {e}")
        return []

# --- Non-LLM Validation Functions ---

def extract_keywords_from_text(text: str) -> set:
    """Extracts keywords from text, removing common stop words."""
    # Common stop words to filter out
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must',
        'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'years', 'year', 'experience', 'skill', 'skills', 'required', 'need', 'needed'
    }
    
    # Convert to lowercase and split
    words = re.findall(r'\b[a-z]+\b', text.lower())
    # Filter out stop words and short words
    keywords = {w for w in words if len(w) > 2 and w not in stop_words}
    return keywords

def extract_experience_years_from_text(text: str) -> float:
    """Extracts total years of experience from resume text using regex patterns.
    Handles both years and months, converting months to decimal years."""
    if not text:
        return 0.0
    
    years_found = []
    months_found = []
    
    # Priority 1: Look for explicit experience mentions with years and/or months
    # e.g., "5 years 3 months", "2 years of experience", "6 months", "18 months experience"
    
    # Pattern 1: "X years Y months" or "X years, Y months"
    years_months_pattern = r'(\d+\.?\d*)\s*(?:years?|yrs?)(?:\s*[,and]?\s*(\d+)\s*(?:months?|mos?))?'
    matches = re.findall(years_months_pattern, text, re.IGNORECASE)
    for year_match, month_match in matches:
        try:
            years = float(year_match) if year_match else 0
            months = float(month_match) if month_match else 0
            total_years = years + (months / 12.0)
            if 0.25 <= total_years <= 50:
                years_found.append(total_years)
        except:
            pass
    
    # Pattern 2: Standalone months (e.g., "6 months", "18 months of experience")
    months_patterns = [
        r'(\d+\.?\d*)\s*(?:months?|mos?)\s*(?:of\s*)?(?:work|professional|industry|experience)',
        r'(?:work|professional|industry|experience)[:\s]*(\d+\.?\d*)\s*(?:months?|mos?)',
        r'(\d+)\+?\s*(?:months?|mos?)',
    ]
    
    for pattern in months_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                months = float(match)
                years = months / 12.0
                if 0.25 <= years <= 50:
                    months_found.append(years)
            except:
                pass
    
    # Pattern 3: Years only (without months)
    explicit_patterns = [
        r'(\d+\.?\d*)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:work|professional|industry|total)\s*(?:experience)?',
        r'(?:work|professional|industry|total)\s*experience[:\s]*(\d+\.?\d*)\s*(?:years?|yrs?)',
        r'(\d+\.?\d*)\s*(?:years?|yrs?)\s*(?:of\s*)?experience',
        r'experience[:\s]*(\d+\.?\d*)\s*(?:years?|yrs?)',
    ]
    
    for pattern in explicit_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                years = float(match)
                if 0.25 <= years <= 50:
                    years_found.append(years)
            except:
                pass
    
    # Combine all found values
    all_experience = years_found + months_found
    
    # If explicit experience found, return it (don't parse dates)
    if all_experience:
        return round(max(all_experience), 2)
    
    # Priority 2: Parse date ranges from work experience sections only
    # Focus on sections that likely contain work experience
    experience_keywords = ['experience', 'work', 'employment', 'professional', 'career', 'intern', 'internship']
    text_lower = text.lower()
    
    # Find experience section indices
    experience_sections = []
    for keyword in experience_keywords:
        indices = [m.start() for m in re.finditer(r'\b' + keyword + r'\b', text_lower)]
        for idx in indices:
            # Extract text around this keyword (next 2000 chars likely contains dates)
            end_idx = min(idx + 2000, len(text))
            experience_sections.append(text[idx:end_idx])
    
    # If no experience section found, use full text but be more careful
    if not experience_sections:
        experience_sections = [text]
    
    from datetime import datetime
    current_year = datetime.now().year
    date_ranges = []
    seen_ranges = set()  # Track seen date ranges to avoid duplicates
    
    # Month name to number mapping
    month_map = {
        'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
        'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6,
        'jul': 7, 'july': 7, 'aug': 8, 'august': 8, 'sep': 9, 'september': 9,
        'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12
    }
    
    # Pattern 1: Month-Year format (most reliable for work experience)
    # e.g., "Jan 2020 - Dec 2023", "August 2023 - October 2023"
    month_year_pattern = r'((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december))\s+(\d{4})\s*[-–—]\s*((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december))\s+(\d{4}|present|current|now)'
    
    for section in experience_sections:
        month_matches = re.findall(month_year_pattern, section, re.IGNORECASE)
        for start_month_str, start_year_str, end_month_str, end_year_str in month_matches:
            try:
                start_month = month_map.get(start_month_str.lower(), 1)
                start_year = int(start_year_str)
                
                if end_year_str.lower() in ['present', 'current', 'now']:
                    end_month = datetime.now().month
                    end_year = current_year
                else:
                    end_month = month_map.get(end_month_str.lower(), 12)
                    end_year = int(end_year_str)
                
                # Create a unique key for this date range to avoid duplicates
                range_key = (start_year, start_month, end_year, end_month)
                if range_key in seen_ranges:
                    continue  # Skip duplicate
                seen_ranges.add(range_key)
                
                # Calculate exact months difference
                start_total_months = start_year * 12 + start_month
                end_total_months = end_year * 12 + end_month
                
                # Calculate months: Aug 2023 to Oct 2023
                # Aug (2023*12 + 8 = 24284), Oct (2023*12 + 10 = 24286)
                # Difference: 24286 - 24284 = 2, but actual months = 3 (Aug, Sep, Oct)
                month_diff = end_total_months - start_total_months
                
                # Calculate actual months worked (inclusive of start and end months)
                # If same month, minimum 1 month
                if month_diff == 0:
                    total_months = 1
                else:
                    # For Aug to Oct: month_diff = 2, but we need to count Aug, Sep, Oct = 3 months
                    # So: month_diff + 1 for inclusive count
                    total_months = month_diff + 1
                
                # Account for partial months: if working from start of one month to end of another,
                # we count almost full months (add small fraction for better accuracy)
                # This makes "Aug - Oct" ≈ 3-4 months instead of exactly 3
                if total_months <= 3:
                    # For short durations (1-3 months), add 0.5 month to account for partial months
                    total_months = total_months + 0.5
                
                # Convert to years (decimal), rounding to 2 decimal places
                # For Aug-Oct: (3 + 0.5) / 12 = 3.5 / 12 = 0.29 ≈ 0.3 years
                years = round(total_months / 12.0, 2)
                
                # Filter: accept if reasonable (0.25 to 50 years)
                if 0.25 <= years <= 50:
                    date_ranges.append(years)
            except:
                pass
    
    # Pattern 2: Simple year ranges, but filter out education dates
    # e.g., "2020 - 2024", but avoid "2021 - 2025" from education
    # Only use year ranges if we didn't find month-year patterns (to avoid double counting)
    if not date_ranges:
        year_pattern = r'(\d{4})\s*[-–—]\s*(\d{4}|present|current|now)'
        year_matches = re.findall(year_pattern, text, re.IGNORECASE)
        
        # Check context - if near "education" or "degree", skip
        for start_year_str, end_year_str in year_matches:
            try:
                start = int(start_year_str)
                if end_year_str.lower() in ['present', 'current', 'now']:
                    end = current_year
                else:
                    end = int(end_year_str)
                
                # Check if this year range was already captured by month-year pattern
                # e.g., if we already found "Aug 2023 - Oct 2023", don't count "2023 - 2023"
                range_key = (start, 1, end, 12)  # Approximate as full year range
                if any(abs(start - existing[0]) <= 1 and abs(end - existing[2]) <= 1 
                       for existing in seen_ranges):
                    continue  # Skip if similar range already found
                
                years = end - start
                
                # Find context around this date range
                date_str = f"{start_year_str}.*?{end_year_str}"
                match_obj = re.search(date_str, text, re.IGNORECASE)
                if match_obj:
                    context_start = max(0, match_obj.start() - 200)
                    context_end = min(len(text), match_obj.end() + 200)
                    context = text[context_start:context_end].lower()
                    
                    # Skip if in education context
                    education_keywords = ['education', 'degree', 'b.tech', 'btech', 'university', 'college', 'cgpa', 'gpa', 'graduation', 'student']
                    if any(keyword in context for keyword in education_keywords):
                        continue
                    
                    # Skip if duration matches typical education (3-5 years) and far in past
                    if 3 <= years <= 5 and start < 2015:
                        continue
                
                # Accept reasonable work experience durations
                if 0.25 <= years <= 50:
                    # Prefer recent dates (likely work) over old dates (likely education)
                    if end >= current_year - 5:  # Recent work experience
                        date_ranges.append(float(years))
                    elif years > 2:  # Longer durations are likely work
                        date_ranges.append(float(years))
            except:
                pass
    
    # Sum all date ranges found (person may have multiple jobs)
    # Remove duplicates to avoid double counting the same period
    if date_ranges:
        # Deduplicate: keep only unique date ranges (with small tolerance for floating point)
        unique_ranges = []
        for years in sorted(set(date_ranges)):
            # Check if this range is similar to an existing one (within 0.01 years tolerance)
            is_duplicate = False
            for existing in unique_ranges:
                if abs(years - existing) < 0.01:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_ranges.append(years)
        
        # Sum unique date ranges
        total_years = sum(unique_ranges)
        return round(min(total_years, 50), 2)  # Cap at 50 years, round to 2 decimals
    
    return 0.0

def extract_required_experience(query: str) -> float:
    """Extracts required years of experience from user query. Handles both years and months."""
    # Pattern 1: "X years Y months" or "X years, Y months"
    years_months_pattern = r'(\d+\.?\d*)\s*(?:years?|yrs?)(?:\s*[,and]?\s*(\d+)\s*(?:months?|mos?))?'
    matches = re.findall(years_months_pattern, query, re.IGNORECASE)
    for year_match, month_match in matches:
        try:
            years = float(year_match) if year_match else 0
            months = float(month_match) if month_match else 0
            total_years = years + (months / 12.0)
            if total_years > 0:
                return total_years
        except:
            pass
    
    # Pattern 2: Standalone months (e.g., "6 months", "18 months")
    months_patterns = [
        r'(\d+\.?\d*)\s*(?:months?|mos?)\s*(?:of\s*)?experience',
        r'(\d+)\+?\s*(?:months?|mos?)',
    ]
    
    for pattern in months_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        if matches:
            try:
                months = max(float(m) for m in matches)
                years = months / 12.0
                if years > 0:
                    return years
            except:
                pass
    
    # Pattern 3: Years only
    patterns = [
        r'(\d+\.?\d*)\s*(?:years?|yrs?)\s*(?:of\s*)?experience',
        r'(\d+)\+?\s*(?:years?|yrs?)',
        r'(\d+)\s*(?:years?|yrs?)\s*(?:experience|exp)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        if matches:
            try:
                return max(float(m) for m in matches)
            except:
                pass
    
    return 0.0

def calculate_keyword_match_score(resume_text: str, query_keywords: set) -> float:
    """Calculates keyword matching score between resume and query."""
    if not query_keywords:
        return 0.0
    
    resume_keywords = extract_keywords_from_text(resume_text)
    
    # Exact matches
    exact_matches = len(query_keywords.intersection(resume_keywords))
    
    # Partial matches (substring matching for technical terms)
    partial_matches = 0
    for query_word in query_keywords:
        for resume_word in resume_keywords:
            if query_word in resume_word or resume_word in query_word:
                partial_matches += 0.5
                break
    
    # Normalize by number of query keywords
    total_score = exact_matches + partial_matches
    max_score = len(query_keywords)
    
    return min(1.0, total_score / max_score) if max_score > 0 else 0.0

def calculate_keyword_density_score(resume_text: str, query_keywords: set) -> float:
    """Calculates keyword density score (how frequently keywords appear)."""
    if not query_keywords:
        return 0.0
    
    resume_lower = resume_text.lower()
    keyword_counts = {}
    
    for keyword in query_keywords:
        # Count occurrences (case-insensitive)
        count = len(re.findall(r'\b' + re.escape(keyword.lower()) + r'\b', resume_lower))
        keyword_counts[keyword] = count
    
    # Normalize: more occurrences = higher score (capped)
    total_occurrences = sum(keyword_counts.values())
    max_occurrences = len(query_keywords) * 5  # Assume 5+ mentions is excellent
    
    return min(1.0, total_occurrences / max_occurrences) if max_occurrences > 0 else 0.0

def calculate_experience_match_score(resume_text: str, required_years: float) -> float:
    """Calculates experience matching score."""
    if required_years == 0:
        return 0.5  # Neutral score if no requirement
    
    candidate_years = extract_experience_years_from_text(resume_text)
    
    if candidate_years == 0:
        return 0.0  # No experience found
    
    if candidate_years >= required_years:
        return 1.0  # Meets or exceeds requirement
    else:
        # Proportional score if below requirement
        return candidate_years / required_years

def calculate_combined_score(
    semantic_score: float,
    keyword_match_score: float,
    keyword_density_score: float,
    experience_score: float,
    weights: dict = None
) -> float:
    """Calculates combined ranking score from multiple factors."""
    if weights is None:
        weights = {
            'semantic': 0.40,      # Semantic similarity (embeddings)
            'keyword_match': 0.30,  # Keyword matching
            'keyword_density': 0.15, # Keyword frequency
            'experience': 0.15     # Experience matching
        }
    
    combined = (
        semantic_score * weights['semantic'] +
        keyword_match_score * weights['keyword_match'] +
        keyword_density_score * weights['keyword_density'] +
        experience_score * weights['experience']
    )
    
    return min(1.0, max(0.0, combined))  # Clamp between 0 and 1

def rank_documents_with_validation(
    candidates: list,
    query: str,
    query_keywords: set = None
) -> list:
    """Ranks documents using multiple validation factors."""
    if query_keywords is None:
        query_keywords = extract_keywords_from_text(query)
    
    required_years = extract_required_experience(query)
    
    for candidate in candidates:
        resume_text = candidate.get('content_preview', '')
        semantic_score = candidate.get('score', 0.0)
        
        # Calculate various scores
        keyword_match = calculate_keyword_match_score(resume_text, query_keywords)
        keyword_density = calculate_keyword_density_score(resume_text, query_keywords)
        experience_match = calculate_experience_match_score(resume_text, required_years)
        
        # Combined score
        combined_score = calculate_combined_score(
            semantic_score,
            keyword_match,
            keyword_density,
            experience_match
        )
        
        # Store individual scores for display
        candidate['validation_scores'] = {
            'semantic': semantic_score,
            'keyword_match': keyword_match,
            'keyword_density': keyword_density,
            'experience': experience_match,
            'combined': combined_score,
            'required_years': required_years,
            'candidate_years': extract_experience_years_from_text(resume_text)
        }
        candidate['final_score'] = combined_score
    
    # Sort by combined score
    candidates.sort(key=lambda x: x.get('final_score', 0), reverse=True)
    
    return candidates

# --- Streamlit UI and Main Logic ---

def highlight_keywords(text: str, keywords: list) -> str:
    """Highlights keywords in the text using HTML markup."""
    if not keywords:
        return text
    
    # Sort keywords by length (longest first) to avoid partial matches
    sorted_keywords = sorted(keywords, key=len, reverse=True)
    
    # Create pattern for case-insensitive matching
    pattern = '|'.join(re.escape(keyword.strip()) for keyword in sorted_keywords if keyword.strip())
    
    if not pattern:
        return text
    
    # Highlight matches with yellow background
    highlighted_text = re.sub(
        f'({pattern})',
        r'<mark style="background-color: #FFEB3B; padding: 2px 4px; border-radius: 3px;">\1</mark>',
        text,
        flags=re.IGNORECASE
    )
    
    return highlighted_text

def extract_keywords_from_query(query: str) -> list:
    """Extracts keywords from the query string (comma-separated or space-separated)."""
    # Split by comma first, then by spaces to get individual keywords
    keywords = []
    for item in query.split(','):
        item = item.strip()
        # Split by spaces and add non-empty parts
        keywords.extend([k.strip() for k in item.split() if k.strip()])
    return keywords

def display_results(results, keywords: list = None):
    """Displays the ranked results with download buttons."""
    st.subheader(f"🏆 Top {len(results)} Ranked Candidates")
    st.caption("Results ranked by: Semantic Similarity (40%) + Keyword Match (30%) + Keyword Density (15%) + Experience (15%)")
    st.markdown("---")

    for i, result in enumerate(results):
        # Get final combined score
        final_score = result.get('final_score', result.get('score', 0))
        validation_scores = result.get('validation_scores', {})
        
        final_percent = max(0, min(100, final_score * 100))
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**{i+1}. Candidate File:** `{result['file_name']}`")
            
            # Score display
            col_score1, col_score2, col_score3, col_score4 = st.columns(4)
            
            with col_score1:
                if final_percent >= 70:
                    color = "#10B981"  # Green
                elif final_percent >= 50:
                    color = "#F59E0B"  # Orange
                else:
                    color = "#EF4444"  # Red
                st.metric("Final Score", f"{final_percent:.1f}%")
                st.caption("Combined")
            
            with col_score2:
                semantic_pct = validation_scores.get('semantic', result.get('score', 0)) * 100
                st.metric("Semantic", f"{semantic_pct:.1f}%")
            
            with col_score3:
                keyword_pct = validation_scores.get('keyword_match', 0) * 100
                st.metric("Keywords", f"{keyword_pct:.1f}%")
            
            with col_score4:
                exp_pct = validation_scores.get('experience', 0) * 100
                st.metric("Experience", f"{exp_pct:.1f}%")
            
            # Detailed validation breakdown
            if validation_scores:
                with st.expander("📊 Detailed Validation Scores"):
                    col_v1, col_v2 = st.columns(2)
                    
                    with col_v1:
                        st.markdown("**Scoring Breakdown:**")
                        st.progress(validation_scores.get('semantic', 0))
                        st.caption(f"Semantic Similarity: {validation_scores.get('semantic', 0)*100:.1f}%")
                        
                        st.progress(validation_scores.get('keyword_match', 0))
                        st.caption(f"Keyword Match: {validation_scores.get('keyword_match', 0)*100:.1f}%")
                        
                        st.progress(validation_scores.get('keyword_density', 0))
                        st.caption(f"Keyword Density: {validation_scores.get('keyword_density', 0)*100:.1f}%")
                    
                    with col_v2:
                        st.markdown("**Experience Analysis:**")
                        req_years = validation_scores.get('required_years', 0)
                        cand_years = validation_scores.get('candidate_years', 0)
                        
                        if req_years > 0:
                            st.info(f"**Required:** {req_years} years")
                            st.info(f"**Candidate Has:** {cand_years:.1f} years")
                            
                            if cand_years >= req_years:
                                st.success("✅ Meets experience requirement")
                            elif cand_years > 0:
                                st.warning(f"⚠️ {req_years - cand_years:.1f} years short")
                            else:
                                st.error("❌ No experience found")
                        else:
                            st.info("No experience requirement specified")
            
            with st.expander("📄 View Extracted Content Preview"):
                preview_text = result['content_preview'][:]
                if keywords:
                    highlighted_text = highlight_keywords(preview_text, keywords)
                    st.markdown(highlighted_text, unsafe_allow_html=True)
                else:
                    st.text(preview_text)

        with col2:
            st.download_button(
                label="📥 Download",
                data=result['file_bytes'],
                file_name=result['file_name'],
                mime="application/pdf",
                type="primary",
                key=f"download_{i}_{result['file_name']}",
                help="Click to download the original resume"
            )
        st.markdown("---")

def main():
    st.set_page_config(
        page_title="Milvus Resume Ranker",
        page_icon="💼",
        layout="wide",
    )
    
    st.title("💼 AI Resume Ranker")
    st.caption("Upload multiple resumes (PDFs) and rank the best candidates using semantic search powered by Milvus + Google Embeddings")

    # 1. Initialize Milvus and Model
    milvus_client = get_milvus_client()
    embeddings_model = get_embedding_model()

    if milvus_client is None or embeddings_model is None:
        st.stop()
        return

    # --- Sidebar for Upload and Configuration ---
    with st.sidebar:
        st.header("📁 Upload & Index Resumes")
        
        uploaded_files = st.file_uploader(
            "Select multiple PDF resumes:",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF resume files to index"
        )
        
        st.caption(f"📊 Uploaded: {len(uploaded_files) if uploaded_files else 0} file(s)")
        
        if st.button("🚀 Process & Index Documents", type="primary", use_container_width=True):
            if uploaded_files:
                success = ingest_documents(milvus_client, embeddings_model, uploaded_files)
                if success:
                    st.rerun()  # Refresh to update count
            else:
                st.warning("⚠️ Please upload files before processing.")

        st.divider()
        
        count = get_collection_count(milvus_client)
        st.metric(label="📑 Total Resumes Indexed", value=count)
        st.caption(f"Vector Dimension: {DIMENSION}D")
        st.caption(f"Similarity Metric: Cosine")
        
    # --- Main Search Area ---
    count = get_collection_count(milvus_client)
    
    if count == 0:
        st.info("👈 **Get started by uploading resumes in the sidebar and clicking 'Process & Index Documents'.**")
        st.markdown("""
        ### How it works:
        1. **Upload** PDF resumes using the sidebar
        2. **Process** the resumes to extract text and create embeddings
        3. **Search** by entering keywords (skills, technologies) and experience in numbers (e.g., "Python, AWS, 5 years")
        4. **Download** the top-ranked candidates
        """)
        return

    st.header("🔍 Search and Ranking Criteria")
    
    search_query = st.text_area(
        "Enter Keywords and Experience (Numbers Only):",
        placeholder="e.g., Python, machine learning, PyTorch, AWS, 5 years, 3 years, distributed systems",
        height=100,
        help="Enter only keywords (skills, technologies) separated by commas and experience in numbers (e.g., '5 years', '3 years'). No full sentences or descriptions."
    )

    col1, col2 = st.columns([2, 1])
    
    with col1:
        top_k_slider = st.slider(
            "Number of Top Candidates to Retrieve", 
            min_value=1, 
            max_value=min(20, count), 
            value=min(5, count),
            help="Select how many top-matching resumes to display"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        search_button = st.button(f"🔎 Find Top {top_k_slider} Candidates", type="primary", use_container_width=True)

    if search_button:
        if not search_query.strip():
            st.warning("⚠️ Please enter keywords and experience (numbers only).")
        else:
            # Step 1: Semantic search to get candidate pool
            results = search_documents(milvus_client, embeddings_model, search_query, top_k_slider * 2)
            
            if results:
                # Step 2: Apply multi-factor validation and ranking
                with st.spinner("Applying multi-factor validation (keyword matching, experience, density)..."):
                    query_keywords = extract_keywords_from_text(search_query)
                    ranked_results = rank_documents_with_validation(results, search_query, query_keywords)
                    # Take top K after ranking
                    ranked_results = ranked_results[:top_k_slider]
                
                # Extract keywords for highlighting
                keywords = extract_keywords_from_query(search_query)
                display_results(ranked_results, keywords)
            else:
                st.warning("No matching resumes found. Try refining your search criteria.")

if __name__ == "__main__":
    main()