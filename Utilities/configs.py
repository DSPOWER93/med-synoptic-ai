chatbot_configs ={
    "default_embedding_model": "amazon.titan-embed-text-v1" ,
    "query_refine_token": 64,
    
}

# bedrock model Dict 
bedrock_model_name_dict = {
    'llama3_70b': "meta.llama3-70b-instruct-v1:0",
    'mistral_8x7':"mistral.mixtral-8x7b-instruct-v0:1"
}


pat_entity_template = """{
    "patient_history": {
        "Diabetes": False,  # Boolean True or False
        "Hypertension": False,  # Boolean True or False
        "Tobacco": False,  # Boolean True or False
        "alcohol_consumption": False,  # Boolean True or False
        "genetic_conditions": [],  # List of genetic conditions observed
        "allergies": [],  # List of past allergies in brief one sentence (for each allergy)
        "past_surgeries": [],  # List of past surgeries
        "family_history": [],  # List of family history of any health complications
        "chronic_illness": [],  # List of chronic illness mentioned
        "other_health_conditions": []  # List of other health conditions not covered above
    },
    "patient_education": []  # List of patient's educational achievements
}"""

bio_llm_promt_dict ={

    "patient_entity_extraction": f"""
                Answer only in python dictionary format based on information provided from patient's discharge summary.
                populate answer only when there is info available else keep the field emplty or mark False as applicable.
                
                
                Reply only and only in the following format, without adding any other value (even something like ```python,
                The redendered text will be used as json in downstream application).
                
                Required Format :: >>

                {pat_entity_template}

               Discharge Summary is as follows >> """,

        "meddra_coding":"""
                Answer only in following nest dictionary format as per MEDDRA standards for the adverse event being mentioned below from given context. 
                Add multiple dictionary's nested in dictionary if required, answer only in the following format:: >>

                {
                "AE_EVENT_1": {
                "High-Level Term (HLT)": "",
                "High-Level Group Term (HLGT)": "",
                "System Organ Class (SOC)": "",
                "Preferred Term (PT)": "",
                "Low-Level Term (LLT)": ""
                },

                "AE_EVENT_2": {} # multiple AE events as follows

                }

                Context>>
            """,
        
        "Medical_billing_codes":"""
            From given text, we are looking to map clinical coding for three fields, 
            Patient's diagnosis, procedure & prescription to ease the process of their medical billing.
            From given extract can you map ICD-10 codes for patient diagnosis, CPT or HCPCS for procedure
            and NDC codes for patient prescriptions. answer in only following format >>

            {
            "diagnosis_codes": {}, # ICD-10 Diagnosis codes in format {"code1": "Description","code12": "Description"...........}
            "procedure_codes": {}, # CPT/ HCPCS procedure codes in format {"code1": "Description","code12": "Description"...........}
            "prescription_codes": {} # NDC codes in format {"code1": "Description","code12": "Description"...........}
            }

            context >>>

            """
        
}

markdown_heading_pat_entity = {
    "Patient_entity":  """This web-app prototype harnesses large language models (LLMs) to automatically extract key patient entities 
    from hospital discharge summaries. It identifies critical details like chronic conditions (e.g., diabetes, hypertension), allergies, 
    family history, and more, outputting them in structured JSON format.
    Users can use two methods <br><br>
    1. Submit Discharge summary manually.<br>
    2. Auto Generate Discharge Summary.<br><br>
    The core goal is to auto-update patient databases efficiently, enabling better-informed 
    treatments and personalized care from healthcare professionals. <br><br>
    Powered By: <a href="https://huggingface.co/aaditya/Llama3-OpenBioLLM-70B"> Llama3-OpenBioLLM-70B</a> ||
                 <a href="https://huggingface.co/openai/gpt-oss-120b"> GPT-OSS-120b </a><br>
    source code: <a href="https://github.com/DSPOWER93/med-synoptic-ai/">Repo</a>""" ,

    "MedDRA_Coder": """MedDRA AE Coder streamlines adverse event reporting by formatting your notes into compliant MedDRA standards,
    ensuring global consistency for safety data in pharma and healthcare. With an intuitive web-app interface, 
    simply submit your own adverse event details or auto-generate a synthetic note for instant testing—get 
    structured JSON outputs that make compliance effortless and error-free. Empower your team to prioritize 
    patient safety and regulatory accuracy without the hassle of manual coding.<br><br>
    Powered By: <a href="https://huggingface.co/aaditya/Llama3-OpenBioLLM-70B"> Llama3-OpenBioLLM-70B</a> ||
                 <a href="https://huggingface.co/openai/gpt-oss-120b"> GPT-OSS-120b </a><br>
    source code: <a href="https://github.com/DSPOWER93/med-synoptic-ai/">Repo</a>""",

    "clincal_coder": """Discover Map Clinical Codes, a user-friendly Web app that simplifies medical coding 
    by transforming doctor's SOAP notes into clear markdown summaries with suggested ICD-10 diagnosis codes 
    and CPT/HCPCS procedure codes. Powered by RAG technology, it pulls from a vector database to provide accurate, 
    explainable matches, saving coders time and reducing errors. Choose to input your own SOAP note or auto-generate 
    a synthetic one for quick testing—results appear instantly in an interactive format. Ideal for healthcare 
    pros seeking efficient, reliable coding support. <br><br>
    <i>(Diagnosis & Procedure notes needs to be tagged uppercase marking as DIAGNSOSIS:/PROCEDURE: for AI model to focus on specific content.)</i> <br><br>
    Powered By: <a href="https://huggingface.co/openai/gpt-oss-120b"> GPT-OSS-120b </a><br>
    source code: <a href="https://github.com/DSPOWER93/med-synoptic-ai/">Repo</a>""",

}

# Create prompt template
system_prompt = (
    """
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know.keep the answer concise"
    "\n\n"
    "{context}"
    """
)

sql_prompt  = (
    """
    You are asked to respond in PrestoDB SQL following ANSI SQL compliance, based on query asked by user.
    Name of database.table will be available in the RAG context. 
    Respond only in SQL query using database.table name from RAG context
    "\n\n"
    "{context}"
    """
)

analysis_prompt = (
    """
    You will receive a JSON object with two keys:

    1. "query" – a description of the information being requested.  
    2. "data"  – a pandas DataFrame serialized to JSON (orient="records").

    Task:  
    • Examine the content under "data".  
    • Identify and describe only clear, distinct patterns, trends, or anomalies. Ignore insignificant or random variations.  
    • Base your commentary solely on the data; use "query" only for context.  
    • Output your analysis in Markdown format only (headings, bullet points, and short paragraphs are acceptable).

    Do not return any other file type or formatting.
    """
)

insight_bridge = (

    """
    Meet :blue[**InsightBridge**], a simple way to extract answers from your data.
    Upload your dataset, choose your source and index from a dropdown, 
    and start asking questions to data in straight forward sentence.

    InsightBridge makes exploratory data analysis convinient in following ways:

    - Easy UI to add data to datasource.
    - User can query data with language Verbatim.   
    - Efficiently retrieves precise records and generates concise, readable summaries, 
    accelerating data exploration and minimizing user scoping time.
  
    Manage your workspace to add, query or delete datasets as convenient. 
    InsightBridge keeps data conversations fast, friendly, and secure.
    """
)