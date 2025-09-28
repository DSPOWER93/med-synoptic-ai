env_config= {
    "athena_configs":{
        "database":"streamlit_demo_tables",
        "s3_path": "s3://streamlit-tables-db/output_tables/",
        "region": 'us-east-1',
        "workgroup": "primary"
    }

}

s3_vector_db_config = {
    "model":"text-embedding-3-small",
    "dimensions":1536,
    "region_name":"us-east-1",
    "bucket_name": "test-vector-bucket",
}


eda_llm_config = {
    "model":"text-embedding-3-small",
    "dimensions":1536,
    "region_name":"us-east-1",
    "bucket_name":"gen-ai-llm-bucket",
}

hf_inference_model = {
    "open_bio_llm":"aaditya/Llama3-OpenBioLLM-70B:nebius",
    "open-ai-oss":"openai/gpt-oss-120b:nebius",
    "model_config":{
        "temprature":0.5,
        "max_tokens":1024
    }
    }


key_index_mapping= {
    "procedure": "cpt-hcpcs-test",
    "diagnosis": "icd-10-test"
 }

replace_text_dict= {
    "diagnoses":"diagnosis",
    "DIAGNOSES":"DIAGNOSIS"
    }

sample_soap_note_path = "datasets/sample_medical_script.csv"
AE_notes_path="datasets/Synthethic_AE_data.csv"


