import json
import boto3
import os
import re
import random
import numpy as np
import pandas as pd
import awswrangler as wr
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_pinecone import PineconeVectorStore  # Updated import
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI as lc_OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from botocore.exceptions import ClientError, BotoCoreError
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Optional, Any
from pydantic import Field

from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from typing import List, Dict, Optional, Any
from pydantic import Field

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

import streamlit as st
import time



# ========================================================================================================================================================

def write_data_to_glue_db(df,
                          table_name,
                          database,
                          s3_path,
                          region_name):
    """
    Save a pandas DataFrame as a Parquet dataset in a specified S3 location and register/update it as a table in AWS Glue Catalog.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be saved.
    table_name : str
        The name of the Glue table to create or overwrite.
    database : str
        The Glue database where the table will be registered.
    s3_path : str
        The S3 bucket path where the Parquet dataset will be stored (e.g., 's3://my-bucket/data/').
    region_name : str
        The AWS region where the Glue Catalog and S3 bucket reside.

    Raises
    ------
    EnvironmentErrorLangchainPinecone
        If the required AWS credentials ('MASTER_ACCESS_KEY' and 'MASTER_SECRET_KEY') are not found in the environment variables.
    Exception
        Propagates any exception raised by boto3 or awswrangler during the write operation.

    Example
    -------
    >>> write_data_to_glue_db(
            df=my_dataframe,
            table_name="my_table",
            database="my_database",
            s3_path="s3://my-bucket/data/",
            region_name="us-east-1"
        )
    """
    access_key = os.getenv("MASTER_ACCESS_KEY")
    secret_key = os.getenv("MASTER_SECRET_KEY")
    if not access_key or not secret_key:
        raise EnvironmentError("AWS credentials not found in environment variables: 'MASTER_ACCESS_KEY' and/or 'MASTER_SECRET_KEY'.")

    # Create boto3 session with credentials
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region_name
    )
    # Save as dataset with Glue Catalog metadata using the session
    wr.s3.to_parquet(
        df=df,
        path=os.path.join(s3_path, table_name),
        dataset=True,
        database=database,
        table=table_name,
        mode="overwrite",
        boto3_session=session  # Pass the session with credentials
    )

# ========================================================================================================================================================

def read_athena_table(sql_query,env_configs):
    """
    Execute SQL query on AWS Athena and return results as a pandas DataFrame.
    
    This function creates an authenticated AWS session using credentials from environment
    variables and executes the provided SQL query against an Athena database. The database
    and region configuration are retrieved from the env_configs parameter.
    
    Args:
        sql_query (str): The SQL query to execute against the Athena database.
            Can include queries spanning multiple databases using fully qualified names.
        env_configs (dict): Configuration dictionary containing Athena settings.
            Must include:
            - athena_configs (dict): Nested dictionary with:
                - database (str): Target database name for query execution
                - region (str): AWS region where Athena service is located
    
    Returns:
        pandas.DataFrame: Query results as a pandas DataFrame. Returns empty DataFrame
            if query produces no results.
    
    Raises:
        EnvironmentError: If required AWS credentials ('MASTER_ACCESS_KEY' or 
            'MASTER_SECRET_KEY') are not found in environment variables.
        KeyError: If required configuration keys are missing from env_configs.
        ClientError: If AWS authentication fails or insufficient permissions.
        Exception: For other AWS Athena query execution errors (timeout, syntax errors, etc.).
    
    Environment Variables Required:
        MASTER_ACCESS_KEY (str): AWS access key ID for authentication
        MASTER_SECRET_KEY (str): AWS secret access key for authentication
    
    Example:
        >>> env_config = {
        ...     "athena_configs": {
        ...         "database": "my_database",
        ...         "region": "us-east-1"
        ...     }
        ... }
        >>> query = "SELECT * FROM table_name LIMIT 10"
        >>> df = read_athena_table(query, env_config)
        >>> print(df.shape)
        (10, 5)
    
    Note:
        - Ensure AWS credentials have appropriate permissions for Athena query execution
        - Query costs are based on data scanned; use LIMIT clauses when appropriate
        - For cross-database queries, use fully qualified table names (database.table)
    """
    access_key = os.getenv("MASTER_ACCESS_KEY")
    secret_key = os.getenv("MASTER_SECRET_KEY")

    if not access_key or not secret_key:
        raise EnvironmentError("AWS credentials not found in environment variables: 'MASTER_ACCESS_KEY' and/or 'MASTER_SECRET_KEY'.")

    # Create boto3 session with credentials
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=env_configs["athena_configs"]["region"]
    )
    
    # Read data using SQL query
    df_load = wr.athena.read_sql_query(
        sql= sql_query,
        database=env_configs["athena_configs"]["database"],
        boto3_session=session
    )

    return df_load


def distinct_values_dict(df, n_values=10):
    result = {}
    for col in df.columns:
        distinct_vals = df[col].dropna().unique()[:n_values]
        result[col] = distinct_vals.tolist()
    return result

# ========================================================================================================================================================

def generate_response_o4_mini(prompt, think=False, effort="medium", max_output_tokens=4000):
    """
    Generate response using OpenAI's o4-mini reasoning model via Responses API
    Tested and corrected based on official API structure
    """
    client = OpenAI()
    
    if think:
        # Enable reasoning mode with summary
        response = client.responses.create(
            model="o4-mini",
            reasoning={
                "effort": effort,
                "summary": "auto"  # This generates reasoning summaries
            },
            max_output_tokens=max_output_tokens,
            input=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        # Extract text content and reasoning from response.output
        text_content = ""
        reasoning_summary = ""
        
        # Iterate through output items to find text and reasoning
        for item in response.output:
            if getattr(item, "type", None) == "text":
                text_content = getattr(item, "text", "")
            elif getattr(item, "type", None) == "reasoning":
                reasoning_summary = getattr(item, "summary", "")
        
        return {
            "response": text_content,
            "thoughts": reasoning_summary
        }
    else:
        # Direct response without reasoning
        response = client.responses.create(
            model="o4-mini",
            max_output_tokens=max_output_tokens,
            input=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        # Extract text content from response.output
        for item in response.output:
            if getattr(item, "type", None) == "text":
                return getattr(item, "text", "No text output found")
        
        # Fallback: check for output_text attribute
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text.strip()
        
        return response # "No valid output returned by the model."

# ========================================================================================================================================================

# Alternative simpler version for testing
def generate_response_o4_mini_simple(prompt, max_output_tokens=1000):
    """
    Simplified version for testing - uses direct output_text if available
    """
    client = OpenAI()
    
    response = client.responses.create(
        model="o4-mini",
        max_output_tokens=max_output_tokens,
        input=[{
            "role": "user",
            "content": prompt
        }]
    )
    
    # Check for direct output_text first (as shown in DataCamp example)
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text.strip()
    
    # Otherwise iterate through output items
    for item in response.output:
        if getattr(item, "type", None) == "text":
            return getattr(item, "text", "")
    
    return "No valid output found"


def create_data_dict_prompt(json_string):
    
    dict_prompt= """You are required to data dictionary on basis of dict passed as an input. the doct contains column names and its unique values in form of
                        '{"column_name": [value1,value2 etc..]}' output to be generated as should be only in python dictionary only in specific mentioned format:                     
                        {"objective": Objective about the dataset,
                         "Columns": {"col1":{
                                            "desc": Description about the column based on its unique values,
                                            "column_data_type": column data type,
                                            "sample_values": ["value 1", "value 2", "value 3"] # total of upto three sample values
                                            }
                                    }
                        } dict values are as follows:: """+ json_string

    return dict_prompt


# ========================================================================================================================================================

def upload_to_pinecone(text,
                       index_name,
                       chunk_size,
                       chunk_overlap,
                       dimension=1536,
                       metric='cosine',
                       model="OpenAI",
                       embedding_model="text-embedding-3-small"):
    """
    Upload text corpus to Pinecone serverless vector database with text chunking and embeddings.
    
    Args:
        text (str): Text corpus to upload and vectorize
        index_name (str): Name of the Pinecone index to create/use
        chunk_size (int): Size of text chunks for splitting
        chunk_overlap (int): Character overlap between consecutive chunks
        dimension (int, optional): Vector embedding dimension. Defaults to 1536.
        metric (str, optional): Distance metric for similarity search. Defaults to 'cosine'.
        model (str, optional): Embedding model provider. Defaults to "OpenAI".
        embedding_model (str, optional): Specific embedding model name. Defaults to "text-embedding-3-small".
    
    Returns:
        PineconeVectorStore: Vector store object for the uploaded documents.
    
    Environment Variables:
        PINECONE_API_KEY: Required for Pinecone authentication
        PINECONE_ENVIRONMENT: AWS region for serverless index (defaults to us-east-1)
    """
    
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # Create index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud='aws',
                region=os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
            )
        )
    
    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n"
    )
    chunks = text_splitter.split_text(text)
    
    if model=="OpenAI":
        # Initialize embeddings (compatible with o3-mini)
        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            dimensions=dimension
    )
    
    # Create vector store and upload documents
    vector_store = PineconeVectorStore.from_texts(
        texts=chunks,
        embedding=embeddings,
        index_name=index_name
    )
    
    print(f"Successfully uploaded {len(chunks)} chunks to Pinecone index '{index_name}'")
    return vector_store


# ========================================================================================================================================================

def retrieve_from_pinecone(
        query,
        index_name, 
        k=4,
        model_name="o3-mini",
        embedding_model="text-embedding-3-small"):
    
    """
    Retrieve and answer questions using RAG (Retrieval-Augmented Generation) from Pinecone vector store.
    
    Args:
        query (str): Question or query to answer
        index_name (str): Name of the existing Pinecone index to search
        k (int, optional): Number of similar documents to retrieve. Defaults to 4.
        model_name (str, optional): ChatOpenAI model for answer generation. Defaults to "o3-mini".
        embedding_model (str, optional): OpenAI embedding model for query vectorization. Defaults to "text-embedding-3-small".
    
    Returns:
        str: Generated answer based on retrieved context, limited to 3 sentences.
    
    Note:
        Requires existing Pinecone index with embedded documents and OpenAI API access.
    """
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        dimensions=1536
    )

    # Connect to existing vector store
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )

    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    # Use ChatOpenAI for chat models
    llm = ChatOpenAI(model=model_name)

    # Create prompt template
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Create the chain using the new pattern
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Use invoke instead of run
    response = rag_chain.invoke({"input": query})

    return response["answer"]

# ========================================================================================================================================================

def upload_df_glue_w_index(
        df,
        database,
        s3_path,
        region_name,
        unique_df_values,
        llm_max_output_tokens,
        pc_index_name,
        chunk_size,
        chunk_overlap,
        dimension=1536,
        model="OpenAI",
        embedding_model="text-embedding-3-small",
        vector_db = "pinecone",
        s3_vector_db_config = None,
        additional_metadata=None
):
    """
    Upload DataFrame to AWS Glue, generate LLM-powered data dictionary, and store in Pinecone vector database.
    
    Combines three operations: (1) saves DataFrame as Parquet to S3 and registers in Glue Catalog, 
    (2) generates structured data dictionary using OpenAI o4-mini model, (3) uploads dictionary to Pinecone for RAG queries.
    
    Args:
        df (pandas.DataFrame): DataFrame to upload and analyze
        table_name (str): Glue table name
        database (str): Glue database name  
        s3_path (str): S3 bucket path for Parquet storage
        region_name (str): AWS region
        unique_df_values (int): Number of unique values per column to analyze
        llm_max_output_tokens (int): Token limit for LLM data dictionary generation
        pc_index_name (str): Pinecone index name for vector storage
        chunk_size (int): Text chunk size for vector embedding
        chunk_overlap (int): Character overlap between chunks
        dimension (int, optional): Vector embedding dimension. Defaults to 1536.
        model (str, optional): Embedding model provider. Defaults to "OpenAI".
        embedding_model (str, optional): Embedding model name. Defaults to "text-embedding-3-small".
    
    Returns:
        PineconeVectorStore: Vector store containing the generated data dictionary.
    
    Environment Variables:
        MASTER_ACCESS_KEY, MASTER_SECRET_KEY: AWS credentials
        PINECONE_API_KEY: Pinecone authentication
        OPENAI_API_KEY: OpenAI API access
    """
    df = clean_col_names(df)
    pc_table_name= pc_index_name.replace("-","_")

    write_data_to_glue_db(df= df,
                        table_name=  pc_table_name,
                        database= database,
                        s3_path= s3_path ,
                        region_name=region_name)
    

    dist_values_dict = distinct_values_dict(df=df,
                                            n_values=unique_df_values)
    dict_string= json.dumps(dist_values_dict)
    final_dict_prompt = create_data_dict_prompt(dict_string)

    # Test with appropriate token limit
    result = generate_response_o4_mini(
        prompt=final_dict_prompt,
        think=False,
        max_output_tokens=llm_max_output_tokens  # Increased from 50
        )
    
    # print(result)
    
    final_input_text = """
                    ### Database Schema
                    This text contains data dictionary for DB.table name as  {db}.{table_name} .
                    table dict has following information: Objectives, columns which has sub heading names as >> "desc", "column_data_type", "sample_values", 
                    the details of dict as follows:
                    {result}
                    """.format(db= database ,
                               table_name= pc_table_name,
                               result=result
                               )
    if vector_db == "pinecone":
        upload_to_pinecone(text=final_input_text,
                            index_name= pc_index_name ,
                            chunk_size= chunk_size ,
                            chunk_overlap=chunk_overlap,
                            dimension= dimension,
                            metric='cosine',
                            model=model,
                            embedding_model= embedding_model
                            )
        print (f"pinecone index name: {pc_index_name} is successfully created")

    elif vector_db == "s3_vector_db":
        if s3_vector_db_config == None:
            raise Exception(f" if vector_db == 's3_vector_db' then param s3_vector_db_config cannot be None")
        
        s3_vector_db = S3VectorDB(
            access_key_id = os.getenv("MASTER_ACCESS_KEY"),
            secret_access_key = os.getenv("MASTER_SECRET_KEY"),
            model=embedding_model,
            dimensions=dimension,
            region_name=region_name
    )

        create_index_response= s3_vector_db.create_vector_index(
            bucket_name=s3_vector_db_config["bucket_name"],
            index_name=pc_index_name,
            dimension= dimension,
            distance_metric = 'cosine'
            )
        if create_index_response == None:
            raise Exception(f"S3 vector DB is not created")
        
        s3_vector_db.process_and_upload_text(
            text= final_input_text, 
            bucket_name=s3_vector_db_config["bucket_name"], 
            index_name=pc_index_name,
            chunk_size= chunk_size,
            chunk_overlap = chunk_overlap,
            additional_metadata= additional_metadata
            )
        print (f"S3 vector DB index name: {pc_index_name} is successfully created")
                
    
# ========================================================================================================================================================

def retrieve_from_pinecone(query, index_name, k=4):
    """
    Retrieve and query data from Pinecone serverless vector database.
    
    Args:
        query (str): The query/question to search for
        index_name (str): Name of the Pinecone index
        k (int): Number of similar documents to retrieve
    
    Returns:
        str: The answer generated by the QA chain
    """
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=1536
    )
    
    # Connect to existing vector store
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    
    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    
    # Use ChatOpenAI instead of OpenAI for o3-mini
    llm = ChatOpenAI(model_name="o3-mini")
    
    # Create prompt template
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know.keep the answer concise"
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Create the chain using the new pattern
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # Use invoke instead of run
    response = rag_chain.invoke({"input": query})
    
    return response["answer"]

# ========================================================================================================================================================

def clean_col_names(df):
    """
    Clean DataFrame column names by:
    1. Converting to lowercase
    2. Replacing all symbols and spaces with underscores
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    
    Returns:
    pandas.DataFrame: DataFrame with cleaned column names
    """
    cleaned_columns = []
    for col in df.columns:
        # 1. Lowercase all column names
        clean_col = col.lower()
        # 2. Replace all symbols and spaces with underscores
        clean_col = re.sub(r'[^a-z0-9]', '_', clean_col)
        cleaned_columns.append(clean_col)
    
    df.columns = cleaned_columns
    return df

# ========================================================================================================================================================

def delete_glue_table(database, table_name, aws_access_key_id=None, 
                     aws_secret_access_key=None, region_name=None, 
                     workgroup=None):
    """
    Delete a Glue table and its associated S3 data with custom AWS credentials.
    
    Args:
        database (str): The Glue database name
        table_name (str): The Glue table name to delete
        aws_access_key_id (str, optional): AWS Access Key ID
        aws_secret_access_key (str, optional): AWS Secret Access Key
        region_name (str, optional): AWS region name
        workgroup (str, optional): Athena workgroup name
    
    Returns:
        dict: Status of the deletion operation
    """
    try:
        # Configure AWS session if credentials are provided
        session = None
        if aws_access_key_id and aws_secret_access_key:
            session = boto3.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name
            )
        elif region_name:
            session = boto3.Session(region_name=region_name)
        
        # Get table information - catalog_id will default to account ID
        s3_location = wr.catalog.get_table_location(
            database=database, 
            table=table_name,
            boto3_session=session
        )
        
        if not s3_location:
            print(f"Warning: No S3 location found for table {table_name}")
        
        # Delete the table from Glue catalog
        wr.catalog.delete_table_if_exists(
            database=database, 
            table=table_name,
            boto3_session=session
        )
        print(f"Successfully deleted Glue table: {database}.{table_name}")
        
        # Delete the S3 data if location exists
        if s3_location:
            wr.s3.delete_objects(
                path=s3_location,
                boto3_session=session
            )
            print(f"Successfully deleted S3 data at: {s3_location}")
        
        return {
            "table_name":f"{database}.{table_name}",
            "status": "success",
            "table_deleted": True,
            "s3_data_deleted": bool(s3_location),
            "s3_location": s3_location,
            "region": region_name,
            "workgroup": workgroup
        }
        
    except Exception as e:
        print(f"Error deleting table {database}.{table_name}: {str(e)}")
        return {
            "table_name":f"{database}.{table_name}",
            "status": "error",
            "error": str(e),
            "table_deleted": False,
            "s3_data_deleted": False
        }

# ========================================================================================================================================================

def delete_pinecone_index(del_index_name):
    # Initialize Pinecone client
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # List and delete all indexes
    indexes = pc.list_indexes()
    for index_name in indexes.names():
        if del_index_name == index_name:
            print(f"Deleting index: {index_name}")
            pc.delete_index(index_name)
    return None

# ========================================================================================================================================================

def delete_dataset_n_attributes(
    database,
    dataset_name,
    access_key_id,
    secret_access_key,
    region_name,
    workgroup,
    vector_db_config,
    vector_db_name

):
    # Step 1: deleting S3 table and it's attributes
    table_name= dataset_name.replace("-","_")
    delete_response = delete_glue_table(

        database=database,
        table_name=table_name, 
        aws_access_key_id= access_key_id ,
        aws_secret_access_key= secret_access_key, 
        region_name=region_name,
        workgroup=workgroup
    )

    # Step 2: deleting pinecone index and it's attributes
    if vector_db_name == "pinecone":
        return delete_pinecone_index(dataset_name)

    if vector_db_name == "S3 Vector db":
        delete_s3_vector_db_index_response = delete_s3_vector_index(
            bucket_name=vector_db_config["bucket_name"],
            index_name=dataset_name,
            access_key_id = access_key_id ,
            secret_access_key = secret_access_key,
            region_name=vector_db_config["region_name"],
            endpoint_url=None
        )

        return delete_s3_vector_db_index_response

# ========================================================================================================================================================


def get_random_txt_content(folder_path):
    """
    Picks a random .txt file from the specified folder, reads its content, 
    and returns it as a string. If no .txt files are found, returns an empty string.
    
    :param folder_path: str - The path to the folder containing .txt files.
    :return: str - The content of the randomly selected .txt file.
    """
    # List all files in the folder that end with .txt
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    if not txt_files:
        return ""  # Or raise an exception if preferred
    
    # Pick a random .txt file
    random_file = random.choice(txt_files)
    
    # Construct the full file path
    file_path = os.path.join(folder_path, random_file)
    
    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    return content

# ========================================================================================================================================================




def inference_open_bio_llm_entity(model,prompt,text_summary, HF_TOKEN):
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN  
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": """{prompt} {text_summary}
               """.format(
                   prompt=prompt,
                   text_summary=text_summary
                          )
            }
        ],
    )

    return completion

# ========================================================================================================================================================


class S3VectorDB:
    """
    A vector database implementation using Amazon S3 with OpenAI embeddings.
    
    This class provides functionality to create vector indices, generate embeddings using OpenAI,
    store vectors in S3, and perform similarity searches for RAG (Retrieval-Augmented Generation) applications.
    
    :param access_key_id: AWS access key ID for authentication
    :type access_key_id: str
    :param secret_access_key: AWS secret access key for authentication
    :type secret_access_key: str
    :param model: OpenAI embedding model name
    :type model: str
    :param dimensions: Number of dimensions for the embedding vectors
    :type dimensions: int
    :param region_name: AWS region name for S3 services
    :type region_name: str
    
    Example:
        >>> s3_vector_db = S3VectorDB(
        ...     access_key_id="your_access_key",
        ...     secret_access_key="your_secret_key",
        ...     model="text-embedding-3-small",
        ...     dimensions=1536,
        ...     region_name="us-west-2"
        ... )
    """
    
    def __init__(
            self,
            access_key_id: str,
            secret_access_key: str,
            model: str = "text-embedding-3-small",
            dimensions: int = 1536,
            client_endpoint: str = "default",
            region_name: str = 'us-east-1'
            ):
        """
        Initialize S3 Vector DB with AWS credentials and OpenAI embedding configuration.
        
        :param access_key_id: AWS access key ID for authentication
        :type access_key_id: str
        :param secret_access_key: AWS secret access key for authentication
        :type secret_access_key: str
        :param model: OpenAI embedding model name, defaults to "text-embedding-3-small"
        :type model: str
        :param dimensions: Number of dimensions for the embedding vectors, defaults to 1536
        :type dimensions: int
        :param region_name: AWS region name for S3 services, defaults to 'us-west-2'
        :type region_name: str
        """
        # Create AWS session with provided credentials
        self.session = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=region_name
        )
        
        if client_endpoint == "default":
            # Initialize S3 vectors client for vector operations
            self.s3vectors = self.session.client('s3vectors')
        elif client_endpoint == "HF":
            # Use the correct S3 Vectors endpoint format
            s3vectors_endpoint = f"https://s3vectors.{region_name}.api.aws"
            self.s3vectors = self.session.client(
                's3vectors',
                region_name=region_name,
                endpoint_url=s3vectors_endpoint
            )

        self.region_name = region_name
        
        # Initialize OpenAI embeddings with specified model and dimensions
        self.embeddings = OpenAIEmbeddings(
            model=model,
            dimensions=dimensions
        )
    
    def create_vector_index(self, bucket_name: str, index_name: str, dimension: int = 1536, distance_metric: str = 'cosine'):
        """
        Create a vector index in the specified S3 vector bucket.
        
        This method creates a new vector index with the specified configuration for storing
        and querying high-dimensional vectors.
        
        :param bucket_name: Name of the S3 vector bucket
        :type bucket_name: str
        :param index_name: Name for the new vector index
        :type index_name: str
        :param dimension: Number of dimensions for vectors in this index, defaults to 1536
        :type dimension: int
        :param distance_metric: Distance metric for similarity calculations, defaults to 'cosine'
        :type distance_metric: str
        :return: Response from S3 vectors service or error message
        :rtype: dict or str or None
        :raises ConflictException: When the vector index already exists
        :raises Exception: For other API errors
        
        Note:
            OpenAI text-embedding-3-small uses 1536 dimensions by default.
            
        Example:
            >>> response = s3_vector_db.create_vector_index(
            ...     bucket_name="my-vector-bucket",
            ...     index_name="my-index",
            ...     dimension=1536,
            ...     distance_metric="cosine"
            ... )
        """
        try:
            # Create vector index with metadata configuration
            response = self.s3vectors.create_index(
                vectorBucketName=bucket_name,
                indexName=index_name,
                dataType='float32',  # Data type for vector components
                dimension=dimension,
                distanceMetric=distance_metric,  # Similarity metric (cosine, euclidean, etc.)
                metadataConfiguration={
                    # Configure non-filterable metadata keys for optimization
                    'nonFilterableMetadataKeys': ['source_text', 'chunk_id']
                }
            )
            print(f"Vector index '{index_name}' created successfully with {dimension} dimensions")
            return response
        except self.s3vectors.exceptions.ConflictException as e:
            # Handle case where vector index already exists
            return f"Vector already exists"
        except Exception as e:
            # Handle other API errors
            print(f"Error creating index: {str(e)}")
            return None
    
    def generate_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text strings using OpenAI through LangChain.
        
        This method processes multiple text chunks in batch to generate their vector
        representations using the configured OpenAI embedding model.
        
        :param texts: List of text strings to generate embeddings for
        :type texts: List[str]
        :return: List of embedding vectors, each as a list of float values
        :rtype: List[List[float]]
        :raises Exception: When OpenAI API call fails
        
        Example:
            >>> texts = ["Hello world", "How are you?"]
            >>> embeddings = s3_vector_db.generate_openai_embeddings(texts)
            >>> print(len(embeddings))  # 2
            >>> print(len(embeddings[0]))  # 1536 (for text-embedding-3-small)
        """
        try:
            # Use LangChain's embed_documents method for efficient batch processing
            embeddings_list = self.embeddings.embed_documents(texts)
            print(f"Generated embeddings for {len(texts)} text chunks")
            return embeddings_list
        except Exception as e:
            # Handle OpenAI API errors or network issues
            print(f"Error generating OpenAI embeddings: {str(e)}")
            return []
    
    def process_and_upload_text(self, text: str, bucket_name: str, index_name: str, 
                               chunk_size: int = 3000, chunk_overlap: int = 10, 
                               additional_metadata: Optional[Dict] = None):
        """
        Process text by splitting into chunks, generating embeddings, and uploading to S3 Vector DB.
        
        This method performs the complete pipeline from text input to vector storage:
        1. Split text into manageable chunks
        2. Generate embeddings for each chunk
        3. Prepare vector data with metadata
        4. Upload vectors to the specified index
        
        :param text: Input text to be processed and stored
        :type text: str
        :param bucket_name: Name of the S3 vector bucket
        :type bucket_name: str
        :param index_name: Name of the vector index to store vectors
        :type index_name: str
        :param chunk_size: Maximum size of each text chunk, defaults to 3000
        :type chunk_size: int
        :param chunk_overlap: Number of overlapping characters between chunks, defaults to 10
        :type chunk_overlap: int
        :param additional_metadata: Optional additional metadata to attach to vectors
        :type additional_metadata: Optional[Dict]
        :return: Response from S3 vectors service or None if failed
        :rtype: dict or None
        
        Example:
            >>> response = s3_vector_db.process_and_upload_text(
            ...     text="Long document content...",
            ...     bucket_name="my-bucket",
            ...     index_name="my-index",
            ...     chunk_size=2000,
            ...     additional_metadata={"source": "document.pdf"}
            ... )
        """
        # Step 1: Split text into chunks using CharacterTextSplitter
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,        # Maximum characters per chunk
            chunk_overlap=chunk_overlap,  # Overlap to maintain context
            separator="\n"                # Split on newlines primarily
        )
        chunks = text_splitter.split_text(text)
        print(f"Split text into {len(chunks)} chunks")
        
        # Step 2: Generate embeddings for all text chunks
        embeddings_list = self.generate_openai_embeddings(chunks)
        
        # Check if embedding generation was successful
        if not embeddings_list:
            print("Failed to generate embeddings")
            return None
        
        # Step 3: Prepare vectors for upload with metadata
        vectors = []
        for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings_list)):
            # Create comprehensive metadata for each chunk
            metadata = {
                "chunk_id": f"chunk_{i}",           # Unique chunk identifier
                "source_text": chunk_text,          # Original text content
                "chunk_index": i,                   # Sequential index
                "chunk_length": len(chunk_text)     # Length for reference
            }
            
            # Merge any additional metadata provided by user
            if additional_metadata:
                metadata.update(additional_metadata)
            
            # Prepare vector data structure for S3 API
            vector_data = {
                "key": f"vector_{i}_{hash(chunk_text) % 1000000}",  # Unique vector identifier
                "data": {"float32": embedding},                      # Embedding vector data
                "metadata": metadata                                 # Associated metadata
            }
            vectors.append(vector_data)

        # Start with estimated batch size
        batch_size = estimate_batch_size(embeddings_list)
        total_vectors_size = len(vectors)

        # Starting from the first index.
        i=0
        
        while i < total_vectors_size:
            batch = vectors[i:i + batch_size]
            # Step 4: Upload all vectors to S3 Vector DB in batch
            try:
                response = self.s3vectors.put_vectors(
                    vectorBucketName=bucket_name,
                    indexName=index_name,
                    vectors=batch  # Batch upload for efficiency
                )
                i+=batch_size
                print(f"Vectors upload completed: {np.round(i/total_vectors_size,3)}. ")
                # return response
            except Exception as e:
                # Handle upload errors
                print(f"Error uploading vectors: {str(e)}")
                i+=batch_size
                # return None
        print(f"Successfully uploaded {len(vectors)} vectors to S3 Vector DB")

    def query_with_openai_embedding(self, bucket_name: str, index_name: str, query_text: str, 
                                   top_k: int = 5, filter_dict: Optional[Dict] = None):
        """
        Query the vector index using OpenAI embeddings for similarity search.
        
        This method converts the query text to an embedding and searches for the most
        similar vectors in the specified index.
        
        :param bucket_name: Name of the S3 vector bucket to query
        :type bucket_name: str
        :param index_name: Name of the vector index to search
        :type index_name: str
        :param query_text: Text query to find similar content for
        :type query_text: str
        :param top_k: Number of most similar results to return, defaults to 5
        :type top_k: int
        :param filter_dict: Optional metadata filters to apply to search
        :type filter_dict: Optional[Dict]
        :return: List of similar vectors with metadata and distances
        :rtype: list
        :raises Exception: When query embedding generation or vector search fails
        
        Example:
            >>> results = s3_vector_db.query_with_openai_embedding(
            ...     bucket_name="my-bucket",
            ...     index_name="my-index",
            ...     query_text="What is machine learning?",
            ...     top_k=3
            ... )
        """
        # Generate embedding for the query text using OpenAI
        try:
            query_embedding = self.embeddings.embed_query(query_text)
        except Exception as e:
            print(f"Error generating query embedding: {str(e)}")
            return []
        
        try:
            # Prepare query parameters for vector search
            query_params = {
                "vectorBucketName": bucket_name,
                "indexName": index_name,
                "queryVector": {"float32": query_embedding},  # Query vector in float32 format
                "topK": top_k,                                # Number of results to return
                "returnDistance": True,                       # Include similarity distances
                "returnMetadata": True                        # Include associated metadata
            }
            
            # Add optional metadata filters if provided
            if filter_dict:
                query_params["filter"] = filter_dict
            
            # Execute vector similarity search
            response = self.s3vectors.query_vectors(**query_params)
            results = response.get("vectors", [])
            # print(f"Found {len(results)} similar vectors")
            
            return results
        except Exception as e:
            # Handle query execution errors
            print(f"Error querying vectors: {str(e)}")
            return []
    
    def get_rag_context(self, bucket_name: str, index_name: str, query_text: str, 
                       similarity_threshold: float = 0.5, max_chunks: int = 5) -> str:
        """
        Get relevant context for RAG (Retrieval-Augmented Generation) applications.
        
        This method retrieves the most relevant text chunks based on similarity to the query
        and combines them into a context string suitable for feeding to language models.
        
        :param bucket_name: Name of the S3 vector bucket to search
        :type bucket_name: str
        :param index_name: Name of the vector index to query
        :type index_name: str
        :param query_text: Query text to find relevant context for
        :type query_text: str
        :param similarity_threshold: Minimum similarity score to include chunks, defaults to 0.5
        :type similarity_threshold: float
        :param max_chunks: Maximum number of chunks to include in context, defaults to 5
        :type max_chunks: int
        :return: Combined context string from relevant chunks
        :rtype: str
        
        Example:
            >>> context = s3_vector_db.get_rag_context(
            ...     bucket_name="knowledge-base",
            ...     index_name="documents",
            ...     query_text="How to train a neural network?",
            ...     similarity_threshold=0.7,
            ...     max_chunks=3
            ... )
            >>> print(len(context))  # Length of combined context
        """
        # Query for more results than needed to allow filtering by threshold
        results = self.query_with_openai_embedding(
            bucket_name=bucket_name,
            index_name=index_name,
            query_text=query_text,
            top_k=max_chunks * 2  # Get extra results for threshold filtering
        )
        
        # Filter results by similarity threshold and collect relevant chunks
        relevant_chunks = []
        for result in results:
            distance = result.get('distance', 1.0)  # Get similarity distance
            # Include chunk if distance is below threshold and we haven't reached max
            if distance <= similarity_threshold and len(relevant_chunks) < max_chunks:
                relevant_chunks.append(result['metadata']['source_text'])
        
        # Combine all relevant chunks into a single context string
        context = "\n\n".join(relevant_chunks)
        return context


# ========================================================================================================================================================


class S3VectorRetriever(BaseRetriever):
    """
    Custom LangChain Retriever for S3 Vector Database integration.
    
    This retriever extends LangChain's BaseRetriever to provide seamless integration
    with S3 Vector DB, enabling vector similarity search within LangChain workflows
    for RAG (Retrieval-Augmented Generation) applications.
    
    The retriever converts search queries into embeddings, performs similarity search
    against the S3 Vector DB, and returns results as LangChain Document objects
    with associated metadata and similarity scores.
    
    :param s3vector_db: Instance of S3VectorDB for performing vector operations
    :type s3vector_db: S3VectorDB
    :param bucket_name: Name of the S3 vector bucket to search
    :type bucket_name: str
    :param index_name: Name of the vector index to query
    :type index_name: str
    :param search_kwargs: Search parameters including 'k' (top results), 'filter' (metadata filters), 'score_threshold' (minimum similarity)
    :type search_kwargs: Dict[str, Any]
    
    Example:
        >>> retriever = S3VectorRetriever(
        ...     s3vector_db=s3_db_instance,
        ...     bucket_name="knowledge-base",
        ...     index_name="documents",
        ...     search_kwargs={"k": 10, "score_threshold": 0.7}
        ... )
        >>> documents = retriever.get_relevant_documents("machine learning")
    """
    
    # Define Pydantic fields with validation and descriptions
    s3vector_db: Any = Field(
        ..., 
        description="S3VectorDB instance for vector operations and similarity search"
    )
    bucket_name: str = Field(
        ..., 
        description="S3 vector bucket name where vectors are stored"
    )
    index_name: str = Field(
        ..., 
        description="Vector index name to query for similarity search"
    )
    search_kwargs: Dict[str, Any] = Field(
        default_factory=lambda: {"k": 5}, 
        description="Search parameters: k (top results), filter (metadata filters), score_threshold (minimum similarity)"
    )
    
    class Config:
        """
        Pydantic configuration for the S3VectorRetriever class.
        
        This configuration allows the use of arbitrary types (like custom S3VectorDB)
        in Pydantic model fields, which is necessary for LangChain integration.
        
        :param arbitrary_types_allowed: Enables custom types in Pydantic fields
        :type arbitrary_types_allowed: bool
        """
        arbitrary_types_allowed = True  # Allow custom types like S3VectorDB instance
    
    # Pydantic handles initialization automatically - no custom __init__ needed
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Retrieve relevant documents from S3 Vector DB based on similarity search.
        
        This method performs the core retrieval functionality by:
        1. Extracting search parameters from search_kwargs
        2. Executing similarity search against S3 Vector DB
        3. Converting results to LangChain Document format
        4. Applying score thresholding if specified
        5. Adding similarity metadata to each document
        
        :param query: Text query to search for similar documents
        :type query: str
        :param run_manager: LangChain callback manager for run tracking
        :type run_manager: CallbackManagerForRetrieverRun
        :return: List of relevant documents with similarity scores and metadata
        :rtype: List[Document]
        :raises Exception: When S3 Vector DB query fails or returns invalid results
        
        Example:
            >>> documents = retriever._get_relevant_documents(
            ...     query="What is deep learning?",
            ...     run_manager=callback_manager
            ... )
            >>> print(f"Found {len(documents)} relevant documents")
        """
        # Extract search parameters from configuration with defaults
        k = self.search_kwargs.get("k", 5)                          # Number of top results to return
        filter_dict = self.search_kwargs.get("filter", None)        # Optional metadata filters
        score_threshold = self.search_kwargs.get("score_threshold", None)  # Minimum similarity threshold
        
        # Execute similarity search against S3 Vector DB
        results = self.s3vector_db.query_with_openai_embedding(
            bucket_name=self.bucket_name,
            index_name=self.index_name,
            query_text=query,
            top_k=k,                    # Request top-k most similar vectors
            filter_dict=filter_dict     # Apply metadata filters if provided
        )
        
        # Convert S3 Vector DB results to LangChain Document objects
        documents = []
        for result in results:
            # Extract similarity distance from result
            distance = result.get('distance', 1.0)  # Default to max distance if missing
            
            # Apply score threshold filtering if specified
            if score_threshold is not None and distance > score_threshold:
                continue  # Skip documents below similarity threshold
            
            # Prepare comprehensive metadata for the document
            metadata = dict(result['metadata'])              # Copy original metadata
            metadata['distance'] = distance                  # Add similarity distance
            metadata['similarity_score'] = 1 - distance     # Convert to similarity score (0-1)
            
            # Create LangChain Document with content and enriched metadata
            doc = Document(
                page_content=result['metadata']['source_text'],  # Original text content
                metadata=metadata                                # Enhanced metadata with similarity info
            )
            documents.append(doc)
        
        return documents

    async def _aget_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Asynchronous version of document retrieval for S3 Vector DB.
        
        This method provides async compatibility for LangChain's asynchronous workflows
        by wrapping the synchronous retrieval method. Currently implemented as a
        simple wrapper around the sync method since S3 Vector DB operations are
        inherently synchronous.
        
        :param query: Text query to search for similar documents
        :type query: str
        :param run_manager: LangChain callback manager for async run tracking
        :type run_manager: CallbackManagerForRetrieverRun
        :return: List of relevant documents with similarity scores and metadata
        :rtype: List[Document]
        
        Note:
            This implementation calls the synchronous method since the underlying
            S3 Vector DB client doesn't support native async operations. Future
            versions could implement true async behavior if the S3 service adds
            async support.
            
        Example:
            >>> documents = await retriever._aget_relevant_documents(
            ...     query="machine learning algorithms",
            ...     run_manager=async_callback_manager
            ... )
        """
        # Delegate to synchronous implementation
        # TODO: Implement true async behavior when S3 Vector DB supports async operations
        return self._get_relevant_documents(query, run_manager=run_manager)


# ========================================================================================================================================================

# Use the composition-based S3VectorRetriever class above

class S3VectorStore:
    """
    Wrapper class providing a Pinecone-like interface for S3 Vector Database operations.
    
    This class serves as an adapter that provides familiar Pinecone-style methods and patterns
    for interacting with S3 Vector DB. It enables easy migration from Pinecone to S3 Vector DB
    by maintaining similar API conventions while leveraging S3's vector capabilities.
    
    The wrapper supports creating retrievers for LangChain integration and provides
    factory methods for connecting to existing vector indices.
    
    :param s3vector_db: Instance of S3VectorDB for backend vector operations
    :type s3vector_db: S3VectorDB
    :param bucket_name: Name of the S3 vector bucket containing the index
    :type bucket_name: str
    :param index_name: Name of the vector index to operate on
    :type index_name: str
    :param embedding: Embedding function or model for text-to-vector conversion
    :type embedding: Any
    
    Example:
        >>> # Create from existing S3 Vector DB setup
        >>> vector_store = S3VectorStore(
        ...     s3vector_db=s3_db_instance,
        ...     bucket_name="my-vectors",
        ...     index_name="documents",
        ...     embedding=openai_embeddings
        ... )
        >>> retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    """
    
    def __init__(self, s3vector_db, bucket_name: str, index_name: str, embedding):
        """
        Initialize S3VectorStore with backend components and configuration.
        
        Sets up the vector store wrapper with all necessary components for
        vector operations including the S3 backend, index location, and
        embedding functionality.
        
        :param s3vector_db: S3VectorDB instance that handles actual vector operations
        :type s3vector_db: S3VectorDB
        :param bucket_name: S3 bucket name where vectors are stored
        :type bucket_name: str
        :param index_name: Name of the specific vector index to use
        :type index_name: str
        :param embedding: Embedding model or function for converting text to vectors
        :type embedding: Any
        
        Example:
            >>> vector_store = S3VectorStore(
            ...     s3vector_db=my_s3_db,
            ...     bucket_name="knowledge-base",
            ...     index_name="documents-v1",
            ...     embedding=openai_embedding_model
            ... )
        """
        # Store reference to the underlying S3 Vector DB instance
        self.s3vector_db = s3vector_db
        
        # Configure S3 location parameters for vector operations
        self.bucket_name = bucket_name  # S3 bucket containing vector data
        self.index_name = index_name    # Specific index within the bucket
        
        # Store embedding function for text-to-vector conversion
        self.embedding = embedding
    
    @classmethod
    def from_existing_index(cls, s3vector_db, bucket_name: str, index_name: str, embedding):
        """
        Create S3VectorStore from an existing vector index using factory pattern.
        
        This class method provides a Pinecone-compatible way to connect to existing
        vector indices without explicitly calling the constructor. It mimics
        Pinecone's from_existing_index method for seamless API compatibility.
        
        :param s3vector_db: S3VectorDB instance for vector operations
        :type s3vector_db: S3VectorDB
        :param bucket_name: Name of S3 bucket containing the existing index
        :type bucket_name: str
        :param index_name: Name of the existing vector index to connect to
        :type index_name: str
        :param embedding: Embedding model compatible with the existing index
        :type embedding: Any
        :return: New S3VectorStore instance connected to the existing index
        :rtype: S3VectorStore
        
        Note:
            This method assumes the vector index already exists in S3. It doesn't
            create new indices, only connects to existing ones.
            
        Example:
            >>> # Connect to existing index (Pinecone-style)
            >>> vector_store = S3VectorStore.from_existing_index(
            ...     s3vector_db=s3_db,
            ...     bucket_name="production-vectors",
            ...     index_name="embeddings-2024",
            ...     embedding=text_embedding_model
            ... )
        """
        # Use factory pattern to create instance - delegates to constructor
        return cls(s3vector_db, bucket_name, index_name, embedding)
    
    def as_retriever(self, search_type: str = "similarity", search_kwargs: Dict[str, Any] = None):
        """
        Create a LangChain-compatible retriever from this vector store.
        
        This method provides Pinecone-style retriever creation, returning a
        retriever object that can be used in LangChain workflows for document
        retrieval and RAG applications.
        
        :param search_type: Type of search to perform, defaults to "similarity"
        :type search_type: str
        :param search_kwargs: Additional search parameters including 'k' (number of results), 'filter' (metadata filters), 'score_threshold' (minimum similarity)
        :type search_kwargs: Dict[str, Any], optional
        :return: Configured retriever instance for LangChain integration
        :rtype: S3VectorRetriever
        
        Note:
            Currently only "similarity" search type is supported. The search_kwargs
            parameter allows fine-tuning of retrieval behavior including result
            count, metadata filtering, and similarity thresholds.
            
        Example:
            >>> # Basic retriever with default settings
            >>> retriever = vector_store.as_retriever()
            >>> 
            >>> # Customized retriever with specific parameters
            >>> retriever = vector_store.as_retriever(
            ...     search_type="similarity",
            ...     search_kwargs={
            ...         "k": 10,
            ...         "score_threshold": 0.8,
            ...         "filter": {"category": "technical"}
            ...     }
            ... )
        """
        # Set default search parameters if not provided
        if search_kwargs is None:
            search_kwargs = {"k": 4}  # Default to top 4 most similar results
        
        # Create and return S3VectorRetriever using composition pattern
        # This provides LangChain compatibility while maintaining S3 Vector DB backend
        return S3VectorRetriever(
            s3vector_db=self.s3vector_db,    # Backend S3 vector database
            bucket_name=self.bucket_name,    # S3 bucket location
            index_name=self.index_name,      # Specific vector index
            search_kwargs=search_kwargs      # Search configuration parameters
        )

# ========================================================================================================================================================


def create_s3_open_ai_rag_chain(s3vector_db,
                              bucket_name: str,
                              index_name: str,
                              system_prompt: str,
                              embedding_model: str = "text-embedding-3-small",
                              model_name: str = "o3-mini",
                              dimensions: int = 1536,
                              k: int = 5,
                              score_threshold: float = 0.5):
    """
    Create a complete RAG (Retrieval-Augmented Generation) chain using S3 Vector DB and OpenAI.
    
    This function constructs an end-to-end RAG pipeline that combines vector similarity search
    with language model generation. It integrates S3 Vector DB for document retrieval with
    OpenAI's embedding and chat models to provide contextually-aware responses.
    
    The resulting RAG chain automatically:
    1. Converts user queries to embeddings
    2. Retrieves relevant documents from S3 Vector DB
    3. Provides retrieved context to the language model
    4. Generates responses based on both query and context
    
    :param s3vector_db: S3VectorDB instance for vector storage and retrieval operations
    :type s3vector_db: S3VectorDB
    :param bucket_name: Name of the S3 bucket containing the vector index
    :type bucket_name: str
    :param index_name: Name of the vector index to query for document retrieval
    :type index_name: str
    :param system_prompt: System message that defines the AI assistant's behavior and context handling
    :type system_prompt: str
    :param embedding_model: OpenAI embedding model for text-to-vector conversion, defaults to "text-embedding-3-small"
    :type embedding_model: str
    :param model_name: OpenAI chat model for response generation, defaults to "o3-mini"
    :type model_name: str
    :param dimensions: Number of dimensions for embedding vectors, defaults to 1536
    :type dimensions: int
    :param k: Number of most similar documents to retrieve for context, defaults to 5
    :type k: int
    :param score_threshold: Minimum similarity score for including documents in context, defaults to 0.5
    :type score_threshold: float
    :return: Configured RAG chain ready for query processing
    :rtype: langchain.chains.retrieval.RetrievalChain
    :raises Exception: When OpenAI API initialization fails or vector store connection errors occur
    
    Example:
        >>> rag_chain = create_s3_open_ai_rag_chain(
        ...     s3vector_db=my_s3_db,
        ...     bucket_name="knowledge-base",
        ...     index_name="documents",
        ...     system_prompt="You are a helpful assistant. Use the provided context to answer questions.",
        ...     embedding_model="text-embedding-3-small",
        ...     model_name="o3-mini",
        ...     k=3,
        ...     score_threshold=0.7
        ... )
        >>> response = rag_chain.invoke({"input": "What is machine learning?"})
    """
    
    # Initialize OpenAI embeddings with specified model and dimensions
    embeddings = OpenAIEmbeddings(
        model=embedding_model,  # Model for converting text to vectors
        dimensions=dimensions   # Vector dimensionality (must match index configuration)
    )
    
    # Create vector store wrapper using factory method for existing index
    vector_store = S3VectorStore.from_existing_index(
        s3vector_db=s3vector_db,  # Backend S3 vector database instance
        bucket_name=bucket_name,   # S3 bucket location
        index_name=index_name,     # Specific vector index to query
        embedding=embeddings       # Embedding function for consistency
    )
    
    # Configure retriever with search parameters for document retrieval
    retriever = vector_store.as_retriever(search_kwargs={
        "k": k,                           # Number of top similar documents to retrieve
        "score_threshold": score_threshold # Minimum similarity threshold for inclusion
    })
    
    # Initialize OpenAI chat model for response generation
    llm = ChatOpenAI(model=model_name)  # Latest OpenAI model for high-quality responses
    
    # Create chat prompt template for structured conversation
    # The 'context' variable will be automatically populated by create_stuff_documents_chain
    # with the retrieved documents from the vector store
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),  # System message defining assistant behavior
        ("human", "{input}"),       # User input placeholder
    ])
    
    # Create document processing chain that combines retrieved docs with LLM
    # This chain automatically injects retrieved documents as context
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create complete RAG chain by combining retriever and QA chain
    # This creates the full pipeline: query -> retrieve -> generate -> response
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

# ========================================================================================================================================================

def s3_vector_db_open_ai_rag_response(rag_chain, query: str):
    """
    Execute a query against the RAG chain and return the response with context analysis.
    
    This function processes user queries through the complete RAG pipeline, providing
    both the generated response and metadata about the retrieval process. It serves
    as a testing and execution interface for the RAG system.
    
    The function invokes the RAG chain which automatically:
    1. Converts the query to embeddings
    2. Retrieves relevant documents from S3 Vector DB
    3. Combines query and retrieved context
    4. Generates a contextually-aware response
    5. Returns response with source documents and metadata
    
    :param rag_chain: Configured RAG chain created by create_s3_open_ai_rag_chain
    :type rag_chain: langchain.chains.retrieval.RetrievalChain
    :param query: User query or question to be processed
    :type query: str
    :return: Complete response object containing generated answer, source documents, and metadata
    :rtype: dict
    :raises Exception: When RAG chain execution fails due to API errors or retrieval issues
    
    The returned dictionary typically contains:
        - 'answer': Generated response from the language model
        - 'context': List of retrieved documents used for context
        - 'source_documents': Original documents with metadata
        - Additional metadata about the retrieval and generation process
    
    Example:
        >>> response = s3_vector_db_open_ai_rag_response(
        ...     rag_chain=my_rag_chain,
        ...     query="What are the benefits of using vector databases?"
        ... )
        >>> print(response['answer'])
        >>> print(f"Used {len(response['context'])} source documents")
        >>> 
        >>> # Access source documents for citation
        >>> for doc in response['context']:
        ...     print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    """
    # Execute the complete RAG pipeline with the user query
    # The rag_chain automatically handles:
    # - Query embedding generation
    # - Vector similarity search
    # - Document retrieval and context preparation  
    # - LLM prompt construction with context
    # - Response generation and metadata collection
    response = rag_chain.invoke({"input": query})
    
    return response

# ========================================================================================================================================================


def estimate_batch_size(vectors, max_size_bytes=20 * 1024 * 1024):
    """
    Estimate how many vectors can fit in a batch under the size limit
    """
    if not vectors:
        return 0
    
    # Estimate size of first vector (including JSON overhead)
    import json
    sample_size = len(json.dumps(vectors[0]).encode('utf-8'))
    estimated_vectors_per_batch = max(1, max_size_bytes // sample_size)
    
    # Cap at AWS limit of 200 vectors per request
    return min(350, estimated_vectors_per_batch)

# ========================================================================================================================================================

class BiomedicalRAG:
    def __init__(self, s3vector_db, bucket_name: str, index_name: str, hf_token: str, system_prompt: str,
                 inference_model: str,temperature: float, max_tokens:int):
        self.s3vector_db = s3vector_db
        self.bucket_name = bucket_name
        self.index_name = index_name
        self.hf_token = hf_token
        self.inference_model=inference_model
        self.system_prompt= system_prompt 
        self.temperature=temperature
        self.max_tokens= max_tokens
        self.rag_chain = self._create_rag_chain()
    
    def _create_rag_chain(self):
        # Setup retrieval with your working S3 Vector DB
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=1536
        )
        
        vector_store = S3VectorStore.from_existing_index(
            s3vector_db=self.s3vector_db,
            bucket_name=self.bucket_name,
            index_name=self.index_name,
            embedding=embeddings
        )
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 30})
        
        # Setup biomedical LLM with HuggingFace
        llm = ChatOpenAI(
            model=self.inference_model,
            openai_api_base="https://router.huggingface.co/v1",
            openai_api_key=self.hf_token,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        system_prompt_input = (
            self.system_prompt + """ \n\n\n{context}" """
            )

        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_input),
            ("human", "{input}"),
        ])
        
        # Create RAG chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        return rag_chain
    
    def query(self, question: str, show_context: bool = False):
        """Query the biomedical RAG system"""
        try:
            response = self.rag_chain.invoke({"input": question})
            
            result = {
                "answer": response["answer"],
                "num_sources": len(response.get("context", []))
            }
            
            if show_context:
                result["sources"] = [
                    {
                        "content": doc.page_content[:200] + "...",
                        "distance": doc.metadata.get("distance", "N/A")
                    }
                    for doc in response.get("context", [])
                ]
            
            return result
        
        except Exception as e:
            print(f"Error during RAG query: {str(e)}")
            return {"answer": f"Error: {str(e)}", "num_sources": 0}
        
# ========================================================================================================================================================

def generate_inference_hf(
        prompt,
        api_key,
        inference_model,
        max_tokens
        ):
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=api_key,
            )

    completion = client.chat.completions.create(
        model=inference_model, # "openai/gpt-oss-120b:nebius"
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.5,
        max_tokens=max_tokens

    )
    return completion.choices[0].message.content


# ========================================================================================================================================================



def split_medical_text(text):
    # Regex pattern to match uppercase indicator words (possibly multi-word with spaces or /) ending with : or ,
    pattern = r'([A-Z/ ]+?):[ ,]*|([A-Z/ ]+?),[ ,]*'
    
    # Find all potential keys
    matches = re.findall(pattern, text)
    unique_keys = []
    for k1, k2 in matches:
        key = (k1 or k2).strip()
        if key and key not in unique_keys:
            unique_keys.append(key)
    
    # If no keys found, return empty dict
    if not unique_keys:
        return {}
    
    # Create split pattern with capturing groups to preserve keys
    escaped_keys = [re.escape(k) for k in unique_keys]
    split_pattern = r'(' + '|'.join(escaped_keys) + r')[:,]?[ ,]*'
    
    # Split the text, capturing keys and values
    splits = re.split(split_pattern, text)
    
    # Build the dictionary
    result = {}
    i = 1  # Start after potential leading empty split
    while i < len(splits) - 1:
        key = splits[i].lower().strip()
        value = splits[i + 1].lower().strip()
        # Clean leading/trailing punctuation from value
        value = re.sub(r'^[ :,]+|[ :,]+$', '', value).strip()
        if key:
            result[key] = value
        i += 2
    
    return result

# =========================================================================================================================

def split_string_on_period(text, min_length=100):
    """
    Fast string splitter that splits on periods after min_length characters.
    """
    if not text:
        return []
    
    result = []
    start = 0
    
    # Find all period positions
    for i, char in enumerate(text):
        if char == '.' and (i - start + 1) >= min_length:
            result.append(text[start:i+1])
            start = i + 1
    
    # Add remaining text if any
    if start < len(text):
        result.append(text[start:])
    
    return result

# =========================================================================================================================

def segregate_soap_notes(input_dict,find_list):
    new_dict = {}
    for condition in find_list:
        sub_dict={}
        for keys in input_dict:
            if keys.find(condition) != -1:
                sub_dict[keys] = input_dict[keys]
        new_dict[condition]=sub_dict
    return new_dict

# =========================================================================================================================

def split_dict_text(input_dict, code_category, find_list = ["procedure","diagnosis", "prescription"]):
    seg_dict_level_1 = segregate_soap_notes(input_dict=input_dict,find_list=find_list)
    for key in seg_dict_level_1.keys():
        if code_category == key:
            sub_dict = seg_dict_level_1[key]
            for sub_key in sub_dict.keys():
                sub_dict[sub_key] = split_string_on_period(sub_dict[sub_key])
    return sub_dict

# =========================================================================================================================


def generate_clinical_rag_prompt(key):
    example_dict = '''
    {{
        "code1": "description1", 
        "code2": "description2"
    }}
    '''
    code_library = {
        "diagnosis":"ICD-10",
        "procedure": "CPT/HCPCS"
        }
    library = code_library[key]
    system_prompt = (
        """
        You are an expert in clinical and medical coding, specializing in {library} {key} codes. 
        You will receive a user query containing doctor's SOAP notes, along with retrieved context containing relevant {key} codes and their descriptions.

        Your task is to:
        1. Analyze the query to identify all mentioned or implied medical {key}s.
        2. Extract ONLY the matching {library} codes and their exact descriptions from the provided context. Do not use any external knowledge or invent codes/descriptions—stick strictly to the context.
        3. If no matching codes are found in the context, return an empty dictionary.
        4. Output a valid Python dictionary where each key is a unique {key} code (as a string), and the value is its corresponding description (as a string, extracted verbatim from the context).

        Respond ONLY with the dictionary in this exact format (no additional text, explanations, or code outside the dict). 
        For multiple codes, include all in one dict, e.g.:
        {example_dict}

        Ensure all extractions are accurate and complete to minimize errors.
        """.format(key=key, example_dict=example_dict, library=library)

    )
    return system_prompt

# =========================================================================================================================

def llm_output_soap_note_summarizer(string):

    context_prompt = f"""
    
    You are a medical documentation assistant. I will provide output from an LLM in the form of a JSON dictionary. This JSON contains SOAP note text extracts (e.g., procedure descriptions/diagnosis description) paired with associated clinical codes (e.g., CPT or diagnosis codes) for procedures or diagnoses.

    Your task is to structure this into a professional, concise Markdown document. Follow these guidelines:

    Parse the JSON: Extract the text extracts and codes accurately. The structure is typically a dictionary like {{'procedure': [[text_extract, {{codes_dict}}]...],'diagnosis': [[text_extract, {{codes_dict}}]...] }}.

    Narrative Style: Create a flowing, narrative summary of the procedure or diagnosis event, integrating the supporting CPT/diagnosis codes inline or in parentheses where relevant. Make it readable and professional, like a clinical report.

    Markdown Format: Use headings (e.g., # Procedure Summary/ Diagnosis Summary), subheadings (e.g., ## Key Steps), bullet points for steps, and tables or bold text for codes if they enhance clarity. Include sections for 'Overview', 'Detailed Steps', and 'Associated Codes' if applicable.

    Handle Empty Codes: If a code dictionary is empty (e.g., '{{}}'), note it as 'No associated codes' without fabricating any.

    Output Only the Markdown: Do not add extra commentary; keep it focused on the summary.

    Here is the JSON input: \n\n\n {string}"""
    return context_prompt

# ========================================================================================================================================================



# def random
def random_choice_list(input_list):
    return random.choice(input_list)


def replace_words_string(text, string_dict):

    for word in string_dict.keys():
        text=text.replace(word,string_dict[word])
    return text

def generate_sample_soap_notes(source_path,replace_text_dict,max_character_size=1000):
    sample_med_script_list = pd.read_csv(source_path)["transcription"].to_list()
    sample_med_script_list_trim = [x for x in sample_med_script_list if len(x)<max_character_size]
    random_choice_text = random_choice_list(sample_med_script_list_trim)
    random_choice_text =replace_words_string(text=random_choice_text, string_dict=replace_text_dict)
    return random_choice_text


# ========================================================================================================================================================


def generate_clinical_coding_summary(
        soap_note,
        embedding_model,
        dimension,
        client_endpoint,
        region_name,
        hf_inference_model,
        key_index_mapping,
        bucket_name,
        temperature,
        max_tokens
        ):

    code_keys= [ "diagnosis","procedure"]

    master_json_string = ""

    s3_vector_db = S3VectorDB(
        access_key_id = os.getenv("MASTER_ACCESS_KEY"),
        secret_access_key = os.getenv("MASTER_SECRET_KEY"),
        model=embedding_model,
        dimensions=dimension,
        client_endpoint= client_endpoint, # "HF"
        region_name=region_name
    )

    for key in code_keys:

        # Create biomedical RAG system
        bio_rag = BiomedicalRAG(
            s3vector_db=s3_vector_db,
            bucket_name=bucket_name,
            index_name=key_index_mapping[key],
            hf_token=os.getenv("HF_TOKEN"),
            system_prompt=generate_clinical_rag_prompt(key=key),
            inference_model=hf_inference_model["open-ai-oss"],
            temperature=temperature,
            max_tokens=max_tokens
        )


        split_soap_note= split_medical_text(soap_note)

        filtered_dict=  split_dict_text(
            input_dict=split_soap_note,
            code_category= key
            )
        
        # print(filtered_dict)
        
        inferred_dict= {}
        for keys in filtered_dict.keys():
            elements_codes= []
            for element in filtered_dict[keys]:
                result = bio_rag.query(element, show_context=True)
                elements_codes.append((element,result['answer']))

            inferred_dict[keys]= elements_codes

        master_json_string = master_json_string + f"  {key}:  " + json.dumps(inferred_dict)
        # print(master_json_string,sep='\n')

    final_model_output = generate_inference_hf(
    prompt=llm_output_soap_note_summarizer(master_json_string),
    api_key = os.getenv("HF_TOKEN"),
    inference_model=hf_inference_model["open-ai-oss"],
    max_tokens=max_tokens
    )

    return final_model_output

# ========================================================================================================================================================


def stream_text(text, delay=0.01):
    placeholder = st.empty()
    streamed_text = ""
    
    for char in text:
        streamed_text += char
        placeholder.markdown(streamed_text)
        time.sleep(delay)

# ========================================================================================================================================================

def arrange_sentence_next_line(text,range_enter=20):
    new_string = []
    for i in range(0,len(text.split(" ")),range_enter):
        new_string = new_string + text.split(" ")[i:i+range_enter] +["\n"]
    return  " ".join(new_string)

# ========================================================================================================================================================

def generate_AE_notes(source_path,max_character_size=1000):
    sample_med_script_list = pd.read_csv(source_path)["input"].to_list()
    sample_med_script_list_trim = [x for x in sample_med_script_list if len(x)<max_character_size]
    random_choice_text = random_choice_list(sample_med_script_list_trim)
    return random_choice_text


# ========================================================================================================================================================

def retrieve_and_infer_vector_db(
        query,
        system_prompt,
        index_name,
        vector_db, 
        s3_vector_db_config=None,
        k=4,
        score_threshold= 1
        ):
    """
    Retrieve and query data from Pinecone serverless vector database.
    
    Args:
        query (str): The query/question to search for
        index_name (str): Name of the Pinecone index
        k (int): Number of similar documents to retrieve
    
    Returns:
        str: The answer generated by the QA chain
    """
    if vector_db == 'pinecone':   
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=1536
        )
        
        # Connect to existing vector store
        vector_store = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        
        # Create retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        
        # Use ChatOpenAI instead of OpenAI for o3-mini
        llm = ChatOpenAI(model_name="o3-mini")
        
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # Create the chain using the new pattern
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        # Use invoke instead of run
        response = rag_chain.invoke({"input": query})
        
        return response["answer"]
    elif vector_db == 'S3 Vector db':
        s3_vector_db = S3VectorDB(
        access_key_id = os.getenv("MASTER_ACCESS_KEY"),
        secret_access_key = os.getenv("MASTER_SECRET_KEY")
    )
        open_ai_rag_chain = create_s3_open_ai_rag_chain(
            s3vector_db=s3_vector_db,
            bucket_name=s3_vector_db_config["bucket_name"],
            index_name=index_name,
            system_prompt= system_prompt,
            embedding_model = s3_vector_db_config["model"],
            model_name = "o3-mini",
            dimensions = s3_vector_db_config["dimensions"],
            k = k,
            score_threshold=score_threshold
    )
        response = s3_vector_db_open_ai_rag_response(
            rag_chain=open_ai_rag_chain,
            query=query
            )
        return response["answer"]


# ========================================================================================================================================================
@st.cache_resource(ttl="60s")
def get_s3_vector_index_names(
        vector_bucket_name,
        access_key_id = os.getenv("MASTER_ACCESS_KEY"),
        secret_access_key = os.getenv("MASTER_SECRET_KEY"),
        region_name="us-east-1"
        ):
    """
    Retrieves a list of index names from an S3 vector bucket using the 's3vectors' client.
    This function lists all vector indexes in the specified vector bucket. If no indexes
    are found, it returns an empty list.

    :param vector_bucket_name: Name of the S3 vector bucket.
    :return: List of index names, or empty list if none.
    """
    session = boto3.Session(
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name=region_name
    )
    client = session.client('s3vectors')
    index_names = []
    
    # Paginate through indexes in the vector bucket
    paginator = client.get_paginator('list_indexes')
    for page in paginator.paginate(vectorBucketName=vector_bucket_name):
        for index in page.get('indexes', []):
            index_names.append(index.get("indexName",""))

    return index_names

# ========================================================================================================================================================

@st.cache_resource(ttl="60s")
def load_pinecone_index_names():
    # Initialize Pinecone client
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    return pc.list_indexes()

# ========================================================================================================================================================

def pandas_to_dict(df,json_string=True):
    result = {}
    for col in df.columns:
        result[col] = df[col].tolist()
    if json_string:
        return json.dumps(result)
    else:
        return result
    

# ========================================================================================================================================================

def colored_header(text,text_color, font_size="24px", border_radius="2%"):
    """
    Renders a header with custom background and text colors.
    """
    st.markdown(
        f'<p style="color:{text_color};font-size:{font_size};border-radius:{border_radius};">{text}</p>',
        unsafe_allow_html=True
    )

# ========================================================================================================================================================

def delete_s3_vector_index(bucket_name, index_name, access_key_id, secret_access_key, region_name, endpoint_url=None):
    """
    Delete an index from S3 vectors service.
    
    Args:
        bucket_name (str): Name of the S3 bucket containing the vector index
        index_name (str): Name of the index to delete
        access_key_id (str): AWS access key ID
        secret_access_key (str): AWS secret access key
        region_name (str): AWS region name
        endpoint_url (str, optional): Custom endpoint URL if using a custom service
    
    Returns:
        dict: Response from the delete operation or error details
    """
    try:
        # Create session with your provided configuration
        session = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=region_name
        )
        
        # Create client - add endpoint_url if using custom service
        if endpoint_url:
            client = session.client('s3vectors', endpoint_url=endpoint_url)
        else:
            client = session.client('s3vectors')
        
        # Delete the index from the specified bucket
        response = client.delete_index(
            vectorBucketName=bucket_name,
            indexName=index_name
        )
        
        print(f"Successfully initiated deletion of index: {index_name} from bucket: {bucket_name}")
        return {
            'success': True,
            'message': f"Index '{index_name}' from bucket '{bucket_name}' deletion initiated successfully",
            'response': response
        }
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        
        print(f"Client error deleting index {index_name} from bucket {bucket_name}: {error_code} - {error_message}")
        return {
            'success': False,
            'error_type': 'ClientError',
            'error_code': error_code,
            'error_message': error_message,
            'bucket_name': bucket_name,
            'index_name': index_name
        }
        
    except BotoCoreError as e:
        print(f"BotoCore error deleting index {index_name} from bucket {bucket_name}: {str(e)}")
        return {
            'success': False,
            'error_type': 'BotoCoreError',
            'error_message': str(e),
            'bucket_name': bucket_name,
            'index_name': index_name
        }
        
    except Exception as e:
        print(f"Unexpected error deleting index {index_name} from bucket {bucket_name}: {str(e)}")
        return {
            'success': False,
            'error_type': 'UnexpectedError',
            'error_message': str(e),
            'bucket_name': bucket_name,
            'index_name': index_name
        }


# ========================================================================================================================================================


def extract_dict_string(text):
    """
    Extracts dictionary patterns from text using regex and returns them as string values in a list.
    
    Args:
        text (str): The input text containing dictionary patterns
        
    Returns:
        list: A list of dictionary strings found in the text
    """
    # Regex pattern to match dictionary-like structures
    # Matches: { ... } that contain at least one colon (key-value separator)
    pattern = r'\{[^{}]*\}'
    
    # Find all potential matches
    potential_matches = re.findall(pattern, text)
    
    # Filter to keep only those that look like dictionaries (contain : )
    dict_matches = [match for match in potential_matches if ':' in match]
    
    return dict_matches