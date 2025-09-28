import os
import re
import json
from pinecone import Pinecone

import streamlit as st
import pandas as pd

from streamlit_scroll_navigation import scroll_navbar

from Utilities.helper_func import (
    upload_df_glue_w_index,
    delete_dataset_n_attributes,
    retrieve_and_infer_vector_db,
    get_s3_vector_index_names,
    read_athena_table,
    load_pinecone_index_names,
    pandas_to_dict,
    generate_inference_hf,
    stream_text,
    colored_header
)

from Utilities.configs import (
    sql_prompt,
    analysis_prompt,
    insight_bridge
)

from config.env_config import env_config, eda_llm_config, hf_inference_model

css = '''
            <style>
                [data-testid='stFileUploader'] {
                    width: max-content;
                }
                [data-testid='stFileUploader'] section {
                    padding: 0;
                    float: left;
                }
                [data-testid='stFileUploader'] section > input + div {
                    display: none;
                }
                [data-testid='stFileUploader'] section + div {
                    float: right;
                    padding-top: 0;
                }

            </style>
            '''
st.markdown(css, unsafe_allow_html=True)


def app():
    _col1_,_col2_,_col3_=st.columns([2,2,1])
    with _col2_:
        colored_header("<b>InsightBridge</b>", "#111a9a", font_size="40px") # Red background, yellow text, larger font

    # st.markdown(f"""<h6 style='text-align: left; font-weight: normal; color:#000080;'>  {insight_bridge}</h6>""",unsafe_allow_html=Fasle)

    st.markdown(f"""{insight_bridge}""",unsafe_allow_html=False)

    st.markdown("""<hr style="border-top: 1px solid blue;">""",unsafe_allow_html=True)

    load_pc_index_names= load_pinecone_index_names().names()

    s3_index_name_list = get_s3_vector_index_names(
        vector_bucket_name=eda_llm_config["bucket_name"],
        access_key_id = os.getenv("MASTER_ACCESS_KEY"),
        secret_access_key = os.getenv("MASTER_SECRET_KEY"),
        region_name=eda_llm_config["region_name"]
        )



    anchor_ids = ["Query with Dataset", "Upload Dataset","Delete Dataset"]

    scroll_navbar(
        anchor_ids=anchor_ids,
        key="navbar4",
        orientation="horizontal",
        override_styles={
            "navbarButtonBase": {
                "backgroundColor": "#007bff",  # Set a custom button background color
                "color": "#ffffff",  # Set custom text color
            },
            "navbarButtonHover": {
                "backgroundColor": "#0036b3",  # Set a custom hover color for the buttons
            },
            "navigationBarBase": {
                "backgroundColor": "#f8f9fa",  # Change the navigation bar background color
            }
        })




    with st.container(border=True):

        st.subheader(f":blue[{anchor_ids[0]}]",anchor=anchor_ids[0], divider= "blue")
        dropdown_col1, dropdown_col2, dropdown_col3 = st.columns([2,2,1])
        with dropdown_col1:
            # First dropdown - Operating System
            dropdown_1 = st.selectbox(
                "Select vector DB:",
                ["Select Vector DB", "S3 Vector db", "pinecone"],
                key="db_select"
            )

            # Second dropdown - Device options based on OS selection
            if dropdown_1 == "S3 Vector db":
                index_name_list = ["Select Index"] + s3_index_name_list
            elif dropdown_1 == "pinecone":
                index_name_list = ["Select Index"] + load_pc_index_names
            else:
                index_name_list = ["Select Index"]

        with dropdown_col2:
            dropdown_2 = st.selectbox(
                "Select Index:",
                index_name_list,
                disabled=(
                    dropdown_1 == "Select Vector DB"),
                    key="Index"
            )

        with dropdown_col3:
            # Create a checkbox
            st.markdown("<br>",unsafe_allow_html=True)
            generate_insights = st.checkbox("**Generate Insights**")

        manual_text = st.text_area("Add Query here", height=90)

        # Form with just the submit button
        with st.form("index_submit"):
            submit = st.form_submit_button("➤ Submit", use_container_width=True)
            
            if submit:
                if dropdown_1 != "Select Vector DB" and dropdown_2 != "Select Index":

                    retrived_anwser = retrieve_and_infer_vector_db(
                        query=manual_text,
                        system_prompt=sql_prompt,
                        index_name=dropdown_2,
                        vector_db='S3 Vector db', 
                        s3_vector_db_config=eda_llm_config,
                        k=4
                        )
                    
                    df_table = read_athena_table(
                        sql_query=retrived_anwser,
                        env_configs=env_config
                        )
                    
                    st.dataframe(df_table, use_container_width=True)

                    if generate_insights:
                        input_dict = pandas_to_dict(df=df_table)

                        analysis_dict = json.dumps(
                            {
                            "query":manual_text,
                            "data":input_dict
                            }
                            )

                        analysis = generate_inference_hf(
                            prompt= analysis_prompt + "dict>>>>" + analysis_dict,
                            api_key = os.getenv("HF_TOKEN"),
                            inference_model=hf_inference_model["open-ai-oss"],
                            max_tokens=hf_inference_model["model_config"]["max_tokens"]
                            )
                        stream_text(text=analysis)

    # Create two columns with equal width (50:50 ratio)
    col1, col2 = st.columns(2)

    with col1:
        
        # Create a form
        with st.form("data_upload_form"):

            st.subheader(f":blue[{anchor_ids[1]}]",anchor=anchor_ids[1], divider= "blue")
            # Dropdown to select data format
            file_format = st.selectbox('Select data format', ['csv', 'parquet'])
            db_source = st.selectbox('Select Database', ['pinecone', 's3_vector_db'])
            # Index Name
            _index_name_= st.text_input("""Dataset Index Name""")
            cleaned_input = re.sub(r'[^a-zA-Z0-9-]', '-', _index_name_)
            # File uploader
            uploaded_file = st.file_uploader('Upload your data file', type=[file_format])
            # Submit button (this becomes the form submit button)
            submit = st.form_submit_button('**Upload**',icon="⬆️",use_container_width=True)

        # Handle form submission
        if submit and uploaded_file is not None:
            if _index_name_ != cleaned_input:
                    st.error("Only letters, numbers, and dashes (-) are allowed")
                    st.info(f"Try output: {cleaned_input}")
            else:
                try:
                    if file_format == 'csv':
                        df = pd.read_csv(uploaded_file)
                    elif file_format == 'parquet':
                        df = pd.read_parquet(uploaded_file)

                    with st.spinner('Uploading...'):
                    
                        upload_df_glue_w_index(
                        df=df,
                        database=env_config["athena_configs"]["database"],
                        s3_path=env_config["athena_configs"]["s3_path"],
                        region_name=env_config["athena_configs"]["region"],
                        unique_df_values=10,
                        llm_max_output_tokens=3500,
                        pc_index_name=_index_name_,
                        chunk_size=3000,
                        chunk_overlap=10,
                        dimension=1536,
                        model="OpenAI",
                        embedding_model="text-embedding-3-small",
                        vector_db = db_source,
                        s3_vector_db_config = eda_llm_config,
                        additional_metadata=None
                        )
                        
                        # Display success message
                        st.success(f"Data successfully Uploaded to {db_source} with index name: {_index_name_} !")
                                    
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")

        elif submit and uploaded_file is None:
            st.warning("Please upload a file before submitting.")




    with col2:
        with st.container(border=True):

            st.subheader(f":red[{anchor_ids[2]}]",anchor=anchor_ids[2], divider= "red")

            # First dropdown - Operating System
            dropdown1 = st.selectbox(
                "Select vector DB:",
                ["Select Vector DB", "S3 Vector db", "pinecone"],
                key="db_select_2"
            )
            # Second dropdown - Device options based on OS selection
            if dropdown1 == "S3 Vector db":
                index_name_list_2 = ["Select Index"] + s3_index_name_list
            elif dropdown1 == "pinecone":
                index_name_list_2 = ["Select Index"] + load_pc_index_names
            else:
                index_name_list_2 = ["Select Index"]

            
            dropdown2 = st.selectbox(
                "Select Index:",
                index_name_list_2,
                disabled=(
                    dropdown1 == "Select Vector DB"),
                    key="Index_2"
            )
            
            st.markdown("""<br><br><br><br><br>""",unsafe_allow_html=True)

            with st.form("Delete_Index"):
                submitted = st.form_submit_button('**Delete**',icon="🗑️",use_container_width=True)
                
                if submitted:
                    if dropdown1 != "Select Vector DB" and dropdown2 != "Select Index":
                        with st.spinner('Deleting...'):
                            delete_dataset_n_attributes(
                                database=env_config["athena_configs"]["database"],
                                dataset_name=dropdown2,
                                access_key_id = os.getenv("MASTER_ACCESS_KEY"),
                                secret_access_key = os.getenv("MASTER_SECRET_KEY"),
                                region_name=env_config["athena_configs"]["region"],
                                workgroup=env_config["athena_configs"]["workgroup"],
                                vector_db_config=eda_llm_config,
                                vector_db_name= dropdown1

                            )
                        st.success(f"Dataset: {dropdown2} is successfully deleted.")