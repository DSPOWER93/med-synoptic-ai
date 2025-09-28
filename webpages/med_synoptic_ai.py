import json
import streamlit as st
from streamlit_scroll_navigation import scroll_navbar
import os
import ast
from Utilities.helper_func import (
    get_random_txt_content,
    inference_open_bio_llm_entity,
    stream_text,
    generate_clinical_coding_summary,
    generate_sample_soap_notes,
    arrange_sentence_next_line,
    generate_AE_notes

)

from Utilities.configs import (
    bio_llm_promt_dict,
    markdown_heading_pat_entity
)

from config.env_config import (
    s3_vector_db_config,
    hf_inference_model,
    key_index_mapping,
    replace_text_dict,
    sample_soap_note_path,
    AE_notes_path
)



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
                h6 {
                text-align: left  !important;;
                font-weight: normal  !important;;
                line-height: 1.6  !important;;
                }   
            </style>
            '''
st.markdown(css, unsafe_allow_html=True)



def app():
    _col1_,_col2_,_col3_=st.columns([1,1,1])
    with _col2_:
        st.header("Med Synoptic AI")

    anchor_ids = [
        "Patient Entity Extractor 📋",
        "MedDRA AE Coder </>",
        "Clinical Coding Assist ꨄ" ]

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
                "backgroundColor": "#0056b3",  # Set a custom hover color for the buttons
            },
            "navigationBarBase": {
                "backgroundColor": "#f8f9fa",  # Change the navigation bar background color
            }
        })


    # Patient Entity extractor

    st.subheader(anchor_ids[0],anchor=anchor_ids[0])
    st.markdown(f"""<h6 style='text-align: left; font-weight: normal; line-height: 1.6;'>  {markdown_heading_pat_entity["Patient_entity"]} <br> </h6>""",unsafe_allow_html=True)

    with st.form(key="text_form"):
        manual_text = st.text_area("Add a Discharge summary here ...", height=150)
        
        col1, col2 = st.columns([1,1])

        with col1:
            # First submit button: Display manual text
            display_manual = st.form_submit_button(" ➤ Submit Discharge Summary" , use_container_width=10)
    
        with col2:
            # Second submit button: Auto-generate and display
            auto_generate = st.form_submit_button("🔁 Auto generate a Discharge Summary",use_container_width=10)


    if display_manual:
        with st.spinner('Summarizing Discharge Summary, might take upto a min...'):
            result = inference_open_bio_llm_entity(
                model=hf_inference_model["open-ai-oss"],
                prompt= bio_llm_promt_dict["patient_entity_extraction"],
                text_summary=manual_text,
                HF_TOKEN=os.getenv("HF_TOKEN")
                )
            json_dict = json.loads(result.choices[0].message.content)

            # Display in an expander
            with st.expander("Click to view complete Discharge summary"):
                st.text(manual_text)  # Full text inside expander, use st.text for plain rendering

            st.markdown("<h3 style='text-align: center;'> Patient Entity Extracted </h3>", unsafe_allow_html=True)
            st.json(json_dict,expanded=2)
        
        

    if auto_generate:
        with st.spinner('Summarizing Discharge Summary, might take upto a min...'):
            auto_generate_text = get_random_txt_content(folder_path="sample_discharge_data/")
            result = inference_open_bio_llm_entity(
                model=hf_inference_model["open-ai-oss"],
                prompt= bio_llm_promt_dict["patient_entity_extraction"],
                text_summary=auto_generate_text,
                HF_TOKEN=os.getenv("HF_TOKEN")
                )
            json_dict = json.loads(result.choices[0].message.content)
            
            # Display in an expander
            with st.expander("Click to view complete Auto-Generated Discharge summary"):
                st.text(auto_generate_text)  # Full text inside expander, use st.text for plain rendering

            st.markdown("<h3 style='text-align: center;'> Patient Entity Extracted </h3>", unsafe_allow_html=True)
            st.json(json_dict,expanded=2)

            st.markdown("---")

        
    # MedDRA Coder

    st.subheader(anchor_ids[1],anchor=anchor_ids[1])
    st.markdown(f"""<h6 style='text-align: left; font-weight: normal; line-height: 1.6;'>  {markdown_heading_pat_entity["MedDRA_Coder"]} <br> </h6>""",unsafe_allow_html=True)


    with st.form(key="MedDRA Form"):
        manual_text_meddra = st.text_area("Write a Adverse Event Note here  ...", height=150)
        
        col1, col2 = st.columns([1,1])

        with col1:
            # First submit button: Display manual text
            display_manual_meddra = st.form_submit_button("➤ Record a Adverse Event", use_container_width=10)
    
        with col2:
            # Second submit button: Auto-generate and display
            auto_generate_meddra = st.form_submit_button("🔁 Auto generate a Adverse Event",use_container_width=10)


    if display_manual_meddra:
        with st.spinner('Summarizing Adverse Event, might take upto a min...'):
            result = inference_open_bio_llm_entity(
                model=hf_inference_model["open-ai-oss"],
                prompt= bio_llm_promt_dict["meddra_coding"],
                text_summary=manual_text_meddra,
                HF_TOKEN=os.getenv("HF_TOKEN")
            )
            json_dict = json.loads(result.choices[0].message.content)
            # json_dict = ast.literal_eval(result.choices[0].message.content)

            # Display in an expander
            with st.expander("Click to view complete Reported AE Event"):
                st.text(manual_text_meddra)  # Full text inside expander, use st.text for plain rendering

            st.markdown("<h3 style='text-align: center;'>AE events as per MeDDRA standards</h3>", unsafe_allow_html=True)
            st.json(json_dict,expanded=2)
        
        

    if auto_generate_meddra:
        with st.spinner('Summarizing Adverse Event, might take upto a min...'):
            sample_AE_note = generate_AE_notes(source_path=AE_notes_path)
            result = inference_open_bio_llm_entity(
                model=hf_inference_model["open-ai-oss"],
                prompt= bio_llm_promt_dict["meddra_coding"],
                text_summary=sample_AE_note,
                HF_TOKEN=os.getenv("HF_TOKEN")
            )
            json_dict = json.loads(result.choices[0].message.content)
            # json_dict = ast.literal_eval(result.choices[0].message.content)
            
            # Display in an expander
            with st.expander("Click to view complete Reported AE Event"):
                st.text(sample_AE_note)  # Full text inside expander, use st.text for plain rendering

            st.markdown("<h3 style='text-align: center;'>AE Event as per MeDDRA standards</h3>", unsafe_allow_html=True)
            st.json(json_dict,expanded=2)

            st.markdown("---")


    # Clinical coder

    st.subheader(anchor_ids[2],anchor=anchor_ids[2])
    st.markdown(f"""<h6 style='text-align: left; font-weight: normal; line-height: 1.6;'>  {markdown_heading_pat_entity["clincal_coder"]} <br> </h6>""",unsafe_allow_html=True)

    with st.form(key="soap_note_form"):
        
        manual_text = st.text_area("Add a SOAP Note here ...", height=100)
        
        col1, col2 = st.columns([1,1])

        with col1:
            # First submit button: Display manual text
            display_manual_soap = st.form_submit_button("➤ Submit SOAP Note", use_container_width=10)
    
        with col2:
            # Second submit button: Auto-generate and display
            auto_generate_soap = st.form_submit_button("🔁 Auto generate SOAP Note",use_container_width=10)


    if auto_generate_soap:
        with st.spinner('Summarizing SOAP Note, might take upto a min...'):
            sample_soap_text = generate_sample_soap_notes(
                source_path=sample_soap_note_path,
                replace_text_dict=replace_text_dict,
                max_character_size=1000
                )
            
            clinical_coding_summary = generate_clinical_coding_summary(
                soap_note=sample_soap_text,
                embedding_model=s3_vector_db_config["model"],
                dimension=s3_vector_db_config["dimensions"],
                client_endpoint="HF",
                region_name=s3_vector_db_config["region_name"],
                hf_inference_model=hf_inference_model,
                key_index_mapping=key_index_mapping,
                bucket_name=s3_vector_db_config["bucket_name"],
                temperature=hf_inference_model["model_config"]["temprature"],
                max_tokens=hf_inference_model["model_config"]["max_tokens"]
                )
            
            # Display in an expander
            with st.expander("Click to view complete Auto-Generated Soap Note"):
                st.text(arrange_sentence_next_line(sample_soap_text))  # Full text inside expander, use st.text for plain rendering

            stream_text(text=clinical_coding_summary)    

            st.markdown("---")

