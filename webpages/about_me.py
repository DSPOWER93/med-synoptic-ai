import streamlit as st
from PIL import Image
import base64
import os
from io import BytesIO

# Page configuration
# st.set_page_config(
#     page_title="About Us",
#     page_icon="👋",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )



# Custom CSS for circular image and styling
def load_css():
    st.markdown("""
    <style>
    .circular-image-wrapper {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }

    .circular-image-container {
        width: 200px !important;
        height: 200px !important;
        border-radius: 50% !important;
        overflow: hidden !important;
        border: 4px solid #f0f0f0 !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
        position: relative !important;
        background-color: #f8f9fa !important;
    }

    .circular-image-container img {
        width: 100% !important;
        height: 100% !important;
        object-fit: cover !important;
        object-position: center center !important;
        display: block !important;
        border-radius: 0 !important;
    }

    .image-placeholder-circular {
        width: 200px !important;
        height: 200px !important;
        border-radius: 50% !important;
        background-color: #e0e0e0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border: 4px solid #f0f0f0 !important;
        margin: 0 auto 20px auto !important;
    }
    
    .main-header {
        font-size: 2rem !important;
        font-weight: bold !important;
        color: #1f77b4 !important;
        margin-bottom: 1px !important;
    }
    
    .subtitle {
        font-size: 1.5rem !important;
        color: #666 !important;
        margin-bottom: 1px !important;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #0a0354;
        margin-top: 15px;
        margin-bottom: 30px;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 5px;
    }
    
    .highlight-box {
        background-color: #F5F9FF;
        padding: 1px;
        border-radius: 1px;
        align-items: center;
        margin: 1px 0;
        text-align: center; 
    }
    
    .contact-item {
        display: flex;
        align-items: center;
        font-weight: bold;
        margin: 10px 0;
        font-size: 1.1rem;
    }
    
    .contact-icon {
        margin-right: 30px;
        margin-left: 10px;
        font-size: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    

def get_base64_encoded_image_from_path(image_path):
    try:
        if os.path.exists(image_path):
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            
            # Crop to square (center crop)
            width, height = image.size
            min_size = min(width, height)
            left = (width - min_size) / 2
            top = (height - min_size) / 2
            right = (width + min_size) / 2
            bottom = (height + min_size) / 2
            
            # Crop to square
            image = image.crop((left, top, right, bottom))
            
            # Resize to standard size for consistency
            image = image.resize((200, 200), Image.Resampling.LANCZOS)
            
            buffered = BytesIO()
            image.save(buffered, format="JPEG", quality=95)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/jpeg;base64,{img_str}"
        else:
            return None
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None


def display_circular_image(image_path):
    if image_path and image_path.strip():
        img_base64 = get_base64_encoded_image_from_path(image_path.strip())
        if img_base64:
            st.markdown(
                f'''
                <div style="display: flex; justify-content: center; margin-bottom: 20px;">
                    <div style="width: 200px; height: 200px; border-radius: 50%; overflow: hidden; border: 4px solid #f0f0f0; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); position: relative; background-color: #f8f9fa;">
                        <img src="{img_base64}" alt="Profile Image" style="width: 100%; height: 100%; object-fit: cover; object-position: center center; display: block;">
                    </div>
                </div>
                ''',
                unsafe_allow_html=True
            )
            return True
        else:
            st.error("❌ Could not load image. Please check the file path.")
            return False
    else:
        # Placeholder for image
        st.markdown(
            '''
            <div style="display: flex; justify-content: center; margin-bottom: 20px;">
                <div style="width: 200px; height: 200px; border-radius: 50%; background-color: #e0e0e0; display: flex; align-items: center; justify-content: center; border: 4px solid #f0f0f0;">
                    <span style="font-size: 3rem;">👤</span>
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )
        return False


def create_about_me_page(
        image_path
        ):
    # Create columns for layout
    col1, col2 = st.columns([1, 2])
    
    # Left column - Image and basic info
    with col1:
        display_circular_image(image_path)
        
        # Contact information in highlight box - ADD THE OPENING TAG
        st.markdown('<div class="highlight-box"> <h3 align-items= center;>📞 Contact Info </h3> </div>', unsafe_allow_html=True)  # ✅ ADD THIS LINE
        st.markdown(
            """
            <div class="contact-item">
                <span class="contact-icon">📧</span>
                <a href="mailto:md786.52@gmail.com" target="_blank" style="text-decoration: none; color: inherit;">Email me @</a>
            </div>
            <div class="contact-item">
                <span class="contact-icon">📱</span>
                <span>+91-96118-05929</span>
            </div>
            <div class="contact-item">
                <span class="contact-icon">🚀</span>
                <a href="https://github.com/DSPOWER93/" target="_blank" style="text-decoration: none; color: inherit;">Github</a>
            </div>
            <div class="contact-item">
                <span class="contact-icon">📍</span>
                <span>Bengaluru, India</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<div class="highlight-box"> <h3 align-items= center;>🎓 Certifications </h3> </div>', unsafe_allow_html=True)


        st.markdown('''
                    <div class="contact-item">
                <span class="contact-icon">📜</span>
                <a href="https://www.credly.com/badges/3b3b466d-b716-456c-ab0f-dd5be9ea92de/" target="_blank" style="text-decoration: none; color: inherit;">
                    AWS ML speciality-(MLS-C01)</a>
                </div>
                    ''', unsafe_allow_html=True)
        
        # st.markdown('''<a href="https://www.credly.com/badges/3b3b466d-b716-456c-ab0f-dd5be9ea92de/" target="_blank" style="text-decoration: none; color: #0a0354;">
        #                 <h5>AWS ML speciality-(MLS-C01)</h5></a>''', unsafe_allow_html=True)
    
    # Right column - Main content
    with col2:
        st.markdown('<h2 class="main-header">Hello, I\'m Mohammed Taher 👋</h2>', unsafe_allow_html=True)
        st.markdown('<h4 class="subtitle">Sr. Data Scientist</h4>', unsafe_allow_html=True)
        
        # About section
        st.markdown('<h2 class="section-header">About Me</h2>', unsafe_allow_html=True)
        st.write("""
        Working as Senior Data Scientist with 6+ years of experience across healthcare and gaming.
                  Specialized in designing and deploying production-grade machine-learning systems 
                 with ML-Ops best practices and Generative AI capabilities to accelerate delivery 
                 and elevate user outcomes.
        """)


        # Skills section
        st.markdown('<h2 class="section-header">Skills & Expertise</h2>', unsafe_allow_html=True)
        
        # Create skill columns
        skill_col1, skill_col2 = st.columns(2)
        
        with skill_col1:
            st.markdown('<h5><b>Technical Skills: </b></h5>', unsafe_allow_html=True)
            st.write("• **Machine Learning**")
            st.write("• **Deep Learning - NLP**")
            st.write("• **AWS**")
            st.write("• **ML-Ops**")
            st.write("• **Rest API**")
        
        with skill_col2:
            st.markdown('<h5><b>Tools & Frameworks:</b></h5>', unsafe_allow_html=True)
            st.write("• **Python**")
            st.write("• **PySpark**")
            st.write("• **Github + Actions**")
            st.write("• **Fast API**")
            st.write("• **Docker**")

        # Experience section
        st.markdown('<h2 class="section-header">Projects</h2>', unsafe_allow_html=True)
        
        with st.expander("# **Gen AI Use Cases (Year-2025)**"):
            st.write("""
            - Created **InsightBridge**, an EDA agent that enables intuitive data exploration without requiring backend database languages.
            - Developed a Gen AI web app to streamline healthcare documentation by automating patient data extraction, MedDRA adverse event coding, and RAG-based clinical code suggestions.
            Project Source : [GitHub Repo](https://github.com/DSPOWER93/med-synoptic-ai)
            """)
        
        with st.expander("**Monitoring Framework for ML-Ops (Year-2024)**"):
            st.write("""
            - Developed a monitoring framework that automates process of identifying Model drift & Data drift in ML pipelines. 
              The framework streamlines the process of integrating montoring modules ML piplines with minimal efforts.
            """)
        


def app():
    load_css()
    create_about_me_page(
        image_path="artifacts/cropped_circle_image.png"
        )
