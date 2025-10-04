import streamlit as st
from streamlit_option_menu import option_menu


st.set_page_config(page_title="Synoptic Gen AI", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)
st.set_option("client.toolbarMode", "viewer")

from webpages import insightbridge, med_synoptic_ai as med_synoptic_ai, about_me


hide_streamlit_style = """
                        <style>
                            #MainMenu {visibility: hidden;}
                            footer {visibility: hidden;}
                            .stAppDeployButton {
                                    display: none;
                                }
                        </style>
                        """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


class MultiApp:

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):

        self.apps.append({
            "title": title,
            "function": func
        })

    def run():
        # app = st.sidebar(
        with st.sidebar:        
            app = option_menu(
                menu_title='Select App', # Heath Wiser
                options=["Med-Synoptic-AI", "InsightBridge", "About-Me" ], #
                icons=['clipboard2-pulse-fill', 'pie-chart', 'person-fill'], # 
                menu_icon='journal-plus',
                default_index=0,
                styles={
                        "container": {"padding": "6!important", "background-color": "#5BB5FBE2"},
                        "icon": {"color": "black", "font-size": "20px"}, 
                        "nav-link": {"font-size": "14px", "text-align": "top", "margin":"0px", "--hover-color": "#B7D0F5"},
                        "nav-link-selected": {"background-color": "#5BB5FBE2"},
                    }
            )
        
        if app == "InsightBridge":
            insightbridge.app()
        if app == "Med-Synoptic-AI":
            med_synoptic_ai.app()
        if app == "About-Me":
            about_me.app()
    
    run()
    