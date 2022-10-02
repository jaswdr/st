import os
import streamlit as st

st.title("Homepage")

def get_page_name(page):
    return page.split('.')[0].replace('_', ' ').title()

def get_page_link(page):
    return page.split('.')[0]

for page in sorted(os.listdir('pages')):
    page_name = get_page_name(page)
    page_link = get_page_link(page)
    st.write(f'- [{page_name}]({page_link})')
