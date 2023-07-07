import datetime
import time

import model
import streamlit as st
import streamlit_redirect as rd


@st.spinner(text="This may take a moment...")
def compute(doc, lang, year):
	if doc is None:
		st.error('Please upload a document')
		return

	st.subheader('Calculation results')
	with rd.stdout(format='text'):
		model.main(doc, lang, year)


def app():
	st.write("# Text analyzer")

	with st.form("input_form"):
		selected_doc = st.file_uploader('Upload the text', type=['pdf'])
		selected_lang = st.selectbox(
			label='Language of the text',
			options=['rus', 'fra', 'eng', 'spa'],
			index=0,
		)
		selected_year = st.number_input(
			'Year of writing',
			min_value=1500,
			max_value=datetime.date.today().year,
			value=1900,
			step=1,
			format='%d'
		)
		submitted = st.form_submit_button('Calculate')
		
	if submitted:
		compute(selected_doc, selected_lang, selected_year)


app()
