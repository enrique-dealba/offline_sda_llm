import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.langchain_structured_outputs import generate_objective_response
from app.utils import get_displayable_fields

st.title("LLM")

user_input = st.text_area("Enter spaceplan objective:", "")

if st.button("Generate Schema"):
    if not user_input:
        st.warning("Please enter a query.")
    else:
        try:
            assert (
                settings.USE_STRUCTURED_OUTPUT
            ), "Structured output is disabled in settings."

            response, time_details = generate_objective_response(user_input)
            # st.info(f"Raw Response: {response}")
            fields = get_displayable_fields(response)

            st.json(fields)
            st.info(f"Total Execution Time: {time_details:.2f} seconds")

        except Exception as e:
            st.error(f"An error occurred: {e}")
