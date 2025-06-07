from typing import Any, Dict, Tuple

import pandas
import streamlit as st

def input_dataset(
    config: Dict[Any, Any], readme: Dict[Any, Any], instructions: Dict[Any, Any]
    ):
    file = st.file_uploader("Upload a csv file", type="csv", help=readme["tooltips"]["dataset_upload"])
