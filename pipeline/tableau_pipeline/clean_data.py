import tabpy
from tabpy.tabpy_tools.client import Client

def clean_text(input_df):
    '''
    This function create preprocessed PAR and output the new dataframe.
    Called in Tableau Prep
    Args:
    ------
        whole dataframe from Tableau
    
    Returns:
    --------
        Returns processed pandas dataframe
    '''
    client = Client("http://10.155.94.140:9004/")
    processed = client.query('clean_text', input_df['X_PAR_COMMENTS'].tolist())['response']
    input_df['PROCESSED_PAR'] = processed
    output_df = input_df
    # return the entire df
    return output_df    

def get_output_schema():
    return pd.DataFrame({
        'CLOSE_DT': prep_datetime(),
        'OPEN_DT': prep_datetime(),
        'AREA': prep_string(),
        'PRIO_CD': prep_string(),
        'RESOLUTION_CD': prep_string(),
        'SEV_CD': prep_string(),
        'SUBTYPE_CD': prep_string(),
        'SUB_AREA': prep_string(),
        'TYPE_CD': prep_string(),
        'W_AREA_CODE': prep_string(),
        'X_PROD_VERSION': prep_string(),
        'X_PRODUCT': prep_string(),
        'X_ENTL_TYPE': prep_string(),
        'X_SR_TITLE': prep_string(),
        'X_SLM_DUE_DT': prep_datetime(),
        'X_ENTL_MTRC_UNIT': prep_string(),
        'X_ENTL_MTRC_VALUE': prep_string(),
        'X_FIRST_RESPONSE_DT': prep_string(),
        'X_SR_PRODUCT_FAMILY': prep_string(),
        'X_PAR_COMMENTS': prep_string(),
        'PROCESSED_PAR': prep_string(),
        'SR_NUM': prep_string()
    })
