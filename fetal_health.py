# Import libraries
import streamlit as st
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

@st.cache_resource
def load_csv(file_name):
  return pd.read_csv(file_name)

@st.cache_resource
def load_model(pickle_file_name):
  with open(pickle_file_name, 'rb') as model_pickle:
    reg_model = pickle.load(model_pickle)
  return reg_model

default_df = load_csv('fetal_health.csv')
default_df = default_df.head(10)

default_df.dropna(inplace = True)

X = default_df.drop(columns='fetal_health')
y = default_df['fetal_health']
default_df.pop('fetal_health')

st.title('Fetal Health Classification Using Machine Learning')
st.image('fetal_health_image.gif', use_column_width=True, caption='Predict fetal health classification')

st.header('Sample Data')
st.write(default_df)

with st.sidebar.form("model_selection"):
    model_selection = st.selectbox(
        label='Select Model',
        options=['Decision Tree', 'Random Forest', 'AdaBoost', 'Soft Voting Classifier'])
    input_CSV = st.file_uploader('Upload CSV', type=['csv'])
    submit_button = st.form_submit_button('Predict')

if submit_button:

  # Load the selected model
  if model_selection == 'Decision Tree':
    model = load_model('decision_tree.pickle')
  elif model_selection == 'Random Forest':
    model = load_model('rnd_forest.pickle')
  elif model_selection == 'AdaBoost':
    model = load_model('adaboost.pickle')
  elif model_selection == 'Soft Voting Classifier':
    model = load_model('soft_voting_classifier.pickle')
  else:
    st.error("Invalid model selection.")
    st.stop()
    
  # Handle uploaded CSV file
  if input_CSV:
    try:
      user_df = pd.read_csv(input_CSV)
      user_X = user_df.dropna()
            
      # Ensure the same feature columns are present in the uploaded file
      missing_cols = set(X.columns) - set(user_X.columns)
      if missing_cols:
        st.error(f"Uploaded CSV is missing columns: {missing_cols}")
        st.stop()

      # Predict on user data
      y_pred_user = model.predict(user_X)
      y_pred_user_proba = model.predict_proba(user_X)

      # Get highest probability class
      max_confidence = y_pred_user_proba.max(axis=1)

      user_df = user_df.assign(PredictionInt=y_pred_user)
      user_df = user_df.assign(Confidence=max_confidence)

      # Move the Prediction column to the start
      confidence_column = user_df.pop('Confidence')

      user_df['Prediction'] = user_df['PredictionInt'].apply(
          lambda x: (
              'Normal' if x == '1.0' else
              'Suspect' if x == '2.0' else
              'Pathological' if x == '3.0' else
              'Unknown'
          )
      )

      prediction_column = user_df.pop('Prediction')

      user_df.insert(0, 'Prediction', prediction_column)
      user_df.insert(1, 'Confidence', confidence_column) 

      # Set conditional styling
      styled_df = user_df.style.apply(
          lambda x: [
            'background-color: lime' if v == 'Normal' else
            'background-color: yellow' if v == 'Suspect' else
            'background-color: orange' if v == 'Pathological' else
            '' for v in x],
            subset=['Prediction'],
            axis=0
            )
      
      # Display user data with predictions
      st.header('Uploaded Data')
      st.write(styled_df)
            
    except Exception as e:
      st.error(f"Error processing uploaded file: {e}")

  st.subheader("Model Insights")
  tab1, tab2, tab3 = st.tabs(["Feature Importance", 
                              "Confusion Matrix", 
                              "Classification Report"])
  
  with tab1:
      st.write("### Feature Importance")

      if model_selection == 'Decision Tree':
        st.image('feat_imp_dt.svg')
      elif model_selection == 'Random Forest':
        st.image('feat_imp_rnd_forest.svg')
      elif model_selection == 'AdaBoost':
        st.image('feat_imp_ada.svg')
      elif model_selection == 'Soft Voting Classifier':
        st.image('feat_imp_sft_vt.svg')
      else:
        st.write('No model selected')

      st.caption("Relative importance of features in prediction.")

  with tab2:
      st.write("### Confusion Matrix")
      
      if model_selection == 'Decision Tree':
        st.image('confusion_mat_dt.svg')
      elif model_selection == 'Random Forest':
        st.image('confusion_mat_rnd_forest.svg')
      elif model_selection == 'AdaBoost':
        st.image('confusion_mat_ada.svg')
      elif model_selection == 'Soft Voting Classifier':
        st.image('confusion_mate_sft_vt.svg')
      else:
        st.write('No model selected')

      st.caption("Confusion matrix results")

  with tab3:
      st.write("### Classification Report")
      
      if model_selection == 'Decision Tree':
        df = pd.read_csv('class_report_dt.csv')
        st.write(df)
      elif model_selection == 'Random Forest':
        df = pd.read_csv('class_report_rnd_forest.csv')
        st.write(df)
      elif model_selection == 'AdaBoost':
        df = pd.read_csv('class_report_ada.csv')
        st.write(df)
      elif model_selection == 'Soft Voting Classifier':
        df = pd.read_csv('class_report_sft_vt.csv')
        st.write(df)
      else:
        st.write('No model selected')

      st.caption("Classification report results")