import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk


# Download NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)


# Load saved models and data
@st.cache_resource
def load_models():
    save_dir = r"C:\Users\Varalakshmi\Desktop\new_new_project\saved_models"

    with open(f"{save_dir}/disease_model.pkl", 'rb') as f:
        model = pickle.load(f)

    with open(f"{save_dir}/label_encoder.pkl", 'rb') as f:
        label_encoder = pickle.load(f)

    with open(f"{save_dir}/symptoms_list.pkl", 'rb') as f:
        symptoms_list = pickle.load(f)

    with open(f"{save_dir}/diseases_data.pkl", 'rb') as f:
        diseases_data = pickle.load(f)

    # ── KEY FIX: align symptoms_list length to exactly what the model expects.
    #    symptoms_list.pkl (136 items) and the trained model (135 features) are
    #    out of sync.  We reconcile here at load-time so the mismatch never
    #    reaches predict_disease().
    n_features = model.n_features_in_          # ground-truth: what the model needs
    symptoms_list = list(symptoms_list)        # ensure it is a plain list
    if len(symptoms_list) > n_features:
        # Extra symptom(s) were added after training — drop the extras from the end
        symptoms_list = symptoms_list[:n_features]
    elif len(symptoms_list) < n_features:
        # Fewer symptoms than expected — pad with safe placeholder names
        symptoms_list += [
            f"unknown_symptom_{i}"
            for i in range(n_features - len(symptoms_list))
        ]
    # Now len(symptoms_list) == n_features guaranteed

    description = pd.read_csv(f"{save_dir}/description.csv")
    precautions = pd.read_csv(f"{save_dir}/precautions.csv")
    medications = pd.read_csv(f"{save_dir}/medications.csv")
    workout = pd.read_csv(f"{save_dir}/workout.csv")

    return model, label_encoder, symptoms_list, diseases_data, description, precautions, medications, workout


# NLP Processor Class
class SymptomNLPProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        self.symptom_synonyms = {
            'headache': ['head_pain', 'headache', 'head_ache', 'migraine_pain'],
            'fever': ['high_temperature', 'fever', 'pyrexia', 'feverish'],
            'cough': ['coughing', 'cough'],
            'cold': ['common_cold', 'runny_nose', 'cold'],
            'stomach_pain': ['stomach_ache', 'abdominal_pain', 'tummy_ache', 'stomach_pain'],
            'vomiting': ['vomiting', 'throwing_up', 'nausea_vomiting'],
            'fatigue': ['tiredness', 'fatigue', 'exhaustion', 'weakness', 'tired'],
            'weight_gain': ['gaining_weight', 'weight_gain', 'increased_weight', 'obesity'],
            'weight_loss': ['losing_weight', 'weight_loss', 'decreased_weight'],
            'skin_rash': ['rash', 'skin_rash', 'skin_eruption', 'rashes'],
            'itching': ['itchy', 'itching', 'scratching', 'itch'],
            'joint_pain': ['joint_ache', 'joint_pain', 'arthralgia', 'painful_joints'],
            'chest_pain': ['chest_ache', 'chest_pain', 'chest_discomfort'],
            'breathlessness': ['shortness_of_breath', 'breathing_difficulty', 'breathlessness', 'dyspnea'],
            'nausea': ['nausea', 'feeling_sick', 'queasy', 'nauseated'],
            'diarrhea': ['loose_stools', 'diarrhea', 'watery_stool'],
            'constipation': ['constipation', 'difficulty_passing_stool', 'hard_stool'],
            'dizziness': ['dizziness', 'vertigo', 'lightheaded', 'dizzy'],
            'anxiety': ['anxiety', 'anxious', 'nervousness', 'worried'],
            'depression': ['depression', 'depressed', 'sad', 'low_mood'],
            'irregular_periods': ['irregular_menstruation', 'irregular_periods', 'period_problems', 'menstrual_irregularity'],
            'acne': ['acne', 'pimples', 'skin_breakout', 'zits'],
            'hair_loss': ['hair_fall', 'hair_loss', 'balding', 'losing_hair'],
            'mood_swings': ['mood_changes', 'mood_swings', 'emotional_changes'],
            'back_pain': ['backache', 'back_pain', 'lower_back_pain'],
            'high_fever': ['high_fever', 'very_high_temperature', 'severe_fever'],
            'muscle_pain': ['muscle_ache', 'muscle_pain', 'myalgia', 'body_pain'],
            'chills': ['shivering', 'chills', 'feeling_cold', 'rigor'],
            'sweating': ['excessive_sweating', 'sweating', 'night_sweats', 'perspiration'],
            'loss_of_appetite': ['no_appetite', 'loss_of_appetite', 'not_hungry', 'anorexia'],
            'yellowish_skin': ['jaundice', 'yellowish_skin', 'yellow_skin', 'yellowing'],
            'dark_urine': ['dark_urine', 'brown_urine', 'cola_colored_urine'],
            'abdominal_pain': ['belly_pain', 'abdominal_pain', 'stomach_ache', 'tummy_pain'],
        }

        self.symptom_to_standard = {}
        for standard, synonyms in self.symptom_synonyms.items():
            for syn in synonyms:
                self.symptom_to_standard[syn.lower().replace('_', ' ')] = standard
                self.symptom_to_standard[syn.lower()] = standard

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens
                  if token not in self.stop_words and len(token) > 2]
        return tokens

    def extract_symptoms(self, text, all_symptoms):
        text_lower = text.lower()
        tokens = self.preprocess_text(text)
        found_symptoms = []

        for symptom in all_symptoms:
            symptom_clean = symptom.replace('_', ' ').lower()
            if symptom_clean in text_lower:
                found_symptoms.append(symptom)

        for key, standard in self.symptom_to_standard.items():
            if key in text_lower and standard not in found_symptoms:
                if standard in all_symptoms:
                    found_symptoms.append(standard)

        for token in tokens:
            for symptom in all_symptoms:
                symptom_words = symptom.lower().replace('_', ' ').split()
                if token in symptom_words and symptom not in found_symptoms:
                    found_symptoms.append(symptom)

        return list(set(found_symptoms))


# Rule-Based QA System
class RuleBasedQA:
    def __init__(self, diseases_data, description, precautions, medications, workout):
        self.diseases_data = diseases_data
        self.description = description
        self.precautions = precautions
        self.medications = medications
        self.workout = workout

        self.question_patterns = {
            'causes': [r'what causes', r'why do i have', r'cause of', r'reason for', r'what is the cause'],
            'symptoms': [r'symptoms of', r'signs of', r'how do i know', r'what are the symptoms'],
            'treatment': [r'how to treat', r'treatment for', r'how to cure', r'medicine for', r'medication'],
            'prevention': [r'how to prevent', r'prevention', r'precaution', r'avoid getting'],
            'diet': [r'what to eat', r'diet for', r'food for', r'what should i eat', r'dietary'],
            'exercise': [r'exercise for', r'workout for', r'physical activity', r'what exercise'],
            'home_remedies': [r'home remed', r'natural cure', r'home treatment', r'natural treatment'],
            'risk_factors': [r'risk factor', r'who is at risk', r'chances of getting'],
            'diagnosis': [r'how is .* diagnosed', r'diagnosis of', r'test for', r'how to diagnose'],
            'doctor': [r'when to see doctor', r'when should i', r'emergency', r'serious'],
            'description': [r'what is', r'tell me about', r'explain', r'describe', r'information about']
        }

        self.greeting_patterns = [r'^hi\b', r'^hello\b', r'^hey\b', r'^good morning',
                                   r'^good afternoon', r'^good evening', r'^greetings']
        self.farewell_patterns = [r'\bbye\b', r'\bgoodbye\b', r'\bsee you\b',
                                   r'\btake care\b', r'\bthank you\b', r'\bthanks\b']

    def find_disease_in_query(self, query):
        query_lower = query.lower()
        for disease in self.diseases_data.keys():
            if disease.lower() in query_lower:
                return disease
        return None

    def classify_question(self, query):
        query_lower = query.lower()
        for q_type, patterns in self.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return q_type
        return 'general'

    def is_greeting(self, query):
        query_lower = query.lower().strip()
        for pattern in self.greeting_patterns:
            if re.search(pattern, query_lower):
                return True
        return False

    def is_farewell(self, query):
        query_lower = query.lower()
        for pattern in self.farewell_patterns:
            if re.search(pattern, query_lower):
                return True
        return False

    def get_disease_info(self, disease, info_type):
        if disease not in self.diseases_data:
            return None

        disease_info = self.diseases_data[disease]

        if info_type == 'causes':
            return disease_info.get('causes', 'Information not available.')
        elif info_type == 'risk_factors':
            return disease_info.get('risk_factors', 'Information not available.')
        elif info_type == 'diagnosis':
            return disease_info.get('diagnosis', 'Information not available.')
        elif info_type == 'home_remedies':
            return disease_info.get('home_remedies', 'Information not available.')
        elif info_type == 'diet':
            return disease_info.get('diet', 'Information not available.')
        elif info_type == 'doctor':
            return disease_info.get('when_to_see_doctor', 'Consult a doctor if symptoms persist.')
        elif info_type == 'description':
            desc_row = self.description[self.description['Disease'] == disease]
            if not desc_row.empty:
                return desc_row['Description'].values[0]
            return disease_info.get('causes', 'Information not available.')
        elif info_type in ['prevention', 'precaution']:
            prec_row = self.precautions[self.precautions['Disease'] == disease]
            if not prec_row.empty:
                precs = [prec_row[f'Precaution_{i}'].values[0] for i in range(1, 5)
                         if f'Precaution_{i}' in prec_row.columns and pd.notna(prec_row[f'Precaution_{i}'].values[0])]
                return ', '.join(precs) if precs else 'Information not available.'
            return 'Information not available.'
        elif info_type == 'treatment':
            med_row = self.medications[self.medications['Disease'] == disease]
            if not med_row.empty:
                return med_row['Medication'].values[0]
            return 'Please consult a doctor for appropriate medication.'
        elif info_type == 'exercise':
            workout_rows = self.workout[self.workout['Disease'] == disease]
            if not workout_rows.empty:
                return '\n• ' + '\n• '.join(workout_rows['workout'].tolist())
            return 'General light exercise recommended. Consult your doctor.'

        return None

    def answer_question(self, query, predicted_disease=None):
        if self.is_greeting(query):
            return "Hello! 👋 I'm your health assistant. Describe your symptoms, and I'll help predict possible conditions. You can also ask about diseases, treatments, diet, and more!"

        if self.is_farewell(query):
            return "Take care! 🌟 Remember to consult a healthcare professional for proper diagnosis. Stay healthy!"

        disease = self.find_disease_in_query(query)
        if disease is None and predicted_disease:
            disease = predicted_disease

        if disease is None:
            return None

        q_type = self.classify_question(query)
        answer = self.get_disease_info(disease, q_type)

        if answer:
            return f"**{disease}** - {q_type.replace('_', ' ').title()}:\n\n{answer}"

        return f"I have information about {disease}. Ask about causes, symptoms, treatment, diet, exercise, prevention, or when to see a doctor."


# Main Chatbot Class
class MedicalChatbot:
    def __init__(self, model, label_encoder, nlp_processor, qa_system,
                 symptoms_list, description, precautions, medications, workout, diseases_data):
        self.model = model
        self.label_encoder = label_encoder
        self.nlp_processor = nlp_processor
        self.qa_system = qa_system
        self.symptoms_list = symptoms_list   # already aligned to n_features_in_ at load time
        self.description = description
        self.precautions = precautions
        self.medications = medications
        self.workout = workout
        self.diseases_data = diseases_data
        self.current_disease = None

    def predict_disease(self, symptoms):
        # symptoms_list is already the same length as n_features_in_ (fixed in load_models)
        # but we keep n_features_in_ as the authoritative size for safety
        n_features = self.model.n_features_in_
        feature_vector = np.zeros(n_features)

        for symptom in symptoms:
            if symptom in self.symptoms_list:
                idx = self.symptoms_list.index(symptom)
                if idx < n_features:
                    feature_vector[idx] = 1

        prediction = self.model.predict([feature_vector])[0]
        probabilities = self.model.predict_proba([feature_vector])[0]

        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_diseases = [(self.label_encoder.classes_[i], probabilities[i])
                        for i in top_indices if probabilities[i] > 0.05]

        return top_diseases

    def get_disease_details(self, disease):
        details = {}

        desc_row = self.description[self.description['Disease'] == disease]
        if not desc_row.empty:
            details['description'] = desc_row['Description'].values[0]

        prec_row = self.precautions[self.precautions['Disease'] == disease]
        if not prec_row.empty:
            details['precautions'] = [prec_row[f'Precaution_{i}'].values[0]
                                      for i in range(1, 5)
                                      if f'Precaution_{i}' in prec_row.columns
                                      and pd.notna(prec_row[f'Precaution_{i}'].values[0])]

        med_row = self.medications[self.medications['Disease'] == disease]
        if not med_row.empty:
            details['medications'] = med_row['Medication'].values[0]

        workout_rows = self.workout[self.workout['Disease'] == disease]
        if not workout_rows.empty:
            details['workout'] = workout_rows['workout'].tolist()

        if disease in self.diseases_data:
            details['causes'] = self.diseases_data[disease].get('causes', '')
            details['diet'] = self.diseases_data[disease].get('diet', '')
            details['home_remedies'] = self.diseases_data[disease].get('home_remedies', '')
            details['when_to_see_doctor'] = self.diseases_data[disease].get('when_to_see_doctor', '')

        return details

    def chat(self, user_input):
        qa_response = self.qa_system.answer_question(user_input, self.current_disease)
        if qa_response:
            return qa_response, None, None

        extracted_symptoms = self.nlp_processor.extract_symptoms(user_input, self.symptoms_list)

        if not extracted_symptoms:
            return "I couldn't identify specific symptoms. Please describe them clearly, e.g., 'I have headache, fever, and fatigue'", None, None

        predictions = self.predict_disease(extracted_symptoms)

        if not predictions:
            return "I couldn't make a prediction. Please try different symptoms.", None, None

        top_disease, top_confidence = predictions[0]
        self.current_disease = top_disease
        details = self.get_disease_details(top_disease)

        return None, (top_disease, top_confidence, details, extracted_symptoms), predictions[1:] if len(predictions) > 1 else []


# ─────────────────────────────────────────────
#  PAGE FUNCTION
# ─────────────────────────────────────────────
def medical_chatbot_page():
    """Render the Medical Health Assistant Streamlit page."""

    # NLTK downloads
    download_nltk_data()



    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        .stTextInput > div > div > input {
            font-size: 1.1rem;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        .user-message {
            background-color: #e3f2fd;
            border-left: 4px solid #1976d2;
        }
        .bot-message {
            background-color: #f5f5f5;
            border-left: 4px solid #4caf50;
        }
        .disclaimer {
            background-color: #fff3e0;
            padding: 1rem;
            border-radius: 5px;
            border-left: 4px solid #ff9800;
            margin-top: 1rem;
        }
        .symptom-tag {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            margin: 0.25rem;
            background-color: #e3f2fd;
            border-radius: 15px;
            font-size: 0.9rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── Session state ──────────────────────────────────────────────────────────
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_disease' not in st.session_state:
        st.session_state.current_disease = None

    # ── Load models ────────────────────────────────────────────────────────────
    try:
        (model, label_encoder, symptoms_list, diseases_data,
         description, precautions, medications, workout) = load_models()

        nlp_processor = SymptomNLPProcessor()
        qa_system = RuleBasedQA(diseases_data, description, precautions, medications, workout)
        chatbot = MedicalChatbot(
            model, label_encoder, nlp_processor, qa_system,
            symptoms_list, description, precautions, medications, workout, diseases_data
        )
        # Sync disease context from session state across reruns
        chatbot.current_disease = st.session_state.current_disease
        models_loaded = True
    except Exception as e:
        models_loaded = False
        error_message = str(e)

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/heart-health.png", width=80)
        st.title("🏥 Health Assistant")
        st.markdown("---")

        st.markdown("### 💡 How to Use")
        st.markdown("""
        1. **Describe symptoms** in natural language
        2. **Ask questions** about diseases
        3. **Get recommendations** for diet & exercise
        """)

        st.markdown("---")
        st.markdown("### 📝 Example Queries")
        st.markdown("""
        - "I have headache and fever"
        - "What causes diabetes?"
        - "Diet for PCOD"
        - "When to see doctor for pneumonia"
        """)

        st.markdown("---")
        st.markdown("### 🩺 Supported Conditions")
        if models_loaded:
            with st.expander("View all conditions"):
                for disease in sorted(diseases_data.keys()):
                    st.markdown(f"• {disease}")

        st.markdown("---")
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.current_disease = None
            st.rerun()

    # ── Main content ───────────────────────────────────────────────────────────
    st.markdown('<h1 class="main-header">🏥 Medical Health Assistant</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Describe your symptoms or ask health-related questions</p>',
        unsafe_allow_html=True
    )

    if not models_loaded:
        st.error(f"⚠️ Error loading models: {error_message}")
        st.info("Please ensure you've run the Jupyter notebook to train and save the models first.")
        st.stop()

    # Display chat history
    for message in st.session_state.messages:
        avatar = "🧑" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # ── Chat input ─────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Describe your symptoms or ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            qa_response, prediction_data, other_predictions = chatbot.chat(prompt)

            if qa_response:
                st.markdown(qa_response)
                response_content = qa_response

            elif prediction_data:
                disease, confidence, details, symptoms = prediction_data
                st.session_state.current_disease = disease

                # Symptoms found
                st.markdown("**🔍 Symptoms Identified:**")
                symptoms_html = " ".join(
                    [f'<span class="symptom-tag">{s.replace("_", " ")}</span>' for s in symptoms]
                )
                st.markdown(symptoms_html, unsafe_allow_html=True)
                st.markdown("---")

                # Prediction result
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"## 🏥 Predicted: **{disease}**")
                with col2:
                    st.metric("Confidence", f"{confidence:.1%}")

                # Info tabs
                tabs = st.tabs([
                    "📋 Overview", "💊 Treatment",
                    "🥗 Diet & Exercise", "🏠 Home Remedies", "👨‍⚕️ When to See Doctor"
                ])

                with tabs[0]:
                    if 'description' in details:
                        st.markdown(details['description'])
                    if details.get('causes'):
                        st.markdown("**Causes:**")
                        st.markdown(details['causes'])

                with tabs[1]:
                    if 'medications' in details:
                        st.markdown("**Medications:**")
                        st.markdown(details['medications'])
                    if 'precautions' in details:
                        st.markdown("**Precautions:**")
                        for p in details['precautions']:
                            st.markdown(f"• {p}")

                with tabs[2]:
                    if details.get('diet'):
                        st.markdown("**Diet Recommendations:**")
                        st.markdown(details['diet'])
                    if 'workout' in details:
                        st.markdown("**Exercise Recommendations:**")
                        for w in details['workout'][:5]:
                            st.markdown(f"• {w}")

                with tabs[3]:
                    if details.get('home_remedies'):
                        st.markdown(details['home_remedies'])
                    else:
                        st.info("No specific home remedies available. Please consult a healthcare provider.")

                with tabs[4]:
                    if details.get('when_to_see_doctor'):
                        st.warning(details['when_to_see_doctor'])
                    else:
                        st.warning("Consult a doctor if symptoms persist or worsen.")

                # Other possible conditions
                if other_predictions:
                    st.markdown("---")
                    st.markdown("**Other Possible Conditions:**")
                    for d, c in other_predictions:
                        st.markdown(f"• {d} ({c:.1%})")

                response_content = f"Predicted: {disease} ({confidence:.1%})"

            else:
                st.markdown("I couldn't process that. Please try again.")
                response_content = "Unable to process request."

            # Disclaimer
            st.markdown("""
            <div class="disclaimer">
            ⚠️ <strong>Disclaimer:</strong> This is for informational purposes only and should not replace
            professional medical advice. Please consult a healthcare provider for proper diagnosis and treatment.
            </div>
            """, unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": response_content})

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🏥 Medical Health Assistant | Built with Streamlit</p>
        <p>⚠️ Always consult a healthcare professional for medical advice</p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Entry point (run this file directly)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    medical_chatbot_page()
