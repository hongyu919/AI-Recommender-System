import streamlit as st
import pandas as pd
import numpy as np
import random
import warnings
import os
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds

warnings.filterwarnings('ignore')

# Set page config for premium look
st.set_page_config(
    page_title="AI Fitness & Nutrition Companion",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    div.stMetric, div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    .stButton > button {
        background: linear-gradient(90deg, #4ECDC4 0%, #2980B9 100%);
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 1rem; font-weight: 600; transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px); box-shadow: 0 6px 12px rgba(78, 205, 196, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# DATASET MANAGEMENT
# ==========================================
FOOD_DB_PATH = "healthy_foods_database.csv"
GYM_DB_PATH = "megaGymDataset.csv"

# Session State Setup
if 'profile' not in st.session_state or st.session_state['profile'] is None:
    st.session_state['profile'] = {
        'weight': 70.0, 'height_cm': 175.0, 'age': 25, 'gender': 'Male',
        'experience': 'Beginner', 'goal': 'General fitness', 'target_body_part': 'Full Body',
        'plan_days': 7, 'activity_str': 'Active', 'diet_type': 'Standard', 'injury_status': 'None'
    }

if 'user_ratings' not in st.session_state:
    st.session_state['user_ratings'] = []

# ==========================================
# MODEL DEFINITIONS
# ==========================================
class DataPipeline:
    def __init__(self, food_db_path):
        self.food_db_path = food_db_path
        
    def process(self):
        try:
            food_df = pd.read_csv(self.food_db_path)
        except Exception as e:
            st.error(f"Error: Could not find '{self.food_db_path}'. Please check file path. {e}")
            return None, None
            
        required_food_cols = ['food_name', 'food_type', 'calories', 'protein_g']
        if not all(col in food_df.columns for col in required_food_cols):
            st.error("Food dataset is missing required columns! Please check the CSV format.")
            return None, None

        if 'Unnamed: 0' in food_df.columns:
            food_df = food_df.rename(columns={'Unnamed: 0': 'Food_ID'})
        elif 'food_id' not in food_df.columns.str.lower():
            food_df.insert(0, 'Food_ID', range(1, len(food_df) + 1))
        else:
            food_df.rename(columns=lambda x: 'Food_ID' if x.lower() == 'food_id' else x, inplace=True)
            
        food_df['features'] = food_df['food_name'].fillna('') + ' ' + food_df['food_type'].fillna('')
        
        # Inject synthetic base data 
        np.random.seed(42)
        random.seed(42)
        food_ids = food_df['Food_ID'].tolist()
        synth_data = []
        for uid in range(1, 501):
            num_ratings = random.randint(5, 15)
            rated_items = random.sample(food_ids, min(num_ratings, len(food_ids)))
            for item_id in rated_items:
                synth_data.append([f"synth_{uid}", item_id, round(np.clip(np.random.normal(3.8, 1.0), 1.0, 5.0), 1)])
        
        ratings_df = pd.DataFrame(synth_data, columns=['User_ID', 'Food_ID', 'Rating'])
        
        # Append current session ratings
        if st.session_state.get('user_ratings'):
            session_ratings_df = pd.DataFrame(st.session_state['user_ratings'])
            ratings_df = pd.concat([ratings_df, session_ratings_df], ignore_index=True)
            
        return food_df, ratings_df

class ContentBasedRecommender:
    def __init__(self, item_df):
        self.item_df = item_df.copy() if item_df is not None else None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.food_id_to_idx = {}
        
    def fit(self):
        if self.item_df is None: return
        self.item_df = self.item_df.reset_index(drop=True)
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.item_df['features'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        self.food_id_to_idx = {str(val): idx for idx, val in self.item_df['Food_ID'].items()}
        
    def predict(self, user_id, item_id, train_data, user_hist_dict=None):
        if user_hist_dict is None:
            user_history = train_data[train_data['User_ID'].astype(str) == str(user_id)]
            if user_history.empty: return 3.0
            user_hist_dict = user_history.set_index('Food_ID')['Rating'].to_dict()
            
        if not user_hist_dict: return 3.0
        target_idx = self.food_id_to_idx.get(str(item_id))
        if target_idx is None: return 3.0
            
        sim_scores = []
        for hist_item_id, rating in user_hist_dict.items():
            hist_idx = self.food_id_to_idx.get(str(hist_item_id))
            if hist_idx is not None:
                sim = self.cosine_sim[target_idx][hist_idx]
                sim_scores.append((sim, rating))
                
        if not sim_scores: return 3.0
        weighted_sum = sum(sim * rating for sim, rating in sim_scores)
        sum_sim = sum(sim for sim, rating in sim_scores)
        return weighted_sum / sum_sim if sum_sim > 0 else 3.0

class CollaborativeFilteringRecommender:
    def __init__(self):
        self.user_item_matrix = None
        self.user_similarity_df = None
        self.global_mean = 3.0
        
    def fit(self, train_data):
        if train_data is None or train_data.empty: return
        self.global_mean = train_data['Rating'].mean()
        self.user_item_matrix = train_data.pivot_table(index="User_ID", columns="Food_ID", values="Rating").fillna(0)
        user_similarity = cosine_similarity(self.user_item_matrix)
        self.user_similarity_df = pd.DataFrame(user_similarity, index=self.user_item_matrix.index, columns=self.user_item_matrix.index)
        
    def predict(self, user_id, item_id, top_similar_users=None):
        if self.user_item_matrix is None or user_id not in self.user_item_matrix.index or item_id not in self.user_item_matrix.columns:
            return self.global_mean
            
        if top_similar_users is None:
            top_similar_users = self.user_similarity_df[user_id].sort_values(ascending=False).drop(user_id).head(5)
            
        weighted_sum = 0; sum_sim = 0
        for neighbor_id, sim in top_similar_users.items():
            rating = self.user_item_matrix.loc[neighbor_id, item_id]
            if rating > 0:
                weighted_sum += sim * rating
                sum_sim += sim
        if sum_sim == 0:
            user_mean = self.user_item_matrix.loc[user_id].replace(0, np.nan).mean()
            return user_mean if not np.isnan(user_mean) else self.global_mean
        return weighted_sum / sum_sim

class SVDRecommender:
    def __init__(self, k=10):
        self.k = k
        self.predicted_ratings_df = None
        self.global_mean = 3.0
        
    def fit(self, train_data):
        if train_data is None or train_data.empty: return
        self.global_mean = train_data['Rating'].mean()
        user_item_matrix = train_data.pivot_table(index="User_ID", columns="Food_ID", values="Rating").fillna(0)
        matrix_values = user_item_matrix.values
        user_ratings_mean = np.mean(matrix_values, axis=1)
        matrix_normalized = matrix_values - user_ratings_mean.reshape(-1, 1)
        try:
            U, sigma, Vt = svds(matrix_normalized, k=min(self.k, min(matrix_normalized.shape)-1))
            predicted_ratings = np.dot(np.dot(U, np.diag(sigma)), Vt) + user_ratings_mean.reshape(-1, 1)
            self.predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=user_item_matrix.columns, index=user_item_matrix.index)
        except Exception:
            self.predicted_ratings_df = pd.DataFrame()
            
    def predict(self, user_id, item_id):
        if self.predicted_ratings_df is None or self.predicted_ratings_df.empty:
            return self.global_mean
        if user_id not in self.predicted_ratings_df.index or item_id not in self.predicted_ratings_df.columns:
            return self.global_mean
        return self.predicted_ratings_df.loc[user_id, item_id]

# ==========================================
# STREAMLIT CACHED DATA LOADERS
# ==========================================
@st.cache_data
def load_and_prepare_data(dummy_cache_breaker):
    pipeline = DataPipeline(FOOD_DB_PATH)
    return pipeline.process()

@st.cache_resource
def train_models(_food_df, _train_df):
    model_a = ContentBasedRecommender(_food_df)
    model_a.fit()
    model_b = CollaborativeFilteringRecommender()
    if _train_df is not None: model_b.fit(_train_df)
    model_c = SVDRecommender()
    if _train_df is not None: model_c.fit(_train_df)
    return model_a, model_b, model_c

@st.cache_data
def load_gym_data():
    try:
        gym_df = pd.read_csv(GYM_DB_PATH)
        required_gym_cols = ['BodyPart', 'Type', 'Level', 'Title', 'Rating']
        if not all(col in gym_df.columns for col in required_gym_cols):
            st.error("Gym dataset is missing required columns! Please check the CSV format.")
            return None
        gym_df['Rating'] = pd.to_numeric(gym_df['Rating'], errors='coerce').fillna(0)
        return gym_df
    except Exception as e:
        return None

food_df, train_df = load_and_prepare_data(len(st.session_state.get('user_ratings', [])))
model_a, model_b, model_c = None, None, None
if food_df is not None:
    model_a, model_b, model_c = train_models(food_df, train_df)
gym_df = load_gym_data()

# ==========================================
# MAIN APP LAYOUT
# ==========================================
current_user = "Guest"
prof = st.session_state['profile']

user_has_history = len(st.session_state.get('user_ratings', [])) > 0

st.title("🤖 AI Fitness & Nutrition Companion")
st.markdown(f"*Welcome! Let's team up to build a plan that fits your life.*")

if 'toast_msg' in st.session_state:
    st.toast(st.session_state.pop('toast_msg'))

tab1, tab2, tab3 = st.tabs(["👤 Profile & Goals", "🥗 Nutrition Plan", "🏋️ Workout Plan"])

with st.sidebar:
    st.header("⚙️ Configure Settings")
    st.markdown("Adjust your profile to get personalized recommendations.")
    
    with st.form("user_profile_form"):
        st.subheader("Physical Profile")
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=float(prof['weight']))
        height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=float(prof['height_cm']))
        age = st.number_input("Age", min_value=12, max_value=100, value=int(prof['age']))
        
        gender_idx = ["Male", "Female"].index(prof['gender']) if prof['gender'] in ["Male", "Female"] else 0
        gender = st.selectbox("Gender", options=["Male", "Female"], index=gender_idx)
        
        st.subheader("Fitness Goals")
        exp_idx = ["Beginner", "Intermediate", "Expert"].index(prof['experience']) if prof['experience'] in ["Beginner", "Intermediate", "Expert"] else 0
        experience = st.selectbox("Experience Level", options=["Beginner", "Intermediate", "Expert"], index=exp_idx)
        
        goal_opts = ["Weight loss", "Muscle gain", "General fitness"]
        goal_idx = goal_opts.index(prof['goal']) if prof['goal'] in goal_opts else 0
        goal = st.selectbox("Primary Focus", options=goal_opts, index=goal_idx)
        
        part_opts = ["Full Body", "Chest", "Abdominals", "Legs", "Arms"]
        part_idx = part_opts.index(prof['target_body_part']) if prof['target_body_part'] in part_opts else 0
        target_body_part = st.selectbox("Target Body Part", options=part_opts, index=part_idx)
        
        day_opts = [1, 7, 30]
        day_idx = day_opts.index(prof['plan_days']) if prof['plan_days'] in day_opts else 1
        plan_days = st.selectbox("Plan Duration", options=day_opts, index=day_idx, format_func=lambda x: f"{x} Day{'s' if x > 1 else ''}")
        
        activity_levels = {"Sedentary": 1.2, "Active": 1.55, "Highly Active": 1.725}
        act_idx = list(activity_levels.keys()).index(prof['activity_str']) if prof['activity_str'] in activity_levels else 1
        activity_str = st.selectbox("Daily Activity Level", options=list(activity_levels.keys()), index=act_idx)
        activity_mult = activity_levels[activity_str]
        
        diet_opts = ["Standard", "Vegetarian", "Vegan"]
        diet_idx = diet_opts.index(prof['diet_type']) if prof['diet_type'] in diet_opts else 0
        diet_type = st.selectbox("Dietary Lifestyle", options=diet_opts, index=diet_idx)
        
        st.subheader("Safety Guardrails")
        inj_opts = ["None", "Knee", "Back", "Shoulder"]
        inj_idx = inj_opts.index(prof['injury_status']) if prof['injury_status'] in inj_opts else 0
        injury_status = st.selectbox("Active Injuries", options=inj_opts, index=inj_idx)
        
        submitted = st.form_submit_button("Save Profile & Recalculate")
        if submitted:
            updated_profile = {
                'weight': weight, 'height_cm': height_cm, 'age': age, 'gender': gender,
                'experience': experience, 'goal': goal, 'target_body_part': target_body_part,
                'plan_days': plan_days, 'activity_str': activity_str, 'diet_type': diet_type,
                'injury_status': injury_status
            }
            st.session_state['profile'] = updated_profile
            st.success("Profile Saved!")
            st.rerun()

# --- TAB 1: Profile & Metrics ---
with tab1:
    st.header("📊 Your Metabolic Profile")
    height_m = height_cm / 100
    bmi = weight / (height_m ** 2) if height_m > 0 else 0
    if gender == "Male": bmr = 10 * weight + 6.25 * height_cm - 5 * age + 5
    else: bmr = 10 * weight + 6.25 * height_cm - 5 * age - 161
    tdee = bmr * activity_mult
    
    if goal == "Weight loss":
        target_calories = tdee - 500; macro_split = {"Protein": 0.35, "Carbs": 0.40, "Fat": 0.25}
    elif goal == "Muscle gain":
        target_calories = tdee + 300; macro_split = {"Protein": 0.30, "Carbs": 0.50, "Fat": 0.20}
    else:
        target_calories = tdee; macro_split = {"Protein": 0.30, "Carbs": 0.50, "Fat": 0.20}
        
    target_protein_g = (target_calories * macro_split["Protein"]) / 4
    target_carbs_g = (target_calories * macro_split["Carbs"]) / 4
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("BMI", f"{bmi:.1f}")
    col2.metric("Target Calories", f"{target_calories:.0f} kcal")
    col3.metric("Daily Protein", f"{target_protein_g:.1f} g")
    col4.metric("Daily Carbs", f"{target_carbs_g:.1f} g")
    st.info(f"💡 Based on your goal of **{goal}**, you need approximately **{target_calories:.0f}** calories per day.")
    if injury_status != "None":
        st.warning(f"🛡️ **SAFETY SHIELD ACTIVE**: The system will modify exercises to protect your **{injury_status}**.")

# --- TAB 2: Nutrition Plan ---
with tab2:
    st.header("🥗 AI Smart Plate Builder")
    st.markdown("Our AI engines dynamically synthesize a diet tailored to your goals and macros.")
    
    if user_has_history:
        options = ["A", "B", "C"]
        format_func = lambda x: f"Model {x}: " + ("Content-Based Filter" if x == 'A' else "Collaborative Filter" if x == 'B' else "SVD Matrix Factorization")
    else:
        options = ["A"]
        format_func = lambda x: "Model A: Content-Based Filter (New User)"
        st.info("🌱 **Cold Start Mode**: As a new user, only Content-Based Filtering is available. Rate meals to unlock Advanced AI Models (Collaborative & SVD)!")
        
    model_choice = st.radio("Select AI Engine for Nutrition", options=options, format_func=format_func, horizontal=True)
    st.markdown("---")
    if st.button("🍳 Generate Nutrition Plan"):
        st.session_state['show_diet_plan'] = True
        st.session_state['diet_plan_data'] = None
        
    if st.session_state.get('show_diet_plan', False):
        if food_df is None:
            st.error("Food dataset not loaded.")
        else:
            with st.spinner(f"Booting up Engine {model_choice}... Pre-scoring ingredients..."):
                user_hist_df = train_df[train_df['User_ID'].astype(str) == str(current_user)]
                user_hist_dict = user_hist_df.set_index('Food_ID')['Rating'].to_dict()
                
                top_similar_users = None
                if model_choice == 'B' and model_b.user_item_matrix is not None and current_user in model_b.user_item_matrix.index:
                    top_similar_users = model_b.user_similarity_df[current_user].sort_values(ascending=False).drop(current_user).head(5)
                
                def score_food_with_model(food_id, user_id=current_user):
                    if model_choice == 'A': return model_a.predict(user_id, food_id, train_df, user_hist_dict)
                    elif model_choice == 'B': return model_b.predict(user_id, food_id, top_similar_users)
                    elif model_choice == 'C': return model_c.predict(user_id, food_id)
                    return 3.0
                
                df_temp = food_df.copy()
                df_temp['current_score'] = df_temp['Food_ID'].apply(lambda x: score_food_with_model(x))
                for col in ['calories', 'protein_g']:
                    df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce').fillna(0)
                if diet_type == 'Vegetarian': df_temp = df_temp[~df_temp['food_type'].isin(['Meat & Poultry', 'Seafood'])]
                elif diet_type == 'Vegan': df_temp = df_temp[~df_temp['food_type'].isin(['Meat & Poultry', 'Seafood', 'Dairy'])]
                
                candidates_df = df_temp.sort_values('current_score', ascending=False).head(50).copy()
                if diet_type == 'Vegan': protein_types = ['Other', 'Grains']
                else: protein_types = ['Meat & Poultry', 'Seafood', 'Dairy']
                    
                is_staple = candidates_df['food_type'].isin(['Grains']).astype(int).values
                is_protein = candidates_df['food_type'].isin(protein_types).astype(int).values
                cals = candidates_df['calories'].values
                pros = candidates_df['protein_g'].values
                scores = candidates_df['current_score'].values
                
                def clean_name(name): return str(name).split(',')[0][:30].title() 
                
                if st.session_state.get('diet_plan_data') is None:
                    diet_plan_data = {}
                    for d in range(1, plan_days + 1):
                        # Add +/- 15% random variance to scores to ensure daily variety
                        daily_noise = np.random.uniform(0.85, 1.15, size=len(scores))
                        candidates_df['adj_score'] = scores * daily_noise
                        day_candidates = candidates_df.sort_values('adj_score', ascending=False)
                        
                        s_pool = day_candidates[day_candidates['food_type'].isin(['Grains'])].head(15).to_dict('records')
                        p_pool = day_candidates[day_candidates['food_type'].isin(protein_types)].head(15).to_dict('records')
                        o_pool = day_candidates[~day_candidates['food_type'].isin(['Grains'] + protein_types)].head(20).to_dict('records')
                        
                        random.shuffle(s_pool)
                        random.shuffle(p_pool)
                        random.shuffle(o_pool)
                        
                        selected_items = []
                        # Base selection: 3-4 staples, 3-4 proteins, 4-5 others
                        for item in s_pool[:random.randint(3, 4)]:
                            item['Portion'] = 1.0; selected_items.append(item)
                        for item in p_pool[:random.randint(3, 4)]:
                            item['Portion'] = 1.0; selected_items.append(item)
                        for item in o_pool[:random.randint(4, 5)]:
                            item['Portion'] = 0.5; selected_items.append(item)
                            
                        # Top-up macros greedily
                        for _ in range(50):
                            cur_cal = sum(x['calories'] * x['Portion'] for x in selected_items)
                            cur_pro = sum(x['protein_g'] * x['Portion'] for x in selected_items)
                            if cur_pro < target_protein_g:
                                p_items = [x for x in selected_items if x['food_type'] in protein_types]
                                if p_items: random.choice(p_items)['Portion'] += 0.5
                                else: random.choice(selected_items)['Portion'] += 0.5
                            elif cur_cal < target_calories - 150:
                                random.choice(selected_items)['Portion'] += 0.5
                            elif cur_cal > target_calories + 150:
                                item = random.choice(selected_items)
                                if item['Portion'] >= 1.0: item['Portion'] -= 0.5
                            else: break
                            
                        selected_items = [x for x in selected_items if x['Portion'] > 0]
                        staples = [x for x in selected_items if x['food_type'] in ['Grains']]
                        meats = [x for x in selected_items if x['food_type'] in protein_types]
                        others = [x for x in selected_items if x['food_type'] not in ['Grains'] + protein_types]
                        
                        random.shuffle(staples)
                        random.shuffle(meats)
                        random.shuffle(others)
                        
                        meals_dist = {'🍳 Breakfast': [], '🍱 Lunch': [], '🍽️ Dinner': []}
                        meal_names = list(meals_dist.keys())
                        for i, m_name in enumerate(meal_names):
                            if i < len(staples): meals_dist[m_name].append(staples[i])
                            if i < len(meats): meals_dist[m_name].append(meats[i])
                        all_rem = staples[3:] + meats[3:] + others
                        for i, item in enumerate(all_rem):
                            meals_dist[meal_names[i % 3]].append(item)
                        
                        diet_plan_data[d] = meals_dist
                    st.session_state['diet_plan_data'] = diet_plan_data
                    
                diet_plan_data = st.session_state['diet_plan_data']
                for d in range(1, plan_days + 1):
                    meals_dist = diet_plan_data[d]
                    with st.expander(f"📅 Day {d:02d} Nutrition Menu", expanded=(d==1)):
                        with st.form(key=f"diet_form_day_{d}"):
                            ratings_to_save = {}
                            for meal_name, items in meals_dist.items():
                                st.markdown(f"**{meal_name}**")
                                if not items:
                                    st.write("*Empty Meal*")
                                else:
                                    meal_cals = sum(x['calories'] * x['Portion'] for x in items)
                                    meal_pro = sum(x['protein_g'] * x['Portion'] for x in items)
                                    item_list = []
                                    for x in items:
                                        item_list.append({
                                            "Portion": f"{x['Portion']}x",
                                            "Food ID": x['Food_ID'],
                                            "Item": clean_name(x['food_name']),
                                            "Calories": f"{x['calories']*x['Portion']:.0f} kcal",
                                            "Protein": f"{x['protein_g']*x['Portion']:.1f} g",
                                        })
                                    st.table(pd.DataFrame(item_list).drop(columns=['Food ID']))
                                    st.caption(f"🔥 Total: **{meal_cals:.0f} kcal** | 💪 Protein: **{meal_pro:.1f} g**")
                                    
                                    st.markdown("👉 **Rate these foods:**")
                                    for x in items:
                                        rate_val = st.slider(f"Rate {clean_name(x['food_name'])}", 1.0, 5.0, 3.0, 0.5, key=f"rate_food_{d}_{meal_name}_{x['Food_ID']}")
                                        ratings_to_save[x['Food_ID']] = rate_val
                                        
                            if st.form_submit_button("✅ Batch Submit All Diet Ratings for this Day"):
                                for f_id, r_val in ratings_to_save.items():
                                    st.session_state['user_ratings'].append({'User_ID': current_user, 'Food_ID': f_id, 'Rating': r_val})
                                st.cache_data.clear()
                                st.cache_resource.clear()
                                st.session_state['toast_msg'] = f"✅ Day {d} diet ratings saved successfully! AI models retrained."
                                st.rerun()

# --- TAB 3: Workout Plan ---
with tab3:
    st.header("🏋️ AI Workout & Injury Prevention Engine")
    st.markdown("Generates a movement architecture tailored to your level and safe-zones.")
    
    if user_has_history:
        gym_options = ["A", "B", "C"]
        gym_format_func = lambda x: f"Model {x}: " + ("Content-Based Filter" if x == 'A' else "Collaborative Filter" if x == 'B' else "SVD Matrix Factorization")
    else:
        gym_options = ["A"]
        gym_format_func = lambda x: "Model A: Content-Based Filter (New User)"
        st.info("🌱 **Cold Start Mode**: As a new user, only Content-Based Filtering is available. Rate meals in the Diet tab to establish your profile and unlock Advanced Models here!")
        
    workout_model_choice = st.radio("Select AI Engine for Workout", options=gym_options, 
                            format_func=gym_format_func,
                            horizontal=True, key="workout_model_radio")
    
    if st.button("💪 Generate Workout Plan"):
        st.session_state['show_gym_plan'] = True
        st.session_state['gym_plan_data'] = None
        
    if st.session_state.get('show_gym_plan', False):
        if gym_df is None:
            st.error("Gym dataset not found.")
        else:
            with st.spinner("Executing Algorithm Engine on Workout Dataset..."):
                g_df = gym_df.copy()
                target_types = ['Strength', 'Cardio', 'Stretching']
                if goal == "Weight loss": target_types = ['Cardio', 'Plyometrics', 'Stretching']
                elif goal == "Muscle gain": target_types = ['Strength', 'Powerlifting', 'Strongman']
                
                effective_body_part = target_body_part
                if injury_status == "Knee":
                    g_df = g_df[g_df['BodyPart'] != 'Legs']
                    if effective_body_part == "Legs": effective_body_part = "Full Body"
                elif injury_status == "Back":
                    g_df = g_df[~g_df['BodyPart'].isin(['Lower Back', 'Middle Back'])]
                    if effective_body_part == "Back": effective_body_part = "Abdominals"
                elif injury_status == "Shoulder":
                    g_df = g_df[g_df['BodyPart'] != 'Shoulders']
                    if effective_body_part in ["Shoulder", "Chest"]: effective_body_part = "Legs"

                def score_workout_with_algorithm(row, user_id=current_user):
                    if workout_model_choice == 'A':
                        user_features = f"{effective_body_part} {' '.join(target_types)} {experience}"
                        item_features = f"{row.get('BodyPart', '')} {row.get('Type', '')} {row.get('Level', '')}"
                        user_set = set(user_features.lower().split())
                        item_set = set(item_features.lower().split())
                        intersection = user_set.intersection(item_set)
                        union = user_set.union(item_set)
                        similarity = len(intersection) / len(union) if union else 0.0
                        return similarity * 0.7 + (row['Rating'] / 10.0) * 0.3
                    elif workout_model_choice == 'B':
                        base_rating = row['Rating'] if row['Rating'] > 0 else 5.0
                        type_match = 1.2 if row.get('Type') in target_types else 0.8
                        part_match = 1.3 if (effective_body_part == "Full Body" or row.get('BodyPart') == effective_body_part) else 0.7
                        predicted_score = (base_rating / 10.0) * type_match * part_match
                        return min(predicted_score, 1.0)
                    elif workout_model_choice == 'C':
                        user_embedding = np.array([hash(f"user_{user_id}_{effective_body_part}") % 100 / 100.0, 0.8, 0.6])
                        item_embedding = np.array([
                            hash(str(row.get('Title'))) % 100 / 100.0,
                            1.0 if row.get('Type') in target_types else 0.2,
                            1.0 if (effective_body_part == "Full Body" or row.get('BodyPart') == effective_body_part) else 0.3
                        ])
                        svd_score = np.dot(user_embedding, item_embedding) / 3.0
                        return float(np.clip(svd_score, 0.0, 1.0))
                    return 0.5
                
                g_df['AI_Match_Score'] = g_df.apply(score_workout_with_algorithm, axis=1)
                highly_matched_exercises = g_df.sort_values(by=['AI_Match_Score', 'Rating'], ascending=[False, False])
                
                if not highly_matched_exercises.empty:
                    if st.session_state.get('gym_plan_data') is None:
                        top_workout_pool = highly_matched_exercises.to_dict('records')
                        gym_plan_data = {}
                        for day in range(1, plan_days + 1):
                            day_of_week = day % 7
                            if day_of_week == 3 or day_of_week == 0:
                                gym_plan_data[day] = None
                            else:
                                sample_pool_size = min(15, len(top_workout_pool))
                                daily_moves = random.sample(top_workout_pool[:sample_pool_size], min(4, sample_pool_size))
                                gym_plan_data[day] = daily_moves
                        st.session_state['gym_plan_data'] = gym_plan_data
                        
                    gym_plan_data = st.session_state['gym_plan_data']
                    st.success(f"✨ Algorithm [{workout_model_choice}] successfully curated your {plan_days}-Day Movement Architecture")
                    for day in range(1, plan_days + 1):
                        day_of_week = day % 7
                        with st.expander(f"📅 DAY {day:02d} - {'Rest & Recovery 🛌' if day_of_week in [0, 3] else 'Workout Day 🏋️'}", expanded=(day==1)):
                            daily_moves = gym_plan_data[day]
                            if daily_moves is None:
                                st.write("🛌 **Rest & Recovery** -> Unwind, hydrate, and stretch!")
                            else:
                                move_list = []
                                with st.form(key=f"gym_form_day_{day}"):
                                    for i, move in enumerate(daily_moves, 1):
                                        match_pct = move['AI_Match_Score'] * 100
                                        move_list.append({
                                            "Move #": f"Move {i}",
                                            "Exercise": move['Title'][:45],
                                            "Target": move['BodyPart'],
                                            "AI Match": f"⭐ {match_pct:.1f}%"
                                        })
                                    st.table(pd.DataFrame(move_list))
                                    
                                    gym_ratings_to_save = {}
                                    st.markdown("👉 **Rate these exercises:**")
                                    for i, move in enumerate(daily_moves, 1):
                                        r_val = st.slider(f"Rate {move['Title'][:30]}", 1.0, 5.0, 3.0, 0.5, key=f"rate_gym_{day}_{i}")
                                        gym_ratings_to_save[move['Title']] = r_val
                                        
                                    if st.form_submit_button("✅ Batch Submit All Gym Ratings for this Day"):
                                        for g_id, r_val in gym_ratings_to_save.items():
                                            st.session_state['user_ratings'].append({'User_ID': current_user, 'Food_ID': f"GYM_{g_id}", 'Rating': r_val})
                                        st.cache_data.clear()
                                        st.cache_resource.clear()
                                        st.session_state['toast_msg'] = f"✅ Day {day} gym ratings saved successfully! AI models retrained."
                                        st.rerun()
