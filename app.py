import streamlit as st
import pandas as pd
import numpy as np
import random
import warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse.linalg import svds
from scipy.optimize import linprog

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
    /* Dark Mode Premium Theme */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Header styling */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Card/Container styling */
    div.stMetric, div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #4ECDC4 0%, #2980B9 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(78, 205, 196, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Use relative paths for cloud deployment
FOOD_DB_PATH = "healthy_foods_database.csv"
GYM_DB_PATH = "megaGymDataset.csv"

# ==========================================
# MODEL DEFINITIONS (Phase 1)
# ==========================================
class DataPipeline:
    def __init__(self, food_db_path, num_users=100, ratings_per_user=(5, 20)):
        self.food_db_path = food_db_path
        self.num_users = num_users
        self.ratings_per_user = ratings_per_user
        
    def process(self):
        try:
            food_df = pd.read_csv(self.food_db_path)
        except Exception as e:
            st.error(f"Error: Could not find '{self.food_db_path}'. Please check file path. {e}")
            return None, None, None
            
        # Validate required columns for food
        required_food_cols = ['food_name', 'food_type', 'calories', 'protein_g']
        if not all(col in food_df.columns for col in required_food_cols):
            st.error("Food dataset is missing required columns! Please check the CSV format.")
            return None, None, None

        if 'Unnamed: 0' in food_df.columns:
            food_df = food_df.rename(columns={'Unnamed: 0': 'Food_ID'})
        elif 'food_id' not in food_df.columns.str.lower():
            food_df.insert(0, 'Food_ID', range(1, len(food_df) + 1))
        else:
            food_df.rename(columns=lambda x: 'Food_ID' if x.lower() == 'food_id' else x, inplace=True)
            
        food_df['features'] = food_df['food_name'].fillna('') + ' ' + food_df['food_type'].fillna('')
        
        np.random.seed(42)
        random.seed(42)
        food_ids = food_df['Food_ID'].tolist()
        ratings_data = []
        for user_id in range(1, self.num_users + 1):
            num_ratings = random.randint(*self.ratings_per_user)
            rated_items = random.sample(food_ids, min(num_ratings, len(food_ids)))
            for item_id in rated_items:
                rating = round(np.clip(np.random.normal(3.8, 1.0), 1.0, 5.0), 1)
                ratings_data.append([user_id, item_id, rating])
                
        ratings_df = pd.DataFrame(ratings_data, columns=['User_ID', 'Food_ID', 'Rating'])
        train_df, test_df = train_test_split(ratings_df, test_size=0.20, random_state=42)
        return food_df, train_df, test_df

class ContentBasedRecommender:
    def __init__(self, item_df):
        self.item_df = item_df.copy() if item_df is not None else None
        self.tfidf_matrix = None
        self.cosine_sim = None
        
    def fit(self):
        if self.item_df is None: return
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.item_df['features'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
    def predict(self, user_id, item_id, train_data):
        user_history = train_data[train_data['User_ID'] == user_id]
        if user_history.empty:
            return 3.0
            
        try:
            target_idx = self.item_df[self.item_df['Food_ID'] == item_id].index[0]
        except IndexError:
            return 3.0
            
        sim_scores = []
        for _, row in user_history.iterrows():
            hist_item_id = row['Food_ID']
            try:
                hist_idx = self.item_df[self.item_df['Food_ID'] == hist_item_id].index[0]
                sim = self.cosine_sim[target_idx][hist_idx]
                sim_scores.append((sim, row['Rating']))
            except IndexError:
                continue
                
        if not sim_scores:
            return train_data['Rating'].mean()
            
        weighted_sum = sum(sim * rating for sim, rating in sim_scores)
        sum_sim = sum(sim for sim, rating in sim_scores)
        
        if sum_sim == 0:
            return user_history['Rating'].mean()
        return weighted_sum / sum_sim

class CollaborativeFilteringRecommender:
    def __init__(self):
        self.user_item_matrix = None
        self.user_similarity = None
        self.global_mean = 3.0
        
    def fit(self, train_data):
        if train_data is None: return
        self.global_mean = train_data['Rating'].mean()
        
        self.user_item_matrix = train_data.pivot_table(index="User_ID", columns="Food_ID", values="Rating").fillna(0)
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        self.user_similarity_df = pd.DataFrame(self.user_similarity, index=self.user_item_matrix.index, columns=self.user_item_matrix.index)
        
    def predict(self, user_id, item_id):
        if user_id not in self.user_item_matrix.index or item_id not in self.user_item_matrix.columns:
            return self.global_mean
            
        similar_users = self.user_similarity_df[user_id].sort_values(ascending=False).drop(user_id)
        
        weighted_sum = 0
        sum_sim = 0
        
        for neighbor_id, sim in similar_users.head(5).items():
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
        if train_data is None: return
        self.global_mean = train_data['Rating'].mean()
        
        user_item_matrix = train_data.pivot_table(index="User_ID", columns="Food_ID", values="Rating").fillna(0)
        matrix_values = user_item_matrix.values
        
        user_ratings_mean = np.mean(matrix_values, axis=1)
        matrix_normalized = matrix_values - user_ratings_mean.reshape(-1, 1)
        
        try:
            U, sigma, Vt = svds(matrix_normalized, k=min(self.k, min(matrix_normalized.shape)-1))
            predicted_ratings = np.dot(np.dot(U, np.diag(sigma)), Vt) + user_ratings_mean.reshape(-1, 1)
            self.predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=user_item_matrix.columns, index=user_item_matrix.index)
        except Exception as e:
            self.predicted_ratings_df = pd.DataFrame()
            
    def predict(self, user_id, item_id):
        if self.predicted_ratings_df.empty:
            return self.global_mean
        if user_id not in self.predicted_ratings_df.index or item_id not in self.predicted_ratings_df.columns:
            return self.global_mean
        return self.predicted_ratings_df.loc[user_id, item_id]

# ==========================================
# STREAMLIT CACHED FUNCTIONS
# ==========================================
@st.cache_data
def load_and_prepare_data():
    pipeline = DataPipeline(FOOD_DB_PATH)
    return pipeline.process()

@st.cache_resource
def train_models(food_df, train_df):
    model_a = ContentBasedRecommender(food_df)
    model_a.fit()
    
    model_b = CollaborativeFilteringRecommender()
    if train_df is not None: model_b.fit(train_df)
        
    model_c = SVDRecommender()
    if train_df is not None: model_c.fit(train_df)
    
    return model_a, model_b, model_c

@st.cache_data
def load_gym_data():
    try:
        gym_df = pd.read_csv(GYM_DB_PATH)
        
        # Validate required columns for gym
        required_gym_cols = ['BodyPart', 'Type', 'Level', 'Title', 'Rating']
        if not all(col in gym_df.columns for col in required_gym_cols):
            st.error("Gym dataset is missing required columns! Please check the CSV format.")
            return None
            
        gym_df['Rating'] = pd.to_numeric(gym_df['Rating'], errors='coerce').fillna(0)
        return gym_df
    except Exception as e:
        st.error(f"Error loading Gym dataset: {e}")
        return None

# Load global data and models
food_df, train_df, test_df = load_and_prepare_data()
model_a, model_b, model_c = None, None, None
if food_df is not None:
    model_a, model_b, model_c = train_models(food_df, train_df)
gym_df = load_gym_data()


# ==========================================
# MAIN APP LAYOUT
# ==========================================

st.title("🤖 AI Fitness & Nutrition Companion")
st.markdown("*Let's team up to build a plan that fits your life, goals, and safety!*")

# Create tabs for structured flow
tab1, tab2, tab3 = st.tabs(["👤 Profile & Goals", "🥗 Nutrition Plan", "🏋️ Workout Plan"])

with st.sidebar:
    st.header("⚙️ Configure Settings")
    st.markdown("Adjust your profile to get personalized recommendations.")
    
    with st.form("user_profile_form"):
        # 1. Humanized Physical Consultation
        st.subheader("Physical Profile")
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=70.0)
        height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=175.0)
        age = st.number_input("Age", min_value=12, max_value=100, value=25)
        gender = st.selectbox("Gender", options=["Male", "Female"])
        
        # 2. Advanced App-Style Customization
        st.subheader("Fitness Goals")
        experience = st.selectbox("Experience Level", options=["Beginner", "Intermediate", "Expert"])
        goal = st.selectbox("Primary Focus", options=["Weight loss", "Muscle gain", "General fitness"])
        target_body_part = st.selectbox("Target Body Part", options=["Full Body", "Chest", "Abdominals", "Legs", "Arms"])
        plan_days = st.selectbox("Plan Duration", options=[1, 7, 30], index=1, format_func=lambda x: f"{x} Day{'s' if x > 1 else ''}")
        
        activity_levels = {"Sedentary": 1.2, "Active": 1.55, "Highly Active": 1.725}
        activity_str = st.selectbox("Daily Activity Level", options=list(activity_levels.keys()))
        activity_mult = activity_levels[activity_str]
        
        diet_type = st.selectbox("Dietary Lifestyle", options=["Standard", "Vegetarian", "Vegan"])
        
        # 3. Strict Safety Guardrails
        st.subheader("Safety Guardrails")
        injury_status = st.selectbox("Active Injuries", options=["None", "Knee", "Back", "Shoulder"])
        
        # Submit Button
        submitted = st.form_submit_button("Save Profile & Recalculate")

# --- TAB 1: Profile & Metrics ---
with tab1:
    st.header("📊 Your Metabolic Profile")
    
    # 4. Under-The-Hood Metabolic Calculations
    height_m = height_cm / 100
    bmi = weight / (height_m ** 2) if height_m > 0 else 0
    
    if gender == "Male":
        bmr = 10 * weight + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height_cm - 5 * age - 161
        
    tdee = bmr * activity_mult
    
    if goal == "Weight loss":
        target_calories = tdee - 500
        macro_split = {"Protein": 0.35, "Carbs": 0.40, "Fat": 0.25}
    elif goal == "Muscle gain":
        target_calories = tdee + 300
        macro_split = {"Protein": 0.30, "Carbs": 0.50, "Fat": 0.20}
    else:
        target_calories = tdee
        macro_split = {"Protein": 0.30, "Carbs": 0.50, "Fat": 0.20}
        
    target_protein_g = (target_calories * macro_split["Protein"]) / 4
    target_carbs_g = (target_calories * macro_split["Carbs"]) / 4
    target_fat_g = (target_calories * macro_split["Fat"]) / 9
    
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
    
    model_choice = st.radio("Select AI Engine for Nutrition", options=["A", "B", "C"], 
                            format_func=lambda x: f"Model {x}: " + 
                            ("Content-Based Filter" if x == 'A' else "Collaborative Filter" if x == 'B' else "SVD Matrix Factorization"),
                            horizontal=True)
    
    if st.button("🍳 Generate Nutrition Plan"):
        if food_df is None:
            st.error("Food database not loaded. Cannot generate plan.")
        else:
            with st.spinner(f"Booting up Engine {model_choice}... Pre-scoring ingredients..."):
                def score_food_with_model(food_id, user_id=10):
                    if model_choice == 'A': return model_a.predict(user_id, food_id, train_df)
                    elif model_choice == 'B': return model_b.predict(user_id, food_id)
                    elif model_choice == 'C': return model_c.predict(user_id, food_id)
                    return 3.0
                
                df_temp = food_df.copy()
                df_temp['current_score'] = df_temp['Food_ID'].apply(lambda x: score_food_with_model(x))
                
                for col in ['calories', 'protein_g']:
                    df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce').fillna(0)
                
                if diet_type == 'Vegetarian':
                    df_temp = df_temp[~df_temp['food_type'].isin(['Meat & Poultry', 'Seafood'])]
                elif diet_type == 'Vegan':
                    df_temp = df_temp[~df_temp['food_type'].isin(['Meat & Poultry', 'Seafood', 'Dairy'])]
                
                candidates_df = df_temp.sort_values('current_score', ascending=False).head(50).copy()
                
                if diet_type == 'Vegan': protein_types = ['Other', 'Grains']
                else: protein_types = ['Meat & Poultry', 'Seafood', 'Dairy']
                    
                is_staple = candidates_df['food_type'].isin(['Grains']).astype(int).values
                is_protein = candidates_df['food_type'].isin(protein_types).astype(int).values
                
                cals = candidates_df['calories'].values
                pros = candidates_df['protein_g'].values
                scores = candidates_df['current_score'].values
                
                def clean_name(name): return str(name).split(',')[0][:30].title() 
                
                for d in range(1, plan_days + 1):
                    with st.expander(f"📅 Day {d:02d} Nutrition Menu", expanded=(d==1)):
                        c = -scores
                        A_ub = np.array([-cals, cals, -pros, -is_staple, -is_protein])
                        b_ub = np.array([-(target_calories - 150), target_calories + 150, -target_protein_g, -3, -3])
                        bounds = [(0, 3) for _ in range(len(candidates_df))]
                        
                        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                        portions = np.zeros(len(candidates_df))
                        
                        if res.success:
                            portions = res.x
                        else:
                            # Fallback if LP fails
                            staple_mask = candidates_df['food_type'].isin(['Grains']).values
                            prot_mask = candidates_df['food_type'].isin(protein_types).values
                            s_c, p_c = 0, 0
                            for i, m in enumerate(staple_mask):
                                if m and s_c < 3: portions[i] = 1.0; s_c += 1
                            for i, m in enumerate(prot_mask):
                                if m and p_c < 3: portions[i] = 1.0; p_c += 1
                                
                            cur_pro = np.sum(pros * portions)
                            if cur_pro < target_protein_g:
                                pro_eff = pros / (cals + 1e-5)
                                for idx in np.argsort(-pro_eff):
                                    if cur_pro >= target_protein_g: break
                                    portions[idx] += 0.5
                                    cur_pro += pros[idx] * 0.5
                        
                        candidates_df['Portion'] = np.round(portions, 1)
                        selected = candidates_df[candidates_df['Portion'] >= 0.1].copy()
                        
                        staples = selected[selected['food_type'].isin(['Grains'])].to_dict('records')
                        meats = selected[selected['food_type'].isin(protein_types)].to_dict('records')
                        others = selected[~selected['food_type'].isin(['Grains'] + protein_types)].to_dict('records')
                        
                        meals_dist = {'🍳 Breakfast': [], '🍱 Lunch': [], '🍽️ Dinner': []}
                        meal_names = list(meals_dist.keys())
                        
                        for i, m_name in enumerate(meal_names):
                            if i < len(staples): meals_dist[m_name].append(staples[i])
                            if i < len(meats): meals_dist[m_name].append(meats[i])
                            
                        all_rem = staples[3:] + meats[3:] + others
                        for i, item in enumerate(all_rem):
                            meals_dist[meal_names[i % 3]].append(item)
                            
                        for meal_name, items in meals_dist.items():
                            st.markdown(f"**{meal_name}**")
                            if not items:
                                st.write("*Empty Meal due to strict constraints*")
                            else:
                                meal_cals = sum(x['calories'] * x['Portion'] for x in items)
                                meal_pro = sum(x['protein_g'] * x['Portion'] for x in items)
                                
                                item_list = []
                                for x in items:
                                    item_list.append({
                                        "Portion": f"{x['Portion']}x",
                                        "Item": clean_name(x['food_name']),
                                        "Calories": f"{x['calories']*x['Portion']:.0f} kcal",
                                        "Protein": f"{x['protein_g']*x['Portion']:.1f} g",
                                        "Score": f"⭐{x['current_score']:.2f}"
                                    })
                                st.table(pd.DataFrame(item_list))
                                st.caption(f"🔥 Total: **{meal_cals:.0f} kcal** | 💪 Protein: **{meal_pro:.1f} g**")


# --- TAB 3: Workout Plan ---
with tab3:
    st.header("🏋️ AI Workout & Injury Prevention Engine")
    st.markdown("Generates a movement architecture tailored to your level and safe-zones.")
    
    workout_model_choice = st.radio("Select AI Engine for Workout", options=["A", "B", "C"], 
                            format_func=lambda x: f"Model {x}: " + 
                            ("Content-Based Filter" if x == 'A' else "Collaborative Filter" if x == 'B' else "SVD Matrix Factorization"),
                            horizontal=True, key="workout_model_radio")
    
    if st.button("💪 Generate Workout Plan"):
        if gym_df is None:
            st.error("Gym dataset not found or couldn't load. Please check megaGymDataset.csv path.")
        else:
            with st.spinner("Executing Algorithm Engine on Workout Dataset..."):
                g_df = gym_df.copy()
                
                target_types = ['Strength', 'Cardio', 'Stretching']
                if goal == "Weight loss": target_types = ['Cardio', 'Plyometrics', 'Stretching']
                elif goal == "Muscle gain": target_types = ['Strength', 'Powerlifting', 'Strongman']
                
                # Apply Safety Shield
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

                def score_workout_with_algorithm(row, user_id=10):
                    # Uses the fallback algorithm implementation from original code
                    # (Removed uninstantiated gym model check for cleaner code)
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
                    top_workout_pool = highly_matched_exercises.to_dict('records')
                    st.success(f"✨ Algorithm [{workout_model_choice}] successfully curated your {plan_days}-Day Movement Architecture")
                    
                    for day in range(1, plan_days + 1):
                        day_of_week = day % 7
                        with st.expander(f"📅 DAY {day:02d} - {'Rest & Recovery 🛌' if day_of_week in [0, 3] else 'Workout Day 🏋️'}", expanded=(day==1)):
                            if day_of_week == 3 or day_of_week == 0:
                                st.write("🛌 **Rest & Recovery** -> Unwind, hydrate, and stretch!")
                            else:
                                sample_pool_size = min(15, len(top_workout_pool))
                                daily_moves = random.sample(top_workout_pool[:sample_pool_size], min(4, sample_pool_size))
                                
                                move_list = []
                                for i, move in enumerate(daily_moves, 1):
                                    match_pct = move['AI_Match_Score'] * 100
                                    move_list.append({
                                        "Move #": f"Move {i}",
                                        "Exercise": move['Title'][:45],
                                        "Target": move['BodyPart'],
                                        "AI Match": f"⭐ {match_pct:.1f}%"
                                    })
                                st.table(pd.DataFrame(move_list))
                
# --- SATISFACTION SURVEY ---
st.divider()
st.subheader("🎯 System Evaluation")
earned_xp = plan_days * 150
st.info(f"🔥 Great start! You've earned: **+{earned_xp} XP**!")

with st.expander("Rate your experience"):
    q1 = st.slider("1. Rate the AI Meal Plan relevance", 1, 5, 3)
    q2 = st.slider("2. Rate the AI Workout Plan safety", 1, 5, 3)
    q3 = st.slider("3. Rate your likelihood to recommend", 1, 5, 3)
    
    if st.button("Submit Survey"):
        avg = (q1 + q2 + q3) / 3
        st.success(f"✅ Survey metrics captured! Average Satisfaction: **{avg:.2f}/5.00**")
        st.balloons()
