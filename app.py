import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# -------------------------
# Configuration / mappings
# -------------------------
TEAM_MAPPING = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Deccan Chargers': 'Sunrisers Hyderabad',
    'Kings XI Punjab': 'Punjab Kings',
    'Pune Warriors': 'Rising Pune Supergiants',
    'Rising Pune Supergiant': 'Rising Pune Supergiants',
    'Gujarat Lions': 'Gujarat Titans',
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru'
}

st.set_page_config(layout="wide")
st.title("IPL Win Probability Predictor")
st.markdown("Predict the **Win Probability** for the chasing team based on real-time match conditions or custom scenarios.")

# -------------------------
# Helper: Resolve file path
# -------------------------
def candidate_paths_for(filename):
    """
    Return a list of candidate Paths to try for a given filename.
    Order: explicit env var -> streamlit secrets -> same dir as app -> data/ -> models/ -> cwd
    """
    candidates = []
    # 1) explicit override via environment variable (uppercase name)
    env_key = filename.upper().replace('.', '_')
    if env_key in os.environ:
        candidates.append(Path(os.environ[env_key]))
    # 2) streamlit secrets (flat)
    try:
        if filename in st.secrets:
            candidates.append(Path(st.secrets[filename]))
    except Exception:
        # In some environments st.secrets access may raise if not configured - ignore
        pass

    here = Path(__file__).resolve().parent
    candidates += [
        here / filename,
        here / "data" / filename,
        here / "models" / filename,
        here / "assets" / filename,
        Path.cwd() / filename,
    ]
    # unique and existing ordering will be checked by loader
    return candidates

@st.cache_data(show_spinner=False)
def resolve_file(filename):
    """
    Return (Path, error) where Path is first readable path or (None, errormsg)
    """
    for p in candidate_paths_for(filename):
        try:
            if p.exists() and p.is_file():
                return p, None
        except Exception:
            # skip broken paths
            continue
    return None, f"{filename} not found in candidate locations."

# -------------------------
# Load models & data
# -------------------------
@st.cache_data(show_spinner=False)
def load_model_with_candidates(filename):
    p, err = resolve_file(filename)
    if p is None:
        return None, err
    try:
        obj = joblib.load(p)
        return obj, p
    except Exception as e:
        return None, f"Failed to load {p}: {e}"

@st.cache_data(show_spinner=False)
def load_dataframe_with_candidates(filename, **read_csv_kwargs):
    p, err = resolve_file(filename)
    if p is None:
        return None, err
    try:
        df = pd.read_csv(p, **read_csv_kwargs)
        return df, p
    except Exception as e:
        return None, f"Failed to read {p}: {e}"

# Try loading required artifacts (search multiple locations)
model, model_path_err = load_model_with_candidates("xgb_ipl_predictor.joblib")
trf, trf_path_err = load_model_with_candidates("column_transformer.joblib")
analysis_df, analysis_path_err = load_model_with_candidates("analysis_data.joblib")  # note: joblib saved df
# If analysis_data.joblib didn't work, try reading analysis_data.csv (fallback)
if analysis_df is None:
    # try CSV fallback
    analysis_df, _ = load_dataframe_with_candidates("analysis_data.csv")

# If any critical artifact is missing, show a helpful message and stop
missing_artifacts = []
if model is None:
    missing_artifacts.append(f"Model: {model_path_err}")
if trf is None:
    missing_artifacts.append(f"Transformer: {trf_path_err}")
if analysis_df is None:
    missing_artifacts.append("analysis_data not found (tried joblib and CSV).")

if missing_artifacts:
    st.error("Required model/data files not found or failed to load:")
    for msg in missing_artifacts:
        st.write(f"- {msg}")
    st.stop()

# Ensure analysis_df is a DataFrame (if loaded via joblib it might already be)
if isinstance(analysis_df, tuple):
    # defensive - should not happen with our wrapper, but just in case
    analysis_df = analysis_df[0]
if not isinstance(analysis_df, pd.DataFrame):
    try:
        analysis_df = pd.DataFrame(analysis_df)
    except Exception:
        st.error("analysis_data could not be coerced to a DataFrame.")
        st.stop()

# Normalize some columns used downstream
analysis_df.columns = analysis_df.columns.str.lower()

# **********


# Ensure 'total_run' column exists, as it's used in the new data table
# This is based on the preprocessing notebook
if 'total_runs' in analysis_df.columns and 'total_run' not in analysis_df.columns:
    analysis_df.rename(columns={'total_runs': 'total_run'}, inplace=True)
elif 'total_run' not in analysis_df.columns:
    # Fallback if neither is present (e.g., set to 0 or load error)
    st.warning("Column 'total_run' (for runs per ball) not found in analysis_data. Setting to 0.")
    analysis_df['total_run'] = 0


#************


# Provide lists for UI controls
all_teams = sorted(analysis_df['batting_team'].unique())
all_venues = sorted(analysis_df['venue'].unique())

# -------------------------
# Prediction logic
# -------------------------
def predict_probability(input_data_df):
    """
    Transforms user input and predicts win probability using the XGBoost model.
    Expects input_data_df with columns matching training names (lowercase).
    """
    expected_cols = ['venue', 'batting_team', 'bowling_team', 'current_score', 
                    'wickets_remaining', 'balls_left', 'runs_to_chase', 
                    'current_run_rate', 'required_run_rate','runs_last_5_overs', 'wickets_last_5_overs']
    # Ensure all columns present; if not, try to rename common variants
    df = input_data_df.copy()
    df.columns = df.columns.str.lower()
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input DataFrame missing columns: {missing}")

    X_input = df[expected_cols]


    # Handle potential extra columns from preprocessing notebook (like 'runs_last_5_overs') *****
    # The model expects *exactly* the columns from the transformer,
    # but the transformer *itself* expects the columns defined in expected_cols.
    
    # Check if a known *extra* column (from preprocessing) is present and drop if so
    # extra_cols_to_check = ['runs_last_5_overs', 'wickets_last_5_overs']
    # cols_to_drop = [col for col in extra_cols_to_check if col in df.columns]
    # if cols_to_drop:
    #     df = df.drop(columns=cols_to_drop)
        
    # # Re-check for missing *required* columns
    # missing = [c for c in expected_cols if c not in df.columns]
    # if missing:
    #     # If we just dropped cols, this might be a mistake. Let's try to proceed.
    #     # But if the core required cols are missing, error out.
    #     core_missing = [c for c in missing if c not in extra_cols_to_check]
    #     if core_missing:
    #         raise ValueError(f"Input DataFrame missing columns: {core_missing}")
    
    # Ensure we only pass the exact columns
    X_input = df[expected_cols] # ***************



    input_transformed = trf.transform(X_input)
    win_proba = model.predict_proba(input_transformed)[0][1]
    return int(round(win_proba * 100))

# -------------------------
# UI: Custom Scenario
# -------------------------
st.header("1. Enter Match Scenario (Custom Scenario)")
st.markdown("Use the dropdowns and sliders below to define the state of the match.")

col1, col2, col3 = st.columns(3)

def format_overs(value):
    whole_overs = int(value)
    balls_in_current_over = round((value - whole_overs) * 10)
    if balls_in_current_over >= 6:
        return float(whole_overs + 1)
    else:
        return float(whole_overs) + balls_in_current_over / 10

with col1:
    batting_team = st.selectbox('Chasing Team (Batting)', all_teams)
    remaining_teams = [t for t in all_teams if t != batting_team]
    bowling_team = st.selectbox('Defending Team (Bowling)', remaining_teams, index=0)
    venue = st.selectbox('Venue', all_venues)

with col2:
    target = st.number_input('Target Score', min_value=1, value=150)
    runs_scored = st.number_input('Current Score (Runs Scored)', min_value=0, value=20)
    wickets_lost = st.number_input('Wickets Lost', min_value=0, max_value=9, value=1)

with col3:
    ## overs_input = st.slider('Overs Completed', min_value=0.0, max_value=19.5, value=5.0, step=0.1, format='%.1f')
    
    #corrected code 
    valid_overs = []
    for over in range(20):  # 0 to 19 overs
        for ball in range(6):  # 0 to 5 legal balls
            valid_overs.append(over + ball/10)
    valid_overs.append(20.0)  # final over completion

    overs_input = st.select_slider(
        'Overs Completed',
        options=valid_overs,
        value=5.0,
        format_func=lambda x: f"{x:.1f}"
    )

    # Convert overs to balls
    whole_overs = int(overs_input)
    balls_in_current_over = int(round((overs_input - whole_overs) * 10))
    balls_completed = whole_overs * 6 + balls_in_current_over

    st.caption(
        f"Actual Overs Completed: **{whole_overs}.{balls_in_current_over}** "
        f"| Total Balls Bowled: {balls_completed}"
    )

    runs_last_5 = st.number_input('Runs in Last 5 Overs', min_value=0, value=30, step=1)
    wickets_last_5 = st.number_input('Wickets in Last 5 Overs', min_value=0, max_value=10, value=1, step=1)

    overs_completed_cricket_format = format_overs(overs_input)
    whole_overs = int(overs_completed_cricket_format)
    balls_in_current_over = round((overs_completed_cricket_format - whole_overs) * 10)
    balls_completed = whole_overs * 6 + balls_in_current_over
    

if st.button('Predict Win Probability'):
    if batting_team == bowling_team:
        st.error("Batting and Bowling teams must be different.")
    elif balls_completed > 120:
        st.error("Balls completed cannot exceed 120.")
    elif runs_scored >= target:
        st.success("The chasing team has already won!")
    else:
        balls_left = 120 - balls_completed
        wickets_remaining = 10 - wickets_lost
        required_runs = target - runs_scored
        current_run_rate = (runs_scored * 6) / balls_completed if balls_completed > 0 else 0
        required_run_rate = (required_runs * 6) / balls_left if balls_left > 0 else 0

        input_data = {
            'venue': venue,
            'batting_team': batting_team,
            'bowling_team': bowling_team,
            'current_score': runs_scored,
            'wickets_remaining': wickets_remaining,
            'balls_left': balls_left,
            'runs_to_chase': target,            # keep same field name as your training; if different, align here
            'current_run_rate': current_run_rate,
            'required_run_rate': required_run_rate,
            'runs_last_5_overs': runs_last_5,
            'wickets_last_5_overs': wickets_last_5
        }
        input_df = pd.DataFrame([input_data])

        try:
            win_proba = predict_probability(input_df)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            win_proba = None

        if win_proba is not None:
            loss_proba = 100 - win_proba
            st.markdown("---")
            st.header("Prediction Result: Win Probability")
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric(f"🏆 {batting_team} Win Probability", f"{win_proba}%")
            with col_res2:
                st.metric(f"📉 {bowling_team} Win Probability", f"{loss_proba}%")

            st.subheader("Key Match Metrics")
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Runs Required", f"{required_runs}")
            col_m2.metric("Required Run Rate", f"{required_run_rate:.2f}")
            col_m3.metric("Current Run Rate", f"{current_run_rate:.2f}")

# -------------------------
# UI: Historical Match Analysis
# -------------------------
st.markdown("---")
st.header("2. Historical Match Analysis (Graphs and Metrics)")
st.caption("Select two teams and a specific match to visualize the chase progression.")

st.sidebar.title("Match Selector")
teams_analysis = sorted(analysis_df['batting_team'].unique())
selected_team1_analysis = st.sidebar.selectbox('Team 1 (Analysis)', teams_analysis)
selected_team2_options_analysis = [t for t in teams_analysis if t != selected_team1_analysis]
selected_team2_analysis = st.sidebar.selectbox('Team 2 (Analysis)', selected_team2_options_analysis, index=0)

# Filter relevant historical matches (second innings)
historical_matches_filtered = analysis_df[
    ((analysis_df['batting_team'] == selected_team1_analysis) & (analysis_df['bowling_team'] == selected_team2_analysis)) |
    ((analysis_df['batting_team'] == selected_team2_analysis) & (analysis_df['bowling_team'] == selected_team1_analysis))
]

if not historical_matches_filtered.empty:
    match_ids = historical_matches_filtered['id'].unique() if 'id' in historical_matches_filtered.columns else historical_matches_filtered['ID'].unique()

    # Attempt to load matches.csv for friendly labels
    matches_full, matches_err = load_dataframe_with_candidates("matches.csv")
    filtered_match_map = {}
    if matches_full is None:
        st.sidebar.warning("Could not find 'matches.csv' for match date and details lookup. Showing Match IDs only.")
        filtered_match_map = {str(mid): f"Match ID: {mid}" for mid in match_ids}
    else:
        # normalize columns for consistency
        matches_full.columns = matches_full.columns.str.lower()
        # apply mapping to team columns if present
        for col in ['team1', 'team2']:
            if col in matches_full.columns:
                matches_full[col].replace(TEAM_MAPPING, inplace=True)

        # Build map id -> label
        if 'id' in matches_full.columns:
            # Convert types to string for consistent matching
            matches_full['id'] = matches_full['id'].astype(str)
            match_ids_str = [str(mid) for mid in match_ids]
            # Build label safely
            def mklabel(row):
                d = row.get('date', '')
                t1 = row.get('team1', '')
                t2 = row.get('team2', '')
                venue = row.get('venue', '')
                base = " - ".join([str(d), f"{t1} vs {t2}"])
                if venue:
                    base = f"{base} @ {venue}"
                return base

            match_date_map = matches_full.set_index('id').apply(mklabel, axis=1).to_dict()
            filtered_match_map = {mid: match_date_map.get(str(mid), f"Match ID: {mid}") for mid in match_ids}
            # fallback to ID strings as keys
            filtered_match_map = {str(k): v for k, v in filtered_match_map.items()}
        else:
            # graceful fallback
            filtered_match_map = {str(mid): f"Match ID: {mid}" for mid in match_ids}

    # Sidebar selectbox uses string keys so we present stable ordering and labels
    selected_key = st.sidebar.selectbox(
        'Select a Specific Match',
        options=sorted(filtered_match_map.keys()),
        format_func=lambda x: filtered_match_map[x]
    )

    # match_data filter: ensure types align
    key_to_match_id = selected_key
    # ensure historical_matches_filtered has lowercase columns
    hist = historical_matches_filtered.copy()
    hist.columns = hist.columns.str.lower()
    # filter by id cast to string
    if 'id' in hist.columns:
        match_data = hist[hist['id'].astype(str) == key_to_match_id].copy()
    elif 'ID' in hist.columns:
        match_data = historical_matches_filtered[historical_matches_filtered['ID'].astype(str) == key_to_match_id].copy()
    else:
        # if no id column, try first match
        match_data = hist

    if match_data.empty:
        st.warning("No data rows found for the selected match.")
    else:
        analysis_batting_team = match_data['batting_team'].iloc[0]
        analysis_bowling_team = match_data['bowling_team'].iloc[0]
        final_target = match_data['target'].iloc[0] if 'target' in match_data.columns else match_data.get('runs_to_chase', pd.Series([None])).iloc[0]

        st.subheader(f"Analyzing: {filtered_match_map.get(selected_key, selected_key)}")
        st.markdown(f"**{analysis_batting_team}** chasing **{final_target}** runs against **{analysis_bowling_team}**")


        # -------------------------
        valid_overs = []
        for over in range(20):  # 0 to 19 overs
            for ball in range(6):  # 0 to 5 legal balls
                valid_overs.append(over + ball/10)
        valid_overs.append(20.0)  # final over completion

        selected_overs = st.select_slider(
        "Select Over Range to Analyze",
        options=valid_overs,
        value=(0.0, 20.0), # Default to the full match
        format_func=lambda x:f"{x:.1f}"
        )
        # -----------------------


        # Feature Calculation
        # Require columns: overs, ballnumber, current_score, wickets_taken
        # Some datasets name these differently; assume your analysis_data matches these names.
        if 'overs' not in match_data.columns or 'ballnumber' not in match_data.columns:
            st.error("Required columns 'overs' and 'ballnumber' not present in historical data for plotting.")
        else:
            match_data['overs'] = match_data['overs'].astype(float)
            match_data['ballnumber'] = match_data['ballnumber'].astype(int)
            match_data['overs_bowled'] = (match_data['overs'] * 6 + match_data['ballnumber']) / 6
            match_data['balls_completed'] = (match_data['overs'] * 6 + match_data['ballnumber'])
            match_data['balls_left'] = 120 - match_data['balls_completed']
            if 'wickets_taken' in match_data.columns:
                match_data['wickets_fallen'] = match_data.groupby('id')['wickets_taken'].cummax() if 'id' in match_data.columns else match_data['wickets_taken'].cummax()
            else:
                match_data['wickets_fallen'] = 0

            match_data['current_run_rate'] = np.where(match_data['balls_completed'] > 0, (match_data['current_score'] * 6) / match_data['balls_completed'], 0)
            match_data['current_run_rate'].replace([np.inf, -np.inf], np.nan, inplace=True)
            match_data['current_run_rate'].fillna(0, inplace=True)
            
            # ** this code here makes the graphs more interactive to the user
            match_data['wickets_remaining'] = 10 - match_data['wickets_fallen']
            match_data['runs_to_chase'] = final_target # You already have this

            # Calculate RRR
            match_data['required_run_rate'] = np.where(
                match_data['balls_left'] > 0, 
                (match_data['runs_to_chase'] - match_data['current_score']) * 6 / match_data['balls_left'], 
                0
            )
            # Fill in constant values for the model
            match_data['venue'] = match_data['venue'].iloc[0]
            match_data['batting_team'] = analysis_batting_team
            match_data['bowling_team'] = analysis_bowling_team
            # **

            # Calculate 'is_wicket' from the change in 'wickets_fallen' (needed for next calc)
            match_data['is_wicket'] = match_data['wickets_fallen'].diff().fillna(0).apply(lambda x: 1 if x > 0 else 0)
            
            # Calculate rolling runs/wickets. .shift(1) gets the sum *before* the current ball.
            # .fillna(0) handles the start of the innings.
            rolling_runs = match_data['total_run'].rolling(window=30, min_periods=1).sum().shift(1).fillna(0)
            rolling_wickets = match_data['is_wicket'].rolling(window=30, min_periods=1).sum().shift(1).fillna(0)
            
            match_data['runs_last_5_overs'] = rolling_runs
            match_data['wickets_last_5_overs'] = rolling_wickets


            # ********
            # Define the columns the model expects
            expected_cols = ['venue', 'batting_team', 'bowling_team', 'current_score', 
                            'wickets_remaining', 'balls_left', 'runs_to_chase', 
                            'current_run_rate', 'required_run_rate','runs_last_5_overs', 'wickets_last_5_overs']

            # Ensure all columns are present
            model_input_df = match_data[expected_cols]

            try:
                # Use the loaded transformer (trf) and model
                input_transformed = trf.transform(model_input_df)
                win_probas_all = model.predict_proba(input_transformed)[:, 1] # Get proba for 'win' (class 1)

                # Add to your dataframe 
                match_data['win_probability'] = (win_probas_all * 100).round(1)
            except Exception as e:
                st.error(f"Failed to generate historical win probability: {e}")
                match_data['win_probability'] = np.nan
            # *********



            # Apply the over range filter from the slider
            analysis_data_filtered = match_data[
                (match_data['overs_bowled'] >= selected_overs[0]) &
                (match_data['overs_bowled'] <= selected_overs[1])
            ].copy()
            

            if analysis_data_filtered.empty:
                st.warning(f"No ball-by-ball data found for the selected over range: {selected_overs[0]} to {selected_overs[1]}")
                st.stop()


            # 1. Run Rate Progression Plot (Worm Graph)
            fig_rr = go.Figure()
            fig_rr.add_trace(go.Scatter(
                x=analysis_data_filtered['overs_bowled'],
                y=analysis_data_filtered['current_run_rate'],
                mode='lines',
                name='Current Run Rate',
                line=dict(width=3)
            ))
            required_rr = (final_target / 20) if final_target else 0
            fig_rr.add_trace(go.Scatter(
                x=[analysis_data_filtered['overs_bowled'].min(), analysis_data_filtered['overs_bowled'].max()],
                y=[required_rr, required_rr],
                mode='lines',
                name='Required Run Rate',
                line=dict(dash='dash', width=3)
            ))
            fig_rr.update_layout(
                title='Run Rate Progression vs. Required Run Rate',
                xaxis_title='Overs Bowled',
                yaxis_title='Run Rate (Runs/Over)',
                hovermode="x unified",
                template="plotly_white"
            )
            st.plotly_chart(fig_rr, use_container_width=True)

            # 2. Wicket Fall Plot
            if 'wickets_taken' in match_data.columns:
                fig_wickets = go.Figure()
                fig_wickets.add_trace(go.Scatter(
                    x=analysis_data_filtered['overs_bowled'],
                    y=analysis_data_filtered['wickets_taken'],
                    mode='lines+markers',
                    name='Wickets Fallen',
                    line=dict(width=3),
                    marker=dict(size=8)
                ))
                fig_wickets.update_layout(
                    title='Wickets Fallen Over Time',
                    xaxis_title='Overs Bowled',
                    yaxis_title='Wickets Fallen',
                    yaxis=dict(tick0=0, dtick=1, range=[-0.5, 10.5]),
                    hovermode="x unified",
                    template="plotly_white"
                )
                st.plotly_chart(fig_wickets, use_container_width=True)
            else:
                st.info("Wickets data not present in this historical dataset; skipping wicket plot.")
            
            # 3. Win Probability Plot
            if 'win_probability' in analysis_data_filtered.columns:
                fig_proba = go.Figure()
                fig_proba.add_trace(go.Scatter(
                    x=analysis_data_filtered['overs_bowled'],
                    y=analysis_data_filtered['win_probability'],
                    mode='lines',
                    name=f'{analysis_batting_team} Win %',
                    line=dict(width=3, color='green')
                ))
                fig_proba.update_layout(
                    title='Win Probability Progression',
                    xaxis_title='Overs Bowled',
                    yaxis_title='Win Probability (%)',
                    yaxis=dict(range=[0, 100]),
                    hovermode="x unified",
                    template="plotly_white"
                )
                st.plotly_chart(fig_proba, use_container_width=True)

        st.subheader(f"Ball-by-Ball Details (Overs {selected_overs[0]} to {selected_overs[1]})")
            
            # Select and rename columns for a cleaner table
        display_columns = {
            'overs_bowled': 'Over',
            'current_score': 'Score',
            'wickets_fallen': 'Wickets',
            'total_run': 'Runs (This Ball)', # From analysis_df
            'current_run_rate': 'Current Run Rate',
            'required_run_rate': 'Required Run Rate',
            'win_probability': 'Win %'
        }
            
        # Filter the dataframe to only show these columns
        # Ensure all display columns are actually in the dataframe before selecting
        cols_to_show = [col for col in display_columns.keys() if col in analysis_data_filtered.columns]
        display_df = analysis_data_filtered[cols_to_show].rename(columns=display_columns)
        if 'Over' in display_df.columns:
            # Format the 'Over' column to be like 5.1, 5.2
            display_df['Over'] = display_df['Over'].apply(lambda x: f"{int(x)}.{int(round((x - int(x)) * 6, 0))}")
            st.dataframe(display_df.set_index('Over'), use_container_width=True)
        else:
                st.dataframe(display_df, use_container_width=True)
else:
    st.warning(f"No complete second-inning matches found between {selected_team1_analysis} and {selected_team2_analysis} in the dataset.")
