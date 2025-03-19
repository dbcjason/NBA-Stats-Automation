import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import random
from scipy.spatial.distance import cdist
from google.colab import drive
import os
import time

SHEET_ID = "1zbLS_oFATsZN4zPHO5x5UZnfMHCgw5eA9iWmy9G9UaU"

# ✅ Set base directory (Change this to your actual folder)
BASE_DIR = "/content/drive/My Drive/"

# ✅ Update paths to use Google Drive
CREDENTIALS_FILE = os.path.join(BASE_DIR, "google_creds.json")


# ✅ Google Sheets Configuration
TRAINING_SHEET_URL = "https://docs.google.com/spreadsheets/d/1aO1TouqQdJPmDXr_PeCRZxyRAflMfqat5CShTdv0Rlo/edit?gid=347820615#gid=347820615"
TRAINING_SHEET_NAME = "Player Peak Projections"  # ✅ Change this to your actual sheet name
PREDICTIONS_SHEET_NAME = "Waffles"
CREDENTIALS_FILE = os.path.join(BASE_DIR, "google_creds.json")


def load_data(sheet_name):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
        client = gspread.authorize(creds)

        # ✅ Open the main spreadsheet (NOT separate sheets)
        spreadsheet = client.open(TRAINING_SHEET_NAME)
        sheet = spreadsheet.worksheet(sheet_name)  # Select the correct tab/sheet
        data = sheet.get_all_records()

        # ✅ Convert to Pandas DataFrame
        df = pd.DataFrame(data)

        print(f"✅ Successfully loaded '{sheet_name}' with {len(df)} rows.")

        return df
    except Exception as e:
        print(f"❌ Error loading data from '{sheet_name}': {e}")
        return None

league_avg_stats = {
    2009: {'Pace': 91.7, 'PPG': 100.0, 'RPG': 41.3, 'APG': 21.0, 'SPG': 7.3, 'BPG': 4.8, 'FTA': 24.7, 'FTM': 19.1, '3PA': 18.1, '3PM': 6.6, 'FGA': 80.9, 'FGM': 37.1, 'TOV': 14.0},
    2010: {'Pace': 92.7, 'PPG': 100.4, 'RPG': 41.7, 'APG': 21.2, 'SPG': 7.2, 'BPG': 4.9, 'FTA': 24.5, 'FTM': 18.6, '3PA': 18.1, '3PM': 6.4, 'FGA': 81.7, 'FGM': 37.7, 'TOV': 14.2},
    2011: {'Pace': 92.1, 'PPG': 99.6, 'RPG': 41.4, 'APG': 21.5, 'SPG': 7.3, 'BPG': 4.9, 'FTA': 24.4, 'FTM': 18.6, '3PA': 18.0, '3PM': 6.5, 'FGA': 81.2, 'FGM': 37.2, 'TOV': 14.3},
    2012: {'Pace': 91.3, 'PPG': 96.3, 'RPG': 42.2, 'APG': 21.0, 'SPG': 7.7, 'BPG': 5.1, 'FTA': 22.5, 'FTM': 16.9, '3PA': 18.4, '3PM': 6.4, 'FGA': 81.4, 'FGM': 36.5, 'TOV': 14.6},
    2013: {'Pace': 92.0, 'PPG': 98.1, 'RPG': 42.1, 'APG': 22.1, 'SPG': 7.8, 'BPG': 5.1, 'FTA': 22.2, 'FTM': 16.7, '3PA': 20.0, '3PM': 7.2, 'FGA': 82.0, 'FGM': 37.1, 'TOV': 14.6},
    2014: {'Pace': 93.9, 'PPG': 101.0, 'RPG': 42.7, 'APG': 22.0, 'SPG': 7.7, 'BPG': 4.7, 'FTA': 23.6, 'FTM': 17.8, '3PA': 21.5, '3PM': 7.7, 'FGA': 83.0, 'FGM': 37.7, 'TOV': 14.6},
    2015: {'Pace': 93.9, 'PPG': 100.0, 'RPG': 43.3, 'APG': 22.0, 'SPG': 7.7, 'BPG': 4.8, 'FTA': 22.8, 'FTM': 17.1, '3PA': 22.4, '3PM': 7.8, 'FGA': 83.6, 'FGM': 37.5, 'TOV': 14.4},
    2016: {'Pace': 95.8, 'PPG': 102.7, 'RPG': 43.8, 'APG': 22.3, 'SPG': 7.8, 'BPG': 5.0, 'FTA': 23.4, 'FTM': 17.7, '3PA': 24.1, '3PM': 8.5, 'FGA': 84.6, 'FGM': 38.2, 'TOV': 14.4},
    2017: {'Pace': 96.4, 'PPG': 105.6, 'RPG': 43.5, 'APG': 22.6, 'SPG': 7.7, 'BPG': 4.7, 'FTA': 23.1, 'FTM': 17.8, '3PA': 27.0, '3PM': 9.7, 'FGA': 85.4, 'FGM': 39.0, 'TOV': 14.0},
    2018: {'Pace': 97.3, 'PPG': 106.3, 'RPG': 43.5, 'APG': 23.2, 'SPG': 7.7, 'BPG': 4.8, 'FTA': 21.7, 'FTM': 16.6, '3PA': 29.0, '3PM': 10.5, 'FGA': 86.1, 'FGM': 39.6, 'TOV': 14.3},
    2019: {'Pace': 100.0, 'PPG': 111.2, 'RPG': 45.2, 'APG': 24.6, 'SPG': 7.6, 'BPG': 5.0, 'FTA': 23.1, 'FTM': 17.7, '3PA': 32.0, '3PM': 11.4, 'FGA': 89.2, 'FGM': 41.1, 'TOV': 14.1},
    2020: {'Pace': 100.3, 'PPG': 111.8, 'RPG': 44.8, 'APG': 24.4, 'SPG': 7.6, 'BPG': 4.9, 'FTA': 23.1, 'FTM': 17.9, '3PA': 34.1, '3PM': 12.2, 'FGA': 88.8, 'FGM': 40.9, 'TOV': 14.5},
    2021: {'Pace': 99.2, 'PPG': 112.1, 'RPG': 44.3, 'APG': 24.8, 'SPG': 7.6, 'BPG': 4.9, 'FTA': 21.8, 'FTM': 17.0, '3PA': 34.6, '3PM': 12.7, 'FGA': 88.4, 'FGM': 41.2, 'TOV': 13.8},
    2022: {'Pace': 98.2, 'PPG': 110.6, 'RPG': 44.5, 'APG': 24.6, 'SPG': 7.6, 'BPG': 4.7, 'FTA': 21.9, 'FTM': 16.9, '3PA': 35.2, '3PM': 12.4, 'FGA': 88.1, 'FGM': 40.6, 'TOV': 13.8},
    2023: {'Pace': 99.2, 'PPG': 114.7, 'RPG': 43.4, 'APG': 25.3, 'SPG': 7.3, 'BPG': 4.7, 'FTA': 23.5, 'FTM': 18.4, '3PA': 34.2, '3PM': 12.3, 'FGA': 88.3, 'FGM': 42.0, 'TOV': 14.1},
    2024: {'Pace': 98.5, 'PPG': 114.2, 'RPG': 43.5, 'APG': 26.7, 'SPG': 7.5, 'BPG': 5.1, 'FTA': 21.7, 'FTM': 17.0, '3PA': 25.1, '3PM': 12.8, 'FGA': 88.9, 'FGM': 42.2, 'TOV': 13.6},
    2025: {'Pace': 99.0, 'PPG': 113.4, 'RPG': 44.2, 'APG': 26.4, 'SPG': 8.3, 'BPG': 5.0, 'FTA': 21.8, 'FTM': 17.0, '3PA': 37.5, '3PM': 13.4, 'FGA': 89.2, 'FGM': 41.5, 'TOV': 14.5}
}



ADJUSTED_PLAYERS = 0  # Track how many players have been adjusted
MAX_PLAYERS_TO_ADJUST = 2  # Only adjust two players for debugging

def scale_player_stats(row):
    """
    Applies progressive scaling to players based on how long they've been in the league.
    - Full scaling if years_since_draft == 6.
    - Gradual reduction as years_since_draft increases.
    - Uses precomputed scaling factors from Year 5 to Peak.
    """
    CURRENT_YEAR = 2025  # ✅ Update this if needed
    years_since_draft = CURRENT_YEAR - row["Draft Year"]

    if years_since_draft < 6:
        return row  # ✅ No scaling if the player hasn't played at least 6 seasons.

    # ✅ Define how much each stat improves from Year 5 to Peak
    scaling_factors_per_stat = {
        "peak_p/g": 1.29,   # ✅ Points per game increase by 29%
        "peak_r/g": 1.29,   # ✅ Rebounds per game increase by 29%
        "peak_a/g": 1.38,   # ✅ Assists per game increase by 38%
        "peak_s/g": 1.25,   # ✅ Steals per game increase by 25%
        "peak_b/g": 1.20    # ✅ Blocks per game increase by 20% (Adjust if needed)
    }

    # ✅ Define progressive scaling factors based on years since draft
    progressive_scaling = {
        6: 1.0,  # ✅ Full boost
        7: 0.75,  # ✅ 80% of the boost
        8: 0.5,  # ✅ 60% of the boost
        9: 0.25,  # ✅ 40% of the boost
        10: 0.0, # ✅ 20% of the boost
    }

    # ✅ Determine the scaling percentage to apply
    scaling_factor = progressive_scaling.get(years_since_draft, 0.0)  # Defaults to 10% boost for older players

    # ✅ Apply scaling to relevant stats
    for col, base_factor in scaling_factors_per_stat.items():
        if col in row and not pd.isna(row[col]):  # ✅ Ensure the column exists
            row[col] *= 1 + ((base_factor - 1) * scaling_factor)  # ✅ Apply progressive scaling

    return row  # ✅ Return adjusted row



def adjust_stats(row):
    """Adjust stats for pace, scoring, and rule changes to match modern NBA (2025)."""
    global ADJUSTED_PLAYERS

    if ADJUSTED_PLAYERS >= MAX_PLAYERS_TO_ADJUST:
        return row  # Skip adjustments after two players

    # ✅ Apply career scaling based on years since draft
    row = scale_player_stats(row)  # 🔥 NEW LINE: Apply progressive scaling first

    # ✅ Get the player's name
    player_name = row["Player"] if "Player" in row and not pd.isna(row["Player"]) else "Unknown Player"

    # ✅ Determine the player's season year
    if "peak_Year" in row and not pd.isna(row["peak_Year"]):
        year = int(row["peak_Year"])  # Use peak_Year for College
    elif "Year" in row and not pd.isna(row["Year"]):
        year = int(row["Year"])  # Use Year for NCAA
    else:
        year = 2025  # Default to current year

    if year not in league_avg_stats:
        return row  # Skip if year is missing

    # ✅ Get league averages for that year
    era_stats = league_avg_stats[year]
    modern_stats = league_avg_stats[2025]  # Use 2025 as benchmark

    # ✅ If the year is already 2025, no adjustment is needed
    if year == 2025:
        print(f"⚠️ Skipping adjustment for {player_name} (Already 2025)")
        return row

    # ✅ Define mappings between column names and league_avg_stats keys
    stat_map = {
        "peak_p/g": "PPG",
        "peak_r/g": "RPG",
        "peak_a/g": "APG",
        "peak_s/g": "SPG",
        "peak_b/g": "BPG",
        "peak_fta/g": "FTA",
        "peak_3/g": "3PM",
        "peak_fga/g": "FGA",
    }

    # ✅ Print BEFORE adjustment
    print(f"\n🔍 BEFORE Adjustment - {player_name} (Year: {year})")
    for stat in stat_map.keys():
        print(f"   {stat}: {row.get(stat, 'N/A')}")

    # ✅ Create a copy of the row to ensure modifications persist
    row_copy = row.copy()

    # ✅ Apply the actual adjustment to each stat
    for stat, league_stat in stat_map.items():
        if stat in row_copy and not pd.isna(row_copy[stat]) and league_stat in modern_stats and league_stat in era_stats:
            adjustment_factor = modern_stats[league_stat] / era_stats[league_stat]
            row_copy[stat] *= adjustment_factor  # ✅ Multiplication now properly applies

    # ✅ Print AFTER adjustment
    print(f"\n✅ AFTER Adjustment - {player_name} (Year: {year})")
    for stat in stat_map.keys():
        print(f"   {stat}: {row_copy.get(stat, 'N/A')}")

    ADJUSTED_PLAYERS += 1  # Increment the count of adjusted players
    return row_copy  # ✅ Return the updated row copy


def preprocess_data(df, model_type):
    # ✅ Ensure "Player" column exists before using it
    if "Player" not in df.columns:
        print("❌ 'Player' column is missing from the dataset!")
        print(f"✅ Columns in dataset: {df.columns.tolist()}")
        return None, None, None, None  # Avoid crashing

    # ✅ Preserve important columns before dropping
    player_column = df[["Player"]].copy()
    year_column = df[["peak_Year"]].copy() if "peak_Year" in df.columns else None

    drop_columns = [
        "Player", "peak_Player", "peak_Value", "NBA_ID", "Team", "peak_Rank", "peak_Year", "Draft Year", "Y1_PG_GP",
        "Y2_PG_GP", "Y3_PG_GP", "Y4_PG_GP", "Y5_PG_GP", "Y1_PG_FGM", "Y2_PG_FGM_x", "Y3_PG_FGM", "Y4_PG_FGM",
        "Y5_PG_FGM", "Y1_P100_FGM", "Y2_P100_FGM_x", "Y3_P100_FGM", "Y4_P100_FGM", "Y5_P100_FGM", "Y1_PG_FGA",
        "Y2_PG_FGA_x", "Y3_PG_FGA", "Y4_PG_FGA", "Y5_PG_FGA", "Y1_P100_FGA", "Y2_P100_FGA_x", "Y3_P100_FGA",
        "Y4_P100_FGA", "Y5_P100_FGA", "Dunks", "G", "ORtg", "D-Rtg", "Rim Att", "Ast'd Rim", "Ast'd 3",
        "Ast'd Total"
    ]

    # ✅ Conditionally drop "Pick" for 3rd-Year, 4th-Year, and 5th-Year models
    if model_type in ["3rd-Year", "4th-Year", "5th-Year"]:
        drop_columns.append("Pick")

    df = df.drop(columns=drop_columns, errors="ignore")

    df = df.apply(pd.to_numeric, errors="coerce")

    target_columns = [
        "peak_g", "peak_m/g", "peak_p/g", "peak_3/g", "peak_3%", "peak_r/g",
        "peak_a/g", "peak_s/g", "peak_b/g", "peak_fg%", "peak_fga/g",
        "peak_ft%", "peak_fta/g"
    ]

    # ✅ Separate rows for training (players with known target values) and prediction (players with missing targets)
    train_df = df.dropna(subset=target_columns)  # Rows where target values are known
    predict_df = df[df[target_columns].isnull().any(axis=1)]  # Rows where some targets are missing

    # ✅ Apply era adjustments to both training and prediction sets
    train_df = train_df.apply(adjust_stats, axis=1)
    predict_df = predict_df.apply(adjust_stats, axis=1)

    # ✅ Restore the "Player" and "peak_Year" column to predict_df
    predict_df = predict_df.merge(player_column, left_index=True, right_index=True, how="left")

    if year_column is not None:
        predict_df = predict_df.merge(year_column, left_index=True, right_index=True, how="left")

        # ✅ Rename any mistakenly duplicated peak_Year columns
        if "peak_Year_x" in predict_df.columns:
            predict_df = predict_df.rename(columns={"peak_Year_x": "peak_Year"})
        if "peak_Year_y" in predict_df.columns:
            predict_df = predict_df.rename(columns={"peak_Year_y": "peak_Year"})

    X_train = train_df.drop(columns=target_columns, errors="ignore")
    y_train = train_df[target_columns]

    X_predict = predict_df.drop(columns=target_columns, errors="ignore")

    return X_train, y_train, X_predict, predict_df

def custom_loss(y_true, y_pred):
    """
    Custom loss function for XGBoost:
    - Penalizes under-predictions more than over-predictions
    - If y_pred < y_true, we apply a higher penalty (1.5x the squared error)
    - If y_pred >= y_true, we apply the normal squared error
    """
    residual = y_true - y_pred
    gradient = np.where(residual > 0, -1.5 * residual, -residual)  # Gradient
    hessian = np.ones_like(residual)  # Hessian is constant (standard in regression)
    return gradient, hessian


def train_and_save_model(sheet_name, model_filename):
    df = load_data(sheet_name)
    if df is None:
        print(f"❌ Failed to load data from {sheet_name}.")
        return

    # ✅ Apply adjustments before splitting into training/prediction sets
    df = df.apply(adjust_stats, axis=1)

    X_train, y_train, _, _ = preprocess_data(df, sheet_name)

    if X_train.empty or y_train.empty:
        print(f"❌ Not enough training data for {sheet_name}. Ensure some rows have complete target values.")
        return

    # ✅ Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000,  # More trees
        learning_rate=0.1,  # Lower learning rate
        max_depth=6,  # Depth of each tree
        colsample_bytree=0.7,  # Use 80% of features per tree
        subsample=0.8,  # Use 80% of data per tree
        reg_alpha=0.05,  # L1 regularization (lasso)
        reg_lambda=0.1,  # L2 regularization (ridge)
        min_child_weight=5,
        gamma=0.2
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],  # Validation data
        verbose=1
    )

    # ✅ Evaluate Model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"✅ {sheet_name} Model trained. MSE: {mse}")

    # ✅ Save Model
    joblib.dump(model, model_filename)
    print(f"✅ Model saved as {model_filename}.")

    feature_importances = model.feature_importances_
    X_train_features = X_train.columns.tolist()

    if len(feature_importances) != len(X_train_features):
        print("⚠️ Warning: Feature importance length mismatch. Adjusting...")

        min_length = min(len(feature_importances), len(X_train_features))
        feature_importances = feature_importances[:min_length]
        feature_names = X_train_features[:min_length]
    else:
        feature_names = X_train_features

    # Normalize so total importance sums to 1
    feature_importances /= feature_importances.sum()


    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    print("\n🔍 Feature Importances:")
    print(importance_df)

    # ✅ Optionally display a bar chart
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["Feature"], importance_df["Importance"], color="blue")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title(f"Feature Importance for {sheet_name} Model")
    plt.gca().invert_yaxis()
    plt.show()




def make_predictions(player_group):
    # Choose the right sheet and model based on experience level
    sheet_map = {
        "College": ("College", "college_model.pkl"),
        "Rookies": ("Rookies", "rookies_model.pkl"),
        "Sophomores": ("Sophomores", "sophomores_model.pkl"),
        "3rd-Year": ("3rd-Year", "3rd-year_model.pkl"),
        "4th-Year": ("4th-Year", "4th-year_model.pkl"),
        "5th-Year": ("5th-Year", "5th-year_model.pkl")
    }

    if player_group not in sheet_map:
        print(f"❌ Invalid player group: {player_group}. Skipping.")
        return None

    sheet_name, model_filename = sheet_map[player_group]

    model_filename = os.path.join(BASE_DIR, model_filename)
    model = joblib.load(model_filename)

    
    df = load_data(sheet_name)
    if df is None:
        return None

    # ✅ Apply adjustments BEFORE predicting
    df = df.apply(adjust_stats, axis=1)

    _, _, X_predict, predict_df = preprocess_data(df, sheet_name)

    if X_predict.empty:
        print(f"🚀 No players in {sheet_name} need predictions. Skipping.")
        return None

    X_predict = X_predict.select_dtypes(include=[np.number])

    # ✅ Make Predictions
    predictions = model.predict(X_predict)

    prediction_columns = [
        "peak_g", "peak_m/g", "peak_p/g", "peak_3/g", "peak_3%", "peak_r/g",
        "peak_a/g", "peak_s/g", "peak_b/g", "peak_fg%", "peak_fga/g",
        "peak_ft%", "peak_fta/g"
    ]

    df_predictions = pd.DataFrame(predictions, columns=prediction_columns)

    # ✅ Restore player names and peak years
    df_predictions.insert(0, "Player", predict_df["Player"].values)
    df_predictions["peak_Year"] = predict_df.iloc[:, 0]  # Selects first column

    # ✅ Print BEFORE adjustment (first 10 rows)
    print("\n🔍 BEFORE Adjustment - First 10 Predictions:")
    print(df_predictions.head(10))

    # ✅ Boost predictions slightly to reflect higher upside
    df_predictions.iloc[:, 1:] *= 1.05  # ✅ Increase all predicted values by 5%

    # ✅ Apply era adjustments to the predictions AFTER they are generated
    df_predictions = df_predictions.apply(adjust_stats, axis=1)

    # ✅ Define the stats that will receive specific boosts
    boost_stats = {
        "peak_p/g": 1.05,  # ✅ Players in top 5% of PPG get a 10% boost
        "peak_b/g": 1.25,  # ✅ Players in top 5% of BPG get a 15% boost
        "peak_a/g": 1.15,  # ✅ Players in top 5% of APG get a 15% boost
        "peak_s/g": 1.15,  # ✅ Players in top 5% of SPG get a 15% boost
        "peak_3/g": 1.10  # ✅ Players in top 5% of 3PM get a 15% boost
    }

    # ✅ Apply boost based on the top 5% threshold for each stat
    for stat, boost_factor in boost_stats.items():
        stat_threshold = df_predictions[stat].quantile(0.95)  # 🔥 Get top 5% threshold
        df_predictions.loc[df_predictions[stat] >= stat_threshold, stat] *= boost_factor  # ✅ Apply boost

    # ✅ Print AFTER adjustment (first 10 rows)
    print("\n✅ AFTER Adjustment - First 10 Predictions:")
    print(df_predictions.head(10))

    return df_predictions

def find_closest_comparisons(df_predictions, training_data):
    """
    Find the closest historical peak season for each predicted peak season using a weighted Euclidean distance.
    - peak_Value is the most important factor (highest weight).
    - peak_p/g, peak_r/g, peak_a/g are second tier (medium weight).
    - All other stats are less important (low weight).
    """

    # ✅ Define the three tiers of importance
    tier_1 = ["peak_s/g", "peak_b/g", "peak_p/g", "peak_r/g", "peak_a/g"]  # ✅ Most important (highest weight)
    tier_2 = ["peak_fg%", "peak_3%", "peak_ft%"]  # ✅ Second most important
    tier_3 = ["peak_3/g", "peak_fga/g", "peak_fta/g"]  # ✅ Least important

    # ✅ Ensure all columns exist in training_data
    all_columns = tier_1 + tier_2 + tier_3
    missing_cols = [col for col in all_columns if col not in training_data.columns]
    if missing_cols:
        print(f"❌ Missing columns in training_data: {missing_cols}")
        return ["N/A"] * len(df_predictions)

    # ✅ Convert to numeric (force conversion for safety)
    training_data.loc[:, all_columns] = training_data[all_columns].apply(pd.to_numeric, errors="coerce")
    df_predictions[all_columns] = df_predictions[all_columns].apply(pd.to_numeric, errors="coerce")

    # ✅ Fill NaN values with 0 to avoid errors in distance computation
    prediction_stats = df_predictions[all_columns].fillna(0).values
    training_stats = training_data[all_columns].infer_objects(copy=False).fillna(0).values

    # ✅ Define Weights for Each Tier
    weights = np.array(
        [5.0] * len(tier_1) +  # ✅ Highest weight for peak_Value
        [3.0] * len(tier_2) +  # ✅ Medium weight for peak_p/g, peak_r/g, peak_a/g
        [1.0] * len(tier_3)    # ✅ Lowest weight for everything else
    )

    # ✅ Compute Weighted Euclidean Distances
    weighted_predictions = prediction_stats * weights  # Apply weights
    weighted_training = training_stats * weights       # Apply weights
    distances = cdist(weighted_predictions, weighted_training, metric="euclidean")

    # ✅ Find the closest match for each prediction
    closest_indices = distances.argmin(axis=1)

    # ✅ Retrieve the corresponding player names from the training data
    closest_players = training_data.iloc[closest_indices]["Player"].tolist()

    return closest_players


def update_google_sheet(df_predictions):
    OUTPUT_SHEET_NAME = "Peak Projection Results"  # 🔥 Different Google Sheet

    # ✅ Authenticate Google Sheets
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
    client = gspread.authorize(creds)

    # ✅ Open the **NEW** output spreadsheet
    try:
        spreadsheet = client.open(OUTPUT_SHEET_NAME)
    except gspread.exceptions.SpreadsheetNotFound:
        print(f"❌ Spreadsheet '{OUTPUT_SHEET_NAME}' not found. Make sure it exists!")
        return

    # ✅ Load training data to get Draft Year information
    sheet_names = ["College", "Rookies", "Sophomores", "3rd-Year", "4th-Year", "5th-Year"]
    training_data = pd.concat([load_data(sheet) for sheet in sheet_names], ignore_index=True)

    if training_data.empty:
        print("❌ Failed to load training data. Skipping closest season comparison.")
        df_predictions["Closest Season Comparison"] = "N/A"
        df_predictions["Draft Year"] = "Unknown"
    else:
        # ✅ Find closest comparisons
        df_predictions["Closest Season Comparison"] = find_closest_comparisons(df_predictions, training_data)

        # ✅ Merge to include Draft Year
        df_predictions = df_predictions.merge(
            training_data[["Player", "Draft Year"]],
            on="Player",
            how="left"
        )

    # ✅ Ensure all required columns exist in df_predictions
    required_columns = [
        "Player", "Closest Season Comparison", "peak_m/g", "peak_p/g", "peak_r/g", "peak_a/g",
        "peak_s/g", "peak_b/g", "peak_fg%", "peak_3%", "peak_ft%", "peak_3/g", "peak_fga/g",
        "peak_fta/g", "Draft Year"
    ]

    for col in required_columns:
        if col not in df_predictions.columns:
            df_predictions[col] = "N/A"  # ✅ Fill missing columns with "N/A" to avoid KeyError

    # ✅ Debug: Print columns in df_predictions
    print(f"🔍 Columns in df_predictions: {df_predictions.columns.tolist()}")

    # ✅ Define the 7 sheets (Cumulative + Draft Classes)
    sheet_names = ["Cumulative", "2020", "2021", "2022", "2023", "2024", "2025"]

    percentage_columns = ["peak_fg%", "peak_3%", "peak_ft%"]

    column_mapping = {
        "Player": "Player",
        "Closest Season Comparison": "Closest Season Comparison",
        "peak_m/g": "Peak MPG",
        "peak_p/g": "Peak PPG",
        "peak_r/g": "Peak RPG",
        "peak_a/g": "Peak APG",
        "peak_s/g": "Peak SPG",
        "peak_b/g": "Peak BPG",
        "peak_fg%": "Peak FG%",
        "peak_3%": "Peak 3P%",
        "peak_ft%": "Peak FT%",
        "peak_3/g": "Peak 3PM/g",
        "peak_fga/g": "Peak FGA/g",
        "peak_fta/g": "Peak FTA/g",
        "Draft Year": "Draft Year"
    }

    # ✅ Loop through each sheet and filter players accordingly
    for sheet_name in sheet_names:
        if sheet_name == "Cumulative":
            filtered_df = df_predictions  # ✅ Save ALL players in Cumulative
        else:
            filtered_df = df_predictions[df_predictions["Draft Year"] == int(sheet_name)]

        if filtered_df.empty:
            print(f"⚠️ No data for sheet '{sheet_name}', skipping.")
            continue

        # ✅ Debug: Print columns in filtered_df before processing
        print(f"🔍 Processing sheet: {sheet_name} | Columns in filtered_df: {filtered_df.columns.tolist()}")

        # ✅ Reorder & clean data
        filtered_df = filtered_df[required_columns]
        filtered_df = filtered_df.rename(columns=column_mapping)
        filtered_df = filtered_df.fillna("")

        # ✅ Only round percentage columns that exist
        available_percentage_columns = [col for col in percentage_columns if col in filtered_df.columns]

        if available_percentage_columns:
            filtered_df.loc[:, available_percentage_columns] = filtered_df[available_percentage_columns].round(3)

        # ✅ Check if sheet exists, create if not
        try:
            sheet = spreadsheet.worksheet(sheet_name)
        except gspread.exceptions.WorksheetNotFound:
            sheet = spreadsheet.add_worksheet(title=sheet_name, rows="1000", cols="20")

        # ✅ Clear existing data & write new data
        sheet.clear()
        sheet.append_row(filtered_df.columns.tolist())  # ✅ Write headers
        sheet.append_rows(filtered_df.values.tolist())  # ✅ Write data
        sheet.freeze(rows=1)  # ✅ Freeze header row
        sheet.set_basic_filter()  # ✅ Enable filtering

        print(f"✅ Google Sheet '{sheet_name}' updated with {len(filtered_df)} players.")


RUN_PREDICTIONS = True  # Change to True when you want predictions


def run_pipeline():
    # Train models for all experience levels
    groups = ["College", "Rookies", "Sophomores", "3rd-Year", "4th-Year", "5th-Year"]

    for group in groups:
        train_and_save_model(group, os.path.join(BASE_DIR, f"{group.lower()}_model.pkl"))


    all_predictions = []

    # Make predictions for all groups
    for group in groups:
        predictions_df = make_predictions(group)
        if predictions_df is not None:
            all_predictions.append(predictions_df)

    # ✅ Combine all predictions into one DataFrame
    if all_predictions:
        final_df = pd.concat(all_predictions)
        final_df = final_df.sort_values(by="Player", ascending=True)  # Sort by peak value
        update_google_sheet(final_df)  # ✅ FIXED: Now correctly calls the function
    else:
        print("❌ No predictions generated.")


if __name__ == "__main__":
    run_pipeline()
