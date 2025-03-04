import pandas as pd
import time
import random
import gspread
import datetime
import os
import json
from nba_api.stats.endpoints import PlayerDashboardByYearOverYear, PlayerCareerStats
from oauth2client.service_account import ServiceAccountCredentials
from tqdm import tqdm

# ✅ Load Google Sheets API credentials securely from GitHub Secrets
google_creds = os.getenv("GOOGLE_CREDS")

if google_creds is None:
    raise ValueError("🚨 GOOGLE_CREDS environment variable not found!")

# ✅ Convert JSON string back into a dictionary
creds_dict = json.loads(google_creds)

# ✅ Authenticate Google Sheets API
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
client = gspread.authorize(creds)

# ✅ Google Sheets setup
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1aO1TouqQdJPmDXr_PeCRZxyRAflMfqat5CShTdv0Rlo/edit#gid=347820615"
SHEET_NAMES = ["Rookies", "Sophomores", "3rd-Year", "4th-Year", "5th-Year"]  # ✅ The five sheets

# ✅ NBA API settings
CURRENT_SEASON = "2024-25"

# ✅ Draft class update schedule (Monday to Friday)
DRAFT_SCHEDULE = {
    0: 2024,  # Monday -> 2024 Draft Class
    1: 2023,  # Tuesday -> 2023 Draft Class
    2: 2022,  # Wednesday -> 2022 Draft Class
    3: 2021,  # Thursday -> 2021 Draft Class
    4: 2020,  # Friday -> 2020 Draft Class
}

# ✅ Determine today's draft class
today = datetime.datetime.today().weekday()
if today not in DRAFT_SCHEDULE:
    print("🛑 Today is not a scheduled update day. Exiting script.")
    exit()

draft_year_to_update = DRAFT_SCHEDULE[today]
print(f"🚀 Updating Draft Class {draft_year_to_update}...")

# ✅ Stats columns
per_game_stat_cols = ["MIN", "GP", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
                      "FTM", "FTA", "FT_PCT", "REB", "AST", "TOV", "STL", "BLK", "PTS"]

per_100_stat_cols = ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "REB", "AST", "TOV", "STL", "BLK", "PTS", "PLUS_MINUS"]

# ✅ Load all sheets into a dictionary {sheet_name: dataframe}
sheet_data = {}
for sheet_name in SHEET_NAMES:
    sheet = client.open_by_url(GOOGLE_SHEET_URL).worksheet(sheet_name)
    sheet_data[sheet_name] = pd.DataFrame(sheet.get_all_records())

# ✅ Ensure all sheets have "Draft Year" and "NBA_ID"
for sheet_name, df in sheet_data.items():
    if "Draft Year" not in df.columns or "NBA_ID" not in df.columns:
        raise ValueError(f"🚨 'Draft Year' and 'NBA_ID' columns missing in {sheet_name} sheet!")

# ✅ Combine all sheets into one DataFrame
full_data = pd.concat(sheet_data.values(), ignore_index=True)

# ✅ Filter only players from today's draft class, regardless of which sheet they're in
filtered_players = full_data[full_data["Draft Year"] == draft_year_to_update]

# ✅ Dictionary to store updated stats
updated_stats = {}

def get_nba_year(nba_id):
    """Determine a player's current NBA Year based on seasons played."""
    try:
        career = PlayerCareerStats(player_id=nba_id).get_data_frames()[0]
        if career.empty:
            return None  # No NBA seasons played
        unique_seasons = career[career["GP"] > 0]["SEASON_ID"].unique()
        return len(unique_seasons)  # Returns Y1, Y2, etc.

    except Exception as e:
        print(f"⚠️ Error determining NBA Year for NBA ID {nba_id}: {e}")
        return None

# ✅ Process each player
for idx, row in tqdm(filtered_players.iterrows(), total=len(filtered_players), desc=f"Fetching NBA Stats for {draft_year_to_update}"):
    nba_id = row["NBA_ID"]

    retries = 2
    while retries:
        try:
            time.sleep(random.uniform(1, 3))  # ✅ Avoid rate limits

            # ✅ Determine correct NBA Year (Y1, Y2, etc.)
            nba_year = get_nba_year(nba_id)
            if not nba_year or nba_year > 5:
                print(f"⚠️ NBA ID {nba_id} has an invalid NBA Year. Skipping...")
                break

            # ✅ Fetch current season per-game stats
            per_game_dashboard = PlayerDashboardByYearOverYear(
                player_id=nba_id, per_mode_detailed="PerGame"
            ).get_data_frames()[1]

            # ✅ Fetch current season per-100 possessions stats
            per_100_dashboard = PlayerDashboardByYearOverYear(
                player_id=nba_id, per_mode_detailed="Per100Possessions"
            ).get_data_frames()[1]

            # ✅ Filter for current season
            per_game_data = per_game_dashboard[
                per_game_dashboard["GROUP_VALUE"] == CURRENT_SEASON
            ]
            per_100_data = per_100_dashboard[
                per_100_dashboard["GROUP_VALUE"] == CURRENT_SEASON
            ]

            # ✅ If no stats exist, skip the player
            if per_game_data.empty or per_100_data.empty:
                print(f"⚠️ No valid stats found for NBA ID {nba_id}. Skipping...")
                updated_stats[nba_id] = {col: None for col in per_game_stat_cols + per_100_stat_cols}
                break

            # ✅ If multiple rows exist, use the highest GP row
            per_game_data = per_game_data.iloc[per_game_data["GP"].idxmax()]
            per_100_data = per_100_data.iloc[per_100_data["GP"].idxmax()]

            # ✅ Map stats to correct columns using determined NBA Year
            prefix = f"Y{nba_year}"
            stats_row = {f"{prefix}_PG_{col}": per_game_data.get(col, None) for col in per_game_stat_cols}
            stats_row.update({f"{prefix}_P100_{col}": per_100_data.get(col, None) for col in per_100_stat_cols})

            # ✅ Store updated data
            updated_stats[nba_id] = stats_row
            break  # Success, move to next player

        except Exception as e:
            print(f"⚠️ Error fetching stats for NBA ID {nba_id} (Attempt {3-retries}/2): {e}")
            retries -= 1
            time.sleep(random.uniform(2, 5))

# ✅ Convert to DataFrame
stats_df = pd.DataFrame.from_dict(updated_stats, orient="index").reset_index()
stats_df.rename(columns={"index": "NBA_ID"}, inplace=True)

# ✅ Merge updated stats back into each sheet
for sheet_name, df in sheet_data.items():
    # ✅ Merge only players that exist in this specific sheet
    df = df.merge(stats_df, on="NBA_ID", how="left")
    df = df.where(pd.notna(df), None)  # Convert NaN to None

    # ✅ Update Google Sheet
    sheet = client.open_by_url(GOOGLE_SHEET_URL).worksheet(sheet_name)
    sheet.update([df.columns.values.tolist()] + df.values.tolist())

print(f"🎉 Finished updating Draft Class {draft_year_to_update} across all sheets!")
