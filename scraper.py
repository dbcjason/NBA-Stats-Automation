import pandas as pd
import time
import random
import gspread
import datetime
import os
import json
import sys
import numpy as np  # ✅ Add this import
from concurrent.futures import ThreadPoolExecutor, as_completed
from nba_api.stats.endpoints import PlayerDashboardByYearOverYear, PlayerCareerStats
from oauth2client.service_account import ServiceAccountCredentials
from tqdm import tqdm

GOOGLE_CREDS_PATH = "/content/drive/My Drive/google_creds.json"  # Update with the correct path

# ✅ Load Google Sheets API credentials
with open(GOOGLE_CREDS_PATH, "r") as creds_file:
    creds_dict = json.load(creds_file)

# ✅ Authenticate Google Sheets API
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
client = gspread.authorize(creds)

print("✅ Google Sheets authentication successful!")

# ✅ Google Sheets setup
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1aO1TouqQdJPmDXr_PeCRZxyRAflMfqat5CShTdv0Rlo/edit#gid=347820615"
SHEET_NAMES = {
    "Rookies": "Y1",
    "Sophomores": "Y2",
    "3rd-Year": "Y3",
    "4th-Year": "Y4",
    "5th-Year": "Y5"
}  # ✅ Each sheet mapped to the correct prefix

# ✅ NBA API settings
CURRENT_SEASON = "2024-25"

# ✅ Draft class update schedule (Monday to Friday)
DRAFT_SCHEDULE = {
    0: 2024,  # Monday -> 2024 Draft Class
    1: 2023,  # Tuesday -> 2023 Draft Class
    2: 2020,  # Wednesday -> 2022 Draft Class
    3: 2021,  # Thursday -> 2021 Draft Class
    4: 2022,  # Friday -> 2020 Draft Class
}

# ✅ Stats columns (Previously missing)
per_game_stat_cols = ["MIN", "GP", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
                      "FTM", "FTA", "FT_PCT", "REB", "AST", "TOV", "STL", "BLK", "PTS"]

per_100_stat_cols = ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "REB", "AST", "TOV", "STL", "BLK", "PTS", "PLUS_MINUS"]

# ✅ Determine today's draft class
today = datetime.datetime.today().weekday()
if today not in DRAFT_SCHEDULE:
    print("🛑 Today is not a scheduled update day. Exiting script.")
    exit()

draft_year_to_update = DRAFT_SCHEDULE[today]
print(f"🚀 Updating Draft Class {draft_year_to_update}...")

# ✅ Load all sheets into a dictionary {sheet_name: dataframe}
sheet_data = {}
for sheet_name in SHEET_NAMES.keys():
    sheet = client.open_by_url(GOOGLE_SHEET_URL).worksheet(sheet_name)
    sheet_data[sheet_name] = pd.DataFrame(sheet.get_all_records())

# ✅ Ensure all sheets have "NBA_ID" and "Draft Year"
for sheet_name, df in sheet_data.items():
    if "NBA_ID" not in df.columns or "Draft Year" not in df.columns:
        raise ValueError(f"🚨 'NBA_ID' or 'Draft Year' column missing in {sheet_name} sheet!")

print("✅ Successfully loaded Google Sheets data.")

# ✅ Filter only players from today's draft class
filtered_players = {}
for sheet_name, df in sheet_data.items():
    filtered_df = df[df["Draft Year"] == draft_year_to_update]
    
    if filtered_df.empty:
        print(f"⚠️ No players found in {sheet_name} for Draft Year {draft_year_to_update}.")
    else:
        print(f"📊 Players found in {sheet_name} for Draft Year {draft_year_to_update}:")
        print(filtered_df[["NBA_ID", "Draft Year", "Player"]].head(10))  # Show first 10 players for debugging
    
    filtered_players[sheet_name] = filtered_df

total_players = sum(len(df) for df in filtered_players.values())

if total_players == 0:
    print("🛑 No players found for today's draft class. Exiting.")
    exit()

print(f"Processing {total_players} players from {draft_year_to_update} Draft Class...")

# ✅ Dictionary to store updated stats
updated_stats = {}
nba_year_cache = {}  # ✅ Cache NBA Year results to avoid redundant failures


def fetch_player_stats(nba_id, prefix):
    """Fetch NBA stats for a single player with exponential backoff and retries."""
    retries = 5
    delay = 5

    while retries:
        try:
            time.sleep(random.uniform(3, 5))

            per_game_dashboard = PlayerDashboardByYearOverYear(player_id=nba_id, per_mode_detailed="PerGame", timeout=60).get_data_frames()[1]
            per_100_dashboard = PlayerDashboardByYearOverYear(player_id=nba_id, per_mode_detailed="Per100Possessions", timeout=60).get_data_frames()[1]

            per_game_data = per_game_dashboard[per_game_dashboard["GROUP_VALUE"] == CURRENT_SEASON]
            per_100_data = per_100_dashboard[per_100_dashboard["GROUP_VALUE"] == CURRENT_SEASON]

            if per_game_data.empty or per_100_data.empty:
                print(f"⚠️ No valid stats found for NBA ID {nba_id}. Skipping...")
                return nba_id, None

            per_game_data = per_game_data[per_game_data["GROUP_VALUE"] == CURRENT_SEASON].reset_index(drop=True)
            per_100_data = per_100_data[per_100_data["GROUP_VALUE"] == CURRENT_SEASON].reset_index(drop=True)

            if per_game_data.empty or per_100_data.empty:
                print(f"⚠️ No valid stats found for NBA ID {nba_id}. Skipping...")
                return nba_id, None

            # ✅ Take the first row instead of max GP (ensures we get the correct season)
            per_game_data = per_game_data.iloc[0]
            per_100_data = per_100_data.iloc[0]

            stats_row = {}
            for col in per_game_stat_cols:
                col_name = f"{prefix}_PG_{col}"  # Ensure exact match with your sheet
                stats_row[col_name] = per_game_data.get(col, None)

            for col in per_100_stat_cols:
                col_name = f"{prefix}_P100_{col}"  # Ensure exact match with your sheet
                stats_row[col_name] = per_100_data.get(col, None)


            print(f"✅ Successfully pulled stats for NBA ID {nba_id}")
            return nba_id, stats_row

        except Exception as e:
            print(f"⚠️ Error fetching stats for NBA ID {nba_id} (Attempt {6 - retries}/5): {e}")
            retries -= 1
            time.sleep(min(delay, 20))
            delay *= 2

    print(f"❌ Failed to fetch stats for NBA ID {nba_id} after multiple attempts.")
    return nba_id, None


# ✅ Process all players for today's draft class in parallel (Limit to 5 threads)
for sheet_name, prefix in SHEET_NAMES.items():
    df = filtered_players[sheet_name]

    if df.empty:
        continue  # ✅ Skip empty sheets

    print(f"📊 Updating {sheet_name} players with {prefix} stats...")
    sheet = client.open_by_url(GOOGLE_SHEET_URL).worksheet(sheet_name)
    existing_data = sheet.get_all_values()
    print(f"📊 Columns Found in {sheet_name} Sheet: {existing_data[0]}")


    player_ids = df["NBA_ID"].tolist()

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_player_stats, nba_id, prefix): nba_id for nba_id in player_ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {sheet_name} Players"):
            nba_id, stats = future.result()
            if stats:
                updated_stats[nba_id] = stats

    # ✅ Prepare batch update data
    batch_updates = []

    for nba_id, stats in updated_stats.items():
        if nba_id in player_ids or True:  # ✅ Force overwriting existing rows
            matching_rows = df[df["NBA_ID"] == nba_id]
            if not matching_rows.empty:
                row_idx = matching_rows.index[0] + 2  # ✅ Find the correct row in the DataFrame, then adjust for Google Sheets index
            else:
                print(f"⚠️ Warning: NBA ID {nba_id} not found in Google Sheets data.")
                continue  # Skip this player if their row isn't found

            row_data = []
            for col_name, value in stats.items():
                # ✅ If column does not exist in the sheet, add it
                if col_name not in existing_data[0]:
                    print(f"🆕 Adding missing column '{col_name}' to {sheet_name}")
                    existing_data[0].append(col_name)
                    col_idx = len(existing_data[0])  # ✅ Assign new index for new column
                else:
                    col_idx = existing_data[0].index(col_name) + 1

                # ✅ Convert NumPy types to standard Python types before updating Google Sheets
                if isinstance(value, (np.int64, np.float64)):  
                    value = value.item()  # Convert to Python int or float

                row_data.append((row_idx, col_idx, value))  # ✅ Append only once

            print(f"📢 Updating Google Sheets with: {row_data}")  # ✅ Debugging output
            print(f"✅ Preparing to update player NBA_ID {nba_id} in row {row_idx}")
            batch_updates.extend(row_data)  # ✅ Extend batch updates properly



    # ✅ Break batch updates into chunks (Google Sheets limit workaround)
    BATCH_SIZE = 100  # ✅ Adjust based on rate limits
    DELAY = 60  # ✅ Wait time in seconds between batches

    if batch_updates:
        print(f"📢 Preparing to update {len(batch_updates)} cells in batches...")

        for i in range(0, len(batch_updates), BATCH_SIZE):
            batch_chunk = batch_updates[i : i + BATCH_SIZE]

            cell_list = [gspread.Cell(row, col, value) for row, col, value in batch_chunk]
            MAX_RETRIES = 3
            for attempt in range(MAX_RETRIES):
                try:
                    sheet.update_cells(cell_list)
                    print(f"✅ Successfully updated {len(batch_chunk)} cells in {sheet_name} (Batch {i // BATCH_SIZE + 1})")
                    break  # ✅ Exit loop if successful
                except gspread.exceptions.APIError as e:
                    print(f"⚠️ API Error: {e}. Retrying {attempt + 1}/{MAX_RETRIES}...")
                    time.sleep(30)  # ✅ Wait before retrying
            else:
                print(f"❌ Failed to update {sheet_name} after {MAX_RETRIES} attempts.")

            print(f"✅ Updated {len(batch_chunk)} cells... Waiting {DELAY} seconds to avoid quota limits.")
            time.sleep(DELAY)  # ✅ Prevents hitting Google API quota

        print("🎉 All updates successfully written to Google Sheets!")





print(f"🎉 Finished updating {draft_year_to_update} Draft Class with Y1-Y5 stats!")
