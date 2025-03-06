import pandas as pd
import time
import random
import gspread
import datetime
import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# ✅ Filter only players from today's draft class
filtered_players = full_data[full_data["Draft Year"] == draft_year_to_update]

# ✅ Handle batch processing (first half vs second half)
batch_type = sys.argv[1] if len(sys.argv) > 1 else "first"
midpoint = len(filtered_players) // 2

if batch_type == "first":
    filtered_players = filtered_players.iloc[:midpoint]  # First half
elif batch_type == "second":
    filtered_players = filtered_players.iloc[midpoint:]  # Second half

print(f"Processing {len(filtered_players)} players for batch: {batch_type}")

# ✅ Dictionary to store updated stats
updated_stats = {}
nba_year_cache = {}  # ✅ Cache NBA Year results to avoid redundant failures

def get_nba_year(nba_id):
    """Determine a player's current NBA Year based on seasons played, handling timeouts and missing data."""
    if nba_id in nba_year_cache:
        return nba_year_cache[nba_id]  # ✅ Use cached value if available

    retries = 3  # ✅ Max retries for API calls
    delay = 5    # ✅ Start delay at 5 seconds

    for attempt in range(retries):
        try:
            time.sleep(random.uniform(3, 5))  # ✅ Stagger requests to avoid rate limits
            career = PlayerCareerStats(player_id=nba_id, timeout=60).get_data_frames()[0]

            # ✅ If there's no game data for this player, log and skip
            if career.empty or "GP" not in career.columns:
                print(f"⚠️ NBA ID {nba_id} has no recorded seasons. Skipping...")
                nba_year_cache[nba_id] = None
                return None  

            unique_seasons = career[career["GP"] > 0]["SEASON_ID"].unique()

            # ✅ If no seasons are found, skip the player
            if len(unique_seasons) == 0:
                print(f"⚠️ NBA ID {nba_id} has no valid NBA seasons. Skipping...")
                nba_year_cache[nba_id] = None
                return None

            nba_year_cache[nba_id] = len(unique_seasons)  # ✅ Store in cache
            return nba_year_cache[nba_id]

        except Exception as e:
            print(f"⚠️ Attempt {attempt+1}/{retries}: Error fetching NBA Year for {nba_id}: {e}")

            if attempt == retries - 1:  # ✅ Last retry failed
                print(f"❌ NBA ID {nba_id} permanently failed. Skipping future queries.")
                nba_year_cache[nba_id] = None  # ✅ Mark permanently failed
                return None

            time.sleep(delay)  # ✅ Increase wait time before retrying
            delay *= 2  # ✅ Exponential backoff (5s, 10s, 20s, etc.)


def fetch_player_stats(nba_id):
    """Fetch NBA stats for a single player with exponential backoff and retries."""
    retries = 5  
    delay = 5  # ✅ Increased from 2s → 5s

    while retries:
        try:
            time.sleep(random.uniform(3, 5))  # ✅ Slower requests to avoid rate limiting

            nba_year = get_nba_year(nba_id)

            if nba_year is None:  # Only skip if the NBA Year is truly unknown
                print(f"⚠️ NBA ID {nba_id} has an unknown NBA Year. Skipping...")
                return nba_id, None

            per_game_dashboard = PlayerDashboardByYearOverYear(player_id=nba_id, per_mode_detailed="PerGame", timeout=60).get_data_frames()[1]
            per_100_dashboard = PlayerDashboardByYearOverYear(player_id=nba_id, per_mode_detailed="Per100Possessions", timeout=60).get_data_frames()[1]

            per_game_data = per_game_dashboard[per_game_dashboard["GROUP_VALUE"] == CURRENT_SEASON]
            per_100_data = per_100_dashboard[per_100_dashboard["GROUP_VALUE"] == CURRENT_SEASON]

            if per_game_data.empty or per_100_data.empty:
                print(f"⚠️ No valid stats found for NBA ID {nba_id}. Skipping...")
                return nba_id, {col: None for col in per_game_stat_cols + per_100_stat_cols}

            per_game_data = per_game_data.iloc[per_game_data["GP"].idxmax()]
            per_100_data = per_100_data.iloc[per_100_data["GP"].idxmax()]

            prefix = f"Y{nba_year}"
            stats_row = {f"{prefix}_PG_{col}": per_game_data.get(col, None) for col in per_game_stat_cols}
            stats_row.update({f"{prefix}_P100_{col}": per_100_data.get(col, None) for col in per_100_stat_cols})

            print(f"✅ Successfully pulled stats for NBA ID {nba_id}")
            return nba_id, stats_row

        except Exception as e:
            print(f"⚠️ Error fetching stats for NBA ID {nba_id} (Attempt {6-retries}/5): {e}")
            retries -= 1
            time.sleep(min(delay, 20))
            delay *= 2  
    
    print(f"❌ Failed to fetch stats for NBA ID {nba_id} after multiple attempts.")
    return nba_id, None

# ✅ Process players in parallel (Limit to 2 threads to avoid rate limiting)
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(fetch_player_stats, row["NBA_ID"]): row["NBA_ID"] for _, row in filtered_players.iterrows()}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Players"):
        nba_id, stats = future.result()
        if stats:
            updated_stats[nba_id] = stats

print(f"🎉 Finished updating Draft Class {draft_year_to_update} across all sheets!")
