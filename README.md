# Football Live Tracker Telegram Bot

Monitors live football matches across all major leagues (including Bulgarian Parva Liga) and sends you a Telegram notification when a team has **shots on target but 0 goals** — along with possession %, corners, and goals info.

## How It Works

1. Every 60 seconds, the bot fetches all **live** fixtures from api-football.com
2. For each match in a tracked league, it pulls live statistics
3. If a team has **>0 shots on target** and **0 goals**, you get a Telegram message with:
   - Shots on target count
   - Possession %
   - Corners
   - Goals scored (if any)
4. Re-notifies you only when shots on target **increase** (avoids spam)
5. Clears the alert once the team scores

## Tracked Leagues

| League | Country |
|---|---|
| Premier League | England |
| La Liga | Spain |
| Bundesliga | Germany |
| Serie A | Italy |
| Ligue 1 | France |
| Champions League | Europe |
| Europa League | Europe |
| Conference League | Europe |
| **Parva Liga** | **Bulgaria** |
| Primeira Liga | Portugal |
| Eredivisie | Netherlands |
| Super Lig | Turkey |
| Liga Profesional | Argentina |
| Serie A | Brazil |
| Liga MX | Mexico |

## Setup (3 steps)

### 1. Get your API keys

**Telegram Bot Token:**
- Message [@BotFather](https://t.me/BotFather) on Telegram
- Send `/newbot`, pick a name, get your token

**Telegram Chat ID:**
- Message [@userinfobot](https://t.me/userinfobot) on Telegram
- It replies with your numeric chat ID

**Football API Key (RapidAPI):**
- Sign up at [api-football.com](https://www.api-football.com/) (free plan: 100 requests/day)
- Go to [RapidAPI dashboard](https://rapidapi.com/api-sports/api/api-football) and copy your key

### 2. Run locally

```bash
cd football-telegram-bot
cp .env.example .env
# Edit .env and fill in your keys
pip install -r requirements.txt
python bot.py
```

### 3. Deploy for free on Railway (recommended)

```bash
# 1. Push to a GitHub repo
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/football-bot.git
git push -u origin main

# 2. Go to https://railway.app, click "New Project"
# 3. Choose "Deploy from GitHub repo"
# 4. Select your repo

# 5. In Railway dashboard, go to Variables tab and add:
#    TELEGRAM_BOT_TOKEN = your_token
#    TELEGRAM_CHAT_ID  = your_chat_id
#    RAPIDAPI_KEY       = your_api_key
#    POLL_INTERVAL      = 60
```

That's it! Railway auto-detects the Dockerfile and deploys.

## Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Yes | — | From @BotFather |
| `TELEGRAM_CHAT_ID` | Yes | — | Your Telegram user/group chat ID |
| `RAPIDAPI_KEY` | Yes | — | From RapidAPI (api-football) |
| `POLL_INTERVAL` | No | `60` | Seconds between polls |

## Free API Limits

- **api-football free plan**: 100 requests/day
- Each poll cycle uses 1 request (live fixtures) + 1 per live match in tracked leagues
- At 60s polling with ~5-10 live matches, that's ~6-11 requests/cycle
- You get roughly 9-16 full cycles per day on the free plan
- Want more? Upgrade to a paid api-football plan ($8/month for 3,000 requests/day)

## Message Example

```
SHOTS ON TARGET BUT NO GOAL

Arsenal  0 - 1  Chelsea
Premier League  67'

Arsenal
   Shots on target: 7
   Goals scored: 0
   Possession: 62%
   Corners: 8
```
