# Reddit Profiler

OSINT tool for analyzing Reddit user behavior and extracting intelligence from publicly available data.

## Features

- User activity analysis and timeline reconstruction
- Geographic location detection with confidence scoring
- Timezone and sleep pattern analysis
- Personal information extraction (age, occupation, family references)
- Communication style and personality assessment
- Social media cross-references
- Privacy risk scoring with detailed recommendations
- Interactive HTML reports with visualizations

## Installation

```bash
git clone https://github.com/unnohwn/reddit-profiler.git
cd reddit-profiler
pip install -r requirements.txt
```

## Setup

Get Reddit API credentials:
1. Visit https://www.reddit.com/prefs/apps
2. Create a "script" application
3. Note your client ID and secret

## Usage

```bash
python reddit_profiler.py USERNAME --client-id YOUR_ID --client-secret YOUR_SECRET
```

Options:
- `--limit 2000` - Number of posts/comments to analyze
- `--output-dir results` - Output directory
- `--user-agent "MyBot:1.0"` - Custom user agent

## Output

Results are saved to `profile_results/`:
- HTML report with interactive charts
- JSON data file with raw analysis
- Visualization images
- Raw posts and comments data

## Privacy & Legal

**Authorized Use:**
- Security research and threat analysis
- Academic studies on social behavior
- Digital forensics investigations

**Prohibited:**
- Harassment or stalking
- Doxxing or privacy violations
- Any illegal activities

This tool analyzes only public Reddit data and is intended for legitimate security research.

## Requirements

- Python 3.8+
- Reddit API access
- Internet connection
