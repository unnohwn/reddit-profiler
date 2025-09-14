#!/usr/bin/env python3

import praw
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import click
import json
import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import time
import re
from tqdm import tqdm
from colorama import init, Fore, Style
from tabulate import tabulate
import pytz
import statistics
import hashlib
import base64
from urllib.parse import urlparse
import ipaddress

init()

class RedditProfiler:
    def __init__(self, client_id=None, client_secret=None, user_agent=None):
        self.reddit = None
        self.user_data = {}
        self.posts_data = []
        self.comments_data = []
        self.network_data = defaultdict(int)
        self.timeline = []
        self.personal_info = defaultdict(list)
        self.privacy_risks = []
        self.behavioral_patterns = {}
        self.communication_style = {}
        self.social_references = defaultdict(int)
        self.technical_indicators = defaultdict(list)
        self.threat_indicators = []
        self.behavioral_anomalies = []
        self.linguistic_fingerprint = {}
        self.account_correlations = []
        
        if client_id and client_secret and user_agent:
            self.setup_reddit_client(client_id, client_secret, user_agent)
    
    def setup_reddit_client(self, client_id, client_secret, user_agent):
        try:
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
            print(f"{Fore.GREEN}Reddit API client initialized successfully{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error initializing Reddit client: {str(e)}{Style.RESET_ALL}")
            sys.exit(1)
    
    def validate_username(self, username):
        try:
            user = self.reddit.redditor(username)
            user.id
            return True
        except Exception:
            return False
    
    def collect_user_data(self, username, limit=1000):
        if not self.reddit:
            raise ValueError("Reddit client not initialized")
        
        if not self.validate_username(username):
            raise ValueError(f"Username '{username}' not found or inaccessible")
        
        print(f"{Fore.CYAN}Analyzing user: {username}{Style.RESET_ALL}")
        
        user = self.reddit.redditor(username)
        
        self.user_data = {
            'username': username,
            'created_utc': datetime.fromtimestamp(user.created_utc),
            'link_karma': user.link_karma,
            'comment_karma': user.comment_karma,
            'total_karma': user.link_karma + user.comment_karma,
            'account_age_days': (datetime.now() - datetime.fromtimestamp(user.created_utc)).days,
            'has_verified_email': user.has_verified_email,
            'is_gold': user.is_gold,
            'is_mod': user.is_mod
        }
        
        print(f"{Fore.YELLOW}Collecting posts...{Style.RESET_ALL}")
        posts = []
        try:
            for submission in tqdm(user.submissions.new(limit=limit), desc="Posts"):
                post_data = {
                    'type': 'post',
                    'id': submission.id,
                    'title': submission.title,
                    'text': submission.selftext,
                    'subreddit': str(submission.subreddit),
                    'score': submission.score,
                    'upvote_ratio': submission.upvote_ratio,
                    'num_comments': submission.num_comments,
                    'created_utc': datetime.fromtimestamp(submission.created_utc),
                    'url': submission.url,
                    'is_self': submission.is_self,
                    'stickied': submission.stickied,
                    'over_18': submission.over_18,
                    'spoiler': submission.spoiler,
                    'locked': submission.locked
                }
                posts.append(post_data)
                self.timeline.append({
                    'datetime': post_data['created_utc'],
                    'type': 'post',
                    'subreddit': post_data['subreddit'],
                    'score': post_data['score'],
                    'content': f"{post_data['title']} {post_data['text']}"
                })
        except Exception as e:
            print(f"{Fore.RED}Warning: Error collecting posts: {str(e)}{Style.RESET_ALL}")
        
        self.posts_data = posts
        
        print(f"{Fore.YELLOW}Collecting comments...{Style.RESET_ALL}")
        comments = []
        try:
            for comment in tqdm(user.comments.new(limit=limit), desc="Comments"):
                comment_data = {
                    'type': 'comment',
                    'id': comment.id,
                    'body': comment.body,
                    'subreddit': str(comment.subreddit),
                    'score': comment.score,
                    'created_utc': datetime.fromtimestamp(comment.created_utc),
                    'is_submitter': comment.is_submitter,
                    'stickied': comment.stickied,
                    'parent_id': comment.parent_id,
                    'link_id': comment.link_id
                }
                comments.append(comment_data)
                self.timeline.append({
                    'datetime': comment_data['created_utc'],
                    'type': 'comment',
                    'subreddit': comment_data['subreddit'],
                    'score': comment_data['score'],
                    'content': comment_data['body']
                })
                
                mentioned_users = re.findall(r'/u/(\w+)', comment.body)
                for mentioned_user in mentioned_users:
                    if mentioned_user.lower() != username.lower():
                        self.network_data[mentioned_user] += 1
                        
        except Exception as e:
            print(f"{Fore.RED}Warning: Error collecting comments: {str(e)}{Style.RESET_ALL}")
        
        self.comments_data = comments
        self.timeline.sort(key=lambda x: x['datetime'])
        
        print(f"{Fore.GREEN}Data collection complete:{Style.RESET_ALL}")
        print(f"  Posts: {len(self.posts_data)}")
        print(f"  Comments: {len(self.comments_data)}")
        print(f"  Timeline entries: {len(self.timeline)}")
    
    def analyze_timezone_patterns(self):
        if not self.timeline:
            return {}

        hours = [entry['datetime'].hour for entry in self.timeline]
        hour_counts = Counter(hours)

        most_active_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:8]

        avg_hour = statistics.mean(hours)

        timezone_indicators = {
            'Europe/Stockholm': {'offset': 1, 'common_hours': [7, 8, 9, 18, 19, 20, 21, 22], 'countries': ['Sweden']},
            'Europe/London': {'offset': 0, 'common_hours': [7, 8, 9, 18, 19, 20, 21, 22], 'countries': ['United Kingdom']},
            'Europe/Berlin': {'offset': 1, 'common_hours': [7, 8, 9, 18, 19, 20, 21, 22], 'countries': ['Germany']},
            'Europe/Paris': {'offset': 1, 'common_hours': [7, 8, 9, 18, 19, 20, 21, 22], 'countries': ['France']},
            'US/Eastern': {'offset': -5, 'common_hours': [7, 8, 9, 18, 19, 20, 21, 22], 'countries': ['United States']},
            'US/Central': {'offset': -6, 'common_hours': [7, 8, 9, 18, 19, 20, 21, 22], 'countries': ['United States']},
            'US/Pacific': {'offset': -8, 'common_hours': [7, 8, 9, 18, 19, 20, 21, 22], 'countries': ['United States']},
            'Asia/Tokyo': {'offset': 9, 'common_hours': [7, 8, 9, 18, 19, 20, 21, 22], 'countries': ['Japan']},
            'Australia/Sydney': {'offset': 10, 'common_hours': [7, 8, 9, 18, 19, 20, 21, 22], 'countries': ['Australia']}
        }

        geo_analysis = self.analyze_geographic_indicators()
        likely_country = geo_analysis.get('most_likely_country', 'Unknown')

        timezone_scores = {}
        for tz_name, tz_info in timezone_indicators.items():
            score = 0

            for hour in tz_info['common_hours']:
                score += hour_counts.get(hour, 0) * 2

            sleep_hours = [(h - tz_info['offset']) % 24 for h in range(2, 7)]
            for hour in sleep_hours:
                score -= hour_counts.get(hour, 0) * 1.5

            if likely_country in tz_info['countries']:
                score *= 3

            timezone_scores[tz_name] = max(score, 0)

        if timezone_scores and max(timezone_scores.values()) > 0:
            most_likely_tz = max(timezone_scores.items(), key=lambda x: x[1])[0]
        else:
            country_timezone_map = {
                'Sweden': 'Europe/Stockholm',
                'United Kingdom': 'Europe/London',
                'Germany': 'Europe/Berlin',
                'France': 'Europe/Paris',
                'United States': 'US/Eastern',
                'Japan': 'Asia/Tokyo',
                'Australia': 'Australia/Sydney'
            }
            most_likely_tz = country_timezone_map.get(likely_country, "Unknown")

        return {
            'most_active_hours': most_active_hours,
            'average_posting_hour': round(avg_hour, 1),
            'likely_timezone': most_likely_tz,
            'timezone_scores': {k: round(v, 2) for k, v in timezone_scores.items()},
            'hourly_distribution': dict(hour_counts),
            'sleep_pattern': self._analyze_sleep_pattern(hours),
            'geographic_correlation': likely_country
        }
    
    def _analyze_sleep_pattern(self, hours):
        night_hours = [h for h in hours if h >= 0 and h <= 6]
        morning_hours = [h for h in hours if h >= 6 and h <= 12] 
        evening_hours = [h for h in hours if h >= 18 and h <= 24]
        
        night_activity = len(night_hours)
        total_activity = len(hours)
        
        if total_activity == 0:
            return "Unknown"
        
        night_ratio = night_activity / total_activity
        
        if night_ratio > 0.3:
            return "Night owl"
        elif len(morning_hours) / total_activity > 0.4:
            return "Early bird"
        else:
            return "Normal schedule"
    
    def extract_personal_information(self):
        all_text = []
        
        for post in self.posts_data:
            all_text.append(f"{post['title']} {post['text']}")
        
        for comment in self.comments_data:
            all_text.append(comment['body'])
        
        if not all_text:
            return {}
        
        combined_text = ' '.join(all_text).lower()
        
        personal_info = {
            'age_indicators': [],
            'location_hints': [],
            'specific_locations': [],
            'occupation_hints': [],
            'relationship_status': [],
            'education_level': [],
            'financial_status': [],
            'family_indicators': [],
            'health_mentions': [],
            'technology_usage': []
        }
        
        age_patterns = [
            r'i am (\d{1,2})', r"i'm (\d{1,2})", r'age (\d{1,2})',
            r'(\d{1,2}) years old', r'born in (\d{4})', r'(\d{1,2})yo',
            r'turned (\d{1,2})', r'my (\d{1,2})th birthday'
        ]
        
        for pattern in age_patterns:
            matches = re.findall(pattern, combined_text)
            for match in matches:
                if match.isdigit():
                    age = int(match)
                    if pattern == r'born in (\d{4})':
                        age = datetime.now().year - age
                    if 13 <= age <= 100:
                        personal_info['age_indicators'].append(age)
        
        location_patterns = [
            r'live in ([a-z\s,]+)', r'from ([a-z\s,]+)', r'in ([a-z\s,]+) area',
            r'([a-z\s,]+) resident', r'moved to ([a-z\s,]+)', r'visiting ([a-z\s,]+)',
            r'([a-z\s,]+) weather', r'downtown ([a-z\s,]+)', r'born in ([a-z\s,]+)',
            r'grew up in ([a-z\s,]+)', r'based in ([a-z\s,]+)'
        ]
        
        specific_city_patterns = {
            'United States': {
                'cities': ['new york', 'los angeles', 'chicago', 'houston', 'phoenix', 'philadelphia', 
                          'san antonio', 'san diego', 'dallas', 'san jose', 'austin', 'jacksonville',
                          'fort worth', 'columbus', 'charlotte', 'san francisco', 'indianapolis',
                          'seattle', 'denver', 'washington dc', 'boston', 'el paso', 'detroit',
                          'nashville', 'portland', 'memphis', 'oklahoma city', 'las vegas', 'louisville',
                          'baltimore', 'milwaukee', 'albuquerque', 'tucson', 'fresno', 'mesa',
                          'sacramento', 'atlanta', 'kansas city', 'colorado springs', 'miami',
                          'raleigh', 'omaha', 'long beach', 'virginia beach', 'oakland', 'minneapolis',
                          'tulsa', 'tampa', 'arlington', 'new orleans'],
                'states': ['alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado',
                          'connecticut', 'delaware', 'florida', 'georgia', 'hawaii', 'idaho',
                          'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana',
                          'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota',
                          'mississippi', 'missouri', 'montana', 'nebraska', 'nevada',
                          'new hampshire', 'new jersey', 'new mexico', 'new york', 'north carolina',
                          'north dakota', 'ohio', 'oklahoma', 'oregon', 'pennsylvania',
                          'rhode island', 'south carolina', 'south dakota', 'tennessee', 'texas',
                          'utah', 'vermont', 'virginia', 'washington', 'west virginia', 'wisconsin', 'wyoming']
            },
            'United Kingdom': {
                'cities': ['london', 'birmingham', 'manchester', 'glasgow', 'liverpool', 'leeds',
                          'sheffield', 'edinburgh', 'bristol', 'cardiff', 'belfast', 'newcastle',
                          'nottingham', 'hull', 'plymouth', 'stoke-on-trent', 'wolverhampton',
                          'derby', 'swansea', 'southampton', 'salford', 'aberdeen', 'westminster',
                          'reading', 'luton', 'york', 'stockport', 'bolton', 'wigan'],
                'regions': ['england', 'scotland', 'wales', 'northern ireland', 'yorkshire',
                           'lancashire', 'devon', 'cornwall', 'kent', 'essex', 'surrey']
            },
            'Canada': {
                'cities': ['toronto', 'montreal', 'vancouver', 'calgary', 'edmonton', 'ottawa',
                          'winnipeg', 'quebec city', 'hamilton', 'kitchener', 'london', 'victoria',
                          'halifax', 'oshawa', 'windsor', 'saskatoon', 'st. catharines', 'regina',
                          'kelowna', 'barrie', 'sherbrooke', 'guelph', 'kanata', 'abbotsford'],
                'provinces': ['ontario', 'quebec', 'british columbia', 'alberta', 'manitoba',
                             'saskatchewan', 'nova scotia', 'new brunswick', 'newfoundland',
                             'prince edward island', 'northwest territories', 'nunavut', 'yukon']
            },
            'Australia': {
                'cities': ['sydney', 'melbourne', 'brisbane', 'perth', 'adelaide', 'gold coast',
                          'newcastle', 'canberra', 'sunshine coast', 'wollongong', 'geelong',
                          'hobart', 'townsville', 'cairns', 'toowoomba', 'darwin', 'ballarat',
                          'bendigo', 'albury', 'launceston', 'mackay', 'rockhampton', 'bunbury'],
                'states': ['new south wales', 'victoria', 'queensland', 'western australia',
                          'south australia', 'tasmania', 'northern territory', 'australian capital territory']
            }
        }
        
        for pattern in location_patterns:
            matches = re.findall(pattern, combined_text)
            personal_info['location_hints'].extend(matches)
        
        detected_locations = []
        for country, locations in specific_city_patterns.items():
            for category, places in locations.items():
                for place in places:
                    pattern = r'\b' + re.escape(place.lower()) + r'\b'
                    matches = re.findall(pattern, combined_text)
                    count = len(matches)

                    if count > 0:
                        confidence = min(95, count * 15)

                        context_patterns = [
                            f"from {place}", f"live in {place}", f"born in {place}",
                            f"visiting {place}", f"moved to {place}", f"{place} resident"
                        ]

                        for ctx_pattern in context_patterns:
                            if ctx_pattern.lower() in combined_text:
                                confidence = min(95, confidence + 25)
                                break

                        detected_locations.append({
                            'location': place.title(),
                            'category': category.rstrip('s').title(),
                            'country': country,
                            'mentions': count,
                            'confidence': confidence
                        })

        country_locations = {
            'United States': {
                'cities': ['new york', 'nyc', 'los angeles', 'chicago', 'houston', 'phoenix', 'philadelphia',
                          'san antonio', 'san diego', 'dallas', 'san jose', 'austin', 'jacksonville', 'fort worth',
                          'columbus', 'charlotte', 'san francisco', 'indianapolis', 'seattle', 'denver',
                          'washington dc', 'boston', 'el paso', 'nashville', 'detroit', 'oklahoma city',
                          'portland', 'las vegas', 'memphis', 'louisville', 'baltimore', 'milwaukee'],
                'regions': ['california', 'texas', 'florida', 'new york', 'pennsylvania', 'illinois', 'ohio',
                           'georgia', 'north carolina', 'michigan', 'new jersey', 'virginia', 'washington',
                           'arizona', 'massachusetts', 'tennessee', 'indiana', 'maryland', 'missouri',
                           'wisconsin', 'colorado', 'minnesota', 'south carolina', 'alabama', 'louisiana'],
                'landmarks': ['statue of liberty', 'golden gate bridge', 'times square', 'central park',
                             'grand canyon', 'yellowstone', 'mount rushmore', 'niagara falls', 'disneyland',
                             'disney world', 'hollywood sign', 'empire state building', 'space needle']
            },
            'United Kingdom': {
                'cities': ['london', 'birmingham', 'manchester', 'glasgow', 'liverpool', 'leeds', 'sheffield',
                          'edinburgh', 'bristol', 'cardiff', 'belfast', 'nottingham', 'leicester', 'coventry',
                          'hull', 'bradford', 'stoke', 'wolverhampton', 'plymouth', 'derby', 'southampton',
                          'swansea', 'reading', 'bournemouth', 'middlesbrough', 'sunderland', 'brighton'],
                'regions': ['england', 'scotland', 'wales', 'northern ireland', 'yorkshire', 'lancashire',
                           'cornwall', 'devon', 'kent', 'essex', 'sussex', 'surrey', 'hampshire', 'dorset',
                           'somerset', 'gloucestershire', 'oxfordshire', 'buckinghamshire', 'hertfordshire'],
                'landmarks': ['big ben', 'tower of london', 'stonehenge', 'buckingham palace', 'westminster abbey',
                             'london eye', 'edinburgh castle', 'hadrians wall', 'windsor castle', 'canterbury cathedral',
                             'bath', 'tower bridge', 'trafalgar square', 'piccadilly circus', 'loch ness']
            },
            'Canada': {
                'cities': ['toronto', 'montreal', 'vancouver', 'calgary', 'edmonton', 'ottawa', 'winnipeg',
                          'quebec city', 'hamilton', 'kitchener', 'london', 'victoria', 'halifax', 'oshawa',
                          'windsor', 'saskatoon', 'regina', 'sherbrooke', 'st johns', 'barrie', 'kelowna',
                          'abbotsford', 'sudbury', 'kingston', 'saguenay', 'trois rivieres', 'guelph'],
                'regions': ['ontario', 'quebec', 'british columbia', 'alberta', 'manitoba', 'saskatchewan',
                           'nova scotia', 'new brunswick', 'newfoundland', 'prince edward island', 'yukon',
                           'northwest territories', 'nunavut', 'labrador'],
                'landmarks': ['cn tower', 'niagara falls', 'banff', 'jasper', 'old quebec', 'parliament hill',
                             'chateau frontenac', 'cabot trail', 'bay of fundy', 'lake louise', 'whistler',
                             'mont tremblant', 'stanley park', 'algonquin park']
            },
            'Australia': {
                'cities': ['sydney', 'melbourne', 'brisbane', 'perth', 'adelaide', 'gold coast', 'newcastle',
                          'canberra', 'sunshine coast', 'wollongong', 'hobart', 'geelong', 'townsville',
                          'cairns', 'darwin', 'toowoomba', 'ballarat', 'bendigo', 'albury', 'launceston',
                          'mackay', 'rockhampton', 'bunbury', 'bundaberg', 'wagga wagga', 'coffs harbour'],
                'regions': ['new south wales', 'victoria', 'queensland', 'western australia', 'south australia',
                           'tasmania', 'northern territory', 'australian capital territory'],
                'landmarks': ['sydney opera house', 'sydney harbour bridge', 'uluru', 'ayers rock', 'great barrier reef',
                             'twelve apostles', 'blue mountains', 'kakadu', 'cradle mountain', 'great ocean road',
                             'bondi beach', 'fraser island', 'daintree rainforest', 'port arthur']
            },
            'Germany': {
                'cities': ['berlin', 'hamburg', 'munich', 'cologne', 'frankfurt', 'stuttgart', 'düsseldorf',
                          'dortmund', 'essen', 'leipzig', 'bremen', 'dresden', 'hannover', 'nuremberg',
                          'duisburg', 'bochum', 'wuppertal', 'bielefeld', 'bonn', 'münster', 'karlsruhe',
                          'mannheim', 'augsburg', 'wiesbaden', 'gelsenkirchen', 'mönchengladbach'],
                'regions': ['bayern', 'bavaria', 'baden württemberg', 'nordrhein westfalen', 'hessen', 'sachsen',
                           'niedersachsen', 'rheinland pfalz', 'thüringen', 'brandenburg', 'sachsen anhalt',
                           'schleswig holstein', 'mecklenburg vorpommern', 'saarland', 'bremen', 'hamburg', 'berlin'],
                'landmarks': ['brandenburg gate', 'neuschwanstein castle', 'cologne cathedral', 'black forest',
                             'rhine valley', 'oktoberfest', 'zugspitze', 'rothenburg', 'heidelberg castle',
                             'sanssouci palace', 'warburg castle', 'lorelei', 'checkpoint charlie']
            },
            'France': {
                'cities': ['paris', 'marseille', 'lyon', 'toulouse', 'nantes', 'strasbourg', 'montpellier',
                          'bordeaux', 'lille', 'rennes', 'reims', 'saint étienne', 'toulon', 'angers', 'grenoble',
                          'dijon', 'nîmes', 'aix en provence', 'brest', 'le mans', 'amiens', 'tours', 'limoges',
                          'clermont ferrand', 'villeurbanne', 'besançon', 'orléans', 'mulhouse'],
                'regions': ['ile de france', 'provence', 'normandy', 'normandie', 'brittany', 'bretagne', 'burgundy',
                           'bourgogne', 'loire valley', 'aquitaine', 'languedoc', 'alsace', 'champagne', 'corsica',
                           'corse', 'rhône alpes', 'midi pyrénées', 'nord pas de calais', 'pays de la loire'],
                'landmarks': ['eiffel tower', 'louvre', 'notre dame', 'versailles', 'mont blanc', 'loire castles',
                             'mont saint michel', 'champs élysées', 'arc de triomphe', 'sacré cœur', 'palace of versailles',
                             'côte dazur', 'french riviera', 'cannes', 'avignon', 'carcassonne']
            },
            'Netherlands': {
                'cities': ['amsterdam', 'rotterdam', 'the hague', 'utrecht', 'eindhoven', 'tilburg', 'groningen',
                          'almere', 'breda', 'nijmegen', 'enschede', 'haarlem', 'arnhem', 'zaanstad', 'amersfoort',
                          'apeldoorn', 'hoofddorp', 'maastricht', 'leiden', 'dordrecht', 'zoetermeer', 'zwolle',
                          'deventer', 'delft', 'alkmaar', 'leeuwarden', 'venlo'],
                'regions': ['north holland', 'south holland', 'noord holland', 'zuid holland', 'zeeland', 'utrecht',
                           'gelderland', 'overijssel', 'drenthe', 'friesland', 'groningen', 'flevoland',
                           'north brabant', 'noord brabant', 'limburg'],
                'landmarks': ['anne frank house', 'keukenhof', 'kinderdijk', 'rijksmuseum', 'van gogh museum',
                             'red light district', 'canal ring', 'zaanse schans', 'giethoorn', 'madurodam',
                             'binnenhof', 'peace palace', 'delta works']
            },
            'Sweden': {
                'cities': ['stockholm', 'göteborg', 'gothenburg', 'malmö', 'malmo', 'uppsala', 'västerås', 'vasteras',
                          'örebro', 'orebro', 'linköping', 'linkoping', 'helsingborg', 'jönköping', 'jonkoping',
                          'norrköping', 'norrkoping', 'lund', 'umeå', 'umea', 'gävle', 'gavle', 'borås', 'boras',
                          'eskilstuna', 'södertälje', 'sodertalje', 'karlstad', 'täby', 'taby', 'växjö', 'vaxjo'],
                'regions': ['skåne', 'skane', 'småland', 'smaland', 'västergötland', 'vastergotland', 'östergötland',
                           'ostergotland', 'dalarna', 'värmland', 'varmland', 'gävleborg', 'gavleborg', 'västerbotten',
                           'vasterbotten', 'norrbotten', 'jämtland', 'jamtland', 'västmanland', 'vastmanland'],
                'landmarks': ['gamla stan', 'drottningholm', 'vasa museum', 'abba museum', 'skansen', 'fotografiska',
                             'turning torso', 'liseberg', 'göteborgs konstmuseum', 'universeum', 'icehotel',
                             'visby', 'kiruna', 'midnight sun', 'northern lights', 'aurora borealis']
            },
            'Norway': {
                'cities': ['oslo', 'bergen', 'trondheim', 'stavanger', 'drammen', 'fredrikstad', 'kristiansand',
                          'sandnes', 'tromsø', 'tromso', 'sarpsborg', 'skien', 'ålesund', 'alesund', 'sandefjord',
                          'haugesund', 'tønsberg', 'tonsberg', 'moss', 'bodø', 'bodo', 'arendal', 'hamar',
                          'ytrebygda', 'larvik', 'halden', 'askøy', 'askoy', 'lillehammer'],
                'regions': ['oslo', 'akershus', 'østfold', 'ostfold', 'vestfold', 'buskerud', 'oppland', 'hedmark',
                           'telemark', 'aust agder', 'vest agder', 'rogaland', 'hordaland', 'sogn og fjordane',
                           'møre og romsdal', 'more og romsdal', 'sør trøndelag', 'sor trondelag', 'nord trøndelag',
                           'nordland', 'troms', 'finnmark'],
                'landmarks': ['preikestolen', 'geirangerfjord', 'nærøyfjord', 'naeroyfjord', 'lofoten', 'midnight sun',
                             'northern lights', 'aurora borealis', 'vigeland park', 'bryggen', 'nidaros cathedral',
                             'atlantic road', 'trolltunga', 'nordkapp', 'north cape', 'flåm railway']
            },
            'Denmark': {
                'cities': ['copenhagen', 'aarhus', 'odense', 'aalborg', 'esbjerg', 'randers', 'kolding', 'horsens',
                          'vejle', 'roskilde', 'herning', 'silkeborg', 'næstved', 'naestved', 'fredericia',
                          'viborg', 'køge', 'koge', 'holstebro', 'taastrup', 'slagelse', 'hillerød', 'hillerod',
                          'sønderborg', 'sonderborg', 'hjørring', 'hjorring', 'frederikshavn', 'nørresundby'],
                'regions': ['zealand', 'sjælland', 'sjaelland', 'jutland', 'jylland', 'funen', 'fyn', 'bornholm',
                           'copenhagen', 'frederiksberg', 'north jutland', 'central jutland', 'south jutland',
                           'region zealand', 'capital region'],
                'landmarks': ['little mermaid', 'tivoli gardens', 'kronborg castle', 'legoland', 'amalienborg',
                             'rosenborg castle', 'nyhavn', 'round tower', 'christiansborg', 'frederiksborg castle',
                             'møns klint', 'mons klint', 'skagen', 'råbjerg mile', 'rabjerg mile']
            }
        }

        for country, location_data in country_locations.items():
            for category, places in location_data.items():
                for place in places:
                    pattern = r'\b' + re.escape(place.lower()) + r'\b'
                    matches = re.findall(pattern, combined_text)
                    count = len(matches)

                    if count > 0:
                        confidence = min(90, count * 20)
                        detected_locations.append({
                            'location': place.title(),
                            'category': category.rstrip('s').title(),
                            'country': country,
                            'mentions': count,
                            'confidence': confidence
                        })

        detected_locations.sort(key=lambda x: (x['confidence'], x['mentions']), reverse=True)
        personal_info['specific_locations'] = detected_locations[:15]
        
        occupation_patterns = [
            r'work as ([a-z\s]+)', r'job as ([a-z\s]+)', r'employed as ([a-z\s]+)',
            r'career in ([a-z\s]+)', r'profession is ([a-z\s]+)', r'i am a ([a-z\s]+)',
            r'work at ([a-z\s]+)', r'my boss', r'my coworker', r'office job',
            r'remote work', r'work from home', r'unemployed', r'retired'
        ]
        
        for pattern in occupation_patterns:
            matches = re.findall(pattern, combined_text)
            personal_info['occupation_hints'].extend(matches)
        
        relationship_patterns = [
            r'my wife', r'my husband', r'my girlfriend', r'my boyfriend',
            r'my partner', r'married', r'single', r'divorced', r'dating',
            r'my ex', r'relationship', r'engaged'
        ]
        
        for pattern in relationship_patterns:
            if re.search(pattern, combined_text):
                personal_info['relationship_status'].append(pattern)
        
        education_patterns = [
            r'college', r'university', r'phd', r'masters', r'bachelors',
            r'student', r'school', r'graduate', r'degree', r'education',
            r'studying', r'professor', r'teacher'
        ]
        
        for pattern in education_patterns:
            if re.search(pattern, combined_text):
                personal_info['education_level'].append(pattern)
        
        financial_patterns = [
            r'salary', r'income', r'mortgage', r'rent', r'expensive',
            r'cheap', r'afford', r'budget', r'money', r'debt',
            r'investment', r'stocks', r'crypto', r'wealthy', r'poor'
        ]
        
        for pattern in financial_patterns:
            if re.search(pattern, combined_text):
                personal_info['financial_status'].append(pattern)
        
        family_patterns = [
            r'my mom', r'my dad', r'my mother', r'my father',
            r'my son', r'my daughter', r'my kids', r'my children',
            r'my brother', r'my sister', r'family', r'parents'
        ]
        
        for pattern in family_patterns:
            if re.search(pattern, combined_text):
                personal_info['family_indicators'].append(pattern)
        
        health_patterns = [
            r'doctor', r'hospital', r'sick', r'illness', r'medication',
            r'therapy', r'depression', r'anxiety', r'health', r'medical'
        ]
        
        for pattern in health_patterns:
            if re.search(pattern, combined_text):
                personal_info['health_mentions'].append(pattern)
        
        tech_patterns = [
            r'iphone', r'android', r'windows', r'mac', r'linux',
            r'programming', r'coding', r'developer', r'computer',
            r'laptop', r'desktop', r'gaming', r'pc'
        ]
        
        for pattern in tech_patterns:
            if re.search(pattern, combined_text):
                personal_info['technology_usage'].append(pattern)
        
        return personal_info
    
    def analyze_communication_style(self):
        if not self.timeline:
            return {}
        
        all_text = [entry['content'] for entry in self.timeline if entry['content']]
        if not all_text:
            return {}
        
        combined_text = ' '.join(all_text)
        
        words = combined_text.split()
        total_words = len(words)
        
        if total_words == 0:
            return {}
        
        sentences = re.split(r'[.!?]+', combined_text)
        avg_sentence_length = total_words / max(len(sentences), 1)
        
        complex_words = [word for word in words if len(word) > 6]
        complexity_ratio = len(complex_words) / total_words
        
        formal_indicators = ['therefore', 'however', 'furthermore', 'moreover', 'consequently']
        informal_indicators = ['lol', 'omg', 'tbh', 'imo', 'btw', 'gonna', 'wanna']
        
        formal_count = sum(1 for indicator in formal_indicators if indicator in combined_text.lower())
        informal_count = sum(1 for indicator in informal_indicators if indicator in combined_text.lower())
        
        capitalization_errors = len(re.findall(r'\bi\b', combined_text))
        punctuation_usage = len(re.findall(r'[.!?]', combined_text)) / max(len(sentences), 1)
        
        profanity_patterns = r'\b(damn|shit|fuck|ass|bitch|hell)\b'
        profanity_count = len(re.findall(profanity_patterns, combined_text.lower()))
        
        question_ratio = len(re.findall(r'\?', combined_text)) / max(len(sentences), 1)
        exclamation_ratio = len(re.findall(r'!', combined_text)) / max(len(sentences), 1)
        
        emojis = len(re.findall(r'[😀-🿿]|[🀀-🃿]|[🄀-🇿]', combined_text))
        emoticons = len(re.findall(r':\)|:\(|:D|;D|XD|:\||:P', combined_text))
        
        education_level = "Unknown"
        if complexity_ratio > 0.3 and formal_count > informal_count:
            education_level = "Higher education"
        elif complexity_ratio > 0.2:
            education_level = "Some college"
        elif informal_count > formal_count * 2:
            education_level = "High school or less"
        else:
            education_level = "Average"
        
        personality_traits = []
        if question_ratio > 0.2:
            personality_traits.append("Curious/Inquisitive")
        if exclamation_ratio > 0.1:
            personality_traits.append("Enthusiastic")
        if profanity_count > total_words * 0.02:
            personality_traits.append("Casual/Informal")
        if formal_count > 5:
            personality_traits.append("Professional")
        if emojis + emoticons > 10:
            personality_traits.append("Expressive")
        
        return {
            'avg_sentence_length': round(avg_sentence_length, 1),
            'vocabulary_complexity': round(complexity_ratio, 3),
            'formality_score': formal_count - informal_count,
            'estimated_education': education_level,
            'personality_traits': personality_traits,
            'punctuation_usage': round(punctuation_usage, 2),
            'profanity_usage': profanity_count,
            'question_frequency': round(question_ratio, 3),
            'exclamation_frequency': round(exclamation_ratio, 3),
            'emoji_usage': emojis + emoticons,
            'total_words_analyzed': total_words
        }
    
    def detect_social_media_references(self):
        all_text = []
        
        for post in self.posts_data:
            all_text.append(f"{post['title']} {post['text']}")
        
        for comment in self.comments_data:
            all_text.append(comment['body'])
        
        if not all_text:
            return {}
        
        combined_text = ' '.join(all_text).lower()
        
        social_platforms = {
            'twitter': [r'twitter', r'tweet', r'@\w+', r'twitter\.com'],
            'instagram': [r'instagram', r'insta', r'ig', r'instagram\.com'],
            'facebook': [r'facebook', r'fb', r'facebook\.com'],
            'linkedin': [r'linkedin', r'linkedin\.com'],
            'tiktok': [r'tiktok', r'tik tok', r'tiktok\.com'],
            'youtube': [r'youtube', r'yt', r'youtube\.com', r'youtu\.be'],
            'discord': [r'discord', r'discord\.gg'],
            'snapchat': [r'snapchat', r'snap'],
            'telegram': [r'telegram', r't\.me'],
            'twitch': [r'twitch', r'twitch\.tv'],
            'steam': [r'steam', r'steamcommunity'],
            'github': [r'github', r'github\.com'],
            'pinterest': [r'pinterest', r'pinterest\.com']
        }
        
        platform_mentions = {}
        username_patterns = []
        
        for platform, patterns in social_platforms.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, combined_text)
                count += len(matches)
                if matches and platform in ['twitter', 'instagram', 'github']:
                    username_patterns.extend(matches)
            
            if count > 0:
                platform_mentions[platform] = count
        
        email_patterns = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', combined_text)
        phone_patterns = re.findall(r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\) \d{3}-\d{4}\b', combined_text)
        
        return {
            'platform_mentions': platform_mentions,
            'potential_usernames': list(set(username_patterns))[:10],
            'email_mentions': len(email_patterns),
            'phone_mentions': len(phone_patterns),
            'total_social_references': sum(platform_mentions.values())
        }
    
    def analyze_technical_indicators(self):
        all_text = []
        
        for post in self.posts_data:
            all_text.append(f"{post['title']} {post['text']}")
        
        for comment in self.comments_data:
            all_text.append(comment['body'])
        
        if not all_text:
            return {}
        
        combined_text = ' '.join(all_text).lower()
        
        technical_categories = {
            'programming': ['python', 'javascript', 'java', 'cpp', 'html', 'css', 'sql', 'react', 'nodejs', 'github', 'stackoverflow', 'coding', 'programming', 'developer', 'software', 'algorithm', 'database', 'api', 'framework', 'library', 'debug', 'compile'],
            'cybersecurity': ['security', 'hacking', 'penetration', 'vulnerability', 'exploit', 'malware', 'firewall', 'encryption', 'vpn', 'tor', 'phishing', 'ddos', 'breach', 'forensics', 'incident', 'threat', 'cyber', 'infosec'],
            'networking': ['router', 'switch', 'tcp', 'udp', 'dns', 'dhcp', 'subnet', 'vlan', 'protocol', 'bandwidth', 'latency', 'packet', 'ethernet', 'wifi', 'bluetooth', 'network', 'infrastructure'],
            'gaming': ['steam', 'xbox', 'playstation', 'nintendo', 'twitch', 'discord', 'fps', 'mmo', 'rpg', 'gaming', 'esports', 'streamer', 'gpu', 'graphics card'],
            'cryptocurrency': ['bitcoin', 'ethereum', 'blockchain', 'crypto', 'mining', 'wallet', 'exchange', 'defi', 'nft', 'altcoin', 'hodl', 'trading'],
            'operating_systems': ['windows', 'linux', 'macos', 'ubuntu', 'debian', 'centos', 'android', 'ios', 'kernel', 'terminal', 'bash', 'powershell']
        }
        
        tech_scores = {}
        for category, keywords in technical_categories.items():
            score = 0
            matches = []
            for keyword in keywords:
                count = combined_text.count(keyword)
                if count > 0:
                    score += count
                    matches.append((keyword, count))
            
            if score > 0:
                tech_scores[category] = {
                    'score': score,
                    'matches': matches,
                    'expertise_level': 'Expert' if score > 20 else 'Advanced' if score > 10 else 'Intermediate' if score > 5 else 'Novice'
                }
        
        ip_patterns = re.findall(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', combined_text)
        domain_patterns = re.findall(r'\b[a-zA-Z0-9-]+\.[a-zA-Z]{2,}\b', combined_text)
        hash_patterns = re.findall(r'\b[a-fA-F0-9]{32,}\b', combined_text)
        
        return {
            'technical_expertise': tech_scores,
            'ip_addresses_mentioned': len(set(ip_patterns)),
            'domains_mentioned': len(set(domain_patterns)),
            'hash_values_mentioned': len(hash_patterns),
            'overall_tech_level': max(tech_scores.items(), key=lambda x: x[1]['score'])[0] if tech_scores else 'Non-technical'
        }
    
    def detect_threat_indicators(self):
        all_text = []
        
        for post in self.posts_data:
            all_text.append(f"{post['title']} {post['text']}")
        
        for comment in self.comments_data:
            all_text.append(comment['body'])
        
        if not all_text:
            return {}
        
        combined_text = ' '.join(all_text).lower()
        
        threat_keywords = {
            'malicious_tools': ['metasploit', 'nmap', 'wireshark', 'burp suite', 'kali linux', 'hydra', 'john the ripper', 'aircrack', 'sqlmap'],
            'illegal_activities': ['fraud', 'scam', 'phishing', 'identity theft', 'credit card', 'stolen', 'hacked account', 'illegal', 'darkweb', 'deep web'],
            'violence_indicators': ['bomb', 'weapon', 'kill', 'hurt', 'violence', 'attack', 'revenge', 'hate'],
            'extremism': ['radical', 'extremist', 'terrorist', 'jihad', 'supremacist', 'militia'],
            'drug_references': ['drugs', 'cocaine', 'heroin', 'methamphetamine', 'dealer', 'trafficking']
        }
        
        threat_score = 0
        detected_threats = {}
        
        for category, keywords in threat_keywords.items():
            matches = []
            for keyword in keywords:
                if keyword in combined_text:
                    count = combined_text.count(keyword)
                    matches.append((keyword, count))
                    threat_score += count * 2
            
            if matches:
                detected_threats[category] = matches
        
        suspicious_patterns = []
        
        if re.search(r'anonymous|proxy|vpn|tor browser', combined_text):
            suspicious_patterns.append('Privacy tools usage')
        
        if re.search(r'fake|false|identity|persona', combined_text):
            suspicious_patterns.append('Identity manipulation references')
        
        if re.search(r'burner|throwaway|temporary', combined_text):
            suspicious_patterns.append('Disposable account indicators')
        
        return {
            'threat_score': threat_score,
            'threat_categories': detected_threats,
            'suspicious_patterns': suspicious_patterns,
            'risk_level': 'High' if threat_score > 20 else 'Medium' if threat_score > 10 else 'Low'
        }
    
    def analyze_behavioral_anomalies(self):
        if not self.timeline:
            return {}
        
        df = pd.DataFrame(self.timeline)
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        
        anomalies = []
        
        post_times = df[df['type'] == 'post']['hour'].tolist()
        comment_times = df[df['type'] == 'comment']['hour'].tolist()
        
        night_posts = sum(1 for hour in post_times if 0 <= hour <= 5)
        total_posts = len(post_times)
        
        if total_posts > 0 and (night_posts / total_posts) > 0.4:
            anomalies.append('Excessive late-night activity')
        
        daily_activity = df['day_of_week'].value_counts()
        if len(daily_activity) > 0:
            weekend_activity = daily_activity.get(5, 0) + daily_activity.get(6, 0)
            weekday_activity = sum(daily_activity.get(i, 0) for i in range(5))
            
            if weekend_activity > 0 and weekday_activity > 0:
                weekend_ratio = weekend_activity / (weekend_activity + weekday_activity)
                if weekend_ratio > 0.7:
                    anomalies.append('Primarily weekend activity pattern')
        
        if len(self.posts_data) > 0 and len(self.comments_data) > 0:
            post_comment_ratio = len(self.posts_data) / len(self.comments_data)
            if post_comment_ratio > 2:
                anomalies.append('Unusually high post-to-comment ratio')
            elif post_comment_ratio < 0.1:
                anomalies.append('Unusually low post-to-comment ratio')
        
        recent_activity = [entry for entry in self.timeline if (datetime.now() - entry['datetime']).days <= 30]
        if len(recent_activity) == 0 and self.user_data.get('account_age_days', 0) < 365:
            anomalies.append('No recent activity on relatively new account')
        
        unique_subreddits = set([entry['subreddit'] for entry in self.timeline])
        if len(unique_subreddits) > 50:
            anomalies.append('Activity across unusually high number of subreddits')
        
        return {
            'anomalies_detected': anomalies,
            'anomaly_count': len(anomalies),
            'risk_assessment': 'High' if len(anomalies) >= 3 else 'Medium' if len(anomalies) >= 2 else 'Low'
        }
    
    def create_linguistic_fingerprint(self):
        all_text = []
        
        for post in self.posts_data:
            all_text.append(f"{post['title']} {post['text']}")
        
        for comment in self.comments_data:
            all_text.append(comment['body'])
        
        if not all_text:
            return {}
        
        combined_text = ' '.join(all_text)
        
        punctuation_usage = {
            'exclamation_marks': combined_text.count('!'),
            'question_marks': combined_text.count('?'),
            'ellipsis': combined_text.count('...'),
            'parentheses': combined_text.count('('),
            'quotation_marks': combined_text.count('"') + combined_text.count("'")
        }
        
        capitalization_patterns = {
            'all_caps_words': len(re.findall(r'\b[A-Z]{2,}\b', combined_text)),
            'lowercase_i': combined_text.count(' i '),
            'sentence_case_errors': len(re.findall(r'\. [a-z]', combined_text))
        }
        
        common_typos = {
            'teh': combined_text.lower().count('teh'),
            'recieve': combined_text.lower().count('recieve'),
            'seperate': combined_text.lower().count('seperate'),
            'definately': combined_text.lower().count('definately'),
            'alot': combined_text.lower().count('alot')
        }
        
        slang_usage = {
            'lol': combined_text.lower().count('lol'),
            'omg': combined_text.lower().count('omg'),
            'btw': combined_text.lower().count('btw'),
            'imo': combined_text.lower().count('imo'),
            'tbh': combined_text.lower().count('tbh')
        }
        
        return {
            'punctuation_patterns': punctuation_usage,
            'capitalization_patterns': capitalization_patterns,
            'common_typos': common_typos,
            'slang_frequency': slang_usage,
            'total_characters': len(combined_text),
            'unique_fingerprint_hash': hashlib.md5(str(sorted({**punctuation_usage, **capitalization_patterns}.items())).encode()).hexdigest()[:16]
        }
    
    def analyze_account_correlations(self):
        correlations = []
        
        if self.user_data.get('account_age_days', 0) < 30 and len(self.posts_data) + len(self.comments_data) > 100:
            correlations.append('New account with high activity (possible sockpuppet)')
        
        if self.user_data.get('link_karma', 0) == 0 and len(self.posts_data) == 0:
            correlations.append('Comment-only account (possible throwaway)')
        
        if self.user_data.get('comment_karma', 0) < 0:
            correlations.append('Negative comment karma (possible troll account)')
        
        subreddit_diversity = len(set([p['subreddit'] for p in self.posts_data + self.comments_data]))
        if subreddit_diversity == 1:
            correlations.append('Single-subreddit focus (possible specialized/bot account)')
        
        activity_pattern = self.analyze_timezone_patterns()
        if activity_pattern.get('sleep_pattern') == 'Night owl' and self.user_data.get('account_age_days', 0) < 90:
            correlations.append('New night owl account (possible foreign actor)')
        
        return {
            'correlation_indicators': correlations,
            'correlation_count': len(correlations),
            'account_legitimacy_score': max(0, 10 - len(correlations) * 2)
        }
    
    def assess_privacy_risks(self):
        risks = []
        risk_score = 0
        
        personal_info = self.extract_personal_information()
        social_refs = self.detect_social_media_references()
        
        if personal_info.get('age_indicators'):
            risks.append("Age information disclosed")
            risk_score += 2
        
        if personal_info.get('location_hints'):
            risks.append("Location information shared")
            risk_score += 3
        
        if personal_info.get('occupation_hints'):
            risks.append("Occupation/workplace details mentioned")
            risk_score += 2
        
        if personal_info.get('family_indicators'):
            risks.append("Family information disclosed")
            risk_score += 2
        
        if social_refs.get('email_mentions', 0) > 0:
            risks.append("Email addresses mentioned")
            risk_score += 4
        
        if social_refs.get('phone_mentions', 0) > 0:
            risks.append("Phone numbers mentioned")
            risk_score += 5
        
        if social_refs.get('total_social_references', 0) > 5:
            risks.append("Multiple social media platforms referenced")
            risk_score += 2
        
        if len(self.posts_data) + len(self.comments_data) > 1000:
            risks.append("High volume of public content")
            risk_score += 1
        
        account_age = self.user_data.get('account_age_days', 0)
        total_activity = len(self.posts_data) + len(self.comments_data)
        
        if account_age > 0 and (total_activity / account_age) > 5:
            risks.append("Very high posting frequency")
            risk_score += 1
        
        unique_subreddits = len(set([p['subreddit'] for p in self.posts_data + self.comments_data]))
        if unique_subreddits > 50:
            risks.append("Active across many communities")
            risk_score += 1
        
        risk_level = "Low"
        if risk_score >= 15:
            risk_level = "Critical"
        elif risk_score >= 10:
            risk_level = "High"
        elif risk_score >= 5:
            risk_level = "Medium"
        
        return {
            'risk_factors': risks,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'privacy_recommendations': self._generate_old_privacy_recommendations(risks)
        }
    
    def _generate_old_privacy_recommendations(self, risks):
        recommendations = []
        
        if "Age information disclosed" in risks:
            recommendations.append("Avoid mentioning specific age or birth year")
        
        if "Location information shared" in risks:
            recommendations.append("Use general geographic terms instead of specific locations")
        
        if "Email addresses mentioned" in risks:
            recommendations.append("Never share contact information in public posts")
        
        if "Phone numbers mentioned" in risks:
            recommendations.append("Remove any phone number references")
        
        if "Family information disclosed" in risks:
            recommendations.append("Limit sharing details about family members")
        
        return recommendations
    
    def analyze_activity_patterns(self):
        if not self.timeline:
            return {}
        
        df = pd.DataFrame(self.timeline)
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        
        subreddit_activity = df['subreddit'].value_counts().to_dict()
        hourly_activity = df['hour'].value_counts().sort_index().to_dict()
        daily_activity = df['day_of_week'].value_counts().sort_index().to_dict()
        
        df['date'] = df['datetime'].dt.date
        date_activity = df.groupby(['date', 'type']).size().unstack(fill_value=0)
        
        date_activity_dict = {}
        if not date_activity.empty:
            for date, row in date_activity.iterrows():
                date_str = date.strftime('%Y-%m-%d')
                date_activity_dict[date_str] = row.to_dict()
        
        return {
            'subreddit_activity': subreddit_activity,
            'hourly_activity': hourly_activity,
            'daily_activity': daily_activity,
            'date_activity': date_activity_dict,
            'most_active_subreddits': list(subreddit_activity.keys())[:10],
            'total_subreddits': len(subreddit_activity),
            'avg_posts_per_day': len(self.posts_data) / max(self.user_data['account_age_days'], 1),
            'avg_comments_per_day': len(self.comments_data) / max(self.user_data['account_age_days'], 1)
        }
    
    def analyze_content_sentiment(self):
        all_text = []
        
        for post in self.posts_data:
            text = f"{post['title']} {post['text']}"
            all_text.append(text)
        
        for comment in self.comments_data:
            all_text.append(comment['body'])
        
        if not all_text:
            return {}
        
        sentiments = []
        emotions = {'anger': 0, 'fear': 0, 'joy': 0, 'sadness': 0}
        
        for text in tqdm(all_text, desc="Analyzing sentiment"):
            if text and len(text.strip()) > 0:
                blob = TextBlob(text)
                sentiments.append({
                    'polarity': blob.sentiment.polarity,
                    'subjectivity': blob.sentiment.subjectivity
                })
                
                text_lower = text.lower()
                
                anger_words = ['angry', 'mad', 'furious', 'hate', 'rage', 'pissed']
                fear_words = ['afraid', 'scared', 'worried', 'anxious', 'terrified']
                joy_words = ['happy', 'excited', 'love', 'great', 'awesome', 'amazing']
                sadness_words = ['sad', 'depressed', 'crying', 'lonely', 'miserable']
                
                for word in anger_words:
                    emotions['anger'] += text_lower.count(word)
                for word in fear_words:
                    emotions['fear'] += text_lower.count(word)
                for word in joy_words:
                    emotions['joy'] += text_lower.count(word)
                for word in sadness_words:
                    emotions['sadness'] += text_lower.count(word)
        
        if not sentiments:
            return {}
        
        avg_polarity = np.mean([s['polarity'] for s in sentiments])
        avg_subjectivity = np.mean([s['subjectivity'] for s in sentiments])
        
        if avg_polarity > 0.1:
            sentiment_label = "Positive"
        elif avg_polarity < -0.1:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
        
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0] if any(emotions.values()) else "neutral"
        
        return {
            'avg_polarity': avg_polarity,
            'avg_subjectivity': avg_subjectivity,
            'sentiment_label': sentiment_label,
            'sentiment_distribution': sentiments,
            'emotional_indicators': emotions,
            'dominant_emotion': dominant_emotion
        }
    
    def extract_keywords_and_themes(self):
        all_text = []
        
        for post in self.posts_data:
            text = f"{post['title']} {post['text']}"
            all_text.append(text)
        
        for comment in self.comments_data:
            all_text.append(comment['body'])
        
        if not all_text:
            return {}
        
        cleaned_text = []
        for text in all_text:
            if text and len(text.strip()) > 10:
                clean = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
                clean = re.sub(r'[^a-zA-Z\s]', ' ', clean)
                clean = ' '.join(clean.split())
                if len(clean) > 10:
                    cleaned_text.append(clean.lower())
        
        if len(cleaned_text) < 2:
            return {}
        
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(cleaned_text)
            feature_names = vectorizer.get_feature_names_out()
            
            scores = tfidf_matrix.sum(axis=0).A1
            keyword_scores = list(zip(feature_names, scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return {
                'top_keywords': keyword_scores[:20],
                'total_documents': len(cleaned_text),
                'vocabulary_size': len(feature_names)
            }
        except Exception as e:
            print(f"{Fore.RED}Warning: Error in keyword extraction: {str(e)}{Style.RESET_ALL}")
            return {}
    
    def calculate_influence_metrics(self):
        if not self.posts_data and not self.comments_data:
            return {}
        
        total_post_karma = sum(post['score'] for post in self.posts_data)
        total_comment_karma = sum(comment['score'] for comment in self.comments_data)
        
        avg_post_karma = total_post_karma / max(len(self.posts_data), 1)
        avg_comment_karma = total_comment_karma / max(len(self.comments_data), 1)
        
        total_comments_received = sum(post['num_comments'] for post in self.posts_data)
        
        all_scores = [post['score'] for post in self.posts_data] + [comment['score'] for comment in self.comments_data]
        if all_scores:
            high_engagement_threshold = np.percentile(all_scores, 90)
            high_engagement_count = sum(1 for score in all_scores if score >= high_engagement_threshold)
        else:
            high_engagement_threshold = 0
            high_engagement_count = 0
        
        controversial_posts = [post for post in self.posts_data if post['upvote_ratio'] < 0.6]
        
        return {
            'total_post_karma': total_post_karma,
            'total_comment_karma': total_comment_karma,
            'avg_post_karma': avg_post_karma,
            'avg_comment_karma': avg_comment_karma,
            'total_comments_received': total_comments_received,
            'high_engagement_count': high_engagement_count,
            'high_engagement_threshold': high_engagement_threshold,
            'controversial_posts_count': len(controversial_posts),
            'karma_per_day': (total_post_karma + total_comment_karma) / max(self.user_data['account_age_days'], 1)
        }
    
    def analyze_network_connections(self):
        if not self.network_data:
            return {}
        
        top_connections = sorted(self.network_data.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_unique_mentions': len(self.network_data),
            'top_connections': top_connections[:20],
            'total_mentions': sum(self.network_data.values()),
            'avg_mentions_per_user': sum(self.network_data.values()) / max(len(self.network_data), 1)
        }
    
    def analyze_geographic_indicators(self):
        if not self.posts_data and not self.comments_data:
            return {}
        
        country_indicators = {
            'United States': {
                'keywords': ['usa', 'america', 'american', 'us', 'states', 'dollar', 'usd', 'fahrenheit',
                           'college', 'freshman', 'sophomore', 'junior', 'senior', 'mom', 'soccer',
                           'apartment', 'walmart', 'target', 'starbucks', 'mcdonalds', 'thanksgiving',
                           'fourth of july', '4th of july', 'super bowl', 'nfl', 'nba', 'mlb', 'baseball',
                           'costco', 'cvs', 'walgreens', 'home depot', 'best buy', 'fedex', 'ups'],
                'subreddits': ['usa', 'america', 'murica', 'askamerica', 'unitedstates', 'newyork',
                             'california', 'texas', 'florida', 'chicago', 'boston', 'seattle',
                             'losangeles', 'asknyc', 'nyc', 'sanfrancisco', 'washingtondc', 'philadelphia',
                             'atlanta', 'denver', 'phoenix', 'detroit', 'minneapolis', 'cleveland'],
                'spellings': ['color', 'honor', 'favor', 'center', 'theater', 'meter'],
                'phrases': ['high school', 'middle school', 'elementary school', 'gas station', 'zip code',
                          'social security', 'drivers license', 'state university']
            },
            'United Kingdom': {
                'keywords': ['uk', 'britain', 'british', 'england', 'scotland', 'wales', 'pound', 'gbp',
                           'celsius', 'university', 'uni', 'mum', 'football', 'flat', 'tesco', 'asda',
                           'marks spencer', 'nhs', 'quid', 'bin', 'lorry', 'lift', 'queue', 'postcode',
                           'bloody hell', 'blimey', 'brilliant', 'posh', 'pint', 'pub', 'chippy'],
                'subreddits': ['unitedkingdom', 'uk', 'britishproblems', 'casualuk', 'askuk', 'london',
                             'scotland', 'wales', 'england', 'manchester', 'birmingham', 'liverpool',
                             'glasgow', 'edinburgh', 'cardiff', 'belfast', 'leeds', 'sheffield',
                             'bristol', 'newcastle', 'nottingham', 'leicester'],
                'spellings': ['colour', 'honour', 'favour', 'centre', 'theatre', 'metre'],
                'phrases': ['secondary school', 'primary school', 'petrol station', 'car park',
                          'council estate', 'year 11', 'a levels', 'gcse']
            },
            'Canada': {
                'keywords': ['canada', 'canadian', 'eh', 'maple', 'hockey', 'tim hortons', 'cad', 'toonie',
                           'loonie', 'celsius', 'university', 'college', 'mum', 'mom', 'toque', 'poutine',
                           'mounties', 'hydro', 'washroom', 'chesterfield', 'double double', 'timmies'],
                'subreddits': ['canada', 'onguardforthee', 'askcanada', 'toronto', 'vancouver',
                             'montreal', 'calgary', 'edmonton', 'ottawa', 'winnipeg', 'quebec',
                             'britishcolumbia', 'alberta', 'ontario', 'saskatchewan', 'manitoba',
                             'novascotia', 'newbrunswick', 'newfoundland'],
                'spellings': ['colour', 'honour', 'favour', 'centre', 'theatre', 'metre'],
                'phrases': ['high school', 'elementary school', 'gas station', 'grade', 'postal code',
                          'health card', 'provincial']
            },
            'Australia': {
                'keywords': ['australia', 'aussie', 'mate', 'aud', 'celsius', 'university', 'uni', 'mum',
                           'footy', 'woolworths', 'coles', 'servo', 'bottle shop', 'arvo', 'brekkie',
                           'barbie', 'dunny', 'ute', 'thongs', 'bogan', 'fair dinkum', 'bloody oath'],
                'subreddits': ['australia', 'straya', 'melbourne', 'sydney', 'brisbane', 'perth',
                             'adelaide', 'canberra', 'darwin', 'hobart', 'tasmania', 'queensland',
                             'newsouthwales', 'victoria', 'southaustralia', 'westernaustralia',
                             'northernterritory', 'tasmania'],
                'spellings': ['colour', 'honour', 'favour', 'centre', 'theatre', 'metre'],
                'phrases': ['year 12', 'primary school', 'petrol station', 'postcode', 'medicare']
            },
            'Germany': {
                'keywords': ['germany', 'german', 'deutschland', 'euro', 'eur', 'celsius', 'university',
                           'uni', 'mama', 'papa', 'fussball', 'rewe', 'edeka', 'aldi', 'lidl', 'dm',
                           'autobahn', 'oktoberfest', 'bundesland', 'krankenkasse', 'gymnasium'],
                'subreddits': ['germany', 'deutschland', 'berlin', 'munich', 'hamburg', 'cologne',
                             'frankfurt', 'dusseldorf', 'stuttgart', 'dortmund', 'essen', 'leipzig',
                             'dresden', 'hannover', 'nuremberg', 'duisburg', 'bochum', 'wuppertal'],
                'spellings': [],
                'phrases': ['gymnasium', 'grundschule', 'tankstelle', 'postleitzahl', 'krankenkasse']
            },
            'France': {
                'keywords': ['france', 'french', 'euro', 'eur', 'celsius', 'université', 'maman', 'papa',
                           'football', 'carrefour', 'leclerc', 'fnac', 'sncf', 'prefecture', 'mairie',
                           'boulangerie', 'pharmacie', 'tabac'],
                'subreddits': ['france', 'french', 'paris', 'lyon', 'marseille', 'toulouse',
                             'nice', 'nantes', 'strasbourg', 'montpellier', 'bordeaux', 'lille',
                             'rennes', 'reims', 'saintetienne', 'toulon', 'angers', 'grenoble'],
                'spellings': [],
                'phrases': ['lycée', 'collège', 'école primaire', 'station essence', 'code postal',
                          'carte vitale', 'securité sociale']
            },
            'Netherlands': {
                'keywords': ['netherlands', 'dutch', 'holland', 'euro', 'eur', 'celsius', 'universiteit',
                           'mama', 'papa', 'voetbal', 'albert heijn', 'jumbo', 'hema', 'mediamarkt',
                           'fiets', 'gracht', 'gezellig', 'stroopwafel', 'bitterballen'],
                'subreddits': ['netherlands', 'thenetherlands', 'dutch', 'amsterdam', 'rotterdam',
                             'utrecht', 'eindhoven', 'tilburg', 'groningen', 'breda', 'nijmegen',
                             'enschede', 'haarlem', 'almere', 'zaanstad', 'haarlemmermeer'],
                'spellings': [],
                'phrases': ['middelbare school', 'basisschool', 'tankstation', 'postcode', 'bsn nummer']
            },
            'Sweden': {
                'keywords': ['sweden', 'swedish', 'sverige', 'krona', 'sek', 'celsius', 'universitet',
                           'mamma', 'pappa', 'fotboll', 'ica', 'coop', 'systembolaget', 'allemansratten',
                           'lagom', 'fika', 'midsommar', 'lucia', 'jantelagen'],
                'subreddits': ['sweden', 'sverige', 'stockholm', 'gothenburg', 'malmo',
                             'uppsala', 'vasteras', 'orebro', 'linkoping', 'helsingborg',
                             'jonkoping', 'norrkoping', 'lund', 'umea', 'gavle', 'boras'],
                'spellings': [],
                'phrases': ['gymnasium', 'grundskola', 'bensinstation', 'postnummer', 'personnummer']
            },
            'Norway': {
                'keywords': ['norway', 'norwegian', 'norge', 'kroner', 'nok', 'celsius', 'universitet',
                           'mamma', 'pappa', 'fotball', 'rema', 'kiwi', 'coop', 'vinmonopolet',
                           'allemannsretten', 'hygge', 'bunad', 'syttende mai', 'midnight sun'],
                'subreddits': ['norway', 'norge', 'oslo', 'bergen', 'trondheim', 'stavanger',
                             'drammen', 'fredrikstad', 'kristiansand', 'sandnes', 'tromsø', 'sarpsborg'],
                'spellings': [],
                'phrases': ['videregaende skole', 'barneskole', 'bensinstation', 'postnummer', 'fodselsnummer']
            },
            'Denmark': {
                'keywords': ['denmark', 'danish', 'danmark', 'kroner', 'dkk', 'celsius', 'universitet',
                           'mor', 'far', 'fodbold', 'netto', 'bilka', 'hygge', 'smorrebrod',
                           'janteloven', 'legoland'],
                'subreddits': ['denmark', 'danmark', 'copenhagen', 'aarhus', 'odense', 'aalborg',
                             'esbjerg', 'randers', 'kolding', 'horsens', 'vejle', 'roskilde'],
                'spellings': [],
                'phrases': ['gymnasium', 'folkeskole', 'tankstation', 'postnummer', 'cpr nummer']
            }
        }
        
        all_text = []
        user_subreddits = []
        
        for post in self.posts_data:
            all_text.append(f"{post['title']} {post['text']}".lower())
            user_subreddits.append(post['subreddit'].lower())
        
        for comment in self.comments_data:
            all_text.append(comment['body'].lower())
            user_subreddits.append(comment['subreddit'].lower())
        
        if not all_text:
            return {}
        
        combined_text = ' '.join(all_text)
        
        country_scores = {}
        
        for country, indicators in country_indicators.items():
            score = 0
            details = {
                'keyword_matches': [],
                'subreddit_matches': [],
                'spelling_matches': [],
                'phrase_matches': []
            }
            
            for keyword in indicators['keywords']:
                count = combined_text.count(keyword.lower())
                if count > 0:
                    score += count * 2
                    details['keyword_matches'].append((keyword, count))
            
            for subreddit in indicators['subreddits']:
                count = user_subreddits.count(subreddit.lower())
                if count > 0:
                    score += count * 5
                    details['subreddit_matches'].append((subreddit, count))
            
            for spelling in indicators['spellings']:
                count = combined_text.count(spelling.lower())
                if count > 0:
                    score += count * 3
                    details['spelling_matches'].append((spelling, count))
            
            for phrase in indicators['phrases']:
                count = combined_text.count(phrase.lower())
                if count > 0:
                    score += count * 3
                    details['phrase_matches'].append((phrase, count))
            
            if score > 0:
                country_scores[country] = {
                    'score': score,
                    'details': details
                }
        
        if not country_scores:
            return {
                'most_likely_country': 'Unknown',
                'confidence_percentage': 0,
                'country_probabilities': {},
                'analysis_note': 'Insufficient geographic indicators found'
            }
        
        total_score = sum(data['score'] for data in country_scores.values())
        country_probabilities = {}
        
        for country, data in country_scores.items():
            percentage = (data['score'] / total_score) * 100
            country_probabilities[country] = {
                'percentage': round(percentage, 1),
                'score': data['score'],
                'details': data['details']
            }
        
        sorted_countries = sorted(country_probabilities.items(), 
                                key=lambda x: x[1]['percentage'], reverse=True)
        
        most_likely = sorted_countries[0] if sorted_countries else ('Unknown', {'percentage': 0})
        
        return {
            'most_likely_country': most_likely[0],
            'confidence_percentage': most_likely[1]['percentage'],
            'country_probabilities': dict(sorted_countries),
            'analysis_note': f'Analysis based on {len(all_text)} text samples',
            'total_indicators_found': sum(len(data['details']['keyword_matches']) + 
                                        len(data['details']['subreddit_matches']) + 
                                        len(data['details']['spelling_matches']) + 
                                        len(data['details']['phrase_matches']) 
                                        for data in country_scores.values())
        }
    
    def generate_timeline_analysis(self):
        if not self.timeline:
            return {}
        
        df = pd.DataFrame(self.timeline)
        df['date'] = df['datetime'].dt.date
        
        daily_activity = df.groupby('date').size()
        
        mean_activity = daily_activity.mean()
        std_activity = daily_activity.std()
        spike_threshold = mean_activity + (2 * std_activity)
        
        activity_spikes = daily_activity[daily_activity > spike_threshold]
        
        df['week'] = df['datetime'].dt.isocalendar().week
        df['year'] = df['datetime'].dt.year
        weekly_activity = df.groupby(['year', 'week']).size().sort_values(ascending=False)
        
        spikes_dict = {}
        for date, count in activity_spikes.items():
            spikes_dict[date.strftime('%Y-%m-%d')] = int(count)
        
        weeks_dict = {}
        for (year, week), count in weekly_activity.head(10).items():
            weeks_dict[f"{year}-W{week:02d}"] = int(count)
        
        return {
            'total_days_active': len(daily_activity),
            'avg_daily_activity': float(mean_activity),
            'activity_spikes': spikes_dict,
            'most_active_weeks': weeks_dict,
            'first_activity': min(df['datetime']).strftime('%Y-%m-%d'),
            'last_activity': max(df['datetime']).strftime('%Y-%m-%d')
        }
    
    def advanced_osint_analysis(self):
        """Premium OSINT analysis with cross-reference validation"""
        if not self.posts_data and not self.comments_data:
            return {}

        all_text = []
        for post in self.posts_data:
            all_text.append(f"{post['title']} {post['text']}".lower())
        for comment in self.comments_data:
            all_text.append(comment['body'].lower())

        combined_text = ' '.join(all_text)

        digital_footprints = {
            'gaming_platforms': {
                'steam': r'\bsteam\b',
                'playstation': r'\b(?:ps4|ps5|playstation|psn)\b',
                'xbox': r'\b(?:xbox|xbl|gamertag)\b',
                'nintendo': r'\b(?:nintendo|switch|3ds)\b',
                'epic_games': r'\b(?:epic games|fortnite)\b',
                'discord': r'\b(?:discord|disc)\b'
            },
            'crypto_indicators': {
                'bitcoin': r'\b(?:bitcoin|btc)\b',
                'ethereum': r'\b(?:ethereum|eth)\b',
                'dogecoin': r'\b(?:dogecoin|doge)\b',
                'wallet_references': r'\b(?:wallet|seed phrase|private key|metamask)\b',
                'exchanges': r'\b(?:binance|coinbase|kraken|bitfinex)\b'
            },
            'device_indicators': {
                'mobile': r'\b(?:iphone|android|samsung|pixel|oneplus)\b',
                'computers': r'\b(?:macbook|imac|windows|linux|ubuntu|dell|hp|lenovo)\b',
                'browsers': r'\b(?:chrome|firefox|safari|edge)\b'
            },
            'software_usage': {
                'development': r'\b(?:github|python|javascript|java|c\+\+|react|node)\b',
                'design': r'\b(?:photoshop|illustrator|figma|sketch)\b',
                'productivity': r'\b(?:office|excel|word|powerpoint|google docs)\b'
            }
        }

        footprint_analysis = {}
        for category, patterns in digital_footprints.items():
            footprint_analysis[category] = {}
            for platform, pattern in patterns.items():
                matches = re.findall(pattern, combined_text)
                if matches:
                    footprint_analysis[category][platform] = len(matches)

        personal_identifiers = {
            'age_indicators': re.findall(r'\b(?:i am|age|born in|year old|turning) (\d{1,2})\b', combined_text),
            'education_specific': re.findall(r'\b(?:studying|graduated from|attended|university of|college of) ([a-zA-Z\s]+)\b', combined_text),
            'workplace_hints': re.findall(r'\b(?:work at|employed by|company called) ([a-zA-Z\s]+)\b', combined_text),
            'vehicle_ownership': re.findall(r'\b(?:drive a|own a|my car is|bought a) ([a-zA-Z0-9\s]+)\b', combined_text),
            'pet_ownership': re.findall(r'\b(?:my cat|my dog|my pet) ([a-zA-Z\s]+)\b', combined_text)
        }

        username_patterns = {
            'mentioned_usernames': re.findall(r'\b@([a-zA-Z0-9_]+)\b', combined_text),
            'gaming_tags': re.findall(r'\b(?:gamertag|psn|steam)\s*:?\s*([a-zA-Z0-9_]+)\b', combined_text),
            'social_handles': re.findall(r'\b(?:instagram|ig|twitter|tiktok)\s*:?\s*([a-zA-Z0-9_]+)\b', combined_text)
        }

        financial_analysis = {
            'income_indicators': len(re.findall(r'\b(?:salary|wage|earn|income|paycheck)\b', combined_text)),
            'spending_patterns': len(re.findall(r'\b(?:expensive|cheap|afford|budget|cost|price)\b', combined_text)),
            'investment_mentions': len(re.findall(r'\b(?:invest|stocks|portfolio|401k|retirement)\b', combined_text)),
            'debt_indicators': len(re.findall(r'\b(?:loan|debt|mortgage|credit card|owe)\b', combined_text))
        }

        relationship_network = {
            'family_members': re.findall(r'\bmy (?:mom|dad|mother|father|sister|brother|son|daughter|wife|husband)\b', combined_text),
            'friend_mentions': len(re.findall(r'\bmy (?:friend|buddy|mate)\b', combined_text)),
            'colleague_mentions': len(re.findall(r'\b(?:coworker|colleague|boss|manager)\b', combined_text)),
            'partner_references': len(re.findall(r'\b(?:girlfriend|boyfriend|partner|spouse)\b', combined_text))
        }

        health_lifestyle = {
            'health_conditions': re.findall(r'\b(?:suffer from|diagnosed with|have) ([a-zA-Z\s]+)\b', combined_text),
            'medications': re.findall(r'\b(?:taking|prescribed|medication) ([a-zA-Z\s]+)\b', combined_text),
            'fitness_activities': len(re.findall(r'\b(?:gym|workout|running|cycling|swimming|yoga)\b', combined_text)),
            'diet_preferences': re.findall(r'\b(vegan|vegetarian|keto|paleo|gluten.free)\b', combined_text)
        }

        travel_analysis = {
            'countries_visited': re.findall(r'\b(?:visited|traveled to|been to|trip to) ([A-Z][a-zA-Z\s]+)\b', combined_text),
            'travel_frequency': len(re.findall(r'\b(?:travel|trip|vacation|holiday|flight)\b', combined_text)),
            'transportation_modes': len(re.findall(r'\b(?:flight|train|bus|uber|taxi|drive)\b', combined_text))
        }

        return {
            'digital_footprints': footprint_analysis,
            'personal_identifiers': personal_identifiers,
            'username_analysis': username_patterns,
            'financial_profile': financial_analysis,
            'relationship_network': relationship_network,
            'health_lifestyle': health_lifestyle,
            'travel_patterns': travel_analysis,
            'risk_assessment': self._calculate_privacy_risk(combined_text),
            'data_correlation_score': self._calculate_correlation_confidence()
        }

    def _calculate_privacy_risk(self, text):
        """Calculate privacy risk based on disclosed information"""
        risk_factors = {
            'location_specific': len(re.findall(r'\b(?:live at|address|home|house number)\b', text)),
            'personal_details': len(re.findall(r'\b(?:full name|phone|email|birthday|ssn)\b', text)),
            'financial_details': len(re.findall(r'\b(?:credit card|bank account|salary)\b', text)),
            'family_info': len(re.findall(r'\b(?:my kid|my child|my daughter|my son)\b', text)),
            'workplace_details': len(re.findall(r'\b(?:work at|company|office|employer)\b', text))
        }

        total_risk = sum(risk_factors.values())
        risk_level = "Low" if total_risk < 3 else "Medium" if total_risk < 7 else "High"

        return {
            'total_risk_score': total_risk,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'recommendations': self._generate_privacy_recommendations(risk_factors)
        }

    def _generate_privacy_recommendations(self, risk_factors):
        """Generate specific privacy recommendations based on risk factors"""
        recommendations = []

        if risk_factors['location_specific'] > 0:
            recommendations.append("Avoid sharing specific addresses or exact locations")

        if risk_factors['personal_details'] > 0:
            recommendations.append("Never share personal identification information")

        if risk_factors['financial_details'] > 0:
            recommendations.append("Keep financial information completely private")

        if risk_factors['family_info'] > 2:
            recommendations.append("Limit sharing details about family members, especially children")

        if risk_factors['workplace_details'] > 1:
            recommendations.append("Be cautious about workplace and employer details")

        recommendations.extend([
            "Use VPN for browsing privacy",
            "Review and adjust social media privacy settings",
            "Consider using pseudonyms for online activities"
        ])

        return recommendations

    def _format_hour(self, hour_float):
        """Convert decimal hour to HH:MM format"""
        hours = int(hour_float)
        minutes = int((hour_float - hours) * 60)
        return f"{hours:02d}:{minutes:02d}"

    def _calculate_correlation_confidence(self):
        """Calculate confidence level of cross-referenced data"""
        geo_analysis = self.analyze_geographic_indicators()
        timezone_analysis = self.analyze_timezone_patterns()

        confidence_factors = 0
        max_factors = 5

        if geo_analysis.get('most_likely_country') and geo_analysis.get('confidence_percentage', 0) > 50:
            confidence_factors += 1

        if timezone_analysis.get('geographic_correlation') == geo_analysis.get('most_likely_country'):
            confidence_factors += 1

        if len(self.timeline) > 50:
            confidence_factors += 1

        if len(self.posts_data) + len(self.comments_data) > 20:
            confidence_factors += 1

        if self.posts_data and self.comments_data:
            confidence_factors += 1

        confidence_percentage = (confidence_factors / max_factors) * 100
        return round(confidence_percentage, 1)

    def create_visualizations(self, output_dir='visualizations'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"{Fore.CYAN}Creating visualizations...{Style.RESET_ALL}")
        
        activity_analysis = self.analyze_activity_patterns()
        
        if activity_analysis:
            plt.figure(figsize=(12, 6))
            subreddits = list(activity_analysis['subreddit_activity'].keys())[:15]
            counts = [activity_analysis['subreddit_activity'][sr] for sr in subreddits]
            plt.barh(subreddits, counts)
            plt.title('Top Subreddits by Activity')
            plt.xlabel('Number of Posts/Comments')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/subreddit_activity.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            plt.figure(figsize=(10, 6))
            hours = list(range(24))
            activity_counts = [activity_analysis['hourly_activity'].get(h, 0) for h in hours]
            plt.plot(hours, activity_counts, marker='o')
            plt.title('Activity by Hour of Day')
            plt.xlabel('Hour')
            plt.ylabel('Number of Posts/Comments')
            plt.xticks(range(0, 24, 2))
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/hourly_activity.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        sentiment_analysis = self.analyze_content_sentiment()
        if sentiment_analysis and sentiment_analysis.get('sentiment_distribution'):
            sentiments = sentiment_analysis['sentiment_distribution']
            polarities = [s['polarity'] for s in sentiments]
            subjectivities = [s['subjectivity'] for s in sentiments]
            
            plt.figure(figsize=(10, 6))
            plt.scatter(polarities, subjectivities, alpha=0.6)
            plt.xlabel('Polarity (Negative ← → Positive)')
            plt.ylabel('Subjectivity (Objective ← → Subjective)')
            plt.title('Sentiment Analysis Distribution')
            plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/sentiment_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        network_analysis = self.analyze_network_connections()
        if network_analysis and network_analysis.get('top_connections'):
            plt.figure(figsize=(12, 8))
            connections = network_analysis['top_connections'][:15]
            users = [conn[0] for conn in connections]
            counts = [conn[1] for conn in connections]
            plt.barh(users, counts)
            plt.title('Most Mentioned Users')
            plt.xlabel('Number of Mentions')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/network_connections.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        if self.timeline:
            df = pd.DataFrame(self.timeline)
            df['date'] = df['datetime'].dt.date
            daily_activity = df.groupby('date').size()
            
            plt.figure(figsize=(15, 6))
            plt.plot(daily_activity.index, daily_activity.values, linewidth=1)
            plt.title('Activity Timeline')
            plt.xlabel('Date')
            plt.ylabel('Daily Activity Count')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/activity_timeline.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        keyword_analysis = self.extract_keywords_and_themes()
        if keyword_analysis and keyword_analysis.get('top_keywords'):
            keywords = dict(keyword_analysis['top_keywords'][:50])
            if keywords:
                wordcloud = WordCloud(
                    width=800, height=400, 
                    background_color='white',
                    max_words=50
                ).generate_from_frequencies(keywords)
                
                plt.figure(figsize=(12, 6))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('Top Keywords and Themes')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/wordcloud.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"{Fore.GREEN}Visualizations saved to {output_dir}/{Style.RESET_ALL}")
    
    def generate_report(self, output_file='reddit_profile_report.json'):
        print(f"{Fore.CYAN}Generating comprehensive report...{Style.RESET_ALL}")
        
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'tool_version': '4.0.0-Premium',
                'analysis_type': 'Reddit User Profile Analysis - Premium OSINT Edition'
            },
            'user_info': self.user_data,
            'activity_patterns': self.analyze_activity_patterns(),
            'content_sentiment': self.analyze_content_sentiment(),
            'keywords_themes': self.extract_keywords_and_themes(),
            'influence_metrics': self.calculate_influence_metrics(),
            'network_connections': self.analyze_network_connections(),
            'geographic_analysis': self.analyze_geographic_indicators(),
            'timezone_analysis': self.analyze_timezone_patterns(),
            'personal_information': self.extract_personal_information(),
            'communication_style': self.analyze_communication_style(),
            'social_media_references': self.detect_social_media_references(),
            'privacy_assessment': self.assess_privacy_risks(),
            'technical_indicators': self.analyze_technical_indicators(),
            'threat_assessment': self.detect_threat_indicators(),
            'behavioral_anomalies': self.analyze_behavioral_anomalies(),
            'linguistic_fingerprint': self.create_linguistic_fingerprint(),
            'account_correlations': self.analyze_account_correlations(),
            'timeline_analysis': self.generate_timeline_analysis(),
            'premium_osint_analysis': self.advanced_osint_analysis(),
            'privacy_notes': {
                'data_source': 'Publicly available Reddit data only',
                'ethical_considerations': 'This analysis is based solely on public information and should be used for legitimate research purposes only',
                'limitations': 'Analysis quality depends on user activity level and public visibility of content',
                'premium_features': 'Enhanced with advanced OSINT correlation, cross-platform analysis, and privacy risk assessment'
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"{Fore.GREEN}Report saved to {output_file}{Style.RESET_ALL}")
        return report
    
    def generate_html_report(self, output_file='reddit_profile_report.html'):
        print(f"{Fore.CYAN}Generating HTML report...{Style.RESET_ALL}")
        
        user_info = self.user_data
        activity = self.analyze_activity_patterns()
        sentiment = self.analyze_content_sentiment()
        keywords = self.extract_keywords_and_themes()
        influence = self.calculate_influence_metrics()
        network = self.analyze_network_connections()
        geo = self.analyze_geographic_indicators()
        timezone_analysis = self.analyze_timezone_patterns()
        personal_info = self.extract_personal_information()
        comm_style = self.analyze_communication_style()
        social_refs = self.detect_social_media_references()
        privacy = self.assess_privacy_risks()
        tech_indicators = self.analyze_technical_indicators()
        threat_assessment = self.detect_threat_indicators()
        anomalies = self.analyze_behavioral_anomalies()
        fingerprint = self.create_linguistic_fingerprint()
        correlations = self.analyze_account_correlations()
        premium_osint = self.advanced_osint_analysis()
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reddit Profiler - {user_info.get('username', 'Unknown')}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #ff6b6b, #ee5a24);
            color: white;
            padding: 40px;
            text-align: center;
            position: relative;
        }}
        
        .header h1 {{
            font-size: 3rem;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            font-size: 1.2rem;
            opacity: 0.9;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            padding: 40px;
            background: #f8f9fa;
        }}
        
        .stat-card {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            border-left: 4px solid;
            transition: transform 0.3s ease;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        
        .stat-card.primary {{ border-left-color: #3498db; }}
        .stat-card.success {{ border-left-color: #2ecc71; }}
        .stat-card.warning {{ border-left-color: #f39c12; }}
        .stat-card.danger {{ border-left-color: #e74c3c; }}
        .stat-card.info {{ border-left-color: #9b59b6; }}
        
        .stat-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        
        .stat-icon {{
            font-size: 2rem;
            opacity: 0.7;
        }}
        
        .stat-title {{
            font-size: 0.9rem;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
        }}
        
        .stat-value {{
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        
        .stat-subtitle {{
            color: #777;
            font-size: 0.9rem;
        }}
        
        .section {{
            padding: 40px;
            border-bottom: 1px solid #eee;
        }}
        
        .section:last-child {{
            border-bottom: none;
        }}
        
        .section-title {{
            font-size: 2rem;
            margin-bottom: 30px;
            color: #2c3e50;
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        
        .section-title i {{
            color: #3498db;
        }}
        
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        
        .geo-section {{
            background: linear-gradient(135deg, #74b9ff, #0984e3);
            color: white;
            margin: 20px 0;
            padding: 30px;
            border-radius: 15px;
        }}
        
        .country-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        
        .country-item {{
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            backdrop-filter: blur(10px);
        }}
        
        .country-percentage {{
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .keywords-cloud {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 20px;
        }}
        
        .keyword-tag {{
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 8px 16px;
            border-radius: 25px;
            font-size: 0.9rem;
            font-weight: 500;
        }}
        
        .network-connections {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        
        .connection-item {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            border-left: 3px solid #3498db;
        }}
        
        .connection-user {{
            font-weight: bold;
            color: #2c3e50;
        }}
        
        .connection-count {{
            color: #7f8c8d;
            font-size: 0.9rem;
        }}
        
        .footer {{
            background: #2c3e50;
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .disclaimer {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }}
        
        .progress-bar {{
            background: #ecf0f1;
            height: 20px;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 10px;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #3498db, #2ecc71);
            transition: width 0.8s ease;
        }}
        
        .privacy-risk {{
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }}
        
        .risk-low {{ background: #d4edda; border-left: 4px solid #28a745; }}
        .risk-medium {{ background: #fff3cd; border-left: 4px solid #ffc107; }}
        .risk-high {{ background: #f8d7da; border-left: 4px solid #dc3545; }}
        .risk-critical {{ background: #f8d7da; border-left: 4px solid #721c24; }}
        
        .personal-info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .info-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 3px solid #17a2b8;
        }}
        
        .timezone-section {{
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            margin: 20px 0;
            padding: 30px;
            border-radius: 15px;
        }}
        
        @media (max-width: 768px) {{
            .stats-grid {{
                grid-template-columns: 1fr;
                padding: 20px;
            }}
            
            .section {{
                padding: 20px;
            }}
            
            .header h1 {{
                font-size: 2rem;
            }}
        }}
        
        .sentiment-badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
            text-transform: uppercase;
        }}
        
        .sentiment-positive {{
            background: #2ecc71;
            color: white;
        }}
        
        .sentiment-negative {{
            background: #e74c3c;
            color: white;
        }}
        
        .sentiment-neutral {{
            background: #95a5a6;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fab fa-reddit-alien"></i> Reddit Profiler</h1>
            <div class="subtitle">Advanced User Profile Analysis for <strong>u/{user_info.get('username', 'Unknown')}</strong></div>
            <div style="margin-top: 20px; opacity: 0.8;">
                Generated on {datetime.now().strftime('%B %d, %Y at %H:%M UTC')}
            </div>
        </div>
        
        <div class="section">
            <div class="disclaimer">
                <strong><i class="fas fa-exclamation-triangle"></i> Ethical Use Notice:</strong>
                This analysis is based solely on publicly available Reddit data and should be used for legitimate 
                research purposes only. Respect user privacy and comply with applicable laws.
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card primary">
                <div class="stat-header">
                    <div class="stat-title">Account Age</div>
                    <i class="fas fa-calendar-alt stat-icon"></i>
                </div>
                <div class="stat-value">{user_info.get('account_age_days', 0)}</div>
                <div class="stat-subtitle">days old</div>
            </div>
            
            <div class="stat-card success">
                <div class="stat-header">
                    <div class="stat-title">Total Karma</div>
                    <i class="fas fa-arrow-up stat-icon"></i>
                </div>
                <div class="stat-value">{user_info.get('total_karma', 0):,}</div>
                <div class="stat-subtitle">combined karma</div>
            </div>
            
            <div class="stat-card warning">
                <div class="stat-header">
                    <div class="stat-title">Posts</div>
                    <i class="fas fa-edit stat-icon"></i>
                </div>
                <div class="stat-value">{len(self.posts_data)}</div>
                <div class="stat-subtitle">total posts</div>
            </div>
            
            <div class="stat-card danger">
                <div class="stat-header">
                    <div class="stat-title">Comments</div>
                    <i class="fas fa-comment stat-icon"></i>
                </div>
                <div class="stat-value">{len(self.comments_data)}</div>
                <div class="stat-subtitle">total comments</div>
            </div>
            
            <div class="stat-card info">
                <div class="stat-header">
                    <div class="stat-title">Active Subreddits</div>
                    <i class="fas fa-users stat-icon"></i>
                </div>
                <div class="stat-value">{activity.get('total_subreddits', 0)}</div>
                <div class="stat-subtitle">communities</div>
            </div>
            
            <div class="stat-card primary">
                <div class="stat-header">
                    <div class="stat-title">Privacy Risk</div>
                    <i class="fas fa-shield-alt stat-icon"></i>
                </div>
                <div class="stat-value">{privacy.get('risk_level', 'Unknown')}</div>
                <div class="stat-subtitle">risk level</div>
            </div>
            
            <div class="stat-card danger">
                <div class="stat-header">
                    <div class="stat-title">Threat Level</div>
                    <i class="fas fa-exclamation-triangle stat-icon"></i>
                </div>
                <div class="stat-value">{threat_assessment.get('risk_level', 'Unknown')}</div>
                <div class="stat-subtitle">threat assessment</div>
            </div>
            
            <div class="stat-card warning">
                <div class="stat-header">
                    <div class="stat-title">Tech Level</div>
                    <i class="fas fa-code stat-icon"></i>
                </div>
                <div class="stat-value">{tech_indicators.get('overall_tech_level', 'Unknown').replace('_', ' ').title()}</div>
                <div class="stat-subtitle">expertise</div>
            </div>
        </div>
        """
        
        if geo and geo.get('most_likely_country') != 'Unknown':
            html_content += f"""
        <div class="section">
            <h2 class="section-title">
                <i class="fas fa-globe-americas"></i>
                Geographic Analysis
            </h2>
            <div class="geo-section">
                <h3 style="margin-bottom: 20px;">
                    <i class="fas fa-flag"></i>
                    Most Likely Location: <strong>{geo['most_likely_country']}</strong>
                </h3>
                <div style="font-size: 1.1rem; margin-bottom: 20px;">
                    Confidence: <strong>{geo['confidence_percentage']:.1f}%</strong> 
                    (Based on {geo['total_indicators_found']} indicators)
                </div>
                
                <div class="country-grid">
            """
            
            for country, data in list(geo['country_probabilities'].items())[:6]:
                html_content += f"""
                    <div class="country-item">
                        <div class="country-percentage">{data['percentage']:.1f}%</div>
                        <div>{country}</div>
                    </div>
                """
            
            html_content += """
                </div>
            </div>
        </div>
            """
        
        if timezone_analysis:
            html_content += f"""
        <div class="section">
            <h2 class="section-title">
                <i class="fas fa-clock"></i>
                Timezone & Behavioral Patterns
            </h2>
            <div class="timezone-section">
                <h3>Likely Timezone: <strong>{timezone_analysis.get('likely_timezone', 'Unknown')}</strong></h3>
                <div style="margin: 15px 0;">
                    Sleep Pattern: <strong>{timezone_analysis.get('sleep_pattern', 'Unknown')}</strong>
                </div>
                <div style="margin: 15px 0;">
                    Average Posting Hour: <strong>{self._format_hour(timezone_analysis.get('average_posting_hour', 0))}</strong>
                </div>
                <div style="font-size: 0.9rem; opacity: 0.8; margin-top: 20px;">
                    Analysis based on posting time patterns and activity distribution
                </div>
            </div>
        </div>
            """
        
        if personal_info:
            html_content += """
        <div class="section">
            <h2 class="section-title">
                <i class="fas fa-user-secret"></i>
                Personal Information Analysis
            </h2>
            """
            
            if personal_info.get('specific_locations'):
                html_content += f"""
            <div class="privacy-risk risk-high" style="margin-bottom: 30px;">
                <h3><i class="fas fa-map-marker-alt"></i> Specific Locations Detected - HIGH PRIVACY RISK</h3>
                <div style="margin: 15px 0;">
                    <strong>WARNING:</strong> Exact geographic locations have been identified in user content.
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin-top: 20px;">
                """
                
                for location_data in personal_info['specific_locations'][:5]:
                    confidence_color = '#e74c3c' if location_data['confidence'] > 60 else '#f39c12' if location_data['confidence'] > 30 else '#3498db'
                    html_content += f"""
                    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; border-left: 4px solid {confidence_color};">
                        <div style="font-weight: bold; font-size: 1.1rem;">{location_data['location']}</div>
                        <div style="color: #666; margin: 5px 0;">{location_data['category']} in {location_data['country']}</div>
                        <div style="font-size: 0.9rem;">
                            <span style="color: {confidence_color};">Confidence: {location_data['confidence']}%</span> | 
                            {location_data['mentions']} mentions
                        </div>
                    </div>
                    """
                
                html_content += """
                </div>
                <div style="margin-top: 20px; padding: 15px; background: rgba(231,76,60,0.1); border-radius: 8px;">
                    <strong>Privacy Recommendations:</strong>
                    <ul style="margin: 10px 0; padding-left: 20px;">
                        <li>Avoid mentioning specific cities, states, or regions in posts</li>
                        <li>Use general geographic terms instead of exact locations</li>
                        <li>Be aware that location data can be used for doxxing or stalking</li>
                        <li>Consider using VPN and avoiding location-specific discussions</li>
                    </ul>
                </div>
            </div>
                """
            
            html_content += "<div class='personal-info-grid'>"
            
            if personal_info.get('age_indicators'):
                ages = personal_info['age_indicators']
                avg_age = sum(ages) / len(ages) if ages else 0
                html_content += f"""
                <div class="info-card">
                    <h4><i class="fas fa-birthday-cake"></i> Age Indicators</h4>
                    <p>Estimated Age: <strong>{avg_age:.0f} years</strong></p>
                    <small>Based on {len(ages)} references</small>
                </div>
                """
            
            if personal_info.get('location_hints') and not personal_info.get('specific_locations'):
                locations = list(set(personal_info['location_hints']))[:5]
                html_content += f"""
                <div class="info-card">
                    <h4><i class="fas fa-map-marker-alt"></i> Location Hints</h4>
                    <p>{', '.join(locations)}</p>
                    <small>{len(personal_info['location_hints'])} location references found</small>
                </div>
                """
            
            if personal_info.get('occupation_hints'):
                occupations = list(set(personal_info['occupation_hints']))[:3]
                html_content += f"""
                <div class="info-card">
                    <h4><i class="fas fa-briefcase"></i> Occupation Hints</h4>
                    <p>{', '.join(occupations)}</p>
                    <small>{len(personal_info['occupation_hints'])} work-related references</small>
                </div>
                """
            
            if personal_info.get('family_indicators'):
                html_content += f"""
                <div class="info-card">
                    <h4><i class="fas fa-users"></i> Family Information</h4>
                    <p>{len(personal_info['family_indicators'])} family references detected</p>
                    <small>Family details shared publicly</small>
                </div>
                """
            
            html_content += "</div></div>"
        
        if comm_style:
            html_content += f"""
        <div class="section">
            <h2 class="section-title">
                <i class="fas fa-comments"></i>
                Communication Style Analysis
            </h2>
            <div class="chart-container">
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">
                    <div>
                        <h4>Education Level</h4>
                        <p style="font-size: 1.2rem; color: #3498db;"><strong>{comm_style.get('estimated_education', 'Unknown')}</strong></p>
                    </div>
                    <div>
                        <h4>Average Sentence Length</h4>
                        <p style="font-size: 1.2rem; color: #3498db;"><strong>{comm_style.get('avg_sentence_length', 0)} words</strong></p>
                    </div>
                    <div>
                        <h4>Vocabulary Complexity</h4>
                        <p style="font-size: 1.2rem; color: #3498db;"><strong>{comm_style.get('vocabulary_complexity', 0):.1%}</strong></p>
                    </div>
                    <div>
                        <h4>Personality Traits</h4>
                        <p style="font-size: 1rem; color: #666;">{', '.join(comm_style.get('personality_traits', []))}</p>
                    </div>
                </div>
            </div>
        </div>
            """
        
        if social_refs and social_refs.get('platform_mentions'):
            html_content += """
        <div class="section">
            <h2 class="section-title">
                <i class="fas fa-share-alt"></i>
                Social Media Cross-References
            </h2>
            <div class="network-connections">
            """
            
            for platform, count in social_refs['platform_mentions'].items():
                html_content += f"""
                <div class="connection-item">
                    <div class="connection-user">{platform.title()}</div>
                    <div class="connection-count">{count} mentions</div>
                </div>
                """
            
            html_content += "</div></div>"
        
        if privacy:
            risk_class = f"risk-{privacy.get('risk_level', 'low').lower()}"
            html_content += f"""
        <div class="section">
            <h2 class="section-title">
                <i class="fas fa-user-shield"></i>
                Privacy Risk Assessment
            </h2>
            <div class="privacy-risk {risk_class}">
                <h3>Risk Level: {privacy.get('risk_level', 'Unknown')} (Score: {privacy.get('risk_score', 0)})</h3>
                <div style="margin: 15px 0;">
                    <strong>Risk Factors Identified:</strong>
                    <ul style="margin: 10px 0; padding-left: 20px;">
            """
            
            for risk in privacy.get('risk_factors', []):
                html_content += f"<li>{risk}</li>"
            
            html_content += """
                    </ul>
                </div>
            </div>
        </div>
            """
        
        if sentiment:
            sentiment_class = sentiment.get('sentiment_label', 'neutral').lower()
            html_content += f"""
        <div class="section">
            <h2 class="section-title">
                <i class="fas fa-brain"></i>
                Content Analysis
            </h2>
            <div class="chart-container">
                <h3>Overall Sentiment: <span class="sentiment-badge sentiment-{sentiment_class}">{sentiment.get('sentiment_label', 'Unknown')}</span></h3>
                <div style="margin-top: 20px;">
                    <div>Polarity Score: <strong>{sentiment.get('avg_polarity', 0):.3f}</strong></div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {(sentiment.get('avg_polarity', 0) + 1) * 50}%"></div>
                    </div>
                </div>
            """
            
            if sentiment.get('emotional_indicators'):
                emotions = sentiment['emotional_indicators']
                html_content += """
                <div style="margin-top: 25px;">
                    <h4>Emotional Indicators</h4>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-top: 15px;">
                """
                
                for emotion, count in emotions.items():
                    if count > 0:
                        html_content += f"""
                        <div style="text-align: center; padding: 10px; background: #f8f9fa; border-radius: 8px;">
                            <div style="font-weight: bold; text-transform: capitalize;">{emotion}</div>
                            <div style="color: #666; font-size: 1.2rem;">{count}</div>
                        </div>
                        """
                
                html_content += "</div></div>"
            
            html_content += "</div></div>"
        
        if keywords and keywords.get('top_keywords'):
            html_content += """
        <div class="section">
            <h2 class="section-title">
                <i class="fas fa-tags"></i>
                Keywords & Themes
            </h2>
            <div class="keywords-cloud">
            """
            
            for keyword, score in keywords['top_keywords'][:20]:
                size = max(0.8, min(2.0, score * 10))
                html_content += f"""
                <span class="keyword-tag" style="font-size: {size}rem;">{keyword}</span>
                """
            
            html_content += "</div></div>"
        
        if activity and activity.get('most_active_subreddits'):
            html_content += """
        <div class="section">
            <h2 class="section-title">
                <i class="fas fa-chart-bar"></i>
                Activity Patterns
            </h2>
            <div class="chart-container">
                <h3>Most Active Subreddits</h3>
                <div style="margin-top: 20px;">
            """
            
            for subreddit in activity['most_active_subreddits'][:10]:
                count = activity['subreddit_activity'].get(subreddit, 0)
                max_count = max(activity['subreddit_activity'].values()) if activity['subreddit_activity'] else 1
                percentage = (count / max_count) * 100
                
                html_content += f"""
                    <div style="margin-bottom: 15px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <strong>r/{subreddit}</strong>
                            <span>{count} posts/comments</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {percentage}%"></div>
                        </div>
                    </div>
                """
            
            html_content += "</div></div></div>"
        
        if network and network.get('top_connections'):
            html_content += """
        <div class="section">
            <h2 class="section-title">
                <i class="fas fa-project-diagram"></i>
                Network Connections
            </h2>
            <div class="network-connections">
            """
            
            for user, count in network['top_connections'][:12]:
                html_content += f"""
                <div class="connection-item">
                    <div class="connection-user">u/{user}</div>
                    <div class="connection-count">{count} mentions</div>
                </div>
                """
            
            html_content += "</div></div>"
        
        if influence:
            html_content += f"""
        <div class="section">
            <h2 class="section-title">
                <i class="fas fa-trophy"></i>
                Influence Metrics
            </h2>
            <div class="stats-grid">
                <div class="stat-card success">
                    <div class="stat-header">
                        <div class="stat-title">Avg Post Karma</div>
                        <i class="fas fa-thumbs-up stat-icon"></i>
                    </div>
                    <div class="stat-value">{influence.get('avg_post_karma', 0):.1f}</div>
                    <div class="stat-subtitle">per post</div>
                </div>
                
                <div class="stat-card info">
                    <div class="stat-header">
                        <div class="stat-title">Avg Comment Karma</div>
                        <i class="fas fa-comment-alt stat-icon"></i>
                    </div>
                    <div class="stat-value">{influence.get('avg_comment_karma', 0):.1f}</div>
                    <div class="stat-subtitle">per comment</div>
                </div>
                
                <div class="stat-card warning">
                    <div class="stat-header">
                        <div class="stat-title">High Engagement</div>
                        <i class="fas fa-fire stat-icon"></i>
                    </div>
                    <div class="stat-value">{influence.get('high_engagement_count', 0)}</div>
                    <div class="stat-subtitle">viral posts/comments</div>
                </div>
                
                <div class="stat-card danger">
                    <div class="stat-header">
                        <div class="stat-title">Controversial</div>
                        <i class="fas fa-exclamation stat-icon"></i>
                    </div>
                    <div class="stat-value">{influence.get('controversial_posts_count', 0)}</div>
                    <div class="stat-subtitle">controversial posts</div>
                </div>
            </div>
        </div>
            """
        
        html_content += f"""
        <div class="footer">
            <div>
                <h3><i class="fas fa-shield-alt"></i> Reddit Profiler</h3>
                <p>Advanced OSINT Analysis Tool</p>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
                <p style="margin-top: 15px; font-size: 0.9rem; opacity: 0.8;">
                    This analysis is based on publicly available Reddit data and should be used responsibly 
                    for legitimate research purposes only.
                </p>
            </div>
        </div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            const progressBars = document.querySelectorAll('.progress-fill');
            progressBars.forEach(bar => {{
                const width = bar.style.width;
                bar.style.width = '0%';
                setTimeout(() => {{
                    bar.style.width = width;
                }}, 500);
            }});
            
            const cards = document.querySelectorAll('.stat-card, .connection-item');
            cards.forEach(card => {{
                card.addEventListener('mouseenter', function() {{
                    this.style.transform = 'translateY(-5px) scale(1.02)';
                }});
                card.addEventListener('mouseleave', function() {{
                    this.style.transform = 'translateY(0) scale(1)';
                }});
            }});
        }});
    </script>
</body>
</html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"{Fore.GREEN}HTML report saved to {output_file}{Style.RESET_ALL}")
        return output_file
    
    def print_summary(self):
        if not self.user_data:
            print(f"{Fore.RED}No data available for summary{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"REDDIT PROFILER - ADVANCED ANALYSIS SUMMARY")
        print(f"{'='*60}{Style.RESET_ALL}")
        
        print(f"\n{Fore.YELLOW}Basic Information:{Style.RESET_ALL}")
        basic_info = [
            ['Username', self.user_data['username']],
            ['Account Created', self.user_data['created_utc'].strftime('%Y-%m-%d')],
            ['Account Age (days)', self.user_data['account_age_days']],
            ['Total Karma', f"{self.user_data['total_karma']:,}"],
            ['Link Karma', f"{self.user_data['link_karma']:,}"],
            ['Comment Karma', f"{self.user_data['comment_karma']:,}"]
        ]
        print(tabulate(basic_info, tablefmt='grid'))
        
        activity = self.analyze_activity_patterns()
        if activity:
            print(f"\n{Fore.YELLOW}Activity Statistics:{Style.RESET_ALL}")
            activity_info = [
                ['Total Posts', len(self.posts_data)],
                ['Total Comments', len(self.comments_data)],
                ['Active Subreddits', activity['total_subreddits']],
                ['Avg Posts/Day', f"{activity['avg_posts_per_day']:.2f}"],
                ['Avg Comments/Day', f"{activity['avg_comments_per_day']:.2f}"],
                ['Most Active Subreddit', activity['most_active_subreddits'][0] if activity['most_active_subreddits'] else 'N/A']
            ]
            print(tabulate(activity_info, tablefmt='grid'))
        
        timezone_analysis = self.analyze_timezone_patterns()
        if timezone_analysis:
            print(f"\n{Fore.YELLOW}Timezone & Behavioral Patterns:{Style.RESET_ALL}")
            timezone_info = [
                ['Likely Timezone', timezone_analysis.get('likely_timezone', 'Unknown')],
                ['Sleep Pattern', timezone_analysis.get('sleep_pattern', 'Unknown')],
                ['Avg Posting Hour', self._format_hour(timezone_analysis.get('average_posting_hour', 0))]
            ]
            print(tabulate(timezone_info, tablefmt='grid'))
        
        sentiment = self.analyze_content_sentiment()
        if sentiment:
            print(f"\n{Fore.YELLOW}Content Analysis:{Style.RESET_ALL}")
            sentiment_info = [
                ['Overall Sentiment', sentiment['sentiment_label']],
                ['Polarity Score', f"{sentiment['avg_polarity']:.3f}"],
                ['Dominant Emotion', sentiment.get('dominant_emotion', 'Unknown')]
            ]
            print(tabulate(sentiment_info, tablefmt='grid'))
        
        comm_style = self.analyze_communication_style()
        if comm_style:
            print(f"\n{Fore.YELLOW}Communication Style:{Style.RESET_ALL}")
            comm_info = [
                ['Estimated Education', comm_style.get('estimated_education', 'Unknown')],
                ['Avg Sentence Length', f"{comm_style.get('avg_sentence_length', 0)} words"],
                ['Vocabulary Complexity', f"{comm_style.get('vocabulary_complexity', 0):.1%}"],
                ['Personality Traits', ', '.join(comm_style.get('personality_traits', []))]
            ]
            print(tabulate(comm_info, tablefmt='grid'))
        
        geo = self.analyze_geographic_indicators()
        if geo and geo.get('most_likely_country') != 'Unknown':
            print(f"\n{Fore.YELLOW}Geographic Analysis:{Style.RESET_ALL}")
            geo_info = [
                ['Most Likely Country', geo['most_likely_country']],
                ['Confidence', f"{geo['confidence_percentage']:.1f}%"],
                ['Total Indicators Found', geo['total_indicators_found']]
            ]
            print(tabulate(geo_info, tablefmt='grid'))
        
        privacy = self.assess_privacy_risks()
        if privacy:
            print(f"\n{Fore.YELLOW}Privacy Risk Assessment:{Style.RESET_ALL}")
            privacy_info = [
                ['Risk Level', privacy.get('risk_level', 'Unknown')],
                ['Risk Score', privacy.get('risk_score', 0)],
                ['Risk Factors', len(privacy.get('risk_factors', []))]
            ]
            print(tabulate(privacy_info, tablefmt='grid'))
        
        tech_analysis = self.analyze_technical_indicators()
        if tech_analysis and tech_analysis.get('technical_expertise'):
            print(f"\n{Fore.YELLOW}Technical Expertise:{Style.RESET_ALL}")
            tech_info = [
                ['Overall Tech Level', tech_analysis.get('overall_tech_level', 'Unknown').replace('_', ' ').title()],
                ['Categories Found', len(tech_analysis['technical_expertise'])],
                ['IP Addresses Mentioned', tech_analysis.get('ip_addresses_mentioned', 0)],
                ['Domains Referenced', tech_analysis.get('domains_mentioned', 0)]
            ]
            print(tabulate(tech_info, tablefmt='grid'))
        
        threat_analysis = self.detect_threat_indicators()
        if threat_analysis and threat_analysis.get('threat_score', 0) > 0:
            print(f"\n{Fore.YELLOW}Threat Assessment:{Style.RESET_ALL}")
            threat_info = [
                ['Threat Level', threat_analysis.get('risk_level', 'Unknown')],
                ['Threat Score', threat_analysis.get('threat_score', 0)],
                ['Threat Categories', len(threat_analysis.get('threat_categories', {}))],
                ['Suspicious Patterns', len(threat_analysis.get('suspicious_patterns', []))]
            ]
            print(tabulate(threat_info, tablefmt='grid'))
        
        anomaly_analysis = self.analyze_behavioral_anomalies()
        if anomaly_analysis and anomaly_analysis.get('anomalies_detected'):
            print(f"\n{Fore.YELLOW}Behavioral Anomalies:{Style.RESET_ALL}")
            anomaly_info = [
                ['Anomaly Risk', anomaly_analysis.get('risk_assessment', 'Unknown')],
                ['Anomalies Detected', len(anomaly_analysis.get('anomalies_detected', []))],
                ['Top Anomaly', anomaly_analysis['anomalies_detected'][0] if anomaly_analysis['anomalies_detected'] else 'None']
            ]
            print(tabulate(anomaly_info, tablefmt='grid'))
        
        correlation_analysis = self.analyze_account_correlations()
        if correlation_analysis and correlation_analysis.get('correlation_indicators'):
            print(f"\n{Fore.YELLOW}Account Analysis:{Style.RESET_ALL}")
            correlation_info = [
                ['Legitimacy Score', f"{correlation_analysis.get('account_legitimacy_score', 0)}/10"],
                ['Correlation Indicators', len(correlation_analysis.get('correlation_indicators', []))],
                ['Account Type Assessment', 'Suspicious' if correlation_analysis.get('account_legitimacy_score', 0) < 5 else 'Normal']
            ]
            print(tabulate(correlation_info, tablefmt='grid'))
        
        print(f"\n{Fore.GREEN}Advanced analysis complete! Check generated files for detailed results.{Style.RESET_ALL}\n")


@click.command()
@click.argument('username')
@click.option('--client-id', help='Reddit API client ID')
@click.option('--client-secret', help='Reddit API client secret')
@click.option('--user-agent', default='RedditProfiler:4.0 (by /u/researcher)', help='User agent for API requests')
@click.option('--limit', default=1000, help='Maximum number of posts/comments to collect')
@click.option('--output-dir', default='profile_results', help='Output directory for results')
@click.option('--no-viz', is_flag=True, help='Skip visualization generation')
@click.option('--html-report', is_flag=True, help='Generate modern HTML report')
@click.option('--use-existing', is_flag=True, help='Use existing data (for testing/re-analysis)')
def main(username, client_id, client_secret, user_agent, limit, output_dir, no_viz, html_report, use_existing):
    
    print(f"{Fore.CYAN}{'='*60}")
    print("REDDIT PROFILER")
    print("Advanced User Profile Analysis & OSINT")
    print(f"{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}ETHICAL USE NOTICE:")
    print("This tool analyzes only publicly available data.")
    print("Use responsibly and in compliance with applicable laws.")
    print(f"Respect user privacy and Reddit's Terms of Service.{Style.RESET_ALL}\n")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        if use_existing:
            raw_file = os.path.join(output_dir, f'{username}_raw_data.json')
            if not os.path.exists(raw_file):
                print(f"{Fore.RED}Error: No existing data found for {username} in {output_dir}{Style.RESET_ALL}")
                sys.exit(1)

            print(f"{Fore.YELLOW}Loading existing data for re-analysis...{Style.RESET_ALL}")
            with open(raw_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            profiler = RedditProfiler(None, None, None)

            profiler.posts_data = raw_data.get('posts', [])
            profiler.comments_data = raw_data.get('comments', [])
            profiler.network_data = Counter(raw_data.get('network', {}))

            profiler.timeline = []

            for post in profiler.posts_data:
                profiler.timeline.append({
                    'datetime': pd.to_datetime(post['created_utc']),
                    'type': 'post',
                    'subreddit': post['subreddit'],
                    'score': post['score'],
                    'content': f"{post.get('title', '')} {post.get('text', '')}"
                })

            for comment in profiler.comments_data:
                profiler.timeline.append({
                    'datetime': pd.to_datetime(comment['created_utc']),
                    'type': 'comment',
                    'subreddit': comment['subreddit'],
                    'score': comment['score'],
                    'content': comment.get('body', '')
                })

            profiler.timeline.sort(key=lambda x: x['datetime'])

            if profiler.posts_data or profiler.comments_data:
                first_item = profiler.posts_data[0] if profiler.posts_data else profiler.comments_data[0]
                profiler.user_data = {
                    'username': username,
                    'id': first_item.get('author_fullname', 'unknown'),
                    'created_utc': 'unknown',
                    'link_karma': 0,
                    'comment_karma': 0,
                    'total_karma': 0,
                    'account_age_days': 0,
                    'has_verified_email': False,
                    'is_gold': False,
                    'is_mod': False
                }
        else:
            if not client_id or not client_secret:
                client_id = click.prompt('Reddit Client ID')
                client_secret = click.prompt('Reddit Client Secret', hide_input=True)

            profiler = RedditProfiler(client_id, client_secret, user_agent)
            profiler.collect_user_data(username, limit)

        if not no_viz:
            viz_dir = os.path.join(output_dir, 'visualizations')
            profiler.create_visualizations(viz_dir)

        report_file = os.path.join(output_dir, f'{username}_profile_report.json')
        profiler.generate_report(report_file)

        html_file = os.path.join(output_dir, f'{username}_profile_report.html')
        profiler.generate_html_report(html_file)

        profiler.print_summary()

        if not use_existing:
            raw_data = {
                'posts': profiler.posts_data,
                'comments': profiler.comments_data,
                'timeline': profiler.timeline,
                'network': dict(profiler.network_data)
            }

            raw_file = os.path.join(output_dir, f'{username}_raw_data.json')
            with open(raw_file, 'w', encoding='utf-8') as f:
                json.dump(raw_data, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"{Fore.GREEN}All results saved to: {output_dir}/{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Open the HTML report: {output_dir}/{username}_profile_report.html{Style.RESET_ALL}")
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Analysis interrupted by user{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        sys.exit(1)


if __name__ == '__main__':
    main()