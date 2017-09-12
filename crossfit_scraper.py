from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests, re, json, time, random, sys, os, itertools
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
from memoize import persistent_memoize


#@persistent_memoize('requests_get')
def requests_get(url):
	'''
	Keep resending requests until get OK
	'''
	r = requests.get(url)
	while r.status_code != 200:
		print("HTTP", r.status_code,":", url)
		if r.status_code == 404: return None
		time.sleep(10)
		r = requests.get(url)
	return r

@persistent_memoize('scrape_leaderboard_data')
def scrape_leaderboard_data(competition='open', year=2017, division='men', 
	sort='overall', fittest_in='region', region='worldwide'):
	'''
	Extract leaderboard data into pandas dataframe.
	'''
	print("Building leaderboard")

	def get_pages():
		url = get_page_url(1)
		leaderboard_page = requests_get(url)
		data = leaderboard_page.json()
		return int(data['totalpages'])

	def process_page(leaderboard_page):
		data = leaderboard_page.json()
		cols = ['overallrank','userid','overallscore','scores']
		df = pd.DataFrame(columns=cols)
		if competition == 'open':
			for athlete in data['athletes']:
				scores = []
				for workout in athlete['scores']:
					scores += [(workout['workoutrank'], workout['scoredisplay'])]
				df = df.append(
					pd.Series(
						[athlete['overallrank'], athlete['userid'], athlete['overallscore'], scores], 
						index=cols), 
					ignore_index=True)
		elif competition in ['games','regionals']:
			for athlete in data['leaderboardRows']:
				scores = []
				for workout in athlete['scores']:
					scoreDisplay = None if 'scoreDisplay' not in workout else workout['scoreDisplay']
					scores += [(workout['workoutrank'], scoreDisplay)]
				df = df.append(
					pd.Series(
						[athlete['overallRank'], 
						athlete['entrant']['competitorId'], 
						athlete['overallScore'], 
						scores], 
						index=cols), 
					ignore_index=True)

		return df

	def get_page_url(page):
		url = "https://games.crossfit.com/competitions/api/v1/competitions/" + \
			competition + "/" + str(year) + "/leaderboards"
		_division = ['','men','women'].index(division)
		_sort = ['overall','17.1','17.2','17.3','17.4','17.5'].index(sort)
		_fittest_in = ['','region'].index(fittest_in)
		if competition == 'open':
			_region = ['worldwide','africa','asia','australia','canada_east'].index(region) # todo fillin...
			url += "?division=" + str(_division) + \
				"&scaled=" + str(0) + \
				"&sort=" + str(_sort) + \
				"&fittest=" + str(_fittest_in) + \
				"&fittest1=" + str(_region) + \
				"&occupation=" + str(0) + \
				"&page=" + str(page)
		elif competition == 'regionals':
			_region = ['','atlantic','california',
				'central','east','meridian','pacific','south','west'].index(region) 
			url += "?division=" + str(_division) + \
				"&sort=" + str(0) + \
				"&regional=" + str(_region) + \
				"&page=" + str(1)
		elif competition == 'games':
			url += "?division=" + str(_division) + \
				"&sort=" + str(0) + \
				"&page=" + str(1)

		return url

	i = 0
	def get_page_dataframe(page):
		nonlocal i
		i += 1
		print("Fetching page",i,"/",get_pages())
		url = get_page_url(page)
		leaderboard_page = requests_get(url)
		df = process_page(leaderboard_page)
		return df

	pool = ThreadPool(processes=50)
	if competition == 'regionals' and region == 'worldwide':
		regions = ['atlantic','california','central','east','meridian','pacific','south','west']
		regional_scrape_leaderboard_data = lambda r: \
			scrape_leaderboard_data(competition=competition, year=year, division=division, 
				sort=sort, fittest_in=fittest_in, region=r)
		dfs = pool.map(regional_scrape_leaderboard_data, regions)
	else:
		dfs = pool.map(get_page_dataframe, range(1, get_pages()+1))

	df = pd.DataFrame()
	for partial_df in dfs:
		df = df.append(partial_df, ignore_index=True)

	return df

#@persistent_memoize('scrape_athlete_data')
def scrape_athlete_data(athlete_id):
	'''
	Extract athlete stats into dict.
	'''

	athlete_page = requests_get("http://games.crossfit.com/athlete/" + str(athlete_id))
	if athlete_page is None: return None
	soup = BeautifulSoup(athlete_page.content, "html.parser")
	athlete_data = {}

	# name
	stat = soup.title.text[:7]
	metric = soup.title.text[9:-17]
	athlete_data[stat] = metric

	# region, division, age, height, weight, affiliate, team
	stats = soup.find_all(attrs='bg-games-black-overlay infobar-container overlayed')
	stats = stats[0].contents[1].contents
	for s in stats:
		if hasattr(s, 'li'):
			divs = s.find_all('div')
			stat = divs[0].text.strip()
			metric = divs[1].text.strip()
			athlete_data[stat] = metric

	# benchmark stats
	stats = soup.find_all(attrs='stats-header')
	for s in stats:
		stat = s.text.strip()
		metric = s.next_sibling.next_sibling.text.strip()
		athlete_data[stat] = metric

	# rankings (not interested in this)
	'''for row in soup.find_all("tr"):
		if row.th is None:
			stats = row.find_all('td')
			stat = stats[0].text
			metric = stats[1].text
			if metric == '--':
				metric = None
			athlete_data[stat] = metric'''
	return athlete_data

@persistent_memoize('scrape_athletes_data')
def _scrape_athletes_data(athlete_ids):
	pool = ThreadPool(processes=50)
	athlete_datas = pool.map(scrape_athlete_data, athlete_ids)
	return athlete_datas

def scrape_athletes_data(athlete_ids):
	# do this so cache will work better
	# this whole sorting thing is probably pointless...
	sorted_ids = sorted(athlete_ids)
	sorted_athletes_data = _scrape_athletes_data(sorted_ids)
	# now need to unsort
	argsort = np.argsort(athlete_ids)
	athletes_data = [None] * len(athlete_ids)
	for idx, data in zip(argsort, sorted_athletes_data):
		athletes_data[idx] = data
	return athletes_data

def process_athlete_stats(athlete_data):
	'''
	Standardise athlete stats to cm/kg/reps/sec
	'''
	def to_cm(data):
		if data is None: return None
		if 'cm' in data:
			data = float(data[:-3])
		else:
			idx = data.index('\'')
			feet = float(data[idx-1])
			inches = float(data[idx+1:-1])
			data = 2.54*(feet*12+inches)
		return data
	def to_kg(data):
		if data is None: return None
		if 'kg' in data:
			data = float(data[:-3])
		else:
			lbs = float(data[:-3])
			data = lbs/2.2
		return data
	def to_reps(data):
		if data is None: return None
		data = float(data)
		return data
	def to_min(data):
		if data is None: return None
		idx = data.index(':')
		# 1 guys fran is '-2:0-35'
		# assume this means 2:35 (it could mean 20:35 though...)
		# so lets strip non-numeric characters from our min/sec
		non_decimal = re.compile(r'[^\d.]+')
		m = float(non_decimal.sub('',data[:idx]))
		s = float(non_decimal.sub('',data[idx+1:]))
		return m+s/60

	if athlete_data is None: athlete_data = {}

	stats = ['Age','Height','Weight',
		'Back Squat','Clean and Jerk','Snatch',
		'Deadlift','Fight Gone Bad','Max Pull-ups',
		'Fran','Grace','Helen',
		'Filthy 50','Sprint 400m','Run 5k']
	for stat in stats:
		if stat not in athlete_data or athlete_data[stat] == '--':
			athlete_data[stat] = None

	if athlete_data['Age'] is not None:
		athlete_data['Age'] = int(athlete_data['Age'])
	athlete_data['Height'] = to_cm(athlete_data['Height'])
	athlete_data['Weight'] = to_kg(athlete_data['Weight'])

	athlete_data['Back Squat'] = to_kg(athlete_data['Back Squat'])
	athlete_data['Clean and Jerk'] = to_kg(athlete_data['Clean and Jerk'])
	athlete_data['Snatch'] = to_kg(athlete_data['Snatch'])
	athlete_data['Deadlift'] = to_kg(athlete_data['Deadlift'])

	athlete_data['Fight Gone Bad'] = to_reps(athlete_data['Fight Gone Bad'])
	athlete_data['Max Pull-ups'] = to_reps(athlete_data['Max Pull-ups'])

	athlete_data['Fran'] = to_min(athlete_data['Fran'])
	athlete_data['Grace'] = to_min(athlete_data['Grace'])
	athlete_data['Helen'] = to_min(athlete_data['Helen'])
	athlete_data['Filthy 50'] = to_min(athlete_data['Filthy 50'])
	athlete_data['Sprint 400m'] = to_min(athlete_data['Sprint 400m'])
	athlete_data['Run 5k'] = to_min(athlete_data['Run 5k'])

	return athlete_data

@persistent_memoize('populate_leaderboard_with_stats')
def populate_leaderboard_with_stats(*args, **kwargs):
	'''
	Append athlete stats to each row in a leaderboard dataframe
	'''
	leaderboard_df = scrape_leaderboard_data(*args, **kwargs)

	print("Populating leaderboard with athlete data")
	athlete_stats = []

	# leaderboard may be big, iterate over in chunks
	chunk_size = 1000
	for i in range(len(leaderboard_df)//chunk_size+1):
		print("Chunk",i+1,"/",len(leaderboard_df)//chunk_size+1)
		a = i*chunk_size
		b = min((i+1)*chunk_size, len(leaderboard_df))
		chunk = leaderboard_df[a:b]
		userids = list(chunk['userid'])

		athlete_stats.extend(scrape_athletes_data(userids))

	#athlete_stats = pool.map(get_athlete_data, leaderboard_df.iterrows())
	athlete_stats = pd.Series(athlete_stats, name='athlete_stats')
	df = pd.concat([leaderboard_df, athlete_stats], axis=1)
	return df

@persistent_memoize('get_analysis_dataframe')
def _get_analysis_dataframe(*args, **kwargs):
	'''
	Get the dataframe to be used for statistical analysis
	Here we process and standardise all rows to kg/cm/reps/sec
	And only keep the 'useful' information
	'''
	leaderboard_df = populate_leaderboard_with_stats(*args, **kwargs)

	cols = ['overallrank','overallscore',
		'Age','Height','Weight',
		'Back Squat','Clean and Jerk','Snatch',
		'Deadlift','Fight Gone Bad','Max Pull-ups',
		'Fran','Grace','Helen',
		'Filthy 50','Sprint 400m','Run 5k']

	df = pd.DataFrame(columns=cols)

	i = 0
	def get_new_row(row):
		_,row = row
		nonlocal i
		i += 1
		if i%1000==0:
			print("Processing row",i,"/",len(leaderboard_df))
		try:
			athlete_stats = process_athlete_stats(row.athlete_stats)
		except Exception as e:
			# some strange athlete stats
			print(e)
			print(row.athlete_stats)
			raise e
		new_row = pd.Series(
			[
				row.overallrank, row.overallscore,
				athlete_stats['Age'],athlete_stats['Height'],athlete_stats['Weight'],
				athlete_stats['Back Squat'],athlete_stats['Clean and Jerk'],athlete_stats['Snatch'],
				athlete_stats['Deadlift'],athlete_stats['Fight Gone Bad'],athlete_stats['Max Pull-ups'],
				athlete_stats['Fran'],athlete_stats['Grace'],athlete_stats['Helen'],
				athlete_stats['Filthy 50'],athlete_stats['Sprint 400m'],athlete_stats['Run 5k']
			], index=cols)
		return new_row

	pool = ThreadPool(processes=1)
	new_rows = list(map(get_new_row, leaderboard_df.iterrows()))
	df = pd.DataFrame(new_rows)
	return df

def get_analysis_dataframe(competition='open', year=2017, division='men', 
	sort='overall', fittest_in='region', region='worldwide'):
	return _get_analysis_dataframe(competition, year, division, sort, fittest_in, region)
