from crossfit_api import get_analysis_dataframe
import numpy as np
import pandas as pd
from memoize import persistent_memoize, memoize

from functools import partial

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.linear_model import Lasso, RANSACRegressor, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler

from multiprocessing import Pool
import itertools, random, os, sys, time
import fancyimpute

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime

from fancyimpute import Solver

class ScaleKNeighborsRegressor(KNeighborsRegressor):
	def predict(self, X):
		# standardise X
		X = self.scaler.transform(X)
		return super().predict(X)
	def fit(self, X, y):
		# standardise X
		self.scaler = RobustScaler().fit(X)
		X = self.scaler.transform(X)
		return super().fit(X,y)

class RecursiveKNN(Solver):
	def __init__(self, k=5, verbose=0, 
		min_value=None,
		max_value=None,
		normalizer=None,
		feature_selector=None,
		regressor=partial(ScaleKNeighborsRegressor, weights='distance'),
		n_jobs=1):
		Solver.__init__(
			self, 
			min_value=min_value,
			max_value=max_value,
			normalizer=normalizer)
		self.k = k
		self.verbose = verbose
		self.feature_selector = feature_selector
		self.regressor = regressor
		self.n_jobs = n_jobs

	def _transform(self, feature_selector, X):
		# alternative feature selector transform to remove some NaN checks
		mask = feature_selector.get_support()
		if not mask.any():
			warn("No features were selected: either the data is"
				" too noisy or the selection test too strict.",
				UserWarning)
			return np.empty(0).reshape((X.shape[0], 0))
		if len(mask) != X.shape[1]:
			raise ValueError("X has a different shape than during fitting.")
		return X[:, mask]

	def _get_reg(self):
		if self.feature_selector != None:
			reg = Pipeline([
				('feature_selection', SelectFromModel(self.feature_selector())),
				('regression', ScaleKNeighborsRegressor(algorithm='brute'))
				])
		else:
			reg = ScaleKNeighborsRegressor()
		return reg

	def _impute_row(self, i):

		row = self.X[i,:]
		known_idx = np.where(~np.isnan(row))[0]
		unknown_idx = np.where(np.isnan(row))[0]
		# todo make this do one col at a time
		X_ = self.X[:,known_idx]
		y_ = self.X[:,unknown_idx]
		y_pred = np.zeros_like(unknown_idx)
		if unknown_idx.size > 0:
			reg = self.regressor()
			full_rows = np.logical_and(~np.isnan(X_).any(axis=1), ~np.isnan(y_).any(axis=1))
			X_ = X_[full_rows]
			y_ = y_[full_rows]
			reg.fit(X_, y_)
			y_pred = reg.predict(row[known_idx].reshape(1,-1))

		return (i, unknown_idx, y_pred)

	def _impute_unonown_idx(self, unknown_idx):
		known_idx = [x for x in range(self.X.shape[1]) if x not in unknown_idx]
		row_idxs = np.argwhere(np.logical_and(
			np.isnan(self.X[:,unknown_idx]).all(axis=1),
			~np.isnan(self.X[:,known_idx]).any(axis=1)))

		y_pred = np.zeros((len(row_idxs),len(unknown_idx)))
		if len(row_idxs) > 0:
			reg = self.regressor()
			selector = SelectFromModel(self.feature_selector())
			# predict 1 feature at a time
			for i,idx in enumerate(unknown_idx):
				full_rows = np.argwhere(np.logical_and(
					~np.isnan(self.X[:,known_idx]).any(axis=1), 
					~np.isnan(self.X[:,[idx]]).any(axis=1)))
				# use these rows to perform feature selection
				selector.fit(
					self.X[full_rows,known_idx], 
					self.X[full_rows,[idx]])
				# now recalculate full rows based on selected features
				full_rows = np.argwhere(np.logical_and(
					~np.isnan(self._transform(selector, self.X[:,known_idx])).any(axis=1), 
					~np.isnan(self.X[:,[idx]]).any(axis=1)))
				# and fit regression model, then predict
				reg.fit(
					self._transform(selector, self.X[full_rows,known_idx]), 
					self.X[full_rows,[idx]])
				# memory error for predicting too many at once
				# so split into chunks
				chunksize = 10000
				for chunk_idx in range(0, len(row_idxs), chunksize):
					y_pred[chunk_idx:chunk_idx+chunksize, [i]] = \
						reg.predict(
							self._transform(selector, 
								self.X[row_idxs[chunk_idx:chunk_idx+chunksize], known_idx]))

				if self.verbose > 1:
					print("Imputed",len(unknown_idx),"features in",len(row_idxs),"rows\n",
						"\tUsing data from", len(full_rows),"rows")

				#y_pred[:,[i]] = reg.predict(self.X[row_idxs,known_idx])

		return (row_idxs, unknown_idx, y_pred)

	def solve(self, X, missing_mask):
		self.X = np.where(~missing_mask, X, np.nan)
		imputed_X = np.where(~missing_mask, X, np.nan)

		# do rows based on what's missing
		pool = Pool(processes=self.n_jobs)
		cols = np.argwhere(np.isnan(self.X).any(axis=0)).flatten()

		num_combs = [j * len(list(itertools.combinations(cols,j))) for j in range(1,len(cols))]
		cum_num_combs = np.cumsum(num_combs)

		t0 = time.time()
		for j in range(1,len(cols)):
			np.savetxt(str(j)+'.csv', imputed_X, delimiter=',')
			if self.verbose > 0:
				if j > 1:
					print("\tTime elapsed:", time.time()-t0)
					print("\t", round(100*cum_num_combs[j-1]/cum_num_combs[-1],1),"% complete")
					print("\tEstimated total time:", (time.time()-t0)/cum_num_combs[j-1] * \
						cum_num_combs[-1])
				print("Imputing",len(list(itertools.combinations(cols,j))),
					"feature combinations of size",j,"/",len(cols)-1)
			for i, unknown_idx, y_pred in \
				pool.imap(self._impute_unonown_idx, itertools.combinations(cols,j), chunksize=100):

				imputed_X[i,unknown_idx] = y_pred

		return imputed_X


# check for extreme values (eg 666 pullups, 10sec 400m...) 
def clear_outliers(data):
	data = data.copy()
	cols = [
			'Age','Height','Weight',
			'Back Squat','Clean and Jerk','Snatch',
			'Deadlift','Fight Gone Bad','Max Pull-ups',
			'Fran','Grace','Helen',
			'Filthy 50','Sprint 400m','Run 5k']
	ranges = [
			(16,80),(100,250),(30,150),
			(20,300),(20,250),(20,200),
			(20,400),(20,750),(0,150),
			(1.5,30),(1,60),(3,60),
			(10,120),(0.72,3),(12.5,60)
		]
	'''ranges = [
			(16,80),(100,250),(30,150),
			(20,300),(20,250),(20,200),
			(20,400),(20,600),(0,120),
			(1.5,10),(1,15),(3,15),
			(10,60),(0.72,3),(12.5,45)
		]'''
	for col,valid_range in zip(cols, ranges):
		outliers = (valid_range[0] > data[col]) | (data[col] > valid_range[1])

		i = 0
		for idx in np.argwhere(outliers==True).flatten():
			i += 1
		print(i, "outliers in", col)
		data[col] = data[col].where(~outliers, np.nan)
	# check for other outliers
	# this doesn't work so well
	'''clf = IsolationForest(contamination=1/1000)
	clf.fit(data.dropna())
	outliers = clf.predict(data.fillna(data.mean()))
	outliers = outliers == -1
	for idx in np.argwhere(outliers==True).flatten():
		print(pd.DataFrame(pd.DataFrame(data.loc[idx]).transpose()))
	raise Exception'''
	return data

@persistent_memoize('get_imputed_dataframe')
def _get_imputed_dataframe(*args, **kwargs):
	def impute_rows(data, X_cols, y_cols):
		rows_idx = np.argwhere(np.logical_and(
			np.isnan(data[:,y_cols]).all(axis=1),
			~np.isnan(data[:,X_cols]).any(axis=1)))
		y_pred = np.zeros((len(rows_idx),len(y_cols)))
		if len(rows_idx) > 0:
			print("\tImputing",len(rows_idx),"rows")
			full_rows = np.argwhere(np.logical_and(
				~np.isnan(data[:,X_cols]).any(axis=1), 
				~np.isnan(data[:,y_cols]).any(axis=1)))
			reg = RANSACRegressor()
			reg.fit(
				data[full_rows,X_cols], 
				data[full_rows,y_cols])
			y_pred = reg.predict(data[rows_idx,X_cols]).clip(min=0)
		return (rows_idx, y_cols, y_pred)
	def impute_update_data(data, X_cols, y_cols):
		print(X_cols,"predicting",y_cols)
		cols = list(data)
		X_cols = [cols.index(x) for x in X_cols]
		y_cols = [cols.index(y) for y in y_cols]
		matrix = data.as_matrix()
		rows_idx, y_cols, y_pred = impute_rows(matrix, X_cols, y_cols)
		matrix[rows_idx,y_cols] = y_pred
		return pd.DataFrame(matrix, index=data.index, columns=data.columns)

	data = get_analysis_dataframe(*args, **kwargs)
	data = data.astype(float)
	data = clear_outliers(data)

	Xys = [
		#(['Height'],['Weight']),
		#(['Weight'],['Height']),

		(['Snatch'],['Clean and Jerk']),
		(['Clean and Jerk'],['Snatch']),

		(['Snatch','Clean and Jerk'],['Back Squat']),

		(['Snatch','Clean and Jerk','Back Squat'],['Deadlift']),
		(['Back Squat'],['Deadlift']),
		(['Deadlift'],['Back Squat']),

		#(['Run 5k'],['Sprint 400m']),
		#(['Sprint 400m'],['Run 5k']),

		(['Weight','Snatch','Clean and Jerk','Back Squat','Deadlift'],['Max Pull-ups']),
		(['Weight','Back Squat','Deadlift'],['Max Pull-ups']),
		(['Weight','Snatch','Clean and Jerk'],['Max Pull-ups']),

		#(['Filthy 50'],['Fight Gone Bad']),
		#(['Fight Gone Bad'],['Filthy 50']),

		(['Max Pull-ups', 'Clean and Jerk'],['Fran']),
		(['Clean and Jerk', 'Fran'],['Grace']),
		(['Max Pull-ups', 'Sprint 400m', 'Run 5k'],['Helen']),
		#(['Max Pull-ups', 'Grace'],['Fran']),
		]
	for x,y in Xys:
		data = impute_update_data(data, x, y)

	data = clear_outliers(data)

	imputer = RecursiveKNN(verbose=1,n_jobs=4,
		feature_selector=DecisionTreeRegressor)
	data = pd.DataFrame(imputer.complete(data), index=data.index, columns=data.columns)
	return data

def get_imputed_dataframe(competition='open', year=2017, division='men', 
	sort='overall', fittest_in='region', region='worldwide'):
	return _get_imputed_dataframe(competition, year, division, sort, fittest_in, region)

# ANALYSIS
def box_plots(data, title='Open'):
	plt.suptitle(title + " Box Plots")
	kwargs = {'showfliers':False}
	stats = ['Age', 'Height', 'Weight']
	weights = ['Deadlift','Back Squat', 'Clean and Jerk', 'Snatch']
	reps = ['Fight Gone Bad', 'Max Pull-ups']
	times = ['Fran', 'Grace', 'Helen', 'Filthy 50', 'Sprint 400m', 'Run 5k']
	for i,x in enumerate(stats):
		plt.subplot(4,3,i+1)
		plt.boxplot(list(data[x].dropna()),labels=[x], **kwargs)
	plt.subplot(4,1,2)
	plt.boxplot([list(data[x].dropna()) for x in weights], labels=weights, vert=False, **kwargs)
	for i,x in enumerate(reps):
		plt.subplot(4,2,5+i)
		plt.boxplot(list(data[x].dropna()),labels=[x], vert=False, **kwargs)
	plt.subplot(4,1,4)
	for i,x in enumerate(times):
		plt.subplot(4,6,19+i)
		plt.boxplot(list(data[x].dropna()),labels=[x], **kwargs)
	plt.show()

def box_plots_all(open_data, regionals_data, games_data, title, my_data, metric):

	def mouse_click(event):
		ax = event.inaxes
		stat = ''
		if ax in ax_stats:
			stat = stats[ax_stats.index(ax)]
			val = event.ydata
		elif ax in ax_weights:
			stat = weights[ax_weights.index(ax)]
			val = event.ydata
		elif ax in ax_reps:
			stat = reps[ax_reps.index(ax)]
			val = event.xdata
		elif ax in ax_times:
			stat = times[ax_times.index(ax)]
			val = event.ydata
		if event.button == 1:
			my_data[stat]=val
		elif event.button == 2:
			nonlocal box_plots_on, ax_stats, ax_weights, ax_reps, ax_times
			box_plots_on = not(box_plots_on)
			ax_stats, ax_weights, ax_reps, ax_times = draw_plots()
		if event.button == 3:
			if stat in my_data:
				del my_data[stat]
		plot_my_data()

	def plot_my_data():
		nonlocal lines
		for l in lines:
			try:
				l.remove()
			except:
				l.pop(0).remove()
		lines = []

		for x,ax in zip(stats, ax_stats):
			if x in my_data:
				ax.set_prop_cycle(None)
				l = ax.plot([0,4],[my_data[x],my_data[x]])
				lines += [l]

		for x,ax in zip(weights, ax_weights):
			if x in my_data:
				ax.set_prop_cycle(None)
				l = ax.plot([0,4],[my_data[x],my_data[x]])
				lines += [l]

		for x,ax in zip(reps, ax_reps):
			if x in my_data:
				ax.set_prop_cycle(None)
				l = ax.plot([my_data[x],my_data[x]], [0,4])
				lines += [l]

		for x,ax in zip(times, ax_times):
			if x in my_data:
				ax.set_prop_cycle(None)
				l = ax.plot([0,4],[my_data[x],my_data[x]])
				lines += [l]

		rank,pcntile = predict_ranking(open_data, my_data)
		filled_my_data = {}
		for k,v in my_data.items(): filled_my_data[k] = v
		for k in stats+weights+reps+times: 
			if k not in filled_my_data:
				filled_my_data[k] = np.nan
		table_text = []
		table_text.append([
			'Age',fmt_age(filled_my_data['Age'],0),
			'Height',fmt_height(filled_my_data['Height'],0),
			'Weight',fmt_weight(filled_my_data['Weight'],0),
			'',''])
		table_text.append([
			'Back Squat',fmt_weight(filled_my_data['Back Squat'],0),
			'Deadlift',fmt_weight(filled_my_data['Deadlift'],0),
			'Fran',fmt_time(filled_my_data['Fran'],0),
			'Filthy 50',fmt_time(filled_my_data['Filthy 50'],0)])
		table_text.append([
			'Clean and Jerk',fmt_weight(filled_my_data['Clean and Jerk'],0),
			'Fight Gone Bad',fmt_reps(filled_my_data['Fight Gone Bad'],0),
			'Grace',fmt_time(filled_my_data['Grace'],0),
			'Sprint 400m',fmt_time(filled_my_data['Sprint 400m'],0)])
		table_text.append([
			'Snatch',fmt_weight(filled_my_data['Snatch'],0),
			'Max Pull-ups',fmt_reps(filled_my_data['Max Pull-ups'],0),
			'Helen',fmt_time(filled_my_data['Helen'],0),
			'Run 5k',fmt_time(filled_my_data['Run 5k'],0)])
		table_text.append([
			'','',
			'','',
			'Estimated Ranking', str(round(rank,0)),
			'Percentile', str(round(pcntile,2))])
		font = {
			'family': 'monospace',
			'color':  'k',
			'weight': 'heavy',
			'size': 12,
			}
		ax = plt.subplot(5,1,5)
		tab = ax.table(cellText=table_text, loc='center', bbox=[0, -.5, 1, 1.25], fontsize=12,
			colWidths=[1.5,1] * 4)
		cells = tab.properties()['celld']
		for i in range(5):
			for j in range(4):
				cells[i,2*j]._loc = 'right'
				cells[i,2*j+1]._loc = 'left'
				cells[i,2*j].set_linewidth(0)
				cells[i,2*j+1].set_linewidth(0)
		ax.axis('tight')
		ax.axis('off')
		lines += [tab]
		plt.gcf().canvas.draw_idle()

	box_plots_on = True
	lines = []
	
	plt.figure().canvas.mpl_connect('button_press_event', mouse_click)

	maintitle = dict(fontsize=18, fontweight='bold')
	subtitle = dict(fontsize=12, fontweight='bold')
	plt.suptitle(title + " Box Plots", **maintitle)
	plt.rcParams['axes.facecolor'] = 'whitesmoke'

	boxprops = dict(linewidth=1, alpha=0.8)
	medianprops = dict(linewidth=2, color='k', alpha=0.8)
	whiskerprops = dict(linewidth=1, color='k', linestyle='-')
	kwargs = dict(sym='', whis=[1,99], patch_artist=True, widths=0.5, #notch=True, bootstrap=1000,
		medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops)
	stats = ['Age', 'Height', 'Weight']
	weights = ['Deadlift','Back Squat', 'Clean and Jerk', 'Snatch']
	reps = ['Fight Gone Bad', 'Max Pull-ups']
	times = ['Fran', 'Grace', 'Helen', 'Filthy 50', 'Sprint 400m', 'Run 5k']

	colors = ['steelblue', 'olivedrab', 'indianred']
	def add_colors(bplot):
		for patch,color in zip(bplot['boxes'],colors):
			patch.set_facecolor(color)

	def fmt_age(x, pos):
		if np.isnan(x): return ''
		x = round(x)
		return str(x)
	def fmt_height(x, pos):
		if np.isnan(x): return ''
		if metric:
			return str(int(x))+" cm"
		ft, inches = divmod(round(x), 12)
		ft, inches = map(int, [ft, inches])
		return ('{}\''.format(ft) if not inches 
			else '{}\'{}"'.format(ft, inches) if ft
			else '{}"'.format(inches))
	def fmt_weight(x, pos):
		if np.isnan(x): return ''
		x = int(x)
		if metric:
			return str(x)+" kg"
		return str(x)+" lbs"
	def fmt_reps(x, pos):
		if np.isnan(x): return ''
		x = int(x)
		return str(x)+" reps"
	def fmt_time(x, pos):
		if np.isnan(x): return ''
		m, s = divmod(round(x*60), 60)
		m, s = map(int, [m, s])
		return (str(m)+':'+str(s).zfill(2))

	def draw_plots():

		def get_cols(cols):
			if metric:
				return [
					list(open_data[x].dropna()), 
					list(regionals_data[x].dropna()), 
					list(games_data[x].dropna())]
			else:
				if x == 'Height': scaler = 1/2.54
				elif x in ['Weight']+weights: scaler = 2.2
				else: scaler = 1
				return [
					list(open_data[x].dropna()*scaler), 
					list(regionals_data[x].dropna()*scaler), 
					list(games_data[x].dropna()*scaler)]

		labels = ['Open','Regionals','Games']

		ax_stats = []
		for i,x in enumerate(stats):
			ax = plt.subplot(5,3,i+1)
			ax_stats += [ax]
			plt.title(x, **subtitle)
			plt.grid(axis='y',linestyle='dotted')
			bplot = plt.boxplot(get_cols(x),labels=labels, **kwargs)
			add_colors(bplot)
			if x == 'Height':
				plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(fmt_height))
			elif x == 'Weight':
				plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(fmt_weight))
			plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))
		ax_weights = []
		for i,x in enumerate(weights):
			ax = plt.subplot(5,4,5+i)
			ax_weights += [ax]
			plt.title(x, **subtitle)
			plt.grid(axis='y',linestyle='dotted')
			bplot = plt.boxplot(get_cols(x),labels=labels, **kwargs)
			add_colors(bplot)
			plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(fmt_weight))
			plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))
		ax_reps = []
		for i,x in enumerate(reps):
			ax = plt.subplot(5,2,5+i)
			ax_reps += [ax]
			plt.title(x, **subtitle)
			plt.grid(axis='x',linestyle='dotted')
			bplot = plt.boxplot(get_cols(x),labels=labels, vert=False, **kwargs)
			add_colors(bplot)
			plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(fmt_reps))
			plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))
		ax_times = []
		for i,x in enumerate(times):
			ax = plt.subplot(5,6,19+i)
			ax_times += [ax]
			plt.title(x, **subtitle)
			plt.grid(axis='y',linestyle='dotted')
			bplot = plt.boxplot(get_cols(x),labels=labels, **kwargs)
			add_colors(bplot)
			plt.gca().set_yscale('log')
			plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(fmt_time))
			plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))
			plt.minorticks_off()

		plt.subplots_adjust(left=0.125,right=0.9,
			bottom=0.1,top=0.9,wspace=0.3,hspace=0.4)

		return ax_stats, ax_weights, ax_reps, ax_times

	ax_stats, ax_weights, ax_reps, ax_times = draw_plots()

	plot_my_data()

	plt.show()

def ANALYSIS_open(division='men'):
	open_data = get_analysis_dataframe(competition='open', division=division)
	open_data = clear_outliers(open_data)

	box_plots(open_data, 'Open')

def ANALYSIS_regionals(division='men'):
	regionals_data = get_analysis_dataframe(competition='regionals', division=division)
	regionals_data = clear_outliers(regionals_data)

	box_plots(regionals_data, 'Regionals')

def ANALYSIS_games(division='men'):
	games_data = get_analysis_dataframe(competition='games', division=division)
	games_data = clear_outliers(games_data)

	box_plots(games_data, 'Games')

def ANALYSIS_all(division='men', my_data={}, metric=True):
	open_data = get_analysis_dataframe(competition='open', division=division)
	open_data = clear_outliers(open_data)

	regionals_data = get_analysis_dataframe(competition='regionals', division=division)
	regionals_data = clear_outliers(regionals_data)

	games_data = get_analysis_dataframe(competition='games', division=division)
	games_data = clear_outliers(games_data)

	box_plots_all(open_data, regionals_data, games_data, division.title(), my_data, metric)

def ANALYSIS_all_imputed(division='men', my_data={}, metric=True):
	open_data = get_imputed_dataframe(division = division, competition='open')
	regionals_data = get_analysis_dataframe(division = division, competition='regionals')
	games_data = get_analysis_dataframe(division = division, competition='games')

	# use imputed values from open data to fill in athlete stats for regionals/games data
	regionals_data = pd.merge(
		open_data.drop(['overallrank','overallscore'],axis=1), 
		regionals_data[['userid','overallrank','overallscore']], 
		on='userid', how='inner')
	games_data = pd.merge(
		open_data.drop(['overallrank','overallscore'],axis=1), 
		games_data[['userid','overallrank','overallscore']], 
		on='userid', how='inner')

	box_plots_all(open_data, regionals_data, games_data, "Imputed " + division.title(), my_data, metric)

alex = {'Age':23,'Height':165,'Weight':70,
	'Back Squat':175, 'Clean and Jerk':133, 'Snatch':108, 'Deadlift':220,
	'Max Pull-ups':25,
	'Fran': 5}
pan = {'Age':22,'Height':158,'Weight':53,
	'Back Squat':57, 'Clean and Jerk':35, 'Snatch':28, 'Deadlift':70,
	'Max Pull-ups':0}

fraser = get_analysis_dataframe(division='men', competition='games').iloc[0].dropna().drop(['overallscore','userid','overallrank']).to_dict()
tct = get_analysis_dataframe(division='women', competition='games').iloc[0].dropna().drop(['overallscore','userid','overallrank']).to_dict()
sara = get_analysis_dataframe(division='women', competition='open').iloc[0].dropna().drop(['overallscore','userid','overallrank']).to_dict()

import xgboost as xgb

@memoize()
def get_fitted_model(data):

	reg = xgb.XGBRegressor(missing=np.nan)

	X = data.drop(['userid', 'overallrank', 'overallscore'], axis=1).as_matrix()
	y = data['overallrank'].as_matrix()
	reg.fit(X,np.log1p(y))
	return reg

def predict_ranking(data, my_data):
	cols = list(data.drop(['userid', 'overallrank', 'overallscore'], axis=1))
	X_pred = []
	for i,col in enumerate(cols):
		if col in my_data:
			X_pred += [my_data[col]]
		else:
			X_pred += [np.nan]

	reg = get_fitted_model(data)

	known_cols = list(my_data)

	y_pred = np.expm1(reg.predict(X_pred))

	return y_pred[0], (y_pred / data['overallrank'].max()*100)[0]

ANALYSIS_all_imputed(division='men',metric=True, my_data=alex)
#ANALYSIS_all_imputed(division='men',metric=True, my_data=fraser)

raise Exception()

def to_imperial(stats):
	stats['Height'] /= 2.54
	stats['Weight'] *= 2.2
	stats['Back Squat'] *= 2.2
	stats['Clean and Jerk'] *= 2.2
	stats['Snatch'] *= 2.2
	stats['Deadlift'] *= 2.2
	return stats

# lets test some models
data = get_analysis_dataframe()
data = clear_outliers(data)
#data = data[:1000]
X = data.drop(['userid', 'overallrank', 'overallscore'], axis=1)
y = pd.to_numeric(data['overallrank'])
y = np.log1p(y)

from sklearn.preprocessing import Imputer, StandardScaler, RobustScaler, FunctionTransformer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from scipy.stats import skew

def log_skewed_cols(X):
	X = X.apply(lambda x: np.log1p(x) if abs(skew(x.dropna()))>1 else x, axis=0)
	return X

get_score = lambda model: cross_val_score(model, X, y, n_jobs=1).mean()
for reg in [
	LinearRegression(),
	RidgeCV(),
	#ElasticNetCV(),
	#MLPRegressor(hidden_layer_sizes=(100,10,5,)),
	#KNeighborsRegressor(),
	#RandomForestRegressor(),
	SVR(),
	xgb.XGBRegressor(),
	#KerasRegressor(build_fn=create_model, verbose=0),
	]:

	print(reg)
	pipeline = Pipeline([
		('logtransform', FunctionTransformer(log_skewed_cols, validate=False)),
		('imputer', Imputer()),
		('scaler', RobustScaler()),
		('regressor', reg)
		])
	try:
		t = time.time()
		print("\tScore:",get_score(pipeline))
		print("\t",time.time()-t,"seconds")
	except Exception as e:
		raise e
		print(e)
# xgb easily outperforms others
# now look at imputing methods
class SKLearnFancyImputer(Imputer):
	def __init__(self,imputer):
		self.imputer = imputer

	def fit(self, X, y=None):
		self.X = X
		return self

	def transform(self, X):
		if np.array_equal(np.nan_to_num(self.X),np.nan_to_num(X)):
			return self.imputer.complete(X)
		else:
			return self.imputer.complete(X.append(self.X))[:len(X)]

'''for imp in [
	Imputer(), Imputer(strategy='median'),
	SKLearnFancyImputer(fancyimpute.SoftImpute(verbose=0)),
	SKLearnFancyImputer(fancyimpute.IterativeSVD(verbose=0)),
	#SKLearnFancyImputer(fancyimpute.MICE(verbose=0)),
	#SKLearnFancyImputer(fancyimpute.MatrixFactorization(verbose=False)),
	#SKLearnFancyImputer(fancyimpute.NuclearNormMinimization(verbose=0)),
	#SKLearnFancyImputer(fancyimpute.BiScaler(verbose=0)),
	]:

	print(imp)
	pipeline = Pipeline([
		('imputer', imp),
		('regressor', xgb.XGBRegressor())
		])
	try:
		t = time.time()
		print("\tScore:",get_score(pipeline))
		print("\t",time.time()-t,"seconds")
	except Exception as e:
		print(e)'''
