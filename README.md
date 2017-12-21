# Crossfit Web Scraper and Data Analysis

`crossfit_api.py` provides an API for scraping data from the 2017 Crossfit leaderboard. Make sure there's a /cache folder in the same directory as crossfit_scaper.py; big queries will be stored here (including the full leaderboards you get, manually delete the files if you want to refresh your data).

`data_analysis.py` produces some pretty box plots, and allows you to interact with them by left clicking to set 'your' benchmarks. The predicted open placement will then be calculated using an XGB model trained on open leaderboard data. This code needs some pretty serious cleaning up though, I'll get around to that sometime...

[Here](https://imgur.com/a/pCmYr) are some nice box plots of the scraped data.
