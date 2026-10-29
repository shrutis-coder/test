# Greenlight Score Team Plan

Last updated Oct 13, 2026 by Sarah

## Roles

| Person | Brief column | Blocks they own | Expert model |
| --- | --- | --- | --- |
| P1: Aryan | Script: Production | Scene parser, Scene tagger, production features, 10 budget driver scores, budget estimator | Feeds the script expert |
| P2: Sarah | Script: Story | Script rater, Segment scorer (3 curves), story features, LLM controls, 50 human-rated scripts | Script expert |
| P3: Melody | Market | Data pull, News reader, trend features, release window search | Market expert |
| P4: Shruti | Interviews | Transcripts, Claim extractor, Rubric builder, Script judge, Report writer | Interview expert |
| P5: Wesley | Comps | Clean and label the 1M dataset, year split, Embedding model, comps search | Comps expert |
| P6 | Combiner | Out-of-fold setup, 1D CNN, dense layers, three heads, calibration, baselines, ablations, evaluation | Combiner |

## Step 1. Collect the data

Download links are in "Where each person downloads data" below.

| Person | Collect | Minimum | Good |
| --- | --- | --- | --- |
| P1 Script: Production | Full scripts with outcomes | 500 | 2,000+ |
| P1 Script: Production | Scripts with known budgets | 300 | 1,000+ |
| P2 Script: Story | Human-rated scripts (2 to 3 raters each) | 50 | 100 |
| P3 Market | Weekly box office and release calendars | 1990 onward | 1980 onward |
| P3 Market | Trade news articles | 2 years back | 5 years back |
| P3 Market | Streaming Top 10 | All since July 2021 | All since July 2021 |
| P4 Interviews | Interview subjects (FilmCrowd plus public) | 15 | 25+ |
| P5 Comps | Films with budget and worldwide gross | 3,000 | 8,000 to 15,000 |
| P5 Comps | Breakout films (5x or more) | 300 | 1,000+ |
| P5 Comps | Validation films (2018 to 2019) | 300 | 600+ |
| P5 Comps | Test films (2022 to 2025) | 400 | 800+ |
| P5 Comps | Synopses with outcomes | 5,000 | 15,000+ |
| P6 Combiner | Repo, storage and handoff schemas ready for everyone's data | Done | Done |

## Step 2. Build each box in the diagram

Each person builds the boxes in their column of the brief's detailed model (Section 04), top to bottom. Every LLM block uses a fixed prompt, a written rubric where it rates, JSON output, three runs averaged, and a version number.

### P1 Script: Production

| Box | Type | What to do | Data needed |
| --- | --- | --- | --- |
| Scene parser | Rules | Split scripts into scenes by sluglines (INT. KITCHEN - NIGHT), count INT/EXT and day/night. No LLM | 20 test scripts, 5 hand-checked |
| Scene tagger | LLM | Per scene return location, speaking characters, extras, stunts, VFX, special conditions | Every script (500 min, 2,000+ good) |
| Production features | Rules | Genre, estimated rating, runtime from page count, INT/EXT ratio, day/night ratio, locations, scene count, plus 10 budget driver scores (shoot days, cast size, locations, exteriors and nights, sets, stunts, VFX, special conditions, costume and makeup, music) | Scene tagger output |
| Budget estimator | Calibrated model | Map drivers to a band (micro under $1M, low $1M to $10M, mid $10M to $40M, high $40M to $100M, tentpole $100M+). Also builds the budget curve | Scripts with known budgets (300 min, 1,000+ good) |

### P2 Script: Story

| Box | Type | What to do | Data needed |
| --- | --- | --- | --- |
| Script rater | LLM | Rate structure, arc, dialogue, tone, payoff 1 to 5 with reasons and written anchors | Every script, plus 50 human-rated scripts to check it |
| Segment scorer | LLM | Score emotion, energy and genre mix for 100 equal segments | Every script |
| Story features | Rules | Structure, characters, dialogue and writing, concept, entertainment payoff (brief Section 05 table), plus 3 curves of 100 points | Rater and scorer output |
| Script expert | Gradient boosting | Train on story plus P1's production features, out-of-fold, 0 to 100 percentile sub-score | Scripts with outcomes (500 min, 2,000+ good) |

### P3 Market

| Box | Type | What to do | Data needed |
| --- | --- | --- | --- |
| Data pull | Rules | Weekly box office, release calendars, streaming Top 10 | Box office from 1990 (1980 good), Top 10 since 2021 |
| News reader | LLM | Turn trade articles into announced releases and dates | 2 years of news (5 good) |
| Trend features | Rules | Genre share, momentum, saturation, recent genre ROI, seasonality index, competition (optional), streaming demand. Only data before the as-of date. Search every release window when none is given | Data pull and News reader output |
| Market expert | Gradient boosting | Out-of-fold, 0 to 100 sub-score | Films with outcomes (3,000 min, 8,000+ good) |

### P4 Interviews

| Box | Type | What to do | Data needed |
| --- | --- | --- | --- |
| Claim extractor | LLM | Pull every "X makes a film work" or "X kills a film" claim with quote and artist | Interview subjects (15 min, 25+ good) |
| Rubric builder | LLM | Cluster claims into criteria, weight by how many artists raised each, write 1 to 5 anchors. A person reviews it | All extracted claims |
| Script judge | LLM | Score each criterion per script and quote the passage that justifies it | Every script |
| Interview expert | Gradient boosting | Out-of-fold, proprietary 0 to 100 sub-score. Prove it by training the full model with and without it | Scripts with outcomes (500 min, 2,000+ good) |
| Report writer | LLM | Turn drivers and SHAP into plain notes and scene-level fixes, three factors up, three down. Never changes the numbers | Combiner output |

### P5 Comps

| Box | Type | What to do | Data needed |
| --- | --- | --- | --- |
| Clean and label | Rules | Inflation adjust. Labels are breakeven (gross 2.5x budget or more), log ROI, breakout (5x or more), streaming success, legs, audience score. Split train through 2017, validate 2018 to 2019, skip 2020 to 2021, test 2022 to 2025 | Films with budget and gross (3,000 min, 8,000 to 15,000 good), 300+ breakouts, 300+ validation, 400+ test |
| Embedding model | Embedding | Turn every synopsis and script into vectors | Synopses (5,000 min, 15,000+ good) plus all scripts |
| Comps search | Rules | 5 to 10 nearest films, same genre and budget band, before the as-of date, never itself. Outputs comp ROI average and spread, closest similarity, comp recency, number of close comps | 5 comps per film (10 good) |
| Comps expert | Gradient boosting | Out-of-fold, 0 to 100 sub-score | Films with outcomes (3,000 min, 8,000+ good) |

### P6 Combiner

| Box | Type | What to do | Data needed |
| --- | --- | --- | --- |
| 1D CNN | Trained | Reads the 3 curves. Conv 16 (kernel 5, pool 2), conv 32, global pool, out 16 | Curves from P2 |
| Concatenate | Rules | Sub-scores as logits, log budget and budget band, genre (multi-hot), comp ROI spread, missing-data flags, CNN output. About 50 values | All four sub-scores |
| Dense 64, Dense 32 | Trained | ReLU, layer norm, dropout 0.2 | Films with all sub-scores (3,000 min, 8,000+ good) |
| Three heads | Trained | Breakeven (sigmoid), ROI 20th, 50th, 80th percentile (pinball loss, never crosses), breakout 5x or more (weighted cross-entropy). Loss weights 1 : 1 : 0.5 | Same films |
| Calibrate, ensemble | Trained | Temperature scaling on validation, five seeds averaged, seed spread feeds the risk level. Adam, learning rate 1e-3, weight decay 1e-4, batch 256, patience 20 | Validation films (300 min, 600+ good) |
| Investor view | Output | Breakeven probability, ROI range, breakout, risk level, sub-scores, top drivers and fixes, five closest comps, budget curve | Test films, opened once (400 min, 800+ good) |

## Step 3. Order of work

1. Collect data until every minimum in Step 1 is met
2. Push 200 films through every box end to end to catch format problems
3. Scale every box to all films
4. Train the four experts out-of-fold
5. Train the combiner and compare it with gradient boosting on the same inputs. If it doesn't win, ship gradient boosting
6. Run ablations (each sub-score, curves, budget, each LLM block) and the interview proof
7. Open the test set once and report the final numbers

Parked for now are the Talent column, the Audience simulator, best channel and genre gating.

## Where each person downloads data

Everything joins on IMDb ID. Sources marked Scrape need a terms-of-use check before the product ships.

### P1 Script: Production

| Source | What it gives | How many we need | Access |
| --- | --- | --- | --- |
| [MovieSum](https://github.com/saxenarohit/MovieSum) | 2,200 formatted scripts with IMDb IDs | All 2,200, keep every one that joins to an outcome | Download |
| [Movie Script Database](https://github.com/Aveek-Saha/Movie-Script-Database) | Scraper for more scripts | Enough to reach 500 scripts with outcomes (2,000+ good) | Scrape |
| [IMSDb](https://imsdb.com/) | Raw scripts to fill gaps | Only the gaps left after the two above | Scrape |
| [The Numbers budgets](https://www.the-numbers.com/movie/budgets/all) | Known budgets to check the budget estimator | 300 scripts with known budgets (1,000+ good) | Scrape |

### P2 Script: Story

| Source | What it gives | How many we need | Access |
| --- | --- | --- | --- |
| [MovieSum](https://github.com/saxenarohit/MovieSum) | Scripts for rating, same files as P1 | Every script P1 collects, plus 50 picked for human rating (100 good) | Download |
| [MPST](https://www.kaggle.com/datasets/cryptexcode/mpst-movie-plot-synopses-with-tags) | Synopses with tone tags for checks | The synopses matching the 50 human-rated scripts | Download |

### P3 Market

| Source | What it gives | How many we need | Access |
| --- | --- | --- | --- |
| [Box Office Mojo weekly](https://www.boxofficemojo.com/weekly/) | Weekly box office history | Every week since 1990 (1980 good) | Scrape |
| [Box Office Mojo calendar](https://www.boxofficemojo.com/calendar/) | Release dates | Every release since 1990 (1980 good) | Scrape |
| [The Numbers release schedule](https://www.the-numbers.com/movies/release-schedule) | Past and upcoming releases | Same years, plus all announced upcoming releases | Scrape |
| [Netflix Top 10](https://www.netflix.com/tudum/top10) | Streaming Top 10 weeks and hours viewed | Every week since July 2021 | Download |
| [Deadline](https://deadline.com/) | Trade news for the News reader | 2 years of release news (5 good) | Scrape or RSS |
| [Variety](https://variety.com/) | Trade news for the News reader | 2 years of release news (5 good) | Scrape or RSS |

### P4 Interviews

| Source | What it gives | How many we need | Access |
| --- | --- | --- | --- |
| FilmCrowd interviews | 5 proprietary transcripts | All 5 | Internal |
| [Scriptnotes](https://johnaugust.com/scriptnotes) | Screenwriter transcripts | 4 to 6 episodes | Download |
| [BAFTA Screenwriters' Lectures](https://www.bafta.org) | Lecture transcripts | 3 to 5 lectures | Download |
| [Film Courage](https://www.youtube.com/@FilmCourage) | Writer, producer, director interviews | 5 to 8 interviews | Transcribe |
| [KCRW The Business](https://www.kcrw.com/culture/shows/the-business) | Producers and execs | 2 to 3 episodes | Transcribe |
| [The Town](https://www.theringer.com/podcasts/the-town-with-matthew-belloni) | Industry business talk | 2 to 3 episodes | Transcribe |
| **Total** |  | **15 different people minimum, 25+ good** |  |
| [youtube-transcript-api](https://pypi.org/project/youtube-transcript-api/) | Pulls YouTube captions | Tool | Tool |
| [Whisper](https://github.com/openai/whisper) | Transcribes audio | Tool | Tool |
| [pyannote](https://github.com/pyannote/pyannote-audio) | Speaker labels | Tool | Tool |

### P5 Comps

| Source | What it gives | How many we need | Access |
| --- | --- | --- | --- |
| [TMDB 1M movies](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies) | Main film list, budget, revenue, overviews | Whole file, giving 3,000 films with budget and gross (8,000 to 15,000 good) | Download |
| [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) | 45k films with IMDb links for prototyping | Whole file, prototyping only | Download |
| [IMDb datasets](https://developer.imdb.com/non-commercial-datasets/) | Titles, ratings, cast, crew | Whole files | Download |
| [TMDB API](https://developer.themoviedb.org/docs) | Fill gaps, fresh data | Only films missing budget, gross or release date | Free API key |
| [The Numbers budgets](https://www.the-numbers.com/movie/budgets/all) | Reliable budgets and worldwide gross | Every film it lists that matches ours | Scrape |
| [Box Office Mojo](https://www.boxofficemojo.com/) | Grosses and opening weekends | Opening weekend for every labeled film, for legs | Scrape |
| [US CPI (FRED)](https://fred.stlouisfed.org/series/CPIAUCSL) | Inflation adjustment | Full monthly series | Download |
| [CMU Movie Summary Corpus](https://www.cs.cmu.edu/~ark/personas/) | 42k plot summaries | Whole file, toward 5,000 synopses with outcomes (15,000+ good) | Download |
| [Wikipedia Movie Plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots) | 34k plots | Whole file, fills synopses CMU misses | Download |

Across P5's sources, the labeled films must include 300+ breakouts, 300+ validation films (2018 to 2019) and 400+ test films (2022 to 2025).

### P6 Combiner

No downloads. P6 works from the tables P1 to P5 hand over.

## Shared rules

**Handoffs**

- Every table has an `imdb_id` column and an `as_of_date` column
- Every LLM output is JSON with a fixed schema and a prompt version number
- New features go into the feature dictionary before anyone uses them

**Repo layout**

- `raw/` untouched downloads, `clean/` cleaned tables, one cleaning script per source
- `prompts/` versioned prompts, `features/` one folder per stage, `models/` experts and combiner
- Log every dropped row and why
