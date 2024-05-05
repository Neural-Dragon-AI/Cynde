# Combined Text Dir from modal-examples

- Full filepath to the merged directory: `C:\Users\Tommaso\Documents\Dev\modal-examples`

- Created: `2024-05-05T12:47:43.973337`

## generators

import modal

app = modal.App(
    "example-generators"
)  # Note: prior to April 2024, "app" was called "stub"


@app.function()
def f(i):
    for j in range(i):
        yield j


@app.local_entrypoint()
def main():
    for r in f.remote_gen(10):
        print(r)


---

## get started

import modal

app = modal.App(
    "example-get-started"
)  # Note: prior to April 2024, "app" was called "stub"


@app.function()
def square(x):
    print("This code is running on a remote worker!")
    return x**2


@app.local_entrypoint()
def main():
    print("the square is", square.remote(42))


---

## hello world

# # Hello, world!
#
# This tutorial demonstrates some core features of Modal:
#
# * You can run functions on Modal just as easily as you run them locally.
# * Running functions in parallel on Modal is simple and fast.
# * Logs and errors show up immediately, even for functions running on Modal.
#
# ## Importing Modal and setting up
#
# We start by importing `modal` and creating a `App`.
# We build up from our `App` to [define our application](/docs/guide/apps).

import sys

import modal

app = modal.App(
    "example-hello-world"
)  # Note: prior to April 2024, "app" was called "stub"

# ## Defining a function
#
# Modal takes code and runs it in the cloud.
#
# So first we've got to write some code.
#
# Let's write a simple function:
# log `"hello"` to standard out if the input is even
# or `"world"` to standard error if it's not,
# then return the input times itself.
#
# To make this function work with Modal, we just wrap it in a decorator
# from our application `app`,
# [`@app.function`](/docs/reference/modal.App#function).


@app.function()
def f(i):
    if i % 2 == 0:
        print("hello", i)
    else:
        print("world", i, file=sys.stderr)

    return i * i


# ## Running our function locally, remotely, and in parallel
#
# Now let's see three different ways we can call that function:
#
# 1. As a regular `local` call on your computer, with `f.local`
#
# 2. As a `remote` call that runs in the cloud, with `f.remote`
#
# 3. By `map`ping many copies of `f` in the cloud over many inputs, with `f.map`
#
# We call `f` in each of these ways inside a `main` function below.


@app.local_entrypoint()
def main():
    # run the function locally
    print(f.local(1000))

    # run the function remotely on Modal
    print(f.remote(1000))

    # run the function in parallel and remotely on Modal
    total = 0
    for ret in f.map(range(20)):
        total += ret

    print(total)


# Enter `modal run hello_world.py` in a shell and you'll see
# a Modal app initialize.
# You'll then see the `print`ed logs of
# the `main` function and, mixed in with them, all the logs of `f` as it is run
# locally, then remotely, and then remotely and in parallel.
#
# That's all triggered by adding the [`@app.local_entrypoint`](/docs/reference/modal.App#local_entrypoint) decorator on `main`,
# which defines it as the function to start from locally when we invoke `modal run`.
#
# ## What just happened?
#
# When we called `.remote` on `f`, the function was executed
# **in the cloud**, on Modal's infrastructure, not locally on our computer.
#
# In short, we took the function `f`, put it inside a container,
# sent it the inputs, and streamed back the logs and outputs.
#
# ## But why does this matter?
#
# Try doing one of these things next to start seeing the full power of Modal!
#
# ### You can change the code and run it again
#
# For instance, change the `print` statement in the function `f`
# to print `"spam"` and `"eggs"` instead and run the app again.
# You'll see that that your new code is run with no extra work from you --
# and it should even run faster!
#
# Modal's goal is to make running code in the cloud feel like you're
# running code locally. That means no waiting for long image builds when you've just moved a comma,
# no fiddling with container image pushes, and no context-switching to a web UI to inspect logs.
#
# ### You can map over more data
#
# Change the `map` range from `20` to some large number, like `1170`. You'll see
# Modal create and run even more containers in parallel this time.
#
# And it'll happen lightning fast!
#
# ### You can run a more interesting function
#
# The function `f` is obviously silly and doesn't do much, but in its place
# imagine something that matters to you, like:
#
# * Running [language model inference](/docs/examples/vllm_mixtral) or [fine-tuning](/docs/examples/slack-finetune)
# * Manipulating [audio](/docs/examples/discord-musicgen) or [images](stable_diffusion_xl_turbo)
# * [Collecting financial data](/docs/examples/fetch_stock_prices) to backtest a trading algorithm.
#
# Modal lets you parallelize that operation effortlessly by running hundreds or
# thousands of containers in the cloud.


---

## import sklearn

# # Install scikit-learn in a custom image
#
# This builds a custom image which installs the sklearn (scikit-learn) Python package in it.
# It's an example of how you can use packages, even if you don't have them installed locally.
#
# First, the imports

import time

import modal

# Next, define an app, with a custom image that installs `sklearn`.

app = modal.App(
    "import-sklearn",
    image=modal.Image.debian_slim()
    .apt_install("libgomp1")
    .pip_install("scikit-learn"),
)  # Note: prior to April 2024, "app" was called "stub"

# The `app.image.imports()` lets us conditionally import in the global scope.
# This is needed because we might not have sklearn and numpy installed locally,
# but we know they are installed inside the custom image.

with app.image.imports():
    import numpy as np
    from sklearn import datasets, linear_model

# Now, let's define a function that uses one of scikit-learn's built-in datasets
# and fits a very simple model (linear regression) to it


@app.function()
def fit():
    print("Inside run!")
    t0 = time.time()
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
    diabetes_X = diabetes_X[:, np.newaxis, 2]
    regr = linear_model.LinearRegression()
    regr.fit(diabetes_X, diabetes_y)
    return time.time() - t0


# Finally, let's trigger the run locally. We also time this. Note that the first time we run this,
# it will build the image. This might take 1-2 min. When we run this subsequent times, the image
# is already build, and it will run much much faster.


if __name__ == "__main__":
    t0 = time.time()
    with app.run():
        t = fit.remote()
        print("Function time spent:", t)
    print("Full time spent:", time.time() - t0)


---

## install cuda

# # Create an image with CUDA
#
# This example shows how you can use the Nvidia CUDA base image from DockerHub.
# We need to add Python 3 and pip with the `add_python` option because the image
# doesn't have these by default.

from modal import App, Image

image = Image.from_registry(
    "nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.11"
)
app = App(image=image)  # Note: prior to April 2024, "app" was called "stub"

# Now, we can create a function with GPU capabilities. Run this file with
# `modal run install_cuda.py`.


@app.function(gpu="T4")
def f():
    import subprocess

    subprocess.run(["nvidia-smi"])


---

## screenshot

# # Screenshot with Chromium
#
# In this example, we use Modal functions and the `playwright` package to take screenshots
# of websites from a list of URLs in parallel.
#
# You can run this example on the command line with
#
# ```
# modal run 02_building_containers/screenshot.py --url 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
# ```
#
# This should take a few seconds then create a `/tmp/screenshots/screenshot.png` file, shown below.
#
# ![screenshot](./screenshot.png)
#
# ## Setup
#
# First we import the Modal client library.

import pathlib

import modal

app = modal.App(
    "example-screenshot"
)  # Note: prior to April 2024, "app" was called "stub"

# ## Define a custom image
#
# We need an image with the `playwright` Python package as well as its `chromium` plugin pre-installed.
# This requires intalling a few Debian packages, as well as setting up a new Debian repository.
# Modal lets you run arbitrary commands, just like in Docker:


image = modal.Image.debian_slim().run_commands(
    "apt-get update",
    "apt-get install -y software-properties-common",
    "apt-add-repository non-free",
    "apt-add-repository contrib",
    "pip install playwright==1.42.0",
    "playwright install-deps chromium",
    "playwright install chromium",
)

# ## The screenshot function
#
# Next, the scraping function which runs headless Chromium, goes to a website, and takes a screenshot.
# This is a Modal function which runs inside the remote container.


@app.function(image=image)
async def screenshot(url):
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle")
        await page.screenshot(path="screenshot.png")
        await browser.close()
        data = open("screenshot.png", "rb").read()
        print("Screenshot of size %d bytes" % len(data))
        return data


# ## Entrypoint code
#
# Let's kick it off by reading a bunch of URLs from a txt file and scrape some of those.


@app.local_entrypoint()
def main(url: str = "https://modal.com"):
    filename = pathlib.Path("/tmp/screenshots/screenshot.png")
    data = screenshot.remote(url)
    filename.parent.mkdir(exist_ok=True)
    with open(filename, "wb") as f:
        f.write(data)
    print(f"wrote {len(data)} bytes to {filename}")


# And we're done! Please also see our [introductory guide](/docs/examples/web-scraper) for another
# example of a web scraper, with more in-depth logic.


---

## basic grid search

# # Hyperparameter search
#
# This example showcases a simple grid search in one dimension, where we try different
# parameters for a model and pick the one with the best results on a holdout set.
#
# ## Defining the image
#
# First, let's build a custom image and install scikit-learn in it.

import modal

app = modal.App(
    "example-basic-grid-search",
    image=modal.Image.debian_slim().pip_install("scikit-learn~=1.2.2"),
)  # Note: prior to April 2024, "app" was called "stub"

# ## The Modal function
#
# Next, define the function. Note that we use the custom image with scikit-learn in it.
# We also take the hyperparameter `k`, which is how many nearest neighbors we use.


@app.function()
def fit_knn(k):
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier

    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    clf = KNeighborsClassifier(k)
    clf.fit(X_train, y_train)
    score = float(clf.score(X_test, y_test))
    print("k = %3d, score = %.4f" % (k, score))
    return score, k


# ## Parallel search
#
# To do a hyperparameter search, let's map over this function with different values
# for `k`, and then select for the best score on the holdout set:


@app.local_entrypoint()
def main():
    # Do a basic hyperparameter search
    best_score, best_k = max(fit_knn.map(range(1, 100)))
    print("Best k = %3d, score = %.4f" % (best_k, best_score))


---

## fetch stock prices

# ---
# output-directory: "/tmp/"
# runtimes: ["runc", "gvisor"]
# ---
# # Fetching stock prices in parallel
#
# This is a simple example that uses the Yahoo! Finance API to fetch a bunch of ETFs
# We do this in parallel, which demonstrates the ability to map over a set of items
# In this case, we fetch 100 stocks in parallel
#
# You can run this script on the terminal with
#
# ```bash
# modal run 03_scaling_out/fetch_stock_prices.py
# ```
#
# If everything goes well, it should plot something like this:
#
# ![stock prices](./stock_prices.png)
#
#
# ## Setup
#
# For this image, we need
#
# - `httpx` and `beautifulsoup4` to fetch a list of ETFs from a HTML page
# - `yfinance` to fetch stock prices from the Yahoo Finance API
# - `matplotlib` to plot the result

import io
import os

import modal

app = modal.App(
    "example-fetch-stock-prices",
    image=modal.Image.debian_slim().pip_install(
        "httpx~=0.24.0",
        "yfinance~=0.2.31",
        "beautifulsoup4~=4.12.2",
        "matplotlib~=3.7.1",
    ),
)  # Note: prior to April 2024, "app" was called "stub"

# ## Fetch a list of tickers
#
# The `yfinance` package does not have a way to download a list of stocks.
# To get a list of stocks, we parse the HTML from Yahoo Finance using Beautiful Soup
# and ask for the top 100 ETFs.


@app.function()
def get_stocks():
    import bs4
    import httpx

    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36",
        "referer": "https://finance.yahoo.com/",
    }
    url = "https://finance.yahoo.com/etfs?count=100&offset=0"
    res = httpx.get(url, headers=headers)
    res.raise_for_status()
    soup = bs4.BeautifulSoup(res.text, "html.parser")
    for td in soup.find_all("td", {"aria-label": "Symbol"}):
        for link in td.find_all("a", {"data-test": "quoteLink"}):
            symbol = str(link.next)
            print(f"Found symbol {symbol}")
            yield symbol


# ## Fetch stock prices
#
# Now, let's fetch the stock data. This is the function that we will parallelize.
# It's fairly simple and just uses the `yfinance` package.


@app.function()
def get_prices(symbol):
    import yfinance

    print(f"Fetching symbol {symbol}...")
    ticker = yfinance.Ticker(symbol)
    data = ticker.history(period="1Y")["Close"]
    print(f"Done fetching symbol {symbol}!")
    return symbol, data.to_dict()


# ## Plot the result
#
# Here is our plotting code. We run this in Modal, although you could also run it locally.
# Note that the plotting code calls the other two functions.
# Since we plot the data in the cloud, we can't display it, so we generate a PNG
# and return the binary content from the function.


@app.function()
def plot_stocks():
    from matplotlib import pyplot, ticker

    # Setup
    pyplot.style.use("ggplot")
    fig, ax = pyplot.subplots(figsize=(8, 5))

    # Get data
    tickers = list(get_stocks.remote_gen())
    if not tickers:
        raise RuntimeError("Retrieved zero stock tickers!")
    data = list(get_prices.map(tickers))
    first_date = min((min(prices.keys()) for symbol, prices in data if prices))
    last_date = max((max(prices.keys()) for symbol, prices in data if prices))

    # Plot every symbol
    for symbol, prices in data:
        if len(prices) == 0:
            continue
        dates = list(sorted(prices.keys()))
        prices = list(prices[date] for date in dates)
        changes = [
            100.0 * (price / prices[0] - 1) for price in prices
        ]  # Normalize to initial price
        if changes[-1] > 20:
            # Highlight this line
            p = ax.plot(dates, changes, alpha=0.7)
            ax.annotate(
                symbol,
                (last_date, changes[-1]),
                ha="left",
                va="center",
                color=p[0].get_color(),
                alpha=0.7,
            )
        else:
            ax.plot(dates, changes, color="gray", alpha=0.2)

    # Configure axes and title
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.set_title(f"Best ETFs {first_date.date()} - {last_date.date()}")
    ax.set_ylabel(f"% change, {first_date.date()} = 0%")

    # Dump the chart to .png and return the bytes
    with io.BytesIO() as buf:
        pyplot.savefig(buf, format="png", dpi=300)
        return buf.getvalue()


# ## Entrypoint
#
# The entrypoint locally runs the app, gets the chart back as a PNG file, and
# saves it to disk.

OUTPUT_DIR = "/tmp/"


@app.local_entrypoint()
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = plot_stocks.remote()
    filename = os.path.join(OUTPUT_DIR, "stock_prices.png")
    print(f"saving data to {filename}")
    with open(filename, "wb") as f:
        f.write(data)


---

## db to sheet

# # Write to Google Sheets
#
# In this tutorial, we'll show how to use Modal to schedule a daily update of a dataset
# from an analytics database to Google Sheets.
#
# ## Entering credentials
#
# We begin by setting up some credentials that we'll need in order to access our database and output
# spreadsheet. To do that in a secure manner, we log in to our Modal account on the web and go to
# the [Secrets](/secrets) section.
#
# ### Database
#
# First we will enter our database credentials. The easiest way to do this is to click **New
# secret** and select the **Postgres compatible** secret preset and fill in the requested
# information. Then we press **Next** and name our secret "example-postgres-secret" and click **Create**.
#
# ### Google Sheets/GCP
#
# We'll now add another Secret for Google Sheets access through Google Cloud Platform. Click **New
# secret** and select the Google Sheets preset.
#
# In order to access the Google Sheets API, we'll need to create a *Service Account* in Google Cloud
# Platform. You can skip this step if you already have a Service Account json file.
#
# 1. Sign up to Google Cloud Platform or log in if you haven't
#    ([https://cloud.google.com/](https://cloud.google.com/)).
# 2. Go to [https://console.cloud.google.com/](https://console.cloud.google.com/).
# 3. In the navigation pane on the left, go to **IAM & Admin** > **Service Accounts**.
# 4. Click the **+ CREATE SERVICE ACCOUNT** button.
# 5. Give the service account a suitable name, like "sheet-access-bot". Click **Done**. You don't
#    have to grant it any specific access privileges at this time.
# 6. Click your new service account in the list view that appears and navigate to the **Keys**
#    section.
# 7. Click **Add key** and choose **Create new key**. Use the **JSON** key type and confirm by
#    clicking **Create**.
# 8. A json key file should be downloaded to your computer at this point. Copy the contents of that
#    file and use it as the value for the **SERVICE_ACCOUNT_JSON** field in your new secret.
#
# We'll name this other secret "gsheets-secret".
#
# Now you can access the values of your secrets from modal functions that you annotate with the
# corresponding EnvDict includes, e.g.:

import os

import modal

app = modal.App(
    "example-db-to-sheet"
)  # Note: prior to April 2024, "app" was called "stub"


@app.function(secrets=[modal.Secret.from_name("example-postgres-secret")])
def my_func():
    # automatically filled from the specified secret
    print("Host is " + os.environ["PGHOST"])


# In order to connect to the database, we'll use the `psycopg2` Python package. To make it available
# to your Modal function you need to supply it with an `image` argument that tells Modal how to
# build the container image that contains that package. We'll base it off of the `Image.debian_slim` base
# image that's built into modal, and make sure to install the required binary packages as well as
# the psycopg2 package itself:

pg_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libpq-dev")
    .pip_install("psycopg2~=2.9.9")
)

# Since the default keynames for a **Postgres compatible** secret correspond to the environment
# variables that `psycopg2` looks for, you can now easily connect to the database even without
# explicit credentials in your code. We'll create a simple function that queries the city for each
# user in our dummy `users` table:


@app.function(
    image=pg_image,
    secrets=[modal.Secret.from_name("example-postgres-secret")],
)
def get_db_rows():
    import psycopg2

    conn = psycopg2.connect()  # no explicit credentials needed
    cur = conn.cursor()
    cur.execute("SELECT city FROM users")
    return [row[0] for row in cur.fetchall()]


# Note that we import psycopg2 inside our function instead of the global scope. This allows us to
# run this Modal function even from an environment where psycopg2 is not installed. We can test run
# this function using the `modal run` shell command: `modal run db_to_sheet.py::app.get_db_rows`.

# ## Applying Python logic
#
# For each city in our source data we'll make an online lookup of the current weather using the
# [http://openweathermap.org](http://openweathermap.org) API. To do this, we'll add the API key to
# another modal secret. We'll use a custom secret called "weather-secret" with the key
# `OPENWEATHER_API_KEY` containing our API key for OpenWeatherMap.

requests_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "requests~=2.31.0"
)


@app.function(
    image=requests_image,
    secrets=[modal.Secret.from_name("weather-secret")],
)
def city_weather(city):
    import requests

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": os.environ["OPENWEATHER_API_KEY"]}
    response = requests.get(url, params=params)
    weather_label = response.json()["weather"][0]["main"]
    return weather_label


# We'll make use of Modal's built-in `function.map` method to create our report. `function.map`
# makes it really easy to parallelise work by executing a function for a larger sequence of input
# data. For this example we'll make a simple count of rows per weather type, using Python's
# standard library `collections.Counter`.

from collections import Counter


@app.function()
def create_report(cities):
    # run city_weather for each city in parallel
    user_weather = city_weather.map(cities)
    users_by_weather = Counter(user_weather).items()
    return users_by_weather


# Let's try to run this! To make it simple to trigger the function with some
# predefined input data, we create a "local entrypoint" `main` that can be
# easily triggered from the command line:


@app.local_entrypoint()
def main():
    cities = [
        "Stockholm,,Sweden",
        "New York,NY,USA",
        "Tokyo,,Japan",
    ]
    print(create_report.remote(cities))


# Running the local entrypoint using `modal run db_to_sheet.py` should print something like:
# `dict_items([('Clouds', 3)])`.
# Note that since this file only has a single app, and the app has only one local entrypoint
# we only have to specify the file to run - the function/entrypoint is inferred.

# In this case the logic is quite simple, but in a real world context you could have applied a
# machine learning model or any other tool you could build into a container to transform the data.
#
# ## Sending output to a Google Sheet
#
# We'll set up a new Google Sheet to send our report to. Using the "Sharing" dialog in Google
# Sheets, we make sure to share the document to the service account's email address (the value of
# the `client_email` field in the json file) and make the service account an editor of the document.
#
# The URL of a Google Sheet is something like:
# `https://docs.google.com/spreadsheets/d/1wOktal......IJR77jD8Do`.
#
# We copy the part of the URL that comes after `/d/` - that is the *key* of the document which
# we'll refer to in our code. We'll make use of the `pygsheets` python package to authenticate with
# Google Sheets and then update the spreadsheet with information from the report we just created:

pygsheets_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "pygsheets~=2.0.6"
)


@app.function(
    image=pygsheets_image,
    secrets=[modal.Secret.from_name("gsheets-secret")],
)
def update_sheet_report(rows):
    import pygsheets

    gc = pygsheets.authorize(service_account_env_var="SERVICE_ACCOUNT_JSON")
    document_key = "1RqQrJ6Ikf611adKunm8tmL1mKzHLjNwLWm_T7mfXSYA"
    sh = gc.open_by_key(document_key)
    worksheet = sh.sheet1
    worksheet.clear("A2")

    worksheet.update_values("A2", [list(row) for row in rows])


# At this point, we have everything we need in order to run the full program. We can put it all together in
# another Modal function, and add a [schedule](/docs/guide/cron) argument so it runs every day automatically:


@app.function(schedule=modal.Cron("0 0 * * *"))
def db_to_sheet():
    rows = get_db_rows.remote()
    report = create_report.remote(rows)
    update_sheet_report.remote(report)
    print("Updated sheet with new weather distribution")
    for weather, count in report:
        print(f"{weather}: {count}")


# This entire app can now be deployed using `modal deploy db_to_sheet.py`. The [apps page](/apps)
# shows our cron job's execution history and lets you navigate to each invocation's logs.
# To trigger a manual run from your local code during development, you can also trigger this function using the cli:
# `modal run db_to_sheet.py::app.db_to_sheet`

# Note that all of the @app.function() annotated functions above run remotely in isolated containers that are specified per
# function, but they are called as seamlessly as using regular Python functions. This is a simple
# showcase of how you can mix and match functions that use different environments and have them feed
# into each other or even call each other as if they were all functions in the same local program.


---

## hackernews alerts

# ---
# lambda-test: false
# ---
# # Hacker News Slackbot

# In this example, we use Modal to deploy a cron job that periodically queries Hacker News for
# new posts matching a given search term, and posts the results to Slack.

# ## Import and define the app
#
# Let's start off with imports, and defining a Modal app.

import os
from datetime import datetime, timedelta

import modal

app = modal.App(
    "example-hn-bot"
)  # Note: prior to April 2024, "app" was called "stub"

# Now, let's define an image that has the `slack-sdk` package installed, in which we can run a function
# that posts a slack message.

slack_sdk_image = modal.Image.debian_slim().pip_install("slack-sdk")

# ## Defining the function and importing the secret
#
# Our Slack bot will need access to a bot token. We can use Modal's [Secrets](/secrets) interface to accomplish this.
# To quickly create a Slack bot secret, navigate to the [create secret](/secrets/create) page, select the Slack secret template
# from the list options, and follow the instructions in the "Where to find the credentials?" panel.
# Name your secret `hn-bot-slack`, so that the code in this example still works.
#
# Now, we define the function `post_to_slack`, which simply instantiates the Slack client using our token,
# and then uses it to post a message to a given channel name.


@app.function(
    image=slack_sdk_image, secrets=[modal.Secret.from_name("hn-bot-slack")]
)
async def post_to_slack(message: str):
    import slack_sdk

    client = slack_sdk.WebClient(token=os.environ["SLACK_BOT_TOKEN"])
    client.chat_postMessage(channel="hn-alerts", text=message)


# ## Searching Hacker News
#
# We are going to use Algolia's [Hacker News Search API](https://hn.algolia.com/api) to query for posts
# matching a given search term in the past X days. Let's define our search term and query period.

QUERY = "serverless"
WINDOW_SIZE_DAYS = 1

# Let's also define an image that has the `requests` package installed, so we can query the API.

requests_image = modal.Image.debian_slim().pip_install("requests")

# We can now define our main entrypoint, that queries Algolia for the term, and calls `post_to_slack`
# on all the results. We specify a [schedule](/docs/guide/cron) in the function decorator, which
# means that our function will run automatically at the given interval.


@app.function(image=requests_image)
def search_hackernews():
    import requests

    url = "http://hn.algolia.com/api/v1/search"

    threshold = datetime.utcnow() - timedelta(days=WINDOW_SIZE_DAYS)

    params = {
        "query": QUERY,
        "numericFilters": f"created_at_i>{threshold.timestamp()}",
    }

    response = requests.get(url, params, timeout=10).json()
    urls = [item["url"] for item in response["hits"] if item.get("url")]

    print(f"Query returned {len(urls)} items.")

    post_to_slack.for_each(urls)


# ## Test running
#
# We can now test run our scheduled function as follows: `modal run hackernews_alerts.py::app.search_hackernews`

# ## Defining the schedule and deploying
#
# Let's define a function that will be called by Modal every day


@app.function(schedule=modal.Period(days=1))
def run_daily():
    search_hackernews.remote()


# In order to deploy this as a persistent cron job, you can run `modal deploy hackernews_alerts.py`,

# Once the job is deployed, visit the [apps page](/apps) page to see
# its execution history, logs and other stats.


---

## schedule simple

# ---
# cmd: ["python", "-m", "05_scheduling.schedule_simple"]
# ---
import time
from datetime import datetime

import modal

app = modal.App(
    "example-schedule-simple"
)  # Note: prior to April 2024, "app" was called "stub"


@app.function(schedule=modal.Period(seconds=5))
def print_time_1():
    print(
        f'Printing with period 5 seconds: {datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}'
    )


@app.function(schedule=modal.Cron("* * * * *"))
def print_time_2():
    print(
        f'Printing with cron every minute: {datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}'
    )


if __name__ == "__main__":
    with app.run():
        time.sleep(60)


---

## blender video

# ---
# output-directory: "/tmp/render"
# ---
# # Render a video with Blender on many GPUs or CPUs in parallel
#
# This example shows how you can render an animated 3D scene using
# [Blender](https://www.blender.org/)'s Python interface.
#
# You can run it on CPUs to scale out on one hundred containers
# or run it on GPUs to get higher throughput per node.
# Even with this simple scene, GPUs render 10x faster than CPUs.
#
# The final render looks something like this:
#
# ![Spinning Modal logo](https://modal-public-assets.s3.amazonaws.com/modal-blender-render.gif)
#
# ## Defining a Modal app

import io
import math
from pathlib import Path

import modal

# Modal runs your Python functions for you in the cloud.
# You organize your code into apps, collections of functions that work together.

app = modal.App("examples-blender-logo")

# We need to define the environment each function runs in --  its container image.
# The block below defines a container image, starting from a basic Debian Linux image
# adding Blender's system-level dependencies
# and then installing the `bpy` package, which is Blender's Python API.

rendering_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("xorg", "libxkbcommon0")  # X11 (Unix GUI) dependencies
    .pip_install("bpy")  # Blender as a Python package
)

# ## Rendering a single frame
#
# We define a function that renders a single frame. We'll scale this function out on Modal later.
#
# Functions in Modal are defined along with their hardware and their dependencies.
# This function can be run with GPU acceleration or without it, and we'll use a global flag in the code to switch between the two.

WITH_GPU = True  # try changing this to False to run rendering massively in parallel on CPUs!

# We decorate the function with `@app.function` to define it as a Modal function.
# Note that in addition to defining the hardware requirements of the function,
# we also specify the container image that the function runs in (the one we defined above).

# The details of the rendering function aren't too important for this example,
# so we abstract them out into functions defined at the end of the file.
# We draw a simple version of the Modal logo:
# two neon green rectangular prisms facing different directions.
# We include a parameter to rotate the prisms around the vertical/Z axis,
# which we'll use to animate the logo.


@app.function(
    gpu="A10G" if WITH_GPU else None,
    concurrency_limit=10
    if WITH_GPU
    else 100,  # default limits on Modal free tier
    image=rendering_image,
)
def render(angle: int = 0) -> bytes:
    """
    Renders Modal's logo, two neon green rectangular prisms.


    Args:
        angle: How much to rotate the two prisms around the vertical/Z axis, in degrees.

    Returns:
        The rendered frame as a PNG image.
    """
    import bpy

    # clear existing objects
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="MESH")
    bpy.ops.object.delete()

    # ctx: the current Blender state, which we mutate
    ctx = bpy.context

    # scene: the 3D environment we are rendering and its camera(s)
    scene = ctx.scene

    # configure rendering -- CPU or GPU, resolution, etc.
    # see function definition below for details
    configure_rendering(ctx, WITH_GPU)

    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = "output.png"

    # set background to black
    black = (0, 0, 0, 1)
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = black

    # add the Modal logo: two neon green rectangular prisms
    iridescent_material = create_iridescent_material()

    add_prism(ctx, (-2.07, -1, 0), 45, angle, iridescent_material)
    add_prism(ctx, (2.07, 1, 0), -45, angle, iridescent_material)

    # add lighting and camera
    add_lighting()
    bpy.ops.object.camera_add(location=(7, -7, 5))
    scene.camera = bpy.context.object
    ctx.object.rotation_euler = (1.1, 0, 0.785)

    # render
    bpy.ops.render.render(write_still=True)

    # return the bytes to the caller
    with open(scene.render.filepath, "rb") as image_file:
        image_bytes = image_file.read()

    return image_bytes


# ### Rendering with acceleration
#
# We can configure the rendering process to use GPU acceleration with NVIDIA CUDA.
# We select the [Cycles rendering engine](https://www.cycles-renderer.org/), which is compatible with CUDA,
# and then activate the GPU.


def configure_rendering(ctx, with_gpu: bool):
    # configure the rendering process
    ctx.scene.render.engine = "CYCLES"
    ctx.scene.render.resolution_x = 1920
    ctx.scene.render.resolution_y = 1080
    ctx.scene.render.resolution_percentage = 100
    ctx.scene.cycles.samples = 128

    # add GPU acceleration if available
    if with_gpu:
        ctx.preferences.addons[
            "cycles"
        ].preferences.compute_device_type = "CUDA"
        ctx.scene.cycles.device = "GPU"

        # reload the devices to update the configuration
        ctx.preferences.addons["cycles"].preferences.get_devices()
        for device in ctx.preferences.addons["cycles"].preferences.devices:
            device.use = True

    else:
        ctx.scene.cycles.device = "CPU"

    # report rendering devices -- a nice snippet for debugging and ensuring the accelerators are being used
    for dev in ctx.preferences.addons["cycles"].preferences.devices:
        print(
            f"ID:{dev['id']} Name:{dev['name']} Type:{dev['type']} Use:{dev['use']}"
        )


# ## Combining frames into a GIF
#
# Rendering 3D images is fun, and GPUs can make it faster, but rendering 3D videos is better!
# We add another function to our app, running on a different, simpler container image
# and different hardware, to combine the frames into a GIF.

combination_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "pillow==10.3.0"
)

# The video has a few parameters, which we set here.

FPS = 60
FRAME_DURATION_MS = 1000 // FPS
NUM_FRAMES = 360  # drop this for faster iteration while playing around

# The function to combine the frames into a GIF takes a sequence of byte sequences, one for each rendered frame,
# and converts them into a single sequence of bytes, the GIF.


@app.function(image=combination_image)
def combine(
    frames_bytes: list[bytes], frame_duration: int = FRAME_DURATION_MS
) -> bytes:
    print("🎞️ combining frames into a gif")
    from PIL import Image

    frames = [
        Image.open(io.BytesIO(frame_bytes)) for frame_bytes in frames_bytes
    ]

    gif_image = io.BytesIO()
    frames[0].save(
        gif_image,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=0,
    )

    gif_image.seek(0)

    return gif_image.getvalue()


# ## Rendering in parallel in the cloud from the comfort of the command line
#
# With these two functions defined, we need only a few more lines to run our rendering at scale on Modal.
#
# First, we need a function that coordinates our functions to `render` frames and `combine` them.
# We decorate that function with `@app.local_entrypoint` so that we can run it with `modal run blender_video.py`.
#
# In that function, we use `render.map` to map the `render` function over a `range` of `angle`s,
# so that the logo will appear to spin in the final video.
#
# We collect the bytes from each frame into a `list` locally and then send it to `combine` with `.remote`.
#
# The bytes for the video come back to our local machine, and we write them to a file.
#
# The whole rendering process (for six seconds of 1080p 60 FPS video) takes about five minutes to run on 10 A10G GPUs,
# with a per-frame latency of about 10 seconds, and about five minutes to run on 100 CPUs, with a per-frame latency of about one minute.


@app.local_entrypoint()
def main():
    output_directory = Path("/tmp") / "render"
    output_directory.mkdir(parents=True, exist_ok=True)
    filename = output_directory / "output.gif"
    with open(filename, "wb") as out_file:
        out_file.write(
            combine.remote(list(render.map(range(0, 360, 360 // NUM_FRAMES))))
        )
    print(f"Image saved to {filename}")


# ## Addenda
#
# The remainder of the code in this example defines the details of the render.
# It's not particularly interesting, so we put it the end of the file.


def add_prism(ctx, location, initial_rotation, angle, material):
    """Add a prism at a given location, rotation, and angle, made of the provided material."""
    import bpy
    import mathutils

    bpy.ops.mesh.primitive_cube_add(size=2, location=location)
    obj = ctx.object  # the newly created object

    bevel = obj.modifiers.new(name="Bevel", type="BEVEL")
    bevel.width = 0.2
    bevel.segments = 5
    bevel.profile = 1.0

    # assign the material to the object
    obj.data.materials.append(material)

    obj.scale = (1, 1, 2)  # square base, 2x taller than wide
    # Modal logo is rotated 45 degrees
    obj.rotation_euler[1] = math.radians(initial_rotation)

    # apply initial transformations
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # to "animate" the rendering, we rotate the prisms around the Z axis
    angle_radians = math.radians(angle)
    rotation_matrix = mathutils.Matrix.Rotation(angle_radians, 4, "Z")
    obj.matrix_world = rotation_matrix @ obj.matrix_world
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def create_iridescent_material():
    import bpy

    mat = bpy.data.materials.new(name="IridescentGreen")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()

    principled_node = nodes.new(type="ShaderNodeBsdfPrincipled")

    emission_node = nodes.new(type="ShaderNodeEmission")
    layer_weight = nodes.new(type="ShaderNodeLayerWeight")
    color_ramp = nodes.new(type="ShaderNodeValToRGB")

    mix_shader_node = nodes.new(type="ShaderNodeMixShader")

    output_node = nodes.new(type="ShaderNodeOutputMaterial")

    principled_node.inputs["Base Color"].default_value = (1, 1, 1, 1)
    principled_node.inputs["Metallic"].default_value = 1.0
    principled_node.inputs["Roughness"].default_value = 0.5

    color_ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
    color_ramp.color_ramp.elements[1].color = (0, 0.5, 0, 1)
    layer_weight.inputs["Blend"].default_value = 0.4

    links.new(layer_weight.outputs["Fresnel"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], emission_node.inputs["Color"])

    emission_node.inputs["Strength"].default_value = 5.0
    emission_node.inputs["Color"].default_value = (0.0, 1.0, 0.0, 1)

    links.new(emission_node.outputs["Emission"], mix_shader_node.inputs[1])
    links.new(principled_node.outputs["BSDF"], mix_shader_node.inputs[2])
    links.new(layer_weight.outputs["Fresnel"], mix_shader_node.inputs["Fac"])

    links.new(mix_shader_node.outputs["Shader"], output_node.inputs["Surface"])

    return mat


def add_lighting():
    import bpy

    # warm key light
    bpy.ops.object.light_add(type="POINT", location=(5, 5, 5))
    key_light = bpy.context.object
    key_light.data.energy = 100
    key_light.data.color = (1, 0.8, 0.5)  # warm

    # tight, cool spotlight
    bpy.ops.object.light_add(type="SPOT", radius=1, location=(4, 0, 6))
    spot_light = bpy.context.object
    spot_light.data.energy = 500
    spot_light.data.spot_size = 0.5
    spot_light.data.color = (0.8, 0.8, 1)  # cool
    spot_light.rotation_euler = (3.14 / 4, 0, -3.14 / 4)

    # soft overall illumination
    bpy.ops.object.light_add(type="AREA", radius=3, location=(-3, 3, 5))
    area_light = bpy.context.object
    area_light.data.energy = 50  # softer
    area_light.data.size = 5  # larger
    area_light.data.color = (1, 1, 1)  # neutral
    area_light.rotation_euler = (3.14 / 2, 0, 3.14)


---

## init



---

## comfy api

# ---
# lambda-test: false
# ---
#
# # Make API calls to a ComfyUI server
#
# This example shows you how to execute ComfyUI JSON-defined workflows via ComfyUI's API.
# It also provides a helper function `get_python_workflow`` that maps a JSON-defined workflow into Python objects.
# ![example comfyui workspace](./comfyui-hero.png)
import json
import os
import pathlib
import urllib
import uuid

import modal

comfyui_commit_sha = "a38b9b3ac152fb5679dad03813a93c09e0a4d15e"

# This workflow JSON has been exported by running `comfy_ui.py` and downloading the JSON
# using the web UI.
comfyui_workflow_data_path = assets_path = (
    pathlib.Path(__file__).parent / "workflow_api.json"
)

generated_workflow_path = (
    pathlib.Path(__file__).parent / "_generated_workflow_api.py"
)

app = modal.App(
    name="example-comfy-api"
)  # Note: prior to April 2024, "app" was called "stub"
from .comfy_ui import image


def fetch_image(
    filename: str, subfolder: str, folder_type: str, server_address: str
) -> bytes:
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(
        "https://{}/view?{}".format(server_address, url_values)
    ) as response:
        return response.read()


def run_workflow(
    ws, prompt: str, server_address: str, client_id: str
) -> list[bytes]:
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")
    req = urllib.request.Request(
        "https://{}/prompt".format(server_address), data=data
    )
    response_data = json.loads(urllib.request.urlopen(req).read())
    prompt_id = response_data["prompt_id"]
    output_images = {}

    while True:
        out = ws.recv()
        if isinstance(out, str):
            print(f"recieved str msg from websocket. ws msg: {out}")
            try:
                message = json.loads(out)
            except json.JSONDecodeError:
                print(f"expected valid JSON but got: {out}")
                raise
            print(f"received msg from ws: {message}")
            if message["type"] == "executing":
                data = message["data"]
                if data["node"] is None and data["prompt_id"] == prompt_id:
                    break  # Execution is done!
        else:
            continue  # previews are binary data

    # Fetch workflow execution history, which contains references to our completed images.
    with urllib.request.urlopen(
        f"https://{server_address}/history/{prompt_id}"
    ) as response:
        output = json.loads(response.read())
    history = output[prompt_id].get("outputs") if prompt_id in output else None
    if not history:
        raise RuntimeError(
            f"Unexpected missing ComfyUI history for {prompt_id}"
        )
    for node_id in history:
        node_output = history[node_id]
        if "images" in node_output:
            images_output = []
            for image in node_output["images"]:
                image_data = fetch_image(
                    filename=image["filename"],
                    subfolder=image["subfolder"],
                    folder_type=image["type"],
                    server_address=server_address,
                )
                images_output.append(image_data)
        output_images[node_id] = images_output

    return output_images


# Execute a run of a JSON-defined workflow on a remote ComfyUI server
# This is adapted from the ComfyUI script examples: https://github.com/comfyanonymous/ComfyUI/blob/master/script_examples/websockets_api_example.py
# A better way to execute a workflow programmatically is to convert the JSON to Python code using convert_workflow_to_python
# Then importing that generated code into a Modal endpoint; see serve_workflow.py
@app.function(image=image)
def query_comfy_via_api(workflow_data: dict, prompt: str, server_address: str):
    import websocket

    # Modify workflow to use requested prompt.
    workflow_data["2"]["inputs"]["text"] = prompt

    # Make a websocket connection to the ComfyUI server. The server will
    # will stream workflow execution updates over this websocket.
    ws = websocket.WebSocket()
    client_id = str(uuid.uuid4())
    ws_address = f"wss://{server_address}/ws?clientId={client_id}"
    print(f"Connecting to websocket at {ws_address} ...")
    ws.connect(ws_address)
    print(f"Connected at {ws_address}. Running workflow via API")
    images = run_workflow(ws, workflow_data, server_address, client_id)
    image_list = []
    for node_id in images:
        for image_data in images[node_id]:
            image_list.append(image_data)
    return image_list


@app.function(
    image=image,
    gpu="any",
)
def convert_workflow_to_python(workflow: str):
    pathlib.Path("/root/workflow_api.json").write_text(workflow)

    import subprocess

    process = subprocess.Popen(
        ["python", "./ComfyUI-to-Python-Extension/comfyui_to_python.py"]
    )
    process.wait()
    retcode = process.returncode

    if retcode != 0:
        raise RuntimeError(
            f"comfy_api.py exited unexpectedly with code {retcode}"
        )
    else:
        try:
            return pathlib.Path("workflow_api.py").read_text()
        except FileNotFoundError:
            print("Error: File workflow_api.py not found.")


# Generate a Python representation of workflow_api.json using this extension: https://github.com/pydn/ComfyUI-to-Python-Extension
# First, you need to download your workflow_api.json from ComfyUI and save it to this directory.
# Then, this function will generate a Python version to _generated_workflow_api.py, which you'll reference in workflow_api.py.
@app.local_entrypoint()
def get_python_workflow():
    workflow_text = convert_workflow_to_python.remote(
        pathlib.Path(comfyui_workflow_data_path).read_text()
    )
    pathlib.Path(generated_workflow_path).write_text(workflow_text)
    print(f"saved '{generated_workflow_path}'")


@app.local_entrypoint()
def main(prompt: str = "bag of wooden blocks") -> None:
    workflow_data = json.loads(comfyui_workflow_data_path.read_text())

    # Run the ComfyUI server app and make an API call to it.
    # The ComfyUI server app will shutdown on exit of this context manager.
    from comfy_ui import app as comfyui_app

    with comfyui_app.run(
        show_progress=False,  # hide server app's modal progress logs
        stdout=open(os.devnull, "w"),  # hide server app's application logs
    ) as comfyui_app:
        print(f"{comfyui_app.app_id=}")
        comfyui_url = comfyui_app.web.web_url

        server_address = comfyui_url.split("://")[1]  # strip protocol

        image_list = query_comfy_via_api.remote(
            workflow_data=workflow_data,
            prompt=prompt,
            server_address=server_address,
        )

    for i, img_bytes in enumerate(image_list):
        filename = f"comfyui_{i}.png"
        with open(filename, "wb") as f:
            f.write(img_bytes)
            f.close()
        print(f"saved '{filename}'")


---

## comfy ui

# ---
# lambda-test: false
# ---
#
# # Run ComfyUI
#
# This example shows you how to run a ComfyUI workspace with `modal serve`.
#
# If you're unfamiliar with how ComfyUI works we recommend going through Scott Detweiler's
# [tutorials on Youtube](https://www.youtube.com/watch?v=AbB33AxrcZo).
#
# ![example comfyui workspace](./comfyui-hero.png)

import pathlib
import subprocess

import modal

# ## Define container image
#
# Fun with ComfyUI begins with pre-trained model checkpoints.
# Add downloadable checkpoints to CHECKPOINTS e.g. [huggingface.co/dreamlike-art/dreamlike-photoreal-2.0](https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0).
# The ComfyUI repository has other recommendations listed in this file:
# [notebooks/comfyui_colab.ipynb](https://github.com/comfyanonymous/ComfyUI/blob/master/notebooks/comfyui_colab.ipynb).
CHECKPOINTS = [
    "https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/resolve/main/512-inpainting-ema.ckpt"
]


def download_checkpoints():
    import httpx
    from tqdm import tqdm

    for url in CHECKPOINTS:
        checkpoints_directory = "/root/models/checkpoints"
        local_filename = url.split("/")[-1]
        local_filepath = pathlib.Path(checkpoints_directory, local_filename)
        local_filepath.parent.mkdir(parents=True, exist_ok=True)

        print(f"downloading {url} ...")
        with httpx.stream("GET", url, follow_redirects=True) as stream:
            total = int(stream.headers["Content-Length"])
            with open(local_filepath, "wb") as f, tqdm(
                total=total, unit_scale=True, unit_divisor=1024, unit="B"
            ) as progress:
                num_bytes_downloaded = stream.num_bytes_downloaded
                for data in stream.iter_bytes():
                    f.write(data)
                    progress.update(
                        stream.num_bytes_downloaded - num_bytes_downloaded
                    )
                    num_bytes_downloaded = stream.num_bytes_downloaded


# Add plugins to PLUGINS, a list of dictionaries with two keys:
# `url` for the github url and an optional `requirements` for the name of a requirements.txt to pip install (remove this key if there is none for the plugin).
# For recommended plugins, see this list:
# [WASasquatch/comfyui-plugins](https://github.com/WASasquatch/comfyui-plugins).
PLUGINS = [
    {
        "url": "https://github.com/coreyryanhanson/ComfyQR",
        "requirements": "requirements.txt",
    }
]


def download_plugins():
    import subprocess

    for plugin in PLUGINS:
        url = plugin["url"]
        name = url.split("/")[-1]
        command = f"cd /root/custom_nodes && git clone {url}"
        try:
            subprocess.run(command, shell=True, check=True)
            print(f"Repository {url} cloned successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository: {e.stderr}")
        if plugin.get("requirements"):
            pip_command = f"cd /root/custom_nodes/{name} && pip install -r {plugin['requirements']}"
        try:
            subprocess.run(pip_command, shell=True, check=True)
            print(f"Requirements for {url} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error installing requirements: {e.stderr}")


# Pin to a specific commit from https://github.com/comfyanonymous/ComfyUI/commits/master/
# for stability. To update to a later ComfyUI version, change this commit identifier.
comfyui_commit_sha = "a38b9b3ac152fb5679dad03813a93c09e0a4d15e"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    # Here we place the latest ComfyUI repository code into /root.
    # Because /root is almost empty, but not entirely empty
    # as it contains this comfy_ui.py script, `git clone` won't work.
    # As a workaround we `init` inside the non-empty directory, then `checkout`.
    .run_commands(
        "cd /root && git init .",
        "cd /root && git remote add --fetch origin https://github.com/comfyanonymous/ComfyUI",
        f"cd /root && git checkout {comfyui_commit_sha}",
        "cd /root && pip install xformers!=0.0.18 -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121",
        "cd /root && git clone https://github.com/pydn/ComfyUI-to-Python-Extension.git",
        "cd /root/ComfyUI-to-Python-Extension && pip install -r requirements.txt",
    )
    .pip_install(
        "httpx",
        "requests",
        "tqdm",
    )
    .run_function(download_checkpoints)
    .run_function(download_plugins)
)
app = modal.App(
    name="example-comfy-ui", image=image
)  # Note: prior to April 2024, "app" was called "stub"

# ## Start the ComfyUI server
#
# Inside the container, we will run the ComfyUI server and execution queue on port 8188. Then, we
# wrap this function in the `@web_server` decorator to expose the server as a web endpoint.
#
# For ASGI-compatible frameworks, you can also use Modal's `@asgi_app` decorator.


@app.function(
    gpu="any",
    # Allows 100 concurrent requests per container.
    allow_concurrent_inputs=100,
    # Restrict to 1 container because we want our ComfyUI session state
    # to be on a single container.
    concurrency_limit=1,
    keep_warm=1,
    timeout=1800,
)
@modal.web_server(8188, startup_timeout=30)
def web():
    cmd = "python main.py --dont-print-server --multi-user --listen --port 8188"
    subprocess.Popen(cmd, shell=True)


---

## workflow api

# ---
# lambda-test: false
# ---
#
# # Run a ComfyUI workflow in Python
#
# This example serves a ComfyUI [inpainting workflow](https://github.com/comfyanonymous/ComfyUI_examples/tree/master/inpaint) as an endpoint.
# ![example comfyui workspace](./comfyui-hero.png)
import pathlib
import random
from typing import Any, Dict, Mapping, Sequence, Union

from fastapi.responses import HTMLResponse
from modal import App, Volume, web_endpoint

from .comfy_ui import image

app = App(
    name="example-comfy-python-api"
)  # Note: prior to April 2024, "app" was called "stub"
vol_name = "comfyui-images"
vol = Volume.from_name(vol_name, create_if_missing=True)


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


# ComfyUI images expect input images to be saved in the /input directory
def download_image(url, save_path="/root/input/"):
    import requests

    try:
        response = requests.get(url)
        response.raise_for_status()
        pathlib.Path(save_path + url.split("/")[-1]).write_bytes(
            response.content
        )
        print(f"{url} image successfully downloaded")

    except Exception as e:
        print(f"Error downloading {url} image: {e}")


# Adapted from main() in `_generated_workflow_api.py` after running modal run comfyui.comfy_api::get_python_workflow
def run_python_workflow(item: Dict):
    # In the generated version, these are in the global scope, but for Modal we move into the function scope
    import torch
    from nodes import (
        CheckpointLoaderSimple,
        CLIPTextEncode,
        KSampler,
        LoadImage,
        SaveImage,
        VAEDecode,
        VAEEncodeForInpaint,
    )

    download_image(item["image"])
    with torch.inference_mode():
        loadimage = LoadImage()
        loadimage_1 = loadimage.load_image(image=item["image"].split("/")[-1])

        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_2 = checkpointloadersimple.load_checkpoint(
            ckpt_name="512-inpainting-ema.ckpt"
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_3 = cliptextencode.encode(
            text=f"closeup photograph of a {item['prompt']} in the yosemite national park mountains nature",
            clip=get_value_at_index(checkpointloadersimple_2, 1),
        )

        cliptextencode_5 = cliptextencode.encode(
            text="watermark, text",
            clip=get_value_at_index(checkpointloadersimple_2, 1),
        )

        vaeencodeforinpaint = VAEEncodeForInpaint()
        vaeencodeforinpaint_9 = vaeencodeforinpaint.encode(
            grow_mask_by=6,
            pixels=get_value_at_index(loadimage_1, 0),
            vae=get_value_at_index(checkpointloadersimple_2, 2),
            mask=get_value_at_index(loadimage_1, 1),
        )

        ksampler = KSampler()
        vaedecode = VAEDecode()
        saveimage = SaveImage()

        for q in range(10):
            ksampler_6 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=20,
                cfg=8,
                sampler_name="uni_pc_bh2",
                scheduler="normal",
                denoise=1,
                model=get_value_at_index(checkpointloadersimple_2, 0),
                positive=get_value_at_index(cliptextencode_3, 0),
                negative=get_value_at_index(cliptextencode_5, 0),
                latent_image=get_value_at_index(vaeencodeforinpaint_9, 0),
            )

            vaedecode_7 = vaedecode.decode(
                samples=get_value_at_index(ksampler_6, 0),
                vae=get_value_at_index(checkpointloadersimple_2, 2),
            )

            saveimage_8 = saveimage.save_images(
                filename_prefix="ComfyUI",
                images=get_value_at_index(vaedecode_7, 0),
            )

        return saveimage_8


# Serves the python workflow behind a web endpoint
# Generated images are written to a Volume
@app.function(image=image, gpu="any", volumes={"/data": vol})
@web_endpoint(method="POST")
def serve_workflow(item: Dict):
    saved_image = run_python_workflow(item)
    images = saved_image["ui"]["images"]

    for i in images:
        filename = "output/" + i["filename"]
        with open(f'/data/{i["filename"]}', "wb") as f:
            f.write(pathlib.Path(filename).read_bytes())
        vol.commit()

    return HTMLResponse(f"<html>Image saved at volume {vol_name}! </html>")


# Run the workflow as a function rather than an endpoint (for easier local testing)
@app.function(image=image, gpu="any")
def run_workflow(item: Dict):
    saved_image = run_python_workflow(item)
    images = saved_image["ui"]["images"]
    image_list = []

    for i in images:
        filename = "output/" + i["filename"]
        image_list.append(pathlib.Path(filename).read_bytes())
    return image_list


@app.local_entrypoint()
def main() -> None:
    values = {
        "prompt": "white heron",
        "image": "https://raw.githubusercontent.com/comfyanonymous/ComfyUI_examples/master/inpaint/yosemite_inpaint_example.png",
    }
    image_list = run_workflow.remote(values)
    for i, img_bytes in enumerate(image_list):
        filename = f"comfyui_{i}.png"
        with open(filename, "wb") as f:
            f.write(img_bytes)
            f.close()
        print(f"saved '{filename}'")


---

## controlnet gradio demos

# ---
# cmd: ["modal", "serve", "06_gpu_and_ml/controlnet/controlnet_gradio_demos.py"]
# deploy: false
# ---
#
# # Play with the ControlNet demos
#
# This example allows you to play with all 10 demonstration Gradio apps from the new and amazing ControlNet project.
# ControlNet provides a minimal interface allowing users to use images to constrain StableDiffusion's generation process.
# With ControlNet, users can easily condition the StableDiffusion image generation with different spatial contexts
# including a depth maps, segmentation maps, scribble drawings, and keypoints!
#
# <center>
# <video controls>
# <source src="https://user-images.githubusercontent.com/12058921/222927911-3ab52dd1-f2ee-4fb8-97e8-dafbf96ed5c5.mp4" type="video/mp4">
# </video>
# </center>
#
# ## Imports and config preamble

import importlib
import os
import pathlib
from dataclasses import dataclass, field

from fastapi import FastAPI
from modal import App, Image, Secret, asgi_app

# Below are the configuration objects for all **10** demos provided in the original [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet) repo.
# The demos each depend on their own custom pretrained StableDiffusion model, and these models are 5-6GB each.
# We can only run one demo at a time, so this module avoids downloading the model and 'detector' dependencies for
# all 10 demos and instead uses the demo configuration object to download only what's necessary for the chosen demo.
#
# Even just limiting our dependencies setup to what's required for one demo, the resulting container image is *huge*.


@dataclass(frozen=True)
class DemoApp:
    """Config object defining a ControlNet demo app's specific dependencies."""

    name: str
    model_files: list[str]
    detector_files: list[str] = field(default_factory=list)


demos = [
    DemoApp(
        name="canny2image",
        model_files=[
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_canny.pth"
        ],
    ),
    DemoApp(
        name="depth2image",
        model_files=[
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_depth.pth"
        ],
        detector_files=[
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt"
        ],
    ),
    DemoApp(
        name="fake_scribble2image",
        model_files=[
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_scribble.pth"
        ],
        detector_files=[
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/network-bsds500.pth"
        ],
    ),
    DemoApp(
        name="hed2image",
        model_files=[
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_hed.pth"
        ],
        detector_files=[
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/network-bsds500.pth"
        ],
    ),
    DemoApp(
        name="hough2image",
        model_files=[
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_mlsd.pth"
        ],
        detector_files=[
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/mlsd_large_512_fp32.pth",
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/mlsd_tiny_512_fp32.pth",
        ],
    ),
    DemoApp(
        name="normal2image",
        model_files=[
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_normal.pth"
        ],
    ),
    DemoApp(
        name="pose2image",
        model_files=[
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_openpose.pth"
        ],
        detector_files=[
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/body_pose_model.pth",
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/hand_pose_model.pth",
        ],
    ),
    DemoApp(
        name="scribble2image",
        model_files=[
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_scribble.pth"
        ],
    ),
    DemoApp(
        name="scribble2image_interactive",
        model_files=[
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_scribble.pth"
        ],
    ),
    DemoApp(
        name="seg2image",
        model_files=[
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_seg.pth"
        ],
        detector_files=[
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/upernet_global_small.pth"
        ],
    ),
]
demos_map: dict[str, DemoApp] = {d.name: d for d in demos}

# ## Pick a demo, any demo
#
# Simply by changing the `DEMO_NAME` below, you can change which ControlNet demo app is setup
# and run by this Modal script.

DEMO_NAME = "scribble2image"  # Change this value to change the active demo app.
selected_demo = demos_map[DEMO_NAME]

# ## Setting up the dependencies
#
# ControlNet requires *a lot* of dependencies which could be fiddly to setup manually, but Modal's programmatic
# container image building Python APIs handle this complexity straightforwardly and automatically.
#
# To run any of the 10 demo apps, we need the following:
#
# 1. a base Python 3 Linux image (we use Debian Slim)
# 2. a bunch of third party PyPi packages
# 3. `git`, so that we can download the ControlNet source code (there's no `controlnet` PyPi package)
# 4. some image process Linux system packages, including `ffmpeg`
# 5. and demo specific pre-trained model and detector `.pth` files
#
# That's a lot! Fortunately, the code below is already written for you that stitches together a working container image
# ready to produce remarkable ControlNet images.
#
# **Note:** a ControlNet model pipeline is [now available in Huggingface's `diffusers` package](https://huggingface.co/blog/controlnet). But this does not contain the demo apps.


def download_file(url: str, output_path: pathlib.Path):
    import httpx
    from tqdm import tqdm

    with open(output_path, "wb") as download_file:
        with httpx.stream("GET", url, follow_redirects=True) as response:
            total = int(response.headers["Content-Length"])
            with tqdm(
                total=total, unit_scale=True, unit_divisor=1024, unit="B"
            ) as progress:
                num_bytes_downloaded = response.num_bytes_downloaded
                for chunk in response.iter_bytes():
                    download_file.write(chunk)
                    progress.update(
                        response.num_bytes_downloaded - num_bytes_downloaded
                    )
                    num_bytes_downloaded = response.num_bytes_downloaded


def download_demo_files() -> None:
    """
    The ControlNet repo instructs: 'Make sure that SD models are put in "ControlNet/models".'
    'ControlNet' is just the repo root, so we place in /root/models.

    The ControlNet repo also instructs: 'Make sure that... detectors are put in "ControlNet/annotator/ckpts".'
    'ControlNet' is just the repo root, so we place in /root/annotator/ckpts.
    """
    demo = demos_map[os.environ["DEMO_NAME"]]
    models_dir = pathlib.Path("/root/models")
    for url in demo.model_files:
        filepath = pathlib.Path(url).name
        download_file(url=url, output_path=models_dir / filepath)
        print(f"download complete for {filepath}")

    detectors_dir = pathlib.Path("/root/annotator/ckpts")
    for url in demo.detector_files:
        filepath = pathlib.Path(url).name
        download_file(url=url, output_path=detectors_dir / filepath)
        print(f"download complete for {filepath}")
    print("🎉 finished baking demo file(s) into image.")


image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "gradio==3.16.2",
        "albumentations==1.3.0",
        "opencv-contrib-python",
        "imageio==2.9.0",
        "imageio-ffmpeg==0.4.2",
        "pytorch-lightning==1.5.0",
        "omegaconf==2.1.1",
        "test-tube>=0.7.5",
        "streamlit==1.12.1",
        "einops==0.3.0",
        "transformers==4.19.2",
        "webdataset==0.2.5",
        "kornia==0.6",
        "open_clip_torch==2.0.2",
        "invisible-watermark>=0.1.5",
        "streamlit-drawable-canvas==0.8.0",
        "torchmetrics==0.6.0",
        "timm==0.6.12",
        "addict==2.4.0",
        "yapf==0.32.0",
        "prettytable==3.6.0",
        "safetensors==0.2.7",
        "basicsr==1.4.2",
        "tqdm~=4.64.1",
    )
    # xformers library offers performance improvement.
    .pip_install("xformers", pre=True)
    .apt_install("git")
    # Here we place the latest ControlNet repository code into /root.
    # Because /root is almost empty, but not entirely empty, `git clone` won't work,
    # so this `init` then `checkout` workaround is used.
    .run_commands(
        "cd /root && git init .",
        "cd /root && git remote add --fetch origin https://github.com/lllyasviel/ControlNet.git",
        "cd /root && git checkout main",
    )
    .apt_install("ffmpeg", "libsm6", "libxext6")
    .run_function(
        download_demo_files,
        secrets=[Secret.from_dict({"DEMO_NAME": DEMO_NAME})],
    )
)
app = App(
    name="example-controlnet", image=image
)  # Note: prior to April 2024, "app" was called "stub"

web_app = FastAPI()

# ## Serving the Gradio web UI
#
# Each ControlNet gradio demo module exposes a `block` Gradio interface running in queue-mode,
# which is initialized in module scope on import and served on `0.0.0.0`. We want the block interface object,
# but the queueing and launched webserver aren't compatible with Modal's serverless web endpoint interface,
# so in the `import_gradio_app_blocks` function we patch out these behaviors.


def import_gradio_app_blocks(demo: DemoApp):
    from gradio import blocks

    # The ControlNet repo demo scripts are written to be run as
    # standalone scripts, and have a lot of code that executes
    # in global scope on import, including the launch of a Gradio web server.
    # We want Modal to control the Gradio web app serving, so we
    # monkeypatch the .launch() function to be a no-op.
    blocks.Blocks.launch = lambda self, server_name: print(
        "launch() has been monkeypatched to do nothing."
    )

    # each demo app module is a file like gradio_{name}.py
    module_name = f"gradio_{demo.name}"
    mod = importlib.import_module(module_name)
    blocks = mod.block
    # disable queueing mode, which is incompatible with our Modal web app setup.
    blocks.enable_queue = False
    return blocks


# Because the ControlNet gradio apps are so time and compute intensive to cold-start,
# the web app function is limited to running just 1 warm container (concurrency_limit=1).
# This way, while playing with the demos we can pay the cold-start cost once and have
# all web requests hit the same warm container.
# Spinning up extra containers to handle additional requests would not be efficient
# given the cold-start time.
# We set the container_idle_timeout to 600 seconds so the container will be kept
# running for 10 minutes after the last request, to keep the app responsive in case
# of continued experimentation.


@app.function(
    gpu="A10G",
    concurrency_limit=1,
    container_idle_timeout=600,
)
@asgi_app()
def run():
    from gradio.routes import mount_gradio_app

    # mount for execution on Modal
    return mount_gradio_app(
        app=web_app,
        blocks=import_gradio_app_blocks(demo=selected_demo),
        path="/",
    )


# ## Have fun!
#
# Serve your chosen demo app with `modal serve controlnet_gradio_demos.py`. If you don't have any images ready at hand,
# try one that's in the `06_gpu_and_ml/controlnet/demo_images/` folder.
#
# StableDiffusion was already impressive enough, but ControlNet's ability to so accurately and intuitively constrain
# the image generation process is sure to put a big, dumb grin on your face.


---

## dreambooth app

# ---
# deploy: true
# ---
#
# # Pet Art Dreambooth with Hugging Face and Gradio
#
# This example finetunes the [Stable Diffusion XL model](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
# on images of a pet (by default, a puppy named Qwerty)
# using a technique called textual inversion from [the "Dreambooth" paper](https://dreambooth.github.io/).
# Effectively, it teaches a general image generation model a new "proper noun",
# allowing for the personalized generation of art and photos.
#
# It then makes the model shareable with others -- without costing $25/day for a GPU server--
# by hosting a [Gradio app](https://gradio.app/) on Modal.
#
# It demonstrates a simple, productive, and cost-effective pathway
# to building on large pretrained models using Modal's building blocks, like
# [GPU-accelerated](https://modal.com/docs/guide/gpu) Modal functions and classes for compute-intensive work,
# [volumes](https://modal.com/docs/guide/volumes) for storage,
# and [web endpoints](https://modal.com/docs/guide/webhooks) for serving.
#
# And with some light customization, you can use it to generate images of your pet!
#
# ![Gradio.app image generation interface](./gradio-image-generate.png)
#
# ## Imports and setup
#
# We start by importing the necessary libraries and setting up the environment.
# By installing Modal, we already brought in the FastAPI library we'll use to serve our app,
# so we import it here.

from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from modal import (
    App,
    Image,
    Mount,
    Secret,
    Volume,
    asgi_app,
    enter,
    method,
)

# ## Building up the environment
#
# Machine learning environments are complex, and the dependencies can be hard to manage.
# Modal makes creating and working with environments easy via containers and container images.
#
# We start from a base image and specify all of our dependencies.
# We'll call out the interesting ones as they come up below.
# Note that these dependencies are not installed locally
# -- they are only installed in the remote environment where our app runs.

app = App(
    name="example-dreambooth-app"
)  # Note: prior to April 2024, "app" was called "stub"

image = Image.debian_slim(python_version="3.10").pip_install(
    "accelerate==0.27.2",
    "datasets~=2.13.0",
    "ftfy~=6.1.0",
    "gradio~=3.50.2",
    "smart_open~=6.4.0",
    "transformers~=4.38.1",
    "torch~=2.2.0",
    "torchvision~=0.16",
    "triton~=2.2.0",
    "peft==0.7.0",
    "wandb==0.16.3",
)

# ### Downloading scripts and installing a git repo with `run_commands`
#
# We'll use an example script from the `diffusers` library to train the model.
# We acquire it from GitHub and install it in our environment with a series of commands.
# The container environments Modal functions run in are highly flexible --
# see [the docs](https://modal.com/docs/guide/custom-container) for more details.

GIT_SHA = (
    "abd922bd0c43a504e47eca2ed354c3634bd00834"  # specify the commit to fetch
)

image = (
    image.apt_install("git")
    # Perform a shallow fetch of just the target `diffusers` commit, checking out
    # the commit in the container's home directory, /root. Then install `diffusers`
    .run_commands(
        "cd /root && git init .",
        "cd /root && git remote add origin https://github.com/huggingface/diffusers",
        f"cd /root && git fetch --depth=1 origin {GIT_SHA} && git checkout {GIT_SHA}",
        "cd /root && pip install -e .",
    )
)
# ### Configuration with `dataclass`es
#
# Machine learning apps often have a lot of configuration information.
# We collect up all of our configuration into dataclasses to avoid scattering special/magic values throughout code.


@dataclass
class SharedConfig:
    """Configuration information shared across project components."""

    # The instance name is the "proper noun" we're teaching the model
    instance_name: str = "Qwerty"
    # That proper noun is usually a member of some class (person, bird),
    # and sharing that information with the model helps it generalize better.
    class_name: str = "Golden Retriever"
    # identifier for pretrained models on Hugging Face
    model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"
    vae_name: str = "madebyollin/sdxl-vae-fp16-fix"  # required for numerical stability in fp16


# ### Downloading weights with `run_function`
#
# Not everything we need for an ML app like Pet Dreambooth is available as a Python package
# or even on GitHub. Sometimes, there is nothing to be done but to execute some code inside the environment.
# We can do this on Modal with `run_function`.
#
# In our case, we use it to download the pretrained model weights for the Stable Diffusion XL model
# that we'll be finetuning.


def download_models():
    import torch
    from diffusers import AutoencoderKL, DiffusionPipeline
    from transformers.utils import move_cache

    config = SharedConfig()

    DiffusionPipeline.from_pretrained(
        config.model_name,
        vae=AutoencoderKL.from_pretrained(
            config.vae_name, torch_dtype=torch.float16
        ),
        torch_dtype=torch.float16,
    )
    move_cache()


image = image.run_function(download_models)


# ### Storing data generated by our app with `modal.Volume`
#
# The tools we've used so far work well for fetching external information,
# which defines the environment our app runs in,
# but what about data that we create or modify during the app's execution?
# A persisted `modal.Volume` can store and share data across Modal apps or runs of the same app.
#
# We'll use one to store the fine-tuned weights we create during training
# and then load them back in for inference.

volume = Volume.from_name(
    "dreambooth-finetuning-volume", create_if_missing=True
)
MODEL_DIR = "/model"


# ### Load finetuning dataset
#
# Part of the magic of the Dreambooth approach is that we only need 3-10 images for finetuning.
# So we can fetch just a few images, stored on consumer platforms like Imgur or Google Drive,
# whenever we need them -- no need for expensive, hard-to-maintain data pipelines.


def load_images(image_urls: list[str]) -> Path:
    import PIL.Image
    from smart_open import open

    img_path = Path("/img")

    img_path.mkdir(parents=True, exist_ok=True)
    for ii, url in enumerate(image_urls):
        with open(url, "rb") as f:
            image = PIL.Image.open(f)
            image.save(img_path / f"{ii}.png")
    print(f"{ii + 1} images loaded")

    return img_path


# ## Finetuning a text-to-image model
#
# The base model we start from is trained to do a sort of "reverse [ekphrasis](https://en.wikipedia.org/wiki/Ekphrasis)":
# it attempts to recreate a visual work of art or image from only its description.
#
# We can use the model to synthesize wholly new images
# by combining the concepts it has learned from the training data.
#
# We use a pretrained model, the XL version of Stability AI's Stable Diffusion.
# In this example, we "finetune" SDXL, making only small adjustments to the weights.
# Furthermore, we don't change all the weights in the model.
# Instead, using a technique called [_low-rank adaptation_](https://arxiv.org/abs/2106.09685),
# we change a much smaller matrix that works "alongside" the existing weights, nudging the model in the direction we want.
#
# We can get away with such a small and simple training process because we're just teach the model the meaning of a single new word: the name of our pet.
#
# The result is a model that can generate novel images of our pet:
# as an astronaut in space, as painted by Van Gogh or Bastiat, etc.
#
# ### Finetuning with Hugging Face 🧨 Diffusers and Accelerate
#
# The model weights, training libraries, and training script are all provided by [🤗 Hugging Face](https://huggingface.co).
#
# You can kick off a training job with the command `modal run dreambooth_app.py::app.train`.
# It should take under five minutes.
#
# Training machine learning models takes time and produces a lot of metadata --
# metrics for performance and resource utilization,
# metrics for model quality and training stability,
# and model inputs and outputs like images and text.
# This is especially important if you're fiddling around with the configuration parameters.
#
# This example can optionally use [Weights & Biases](https://wandb.ai) to track all of this training information.
# Just sign up for an account, switch the flag below, and add your API key as a [Modal secret](https://modal.com/docs/guide/secrets).

USE_WANDB = False

# You can see an example W&B dashboard [here](https://wandb.ai/cfrye59/dreambooth-lora-sd-xl).
# Check out [this run](https://wandb.ai/cfrye59/dreambooth-lora-sd-xl/runs/ca3v1lsh?workspace=user-cfrye59),
# which [despite having high GPU utilization](https://wandb.ai/cfrye59/dreambooth-lora-sd-xl/runs/ca3v1lsh/system)
# suffered from numerical instability during training and produced only black images -- hard to debug without experiment management logs!
#
# You can read more about how the values in `TrainConfig` are chosen and adjusted [in this blog post on Hugging Face](https://huggingface.co/blog/dreambooth).
# To run training on images of your own pet, upload the images to separate URLs and edit the contents of the file at `TrainConfig.instance_example_urls_file` to point to them.
#
# Tip: if the results you're seeing don't match the prompt too well, and instead produce an image
# of your subject without taking the prompt into account, the model has likely overfit. In this case, repeat training with a lower
# value of `max_train_steps`. If you used W&B, look back at results earlier in training to determine where to stop.
# On the other hand, if the results don't look like your subject, you might need to increase `max_train_steps`.


@dataclass
class TrainConfig(SharedConfig):
    """Configuration for the finetuning step."""

    # training prompt looks like `{PREFIX} {INSTANCE_NAME} the {CLASS_NAME} {POSTFIX}`
    prefix: str = "a photo of"
    postfix: str = ""

    # locator for plaintext file with urls for images of target instance
    instance_example_urls_file: str = str(
        Path(__file__).parent / "instance_example_urls.txt"
    )

    # Hyperparameters/constants from the huggingface training example
    resolution: int = 1024
    train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    max_train_steps: int = 80
    checkpointing_steps: int = 1000
    seed: int = 117


@app.function(
    image=image,
    gpu="A100",  # fine-tuning is VRAM-heavy and requires an A100 GPU
    volumes={MODEL_DIR: volume},  # stores fine-tuned model
    timeout=1800,  # 30 minutes
    secrets=[Secret.from_name("my-wandb-secret")] if USE_WANDB else [],
)
def train(instance_example_urls):
    import subprocess

    from accelerate.utils import write_basic_config

    config = TrainConfig()

    # load data locally
    img_path = load_images(instance_example_urls)

    # set up hugging face accelerate library for fast training
    write_basic_config(mixed_precision="fp16")

    # define the training prompt
    instance_phrase = f"{config.instance_name} the {config.class_name}"
    prompt = f"{config.prefix} {instance_phrase} {config.postfix}".strip()

    # the model training is packaged as a script, so we have to execute it as a subprocess, which adds some boilerplate
    def _exec_subprocess(cmd: list[str]):
        """Executes subprocess and prints log to terminal while subprocess is running."""
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        with process.stdout as pipe:
            for line in iter(pipe.readline, b""):
                line_str = line.decode()
                print(f"{line_str}", end="")

        if exitcode := process.wait() != 0:
            raise subprocess.CalledProcessError(exitcode, "\n".join(cmd))

    # run training -- see huggingface accelerate docs for details
    print("launching dreambooth training script")
    _exec_subprocess(
        [
            "accelerate",
            "launch",
            "examples/dreambooth/train_dreambooth_lora_sdxl.py",
            "--mixed_precision=fp16",  # half-precision floats most of the time for faster training
            f"--pretrained_model_name_or_path={config.model_name}",
            f"--pretrained_vae_model_name_or_path={config.vae_name}",  # required for numerical stability in fp16
            f"--instance_data_dir={img_path}",
            f"--output_dir={MODEL_DIR}",
            f"--instance_prompt={prompt}",
            f"--resolution={config.resolution}",
            f"--train_batch_size={config.train_batch_size}",
            f"--gradient_accumulation_steps={config.gradient_accumulation_steps}",
            f"--learning_rate={config.learning_rate}",
            f"--lr_scheduler={config.lr_scheduler}",
            f"--lr_warmup_steps={config.lr_warmup_steps}",
            f"--max_train_steps={config.max_train_steps}",
            f"--checkpointing_steps={config.checkpointing_steps}",
            f"--seed={config.seed}",  # increased reproducibility by seeding the RNG
        ]
        + (
            [
                "--report_to=wandb",
                f"--validation_prompt={prompt} in space",  # simple test prompt
                f"--validation_epochs={config.max_train_steps // 5}",
            ]
            if USE_WANDB
            else []
        ),
    )
    # The trained model information has been output to the volume mounted at `MODEL_DIR`.
    # To persist this data for use in our web app, we 'commit' the changes
    # to the volume.
    volume.commit()


# ## Running our model
#
# To generate images from prompts using our fine-tuned model, we define a Modal function called `inference`.
#
# Naively, this would seem to be a bad fit for the flexible, serverless infrastructure of Modal:
# wouldn't you need to include the steps to load the model and spin it up in every function call?
#
# In order to initialize the model just once on container startup,
# we use Modal's [container lifecycle](https://modal.com/docs/guide/lifecycle-functions) features, which require the function to be part
# of a class. Note that the `modal.Volume` we saved the model to is mounted here as well,
# so that the fine-tuned model created  by `train` is available to us.


@app.cls(image=image, gpu="A10G", volumes={MODEL_DIR: volume})
class Model:
    @enter()
    def load_model(self):
        import torch
        from diffusers import AutoencoderKL, DiffusionPipeline

        config = TrainConfig()

        # Reload the modal.Volume to ensure the latest state is accessible.
        volume.reload()

        # set up a hugging face inference pipeline using our model
        pipe = DiffusionPipeline.from_pretrained(
            config.model_name,
            vae=AutoencoderKL.from_pretrained(
                config.vae_name, torch_dtype=torch.float16
            ),
            torch_dtype=torch.float16,
        ).to("cuda")
        pipe.load_lora_weights(MODEL_DIR)
        self.pipe = pipe

    @method()
    def inference(self, text, config):
        image = self.pipe(
            text,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
        ).images[0]

        return image


# ## Wrap the trained model in a Gradio web UI
#
# [Gradio](https://gradio.app) makes it super easy to expose a model's functionality
# in an easy-to-use, responsive web interface.
#
# This model is a text-to-image generator,
# so we set up an interface that includes a user-entry text box
# and a frame for displaying images.
#
# We also provide some example text inputs to help
# guide users and to kick-start their creative juices.
#
# And we couldn't resist adding some Modal style to it as well!
#
# You can deploy the app on Modal with the command
# `modal deploy dreambooth_app.py`.
# You'll be able to come back days, weeks, or months later and find it still ready to do,
# even though you don't have to pay for a server to run while you're not using it.

web_app = FastAPI()
assets_path = Path(__file__).parent / "assets"


@dataclass
class AppConfig(SharedConfig):
    """Configuration information for inference."""

    num_inference_steps: int = 25
    guidance_scale: float = 7.5


@app.function(
    image=image,
    concurrency_limit=3,
    mounts=[Mount.from_local_dir(assets_path, remote_path="/assets")],
)
@asgi_app()
def fastapi_app():
    import gradio as gr
    from gradio.routes import mount_gradio_app

    # Call out to the inference in a separate Modal environment with a GPu
    def go(text=""):
        if not text:
            text = example_prompts[0]
        return Model().inference.remote(text, config)

    # set up AppConfig
    config = AppConfig()

    instance_phrase = f"{config.instance_name} the {config.class_name}"

    example_prompts = [
        f"{instance_phrase}",
        f"a painting of {instance_phrase.title()} With A Pearl Earring, by Vermeer",
        f"oil painting of {instance_phrase} flying through space as an astronaut",
        f"a painting of {instance_phrase} in cyberpunk city. character design by cory loftis. volumetric light, detailed, rendered in octane",
        f"drawing of {instance_phrase} high quality, cartoon, path traced, by studio ghibli and don bluth",
    ]

    modal_docs_url = "https://modal.com/docs/guide"
    modal_example_url = f"{modal_docs_url}/examples/dreambooth_app"

    description = f"""Describe what they are doing or how a particular artist or style would depict them. Be fantastical! Try the examples below for inspiration.

### Learn how to make a "Dreambooth" for your own pet [here]({modal_example_url}).
    """

    # custom styles: an icon, a background, and a theme
    @web_app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return FileResponse("/assets/favicon.svg")

    @web_app.get("/assets/background.svg", include_in_schema=False)
    async def background():
        return FileResponse("/assets/background.svg")

    with open("/assets/index.css") as f:
        css = f.read()

    theme = gr.themes.Default(
        primary_hue="green", secondary_hue="emerald", neutral_hue="neutral"
    )

    # add a gradio UI around inference
    with gr.Blocks(
        theme=theme, css=css, title="Pet Dreambooth on Modal"
    ) as interface:
        gr.Markdown(
            f"# Dream up images of {instance_phrase}.\n\n{description}",
        )
        with gr.Row():
            inp = gr.Textbox(  # input text component
                label="",
                placeholder=f"Describe the version of {instance_phrase} you'd like to see",
                lines=10,
            )
            out = gr.Image(  # output image component
                height=512, width=512, label="", min_width=512, elem_id="output"
            )
        with gr.Row():
            btn = gr.Button("Dream", variant="primary", scale=2)
            btn.click(
                fn=go, inputs=inp, outputs=out
            )  # connect inputs and outputs with inference function

            gr.Button(  # shameless plug
                "⚡️ Powered by Modal",
                variant="secondary",
                link="https://modal.com",
            )

        with gr.Column(variant="compact"):
            # add in a few examples to inspire users
            for ii, prompt in enumerate(example_prompts):
                btn = gr.Button(prompt, variant="secondary")
                btn.click(fn=lambda idx=ii: example_prompts[idx], outputs=inp)

    # mount for execution on Modal
    return mount_gradio_app(
        app=web_app,
        blocks=interface,
        path="/",
    )


# ## Running your own Dreambooth from the command line
#
# You can use the `modal` command-line interface to set up, customize, and deploy this app:
#
# - `modal run dreambooth_app.py` will train the model. Change the `instance_example_urls_file` to point to your own pet's images.
# - `modal serve dreambooth_app.py` will [serve](https://modal.com/docs/guide/webhooks#developing-with-modal-serve) the Gradio interface at a temporary location. Great for iterating on code!
# - `modal shell dreambooth_app.py` is a convenient helper to open a bash [shell](https://modal.com/docs/guide/developing-debugging#interactive-shell) in our image. Great for debugging environment issues.
#
# Remember, once you've trained your own fine-tuned model, you can deploy it using `modal deploy dreambooth_app.py`.
#
# If you just want to try the app out, you can find it at https://modal-labs-example-dreambooth-app-fastapi-app.modal.run


@app.local_entrypoint()
def run():
    with open(TrainConfig().instance_example_urls_file) as f:
        instance_example_urls = [line.strip() for line in f.readlines()]
    train.remote(instance_example_urls)


---

## instructor

from modal import App, Image, build, enter, method

MODEL_DIR = "/model"


image = (
    Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .run_commands(
        "git clone https://github.com/HKUNLP/instructor-embedding",
        # Package doesn't define it's requirements properly?
        "cd instructor-embedding && pip install -r requirements.txt",
    )
    .pip_install("InstructorEmbedding")
)

app = App(
    "instructor", image=image
)  # Note: prior to April 2024, "app" was called "stub"

with image.imports():
    from InstructorEmbedding import INSTRUCTOR


@app.cls(gpu="any")
class InstructorModel:
    @build()
    def download_model(self):
        model = INSTRUCTOR("hkunlp/instructor-large")
        model.save(MODEL_DIR)

    @enter()
    def enter(self):
        self.model = INSTRUCTOR(MODEL_DIR, device="cuda")

    @method()
    def compare(self, sentences_a, sentences_b):
        from sklearn.metrics.pairwise import cosine_similarity

        embeddings_a = self.model.encode(sentences_a)
        embeddings_b = self.model.encode(sentences_b)
        similarities = cosine_similarity(embeddings_a, embeddings_b)
        return similarities.tolist()


@app.local_entrypoint()
def run():
    sentences_a = [
        [
            "Represent the Science sentence: ",
            "Parton energy loss in QCD matter",
        ],
        [
            "Represent the Financial statement: ",
            "The Federal Reserve on Wednesday raised its benchmark interest rate.",
        ],
    ]
    sentences_b = [
        [
            "Represent the Science sentence: ",
            "The Chiral Phase Transition in Dissipative Dynamics",
        ],
        [
            "Represent the Financial statement: ",
            "The funds rose less than 0.5 per cent on Friday",
        ],
    ]

    model = InstructorModel()
    similarities = model.compare.remote(sentences_a, sentences_b)
    print(similarities)


---

## text embeddings inference

# ---
# cmd: ["modal", "run", "06_gpu_and_ml/embeddings/text_embeddings_inference.py::embed_dataset"]
# ---
import json
import os
import socket
import subprocess
from pathlib import Path

from modal import App, Image, Secret, Volume, enter, exit, gpu, method

GPU_CONFIG = gpu.A10G()
MODEL_ID = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 32
DOCKER_IMAGE = (
    "ghcr.io/huggingface/text-embeddings-inference:86-0.4.0"  # Ampere 86 for A10s.
    # "ghcr.io/huggingface/text-embeddings-inference:0.4.0" # Ampere 80 for A100s.
    # "ghcr.io/huggingface/text-embeddings-inference:0.3.0"  # Turing for T4s.
)

DATA_PATH = Path("/data/dataset.jsonl")

LAUNCH_FLAGS = [
    "--model-id",
    MODEL_ID,
    "--port",
    "8000",
]


def spawn_server() -> subprocess.Popen:
    process = subprocess.Popen(
        ["text-embeddings-router"] + LAUNCH_FLAGS,
        env={
            **os.environ,
            "HUGGING_FACE_HUB_TOKEN": os.environ["HF_TOKEN"],
        },
    )

    # Poll until webserver at 127.0.0.1:8000 accepts connections before running inputs.
    while True:
        try:
            socket.create_connection(("127.0.0.1", 8000), timeout=1).close()
            print("Webserver ready!")
            return process
        except (socket.timeout, ConnectionRefusedError):
            # Check if launcher webserving process has exited.
            # If so, a connection can never be made.
            retcode = process.poll()
            if retcode is not None:
                raise RuntimeError(
                    f"launcher exited unexpectedly with code {retcode}"
                )


def download_model():
    # Wait for server to start. This downloads the model weights when not present.
    spawn_server().terminate()


volume = Volume.from_name("tei-hn-data", create_if_missing=True)

app = App("example-tei")  # Note: prior to April 2024, "app" was called "stub"


tei_image = (
    Image.from_registry(
        DOCKER_IMAGE,
        add_python="3.10",
    )
    .dockerfile_commands("ENTRYPOINT []")
    .run_function(
        download_model,
        gpu=GPU_CONFIG,
        secrets=[Secret.from_name("huggingface-secret")],
    )
    .pip_install("httpx")
)


with tei_image.imports():
    from httpx import AsyncClient


@app.cls(
    secrets=[Secret.from_name("huggingface-secret")],
    gpu=GPU_CONFIG,
    image=tei_image,
    # Use up to 20 GPU containers at once.
    concurrency_limit=20,
    # Allow each container to process up to 10 batches at once.
    allow_concurrent_inputs=10,
)
class TextEmbeddingsInference:
    @enter()
    def setup_server(self):
        self.process = spawn_server()
        self.client = AsyncClient(base_url="http://127.0.0.1:8000")

    @exit()
    def teardown_server(self):
        self.process.terminate()

    @method()
    async def embed(self, inputs_with_ids: list[tuple[int, str]]):
        ids, inputs = zip(*inputs_with_ids)
        resp = await self.client.post("/embed", json={"inputs": inputs})
        resp.raise_for_status()
        outputs = resp.json()

        return list(zip(ids, outputs))


def download_data():
    service_account_info = json.loads(os.environ["SERVICE_ACCOUNT_JSON"])
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info
    )

    client = bigquery.Client(credentials=credentials)

    iterator = client.list_rows(
        "bigquery-public-data.hacker_news.full",
        max_results=100_000,
    )
    df = iterator.to_dataframe(progress_bar_type="tqdm")
    df["id"] = df["id"].astype(int)
    # TODO: better chunking / splitting.
    df["text"] = df["text"].apply(lambda x: x[:512])

    data = list(zip(df["id"], df["text"]))

    with open(DATA_PATH, "w") as f:
        json.dump(data, f)

    volume.commit()


image = Image.debian_slim(python_version="3.10").pip_install(
    "google-cloud-bigquery", "pandas", "db-dtypes", "tqdm"
)

with image.imports():
    from google.cloud import bigquery
    from google.oauth2 import service_account


@app.function(
    image=image,
    secrets=[Secret.from_name("bigquery")],
    volumes={DATA_PATH.parent: volume},
)
def embed_dataset():
    model = TextEmbeddingsInference()

    if not DATA_PATH.exists():
        print("Downloading data. This takes a while...")
        download_data()

    with open(DATA_PATH) as f:
        data = json.loads(f.read())

    def generate_batches():
        batch = []
        for item in data:
            batch.append(item)

            if len(batch) == BATCH_SIZE:
                yield batch
                batch = []

    # data is of type list[tuple[str, str]].
    # starmap spreads the tuples into positional arguments.
    for output_batch in model.embed.map(
        generate_batches(), order_outputs=False
    ):
        # Do something with the outputs.
        pass


---

## download

from modal import App, Image, Volume

# We first set out configuration variables for our script.
DATASET_DIR = "/data"
DATASET_NAME = "wikipedia"
DATASET_CONFIG = "20220301.en"


# We define our Modal Resources that we'll need
volume = Volume.from_name("embedding-wikipedia", create_if_missing=True)
image = Image.debian_slim(python_version="3.9").pip_install(
    "datasets==2.16.1", "apache_beam==2.53.0"
)
app = App(image=image)  # Note: prior to April 2024, "app" was called "stub"


# The default timeout is 5 minutes re: https://modal.com/docs/guide/timeouts#handling-timeouts
#  but we override this to
# 3000s to avoid any potential timeout issues
@app.function(volumes={DATASET_DIR: volume}, timeout=3000)
def download_dataset():
    # Redownload the dataset
    import time

    from datasets import load_dataset

    start = time.time()
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, num_proc=6)
    end = time.time()
    print(f"Download complete - downloaded files in {end-start}s")

    dataset.save_to_disk(f"{DATASET_DIR}/{DATASET_NAME}")
    volume.commit()


@app.local_entrypoint()
def main():
    download_dataset.remote()


---

## main

import asyncio
import json
import subprocess

from modal import App, Image, Secret, Volume, build, enter, exit, gpu, method

# We first set out configuration variables for our script.
## Embedding Containers Configuration
GPU_CONCURRENCY = 100
GPU_CONFIG = gpu.A10G()
MODEL_ID = "BAAI/bge-small-en-v1.5"
MODEL_SLUG = MODEL_ID.split("/")[-1]
BATCH_SIZE = 512
DOCKER_IMAGE = (
    "ghcr.io/huggingface/text-embeddings-inference:86-0.4.0"  # Ampere 86 for A10s.
    # "ghcr.io/huggingface/text-embeddings-inference:0.4.0" # Ampere 80 for A100s.
    # "ghcr.io/huggingface/text-embeddings-inference:0.3.0"  # Turing for T4s.
)

## Dataset-Specific Configuration
DATASET_NAME = "wikipedia"
DATASET_READ_VOLUME = Volume.from_name(
    "embedding-wikipedia", create_if_missing=True
)
EMBEDDING_CHECKPOINT_VOLUME = Volume.from_name(
    "checkpoint", create_if_missing=True
)
DATASET_DIR = "/data"
CHECKPOINT_DIR = "/checkpoint"
SAVE_TO_DISK = True

## Upload-Specific Configuration
DATASET_HF_UPLOAD_REPO_NAME = "567-labs/upload-test"
UPLOAD_TO_HF = True

## HF Text-Embedding Inference specific Configuration

LAUNCH_FLAGS = [
    "--model-id",
    MODEL_ID,
    "--port",
    "8000",
    "--max-client-batch-size",
    str(BATCH_SIZE),
    "--max-batch-tokens",
    str(BATCH_SIZE * 512),
]


app = App(
    "example-embeddings"
)  # Note: prior to April 2024, "app" was called "stub"


def spawn_server() -> subprocess.Popen:
    import socket

    process = subprocess.Popen(["text-embeddings-router"] + LAUNCH_FLAGS)
    # Poll until webserver at 127.0.0.1:8000 accepts connections before running inputs.
    while True:
        try:
            socket.create_connection(("127.0.0.1", 8000), timeout=1).close()
            print("Webserver ready!")
            return process
        except (socket.timeout, ConnectionRefusedError):
            # Check if launcher webserving process has exited.
            # If so, a connection can never be made.
            retcode = process.poll()
            if retcode is not None:
                raise RuntimeError(
                    f"launcher exited unexpectedly with code {retcode}"
                )


tei_image = (
    Image.from_registry(
        "ghcr.io/huggingface/text-embeddings-inference:86-0.4.0",
        add_python="3.10",
    )
    .dockerfile_commands("ENTRYPOINT []")
    .pip_install("httpx")
)

with tei_image.imports():
    import numpy as np


def generate_chunks_from_dataset(xs, chunk_size: int):
    """
    Generate chunks from a dataset.

    Args:
        xs (list): The dataset containing dictionaries with "id", "url", "title", and "text" keys.
        chunk_size (int): The size of each chunk.

    Yields:
        tuple: A tuple containing the id, url, title, and a chunk of text.

    """
    for data in xs:
        id_ = data["id"]
        url = data["url"]
        title = data["title"]
        text = data["text"]
        for chunk_start in range(0, len(text), chunk_size):
            yield (
                id_,
                url,
                title,
                text[chunk_start : chunk_start + chunk_size],
            )


def generate_batches(xs, batch_size):
    batch = []
    for x in xs:
        batch.append(x)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


@app.cls(
    gpu=GPU_CONFIG,
    image=tei_image,
    concurrency_limit=GPU_CONCURRENCY,
    allow_concurrent_inputs=True,
    retries=3,
)
class TextEmbeddingsInference:
    @build()
    def download_model(self):
        spawn_server()

    @enter()
    def open_connection(self):
        # If the process is running for a long time, the client does not seem to close the connections, results in a pool timeout
        from httpx import AsyncClient

        self.process = spawn_server()
        self.client = AsyncClient(base_url="http://127.0.0.1:8000", timeout=30)

    @exit()
    def terminate_connection(self):
        self.process.terminate()

    async def _embed(self, chunk_batch):
        texts = [chunk[3] for chunk in chunk_batch]
        res = await self.client.post("/embed", json={"inputs": texts})
        return np.array(res.json())

    @method()
    async def embed(self, chunks):
        """Embeds a list of texts.  id, url, title, text = chunks[0]"""
        coros = [
            self._embed(chunk_batch)
            for chunk_batch in generate_batches(chunks, batch_size=BATCH_SIZE)
        ]

        embeddings = np.vstack(await asyncio.gather(*coros))
        return chunks, embeddings


def load_dataset_from_disk(down_scale: float = 0.01):
    """
    Load a dataset from disk and return a subset of the training data.

    Args:
        down_scale (float): The fraction of the training data to select. Defaults to 0.01.

    Returns:
        Dataset: A subset of the training data.
    """
    import time

    from datasets import load_from_disk

    start = time.perf_counter()
    # Load the dataset as a Hugging Face dataset
    print(f"Loading dataset from {DATASET_DIR}/wikipedia")
    dataset = load_from_disk(f"{DATASET_DIR}/wikipedia")
    print(f"Dataset loaded in {time.perf_counter()-start:.2f} seconds")

    # Extract the total size of the dataset
    ttl_size = len(dataset["train"])

    sample_size = int(ttl_size * down_scale)

    return dataset["train"].select(range(sample_size))


def save_dataset_to_intermediate_checkpoint(acc_chunks, embeddings, batch_size):
    """Saves the dataset to an intermediate checkpoint.

    Args:
        acc_chunks (list): Accumulated chunks
        embeddings (list): Accumulated embeddings
        batch_size (int): Batch size
    """
    import pyarrow as pa
    from datasets import Dataset

    table = pa.Table.from_arrays(
        [
            pa.array([chunk[0] for chunk in acc_chunks]),  # id
            pa.array([chunk[1] for chunk in acc_chunks]),  # url
            pa.array([chunk[2] for chunk in acc_chunks]),  # title
            pa.array([chunk[3] for chunk in acc_chunks]),  # text
            pa.array(embeddings),
        ],
        names=["id", "url", "title", "text", "embedding"],
    )
    path_parent_folder = f"{CHECKPOINT_DIR}/{MODEL_SLUG}-{batch_size}"
    dataset = Dataset(table)
    dataset.save_to_disk(path_parent_folder)
    EMBEDDING_CHECKPOINT_VOLUME.commit()
    print(f"Saved checkpoint at {path_parent_folder}")


def upload_result_to_hf(batch_size: int) -> None:
    """
    Uploads the result to the Hugging Face Hub.

    Args:
        batch_size (int): The batch size for the model.

    Returns:
        None
    """
    import os
    import time

    from huggingface_hub import HfApi

    path_parent_folder = f"{CHECKPOINT_DIR}/{MODEL_SLUG}-{batch_size}"
    api = HfApi(token=os.environ["HUGGINGFACE_TOKEN"])
    api.create_repo(
        repo_id=DATASET_HF_UPLOAD_REPO_NAME,
        private=False,
        repo_type="dataset",
        exist_ok=True,
    )

    print(f"Pushing to hub {DATASET_HF_UPLOAD_REPO_NAME}")
    start = time.perf_counter()
    api.upload_folder(
        folder_path=path_parent_folder,
        repo_id=DATASET_HF_UPLOAD_REPO_NAME,
        repo_type="dataset",
        multi_commits=True,
        multi_commits_verbose=True,
    )

    end = time.perf_counter()
    print(f"Uploaded in {end-start}s")


@app.function(
    image=Image.debian_slim().pip_install(
        "datasets", "pyarrow", "hf_transfer", "huggingface_hub"
    ),
    volumes={
        DATASET_DIR: DATASET_READ_VOLUME,
        CHECKPOINT_DIR: EMBEDDING_CHECKPOINT_VOLUME,
    },
    timeout=86400,
    secrets=[Secret.from_name("huggingface-secret")],
)
def embed_dataset(down_scale: float = 1, batch_size: int = 512 * 50):
    """
    Embeds a dataset with the Text Embeddings Inference container.

    Args:
        down_scale (float): The fraction of the training data to select. Defaults to 1.
        batch_size (int): The batch size to use. Defaults to 512 * 50.

    Returns:
        dict: A dictionary containing the benchmark results.
    """
    import datetime
    import time

    if UPLOAD_TO_HF and not SAVE_TO_DISK:
        raise ValueError(
            "Uploading to HF requires SAVE_TO_DISK to be set to true in case of intermediate failure."
        )

    dataset_chars = 19560538957  # sum(map(len, dataset["train"]["text"]))
    subset = load_dataset_from_disk(down_scale)
    model = TextEmbeddingsInference()
    text_chunks = generate_chunks_from_dataset(subset, chunk_size=512)
    batches = generate_batches(text_chunks, batch_size=batch_size)

    start = time.perf_counter()
    acc_chunks = []
    embeddings = []
    for resp in model.embed.map(
        batches, order_outputs=False, return_exceptions=True
    ):
        if isinstance(resp, Exception):
            print(f"Exception: {resp}")
            continue

        batch_chunks, batch_embeddings = resp

        acc_chunks.extend(batch_chunks)
        embeddings.extend(batch_embeddings)

    end = time.perf_counter()

    duration = end - start
    characters = sum(map(len, [chunk[3] for chunk in acc_chunks]))
    characters_per_sec = int(characters / duration)
    extrapolated_duration_cps_fmt = str(
        datetime.timedelta(seconds=dataset_chars / characters_per_sec)
    )
    resp = {
        "downscale": down_scale,
        "batch_size": batch_size,
        "n_gpu": GPU_CONCURRENCY,
        "duration_mins": duration / 60,
        "characters_per_sec": characters_per_sec,
        "extrapolated_duration": extrapolated_duration_cps_fmt,
    }

    if SAVE_TO_DISK:
        save_dataset_to_intermediate_checkpoint(
            acc_chunks, embeddings, batch_size
        )

    if UPLOAD_TO_HF:
        upload_result_to_hf(batch_size)

    return resp


@app.local_entrypoint()
def full_job():
    batch_size = 512 * 150
    with open("benchmarks.json", "a") as f:
        benchmark = embed_dataset.remote(batch_size=batch_size)
        f.write(json.dumps(benchmark, indent=2) + "\n")


---

## flan t5 finetune

# ---
# runtimes: ["runc", "gvisor"]
# ---
#
# # Finetuning Flan-T5
#
# Example by [@anishpdalal](https://github.com/anishpdalal)
#
# [Flan-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5) is a highly versatile model that's been instruction-tuned to
# perform well on a variety of text-based tasks such as question answering and summarization. There are smaller model variants available which makes
# Flan-T5 a great base model to use for finetuning on a specific instruction dataset with just a single GPU. In this example, we'll
# finetune Flan-T5 on the [Extreme Sum ("XSum")](https://huggingface.co/datasets/xsum) dataset to summarize news articles.

# ## Defining dependencies
#
# The example uses the `dataset` package from HuggingFace to load the xsum dataset. It also uses the `transformers`
# and `accelerate` packages with a PyTorch backend to finetune and serve the model. Finally, we also
# install `tensorboard` and serve it via a web app. All packages are installed into a Debian Slim base image
# using the `pip_install` function.
#

from pathlib import Path

import modal
from modal import App, Image, Volume, enter, method, wsgi_app

VOL_MOUNT_PATH = Path("/vol")

# Other Flan-T5 models can be found [here](https://huggingface.co/docs/transformers/model_doc/flan-t5)
BASE_MODEL = "google/flan-t5-base"

image = Image.debian_slim().pip_install(
    "accelerate",
    "transformers",
    "torch",
    "datasets",
    "tensorboard",
)

app = App(
    name="example-news-summarizer", image=image
)  # Note: prior to April 2024, "app" was called "stub"
output_vol = Volume.from_name("finetune-volume", create_if_missing=True)

# ### Handling preemption
#
# As this finetuning job is long-running it's possible that it experiences a preemption.
# The training code is robust to pre-emption events by periodically saving checkpoints and restoring
# from checkpoint on restart. But it's also helpful to observe in logs when a preemption restart has occurred,
# so we track restarts with a `modal.Dict`.
#
# See the [guide on preemptions](/docs/guide/preemption#preemption) for more details on preemption handling.

restart_tracker_dict = modal.Dict.from_name(
    "finetune-restart-tracker", create_if_missing=True
)


def track_restarts(restart_tracker: modal.Dict) -> int:
    if not restart_tracker.contains("count"):
        preemption_count = 0
        print(f"Starting first time. {preemption_count=}")
        restart_tracker["count"] = preemption_count
    else:
        preemption_count = restart_tracker.get("count") + 1
        print(f"Restarting after pre-emption. {preemption_count=}")
        restart_tracker["count"] = preemption_count
    return preemption_count


# ## Finetuning Flan-T5 on XSum dataset
#
# Each row in the dataset has a `document` (input news article) and `summary` column.


@app.function(
    gpu="A10g",
    timeout=7200,
    volumes={VOL_MOUNT_PATH: output_vol},
    _allow_background_volume_commits=True,
)
def finetune(num_train_epochs: int = 1, size_percentage: int = 10):
    from datasets import load_dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    restarts = track_restarts(restart_tracker_dict)

    # Use size percentage to retrieve subset of the dataset to iterate faster
    if size_percentage:
        xsum_train = load_dataset("xsum", split=f"train[:{size_percentage}%]")
        xsum_test = load_dataset("xsum", split=f"test[:{size_percentage}%]")

    # Load the whole dataset
    else:
        xsum = load_dataset("xsum")
        xsum_train = xsum["train"]
        xsum_test = xsum["test"]

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    # Replace all padding tokens with a large negative number so that the loss function ignores them in
    # its calculation
    padding_token_id = -100

    batch_size = 8

    def preprocess(batch):
        # prepend summarize: prefix to document to convert the example to a summarization instruction
        inputs = ["summarize: " + doc for doc in batch["document"]]

        model_inputs = tokenizer(
            inputs, max_length=512, truncation=True, padding="max_length"
        )

        labels = tokenizer(
            text_target=batch["summary"],
            max_length=128,
            truncation=True,
            padding="max_length",
        )

        labels["input_ids"] = [
            [
                l if l != tokenizer.pad_token_id else padding_token_id
                for l in label
            ]
            for label in labels["input_ids"]
        ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_xsum_train = xsum_train.map(
        preprocess, batched=True, remove_columns=["document", "summary", "id"]
    )

    tokenized_xsum_test = xsum_test.map(
        preprocess, batched=True, remove_columns=["document", "summary", "id"]
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=padding_token_id,
        pad_to_multiple_of=batch_size,
    )

    training_args = Seq2SeqTrainingArguments(
        # Save checkpoints to the mounted volume
        output_dir=str(VOL_MOUNT_PATH / "model"),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        learning_rate=3e-5,
        num_train_epochs=num_train_epochs,
        logging_strategy="steps",
        logging_steps=100,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_xsum_train,
        eval_dataset=tokenized_xsum_test,
    )

    try:
        resume = restarts > 0
        if resume:
            print("resuming from checkpoint")
        trainer.train(resume_from_checkpoint=resume)
    except KeyboardInterrupt:  # handle possible preemption
        print("received interrupt; saving state and model")
        trainer.save_state()
        trainer.save_model()
        raise

    # Save the trained model and tokenizer to the mounted volume
    model.save_pretrained(str(VOL_MOUNT_PATH / "model"))
    tokenizer.save_pretrained(str(VOL_MOUNT_PATH / "tokenizer"))
    output_vol.commit()
    print("✅ done")


# ## Monitoring Finetuning with Tensorboard
#
# Tensorboard is an application for visualizing training loss. In this example we
# serve it as a Modal WSGI app.
#
@app.function(volumes={VOL_MOUNT_PATH: output_vol})
@wsgi_app()
def monitor():
    import tensorboard

    board = tensorboard.program.TensorBoard()
    board.configure(logdir=f"{VOL_MOUNT_PATH}/logs")
    (data_provider, deprecated_multiplexer) = board._make_data_provider()
    wsgi_app = tensorboard.backend.application.TensorBoardWSGIApp(
        board.flags,
        board.plugin_loaders,
        data_provider,
        board.assets_zip_provider,
        deprecated_multiplexer,
    )  # Note: prior to April 2024, "app" was called "stub"
    return wsgi_app


# ## Model Inference
#


@app.cls(volumes={VOL_MOUNT_PATH: output_vol})
class Summarizer:
    @enter()
    def load_model(self):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

        # Load saved tokenizer and finetuned from training run
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, cache_dir=VOL_MOUNT_PATH / "tokenizer/"
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL, cache_dir=VOL_MOUNT_PATH / "model/"
        )

        self.summarizer = pipeline(
            "summarization", tokenizer=tokenizer, model=model
        )

    @method()
    def generate(self, input: str) -> str:
        return self.summarizer(input)[0]["summary_text"]


@app.local_entrypoint()
def main():
    input = """
    The 14-time major champion, playing in his first full PGA Tour event for almost 18 months,
    carded a level-par second round of 72, but missed the cut by four shots after his first-round 76.
    World number one Jason Day and US Open champion Dustin Johnson also missed the cut at Torrey Pines in San Diego.
    Overnight leader Rose carded a one-under 71 to put him on eight under. Canada's
    Adam Hadwin and USA's Brandt Snedeker are tied in second on seven under, while US PGA champion
    Jimmy Walker missed the cut as he finished on three over. Woods is playing in just his
    second tournament since 15 months out with a back injury. "It's frustrating not being
    able to have a chance to win the tournament," said the 41-year-old, who won his last major,
    the US Open, at the same course in 2008. "Overall today was a lot better than yesterday.
    I hit it better, I putted well again. I hit a lot of beautiful putts that didn't go in, but
    I hit it much better today, which was nice." Scotland's Martin Laird and England's Paul Casey
    are both on two under, while Ireland's Shane Lowry is on level par.
    """
    model = Summarizer()
    response = model.generate.remote(input)
    print(response)


# ## Run via the CLI
# Invoke model finetuning use the provided command below
#
# ```bash
# modal run --detach flan_t5_finetune.py::finetune --num-train-epochs=1 --size-percentage=10
# View the tensorboard logs at https://<username>--example-news-summarizer-monitor-dev.modal.run
# ```
#
# Invoke finetuned model inference via local entrypoint
#
# ```bash
# modal run flan_t5_finetune.py
# World number one Tiger Woods missed the cut at the US Open as he failed to qualify for the final round of the event in Los Angeles.
# ```
#


---

## import torch

# # PyTorch with CUDA GPU support
#
# This example shows how you can use CUDA GPUs in Modal, with a minimal PyTorch
# image. You can specify GPU requirements in the `app.function` decorator.

import time

import modal

app = modal.App(
    "example-import-torch",
    image=modal.Image.debian_slim().pip_install(
        "torch", find_links="https://download.pytorch.org/whl/cu116"
    ),
)  # Note: prior to April 2024, "app" was called "stub"


@app.function(gpu="any")
def gpu_function():
    import subprocess

    import torch

    subprocess.run(["nvidia-smi"])
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())


if __name__ == "__main__":
    t0 = time.time()
    with app.run():
        gpu_function.remote()
    print("Full time spent:", time.time() - t0)


---

## init



---

## agent

"""This module defines our agent and attaches it to the Modal App.

Our agent is defined as a graph: a collection of nodes and edges,
where nodes represent actions and edges represent transitions between actions.

The meat of the logic is therefore in the edges and nodes modules.

We have a very simple "context-stuffing" retrieval approach in the retrieval module.
Replace this with something that retrieves your documentation and adjust the prompts accordingly.

You can test the agent from the command line with `modal run agent.py --question` followed by your query"""

import edges
import nodes
import retrieval
from common import app


@app.local_entrypoint()
def main(question: str = "How do I build a RAG pipeline?", debug: bool = False):
    """Sends a question to the LCEL code generation agent.

    Switch to debug mode for shorter context and smaller model."""
    if debug:
        if question == "How do I build a RAG pipeline?":
            question = "gm king, how are you?"
    print(go.remote(question, debug=debug)["keys"]["response"])


@app.function()
def go(question: str = "How do I build a RAG pipeline?", debug: bool = False):
    """Compiles the LCEL code generation agent graph and runs it, returning the result."""
    graph = construct_graph(debug=debug)
    runnable = graph.compile()
    result = runnable.invoke(
        {"keys": {"question": question, "iterations": 0}},
        config={"recursion_limit": 50},
    )

    return result


def construct_graph(debug=False):
    from common import GraphState
    from langgraph.graph import StateGraph

    context = retrieval.retrieve_docs(debug=debug)

    graph = StateGraph(GraphState)

    # attach our nodes to the graph
    graph_nodes = nodes.Nodes(context, debug=debug)
    for key, value in graph_nodes.node_map.items():
        graph.add_node(key, value)

    # construct the graph by adding edges
    graph = edges.enrich(graph)

    # set the starting and ending nodes of the graph
    graph.set_entry_point(key="generate")
    graph.set_finish_point(key="finish")

    return graph


---

## app

"""Application serving logic for the CodeLangChain agent."""

import agent
import modal
from agent import app, nodes
from fastapi import FastAPI, responses
from fastapi.middleware.cors import CORSMiddleware

# create a FastAPI app
web_app = FastAPI(
    title="CodeLangChain Server",
    version="1.0",
    description="Answers questions about LangChain Expression Language (LCEL).",
)


# set all CORS enabled origins
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# host it on Modal
@app.function(keep_warm=1)
@modal.asgi_app()
def serve():
    from langchain_core.runnables import RunnableLambda
    from langserve import add_routes

    def inp(question: str) -> dict:
        return {"keys": {"question": question, "iterations": 0}}

    def out(state: dict) -> str:
        if "keys" in state:
            return state["keys"]["response"]
        elif "generate" in state:
            return nodes.extract_response(state["generate"])
        else:
            return str(state)

    graph = agent.construct_graph(debug=False).compile()

    chain = RunnableLambda(inp) | graph | RunnableLambda(out)

    add_routes(
        web_app,
        chain,
        path="/codelangchain",
    )

    # redirect the root to the interactive playground
    @web_app.get("/")
    def redirect():
        return responses.RedirectResponse(url="/codelangchain/playground")

    return web_app


---

## common

"""Shared information: image definitions and common utilities."""

import os
from typing import Dict, TypedDict

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "beautifulsoup4~=4.12.3",
    "langchain==0.1.11",
    "langgraph==0.0.26",
    "langchain_community==0.0.27",
    "langchain-openai==0.0.8",
    "langserve[all]==0.0.46",
)

agent_image = image.pip_install(
    "chromadb==0.4.24",
    "langchainhub==0.1.15",
    "faiss-cpu~=1.8.0",
    "tiktoken==0.6.0",
)

app = modal.App(
    "code-langchain",
    image=image,
    secrets=[
        modal.Secret.from_name("my-openai-secret"),
        modal.Secret.from_name("my-langsmith-secret"),
    ],
)  # Note: prior to April 2024, "app" was called "stub"


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """

    keys: Dict[str, any]


os.environ["LANGCHAIN_PROJECT"] = "codelangchain"

COLOR = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "ENDC": "\033[0m",
}


---

## edges

"""Defines functions that transition our agent from one state to another."""

from typing import Callable

from common import GraphState

EXPECTED_NODES = [
    "generate",
    "check_code_imports",
    "check_code_execution",
    "finish",
]


def enrich(graph):
    """Adds transition edges to the graph."""

    for node_name in set(EXPECTED_NODES):
        assert node_name in graph.nodes, f"Node {node_name} not found in graph"

    graph.add_edge("generate", "check_code_imports")
    graph.add_conditional_edges(
        "check_code_imports",
        EDGE_MAP["decide_to_check_code_exec"],
        {
            "check_code_execution": "check_code_execution",
            "generate": "generate",
        },
    )
    graph.add_conditional_edges(
        "check_code_execution",
        EDGE_MAP["decide_to_finish"],
        {
            "finish": "finish",
            "generate": "generate",
        },
    )
    return graph


def decide_to_check_code_exec(state: GraphState) -> str:
    """
    Determines whether to test code execution, or re-try answer generation.

    Args:
    state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---DECIDE TO TEST CODE EXECUTION---")
    state_dict = state["keys"]
    error = state_dict["error"]

    if error == "None":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: TEST CODE EXECUTION---")
        return "check_code_execution"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: RE-TRY SOLUTION---")
        return "generate"


def decide_to_finish(state: GraphState) -> str:
    """
    Determines whether to finish (re-try code 3 times).

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---DECIDE TO TEST CODE EXECUTION---")
    state_dict = state["keys"]
    error = state_dict["error"]
    iter = state_dict["iterations"]

    if error == "None" or iter >= 3:
        print("---DECISION: FINISH---")
        return "finish"
    else:
        print("---DECISION: RE-TRY SOLUTION---")
        return "generate"


EDGE_MAP: dict[str, Callable] = {
    "decide_to_check_code_exec": decide_to_check_code_exec,
    "decide_to_finish": decide_to_finish,
}


---

## nodes

import sys
from operator import itemgetter

import sandbox
from common import GraphState, agent_image, image

with image.imports():
    from langchain.output_parsers.openai_tools import PydanticToolsParser
    from langchain.prompts import PromptTemplate
    from langchain_core.pydantic_v1 import BaseModel, Field
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from langchain_openai import ChatOpenAI


class Nodes:
    def __init__(self, context: str, debug: bool = False):
        self.context = context
        self.debug = debug
        self.model = (
            "gpt-4-0125-preview" if not self.debug else "gpt-3.5-turbo-0125"
        )
        self.node_map = {
            "generate": self.generate,
            "check_code_imports": self.check_code_imports,
            "check_code_execution": self.check_code_execution,
            "finish": self.finish,
        }

    def generate(self, state: GraphState) -> GraphState:
        """
        Generate a code solution based on LCEL docs and the input question
        with optional feedback from code execution tests

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """

        ## State
        state_dict = state["keys"]
        question = state_dict["question"]
        iter = state_dict["iterations"]

        ## Data model
        class code(BaseModel):
            """Code output"""

            prefix: str = Field(
                description="Description of the problem and approach"
            )
            imports: str = Field(description="Code block import statements")
            code: str = Field(
                description="Code block not including import statements"
            )

        ## LLM
        llm = ChatOpenAI(temperature=0, model=self.model, streaming=True)

        # Tool
        code_tool_oai = convert_to_openai_tool(code)

        # LLM with tool and enforce invocation
        llm_with_tool = llm.bind(
            tools=[code_tool_oai],
            tool_choice={"type": "function", "function": {"name": "code"}},
        )

        # Parser
        parser_tool = PydanticToolsParser(tools=[code])

        ## Prompt
        template = (
            """You are a coding assistant with expertise in LCEL, LangChain expression language. \n
        You are able to execute Python code in a sandbox environment that was constructed by chaining together the following two Dockerfile commands: \n
        """
            + f"{image.dockerfile_commands()}"
            + "\n"
            + f"{agent_image.dockerfile_commands()}"
            + """
            You are tasked with responding to the following user question: {question}
            Your response will be shown to the user.
            Here is a full set of LCEL documentation:
            \n ------- \n
            {context}
            \n ------- \n
            Answer the user question based on the above provided documentation. \n
            Ensure any code you provide can be executed with all required imports and variables defined. \n
            Structure your answer as a description of the code solution, \n
            then a list of the imports, and then finally list the functioning code block. \n
            Here is the user question again: \n --- --- --- \n {question}
        """
        )

        ## Generation
        if "error" in state_dict:
            print("---RE-GENERATE SOLUTION w/ ERROR FEEDBACK---")

            error = state_dict["error"]
            code_solution = state_dict["generation"]

            # Update prompt
            addendum = """  \n --- --- --- \n You previously tried to solve this problem. \n Here is your solution:
                        \n --- --- --- \n {generation}  \n --- --- --- \n  Here is the resulting error from code
                        execution:  \n --- --- --- \n {error}  \n --- --- --- \n Please re-try to answer this.
                        Structure your answer with a description of the code solution. \n Then list the imports.
                        And finally list the functioning code block. Structure your answer with a description of
                        the code solution. \n Then list the imports. And finally list the functioning code block.
                        \n Here is the user question: \n --- --- --- \n {question}"""
            template = template + addendum

            # Prompt
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question", "generation", "error"],
            )

            # Chain
            chain = (
                {
                    "context": lambda _: self.context,
                    "question": itemgetter("question"),
                    "generation": itemgetter("generation"),
                    "error": itemgetter("error"),
                }
                | prompt
                | llm_with_tool
                | parser_tool
            )

            code_solution = chain.invoke(
                {
                    "question": question,
                    "generation": str(code_solution[0]),
                    "error": error,
                }
            )

        else:
            print("---GENERATE SOLUTION---")

            # Prompt
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"],
            )

            # Chain
            chain = (
                {
                    "context": lambda _: self.context,
                    "question": itemgetter("question"),
                }
                | prompt
                | llm_with_tool
                | parser_tool
            )

            code_solution = chain.invoke({"question": question})

        iter = iter + 1
        return {
            "keys": {
                "generation": code_solution,
                "question": question,
                "iterations": iter,
            }
        }

    def check_code_imports(self, state: GraphState) -> GraphState:
        """
        Check imports

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, error
        """

        ## State
        print("---CHECKING CODE IMPORTS---")
        state_dict = state["keys"]
        question = state_dict["question"]
        code_solution = state_dict["generation"]
        imports = code_solution[0].imports
        iter = state_dict["iterations"]

        # Attempt to execute the imports
        sb = sandbox.run(imports)
        output, error = sb.stdout.read(), sb.stderr.read()
        if error:
            print("---CODE IMPORT CHECK: FAILED---")
            # Catch any error during execution (e.g., ImportError, SyntaxError)
            error = f"Execution error: {error}"
            print(f"Error: {error}", file=sys.stderr)
            if "error" in state_dict:
                error_prev_runs = state_dict["error"]
                error = (
                    error_prev_runs
                    + "\n --- Most recent run output and error --- \n"
                    " ------ output ------ \n"
                    + output
                    + "\n ------ error ------ \n"
                    + error
                )
        else:
            print("---CODE IMPORT CHECK: SUCCESS---")
            # No errors occurred
            error = "None"

        return {
            "keys": {
                "generation": code_solution,
                "question": question,
                "error": error,
                "iterations": iter,
            }
        }

    def check_code_execution(self, state: GraphState) -> GraphState:
        """
        Check code block execution

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, error
        """

        ## State
        print("---CHECKING CODE EXECUTION---")
        state_dict = state["keys"]
        question = state_dict["question"]
        code_solution = state_dict["generation"]
        prefix = code_solution[0].prefix
        imports = code_solution[0].imports
        code = code_solution[0].code
        code_block = imports + "\n" + code
        iter = state_dict["iterations"]

        sb = sandbox.run(code_block)
        output, error = sb.stdout.read(), sb.stderr.read()
        if error:
            print("---CODE BLOCK CHECK: FAILED---")
            error = f"Execution error: {error}"
            print(f"Error: {error}", file=sys.stderr)
            if "error" in state_dict:
                error_prev_runs = state_dict["error"]
                error = (
                    error_prev_runs
                    + "\n --- Most recent run output and error --- \n"
                    " ------ output ------ \n"
                    + output
                    + "\n ------ error ------ \n"
                    + error
                )
        else:
            print("---CODE BLOCK CHECK: SUCCESS---")
            # No errors occurred
            error = "None"

        return {
            "keys": {
                "generation": code_solution,
                "question": question,
                "error": error,
                "prefix": prefix,
                "imports": imports,
                "iterations": iter,
                "code": code,
            }
        }

    def finish(self, state: GraphState) -> dict:
        """
        Finish the graph

        Returns:
            dict: Final result
        """

        print("---FINISHING---")

        response = extract_response(state)

        return {"keys": {"response": response}}


def extract_response(state: GraphState) -> str:
    """
    Extract the response from the graph state

    Args:
        state (dict): The current graph state

    Returns:
        str: The response
    """

    state_dict = state["keys"]
    code_solution = state_dict["generation"][0]
    prefix = code_solution.prefix
    imports = code_solution.imports
    code = code_solution.code

    return "\n".join([prefix, imports, code])


---

## retrieval

"""Just as a constant function is _technically_ a polynomial, so too is injecting the same information every time _technically_ RAG."""

from common import COLOR

lcel_docs_url = "https://python.langchain.com/docs/expression_language/"


def retrieve_docs(url: str = lcel_docs_url, debug=False):
    from bs4 import BeautifulSoup as Soup
    from langchain_community.document_loaders.recursive_url_loader import (
        RecursiveUrlLoader,
    )

    print(
        f"{COLOR['HEADER']}📜: Retrieving documents from {url}{COLOR['ENDC']}"
    )
    loader = RecursiveUrlLoader(
        url=lcel_docs_url,
        max_depth=20 // (int(debug) + 1),  # retrieve fewer docs in debug mode
        extractor=lambda x: Soup(x, "html.parser").text,
    )
    docs = loader.load()

    # sort the list based on the URLs
    d_sorted = sorted(docs, key=lambda x: x.metadata["source"], reverse=True)

    # combine them all together
    concatenated_content = "\n\n\n --- \n\n\n".join(
        [
            "## " + doc.metadata["source"] + "\n\n" + doc.page_content.strip()
            for doc in d_sorted
        ]
    )

    print(
        f"{COLOR['HEADER']}📜: Retrieved {len(docs)} documents{COLOR['ENDC']}",
        f"{COLOR['GREEN']}{concatenated_content[:100].strip()}{COLOR['ENDC']}",
        sep="\n",
    )

    if debug:
        print(
            f"{COLOR['HEADER']}📜: Restricting to at most 30,000 characters{COLOR['ENDC']}"
        )
        concatenated_content = concatenated_content[:30_000]

    return concatenated_content


---

## sandbox

"""Defines the logic for running agent code in a sandbox."""

import modal
from common import COLOR, agent_image, app


def run(code: str):
    print(
        f"{COLOR['HEADER']}📦: Running in sandbox{COLOR['ENDC']}",
        f"{COLOR['GREEN']}{code}{COLOR['ENDC']}",
        sep="\n",
    )
    sb = app.spawn_sandbox(
        "python",
        "-c",
        code,
        image=agent_image,
        timeout=60 * 10,  # 10 minutes
        secrets=[
            modal.Secret.from_name(
                "my-openai-secret"
            )  # could be a different secret!
        ],
    )

    sb.wait()

    if sb.returncode != 0:
        print(
            f"{COLOR['HEADER']}📦: Failed with exitcode {sb.returncode}{COLOR['ENDC']}"
        )

    return sb


---

## potus speech qanda

# ---
# args: ["--query", "How many oil barrels were released from reserves"]
# ---
# # Question-answering with LangChain
#
# In this example we create a large-language-model (LLM) powered question answering
# web endpoint and CLI. Only a single document is used as the knowledge-base of the application,
# the 2022 USA State of the Union address by President Joe Biden. However, this same application structure
# could be extended to do question-answering over all State of the Union speeches, or other large text corpuses.
#
# It's the [LangChain](https://github.com/hwchase17/langchain) library that makes this all so easy. This demo is only around 100 lines of code!

# ## Defining dependencies
#
# The example uses three PyPi packages to make scraping easy, and three to build and run the question-answering functionality.
# These are installed into a Debian Slim base image using the `pip_install` function.
#
# Because OpenAI's API is used, we also specify the `openai-secret` Modal Secret, which contains an OpenAI API key.
#
# A `docsearch` global variable is also declared to facilitate caching a slow operation in the code below.
from pathlib import Path

from modal import App, Image, Secret, web_endpoint

image = Image.debian_slim().pip_install(
    # scraping pkgs
    "beautifulsoup4~=4.11.1",
    "httpx~=0.23.3",
    "lxml~=4.9.2",
    # langchain pkgs
    "faiss-cpu~=1.7.3",
    "langchain~=0.0.138",
    "openai~=0.27.4",
    "tiktoken==0.3.0",
)
app = App(
    name="example-langchain-qanda",
    image=image,
    secrets=[Secret.from_name("openai-secret")],
)  # Note: prior to April 2024, "app" was called "stub"
docsearch = None  # embedding index that's relatively expensive to compute, so caching with global var.

# ## Scraping the speech from whitehouse.gov
#
# It's super easy to scrape the transcipt of Biden's speech using `httpx` and `BeautifulSoup`.
# This speech is just one document and it's relatively short, but it's enough to demonstrate
# the question-answering capability of the LLM chain.


def scrape_state_of_the_union() -> str:
    import httpx
    from bs4 import BeautifulSoup

    url = "https://www.whitehouse.gov/state-of-the-union-2022/"

    # fetch article; simulate desktop browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9"
    }
    response = httpx.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "lxml")

    # get all text paragraphs & construct string of article text
    speech_text = ""
    speech_section = soup.find_all(
        "div", {"class": "sotu-annotations__content"}
    )
    if speech_section:
        paragraph_tags = speech_section[0].find_all("p")
        speech_text = "".join([p.get_text() for p in paragraph_tags])

    return speech_text.replace("\t", "")


# ## Constructing the Q&A chain
#
# At a high-level, this LLM chain will be able to answer questions asked about Biden's speech and provide
# references to which parts of the speech contain the evidence for given answers.
#
# The chain combines a text-embedding index over parts of Biden's speech with OpenAI's [GPT-3 LLM](https://openai.com/blog/chatgpt/).
# The index is used to select the most likely relevant parts of the speech given the question, and these
# are used to build a specialized prompt for the OpenAI language model.
#
# For more information on this, see [LangChain's "Question Answering" notebook](https://langchain.readthedocs.io/en/latest/use_cases/evaluation/question_answering.html).


def retrieve_sources(sources_refs: str, texts: list[str]) -> list[str]:
    """
    Map back from the references given by the LLM's output to the original text parts.
    """
    clean_indices = [
        r.replace("-pl", "").strip() for r in sources_refs.split(",")
    ]
    numeric_indices = (int(r) if r.isnumeric() else None for r in clean_indices)
    return [
        texts[i] if i is not None else "INVALID SOURCE" for i in numeric_indices
    ]


def qanda_langchain(query: str) -> tuple[str, list[str]]:
    from langchain.chains.qa_with_sources import load_qa_with_sources_chain
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.llms import OpenAI
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.vectorstores.faiss import FAISS

    # Support caching speech text on disk.
    speech_file_path = Path("state-of-the-union.txt")

    if speech_file_path.exists():
        state_of_the_union = speech_file_path.read_text()
    else:
        print("scraping the 2022 State of the Union speech")
        state_of_the_union = scrape_state_of_the_union()
        speech_file_path.write_text(state_of_the_union)

    # We cannot send the entire speech to the model because OpenAI's model
    # has a maximum limit on input tokens. So we split up the speech
    # into smaller chunks.
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    print("splitting speech into text chunks")
    texts = text_splitter.split_text(state_of_the_union)

    # Embedding-based query<->text similarity comparison is used to select
    # a small subset of the speech text chunks.
    # Generating the `docsearch` index is too slow to re-run on every request,
    # so we do rudimentary caching using a global variable.
    global docsearch

    if not docsearch:
        # New OpenAI accounts have a very low rate-limit for their first 48 hrs.
        # It's too low to embed even just this single Biden speech.
        # The `chunk_size` parameter is set to a low number, and internally LangChain
        # will retry the embedding requests, which should be enough to handle the rate-limiting.
        #
        # Ref: https://platform.openai.com/docs/guides/rate-limits/overview.
        print("generating docsearch indexer")
        docsearch = FAISS.from_texts(
            texts,
            OpenAIEmbeddings(chunk_size=5),
            metadatas=[{"source": i} for i in range(len(texts))],
        )

    print("selecting text parts by similarity to query")
    docs = docsearch.similarity_search(query)

    chain = load_qa_with_sources_chain(
        OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0),
        chain_type="stuff",
    )
    print("running query against Q&A chain.\n")
    result = chain(
        {"input_documents": docs, "question": query}, return_only_outputs=True
    )
    output: str = result["output_text"]
    parts = output.split("SOURCES: ")
    if len(parts) == 2:
        answer, sources_refs = parts
        sources = retrieve_sources(sources_refs, texts)
    elif len(parts) == 1:
        answer = parts[0]
        sources = []
    else:
        raise RuntimeError(
            f"Expected to receive an answer with a single 'SOURCES' block, got:\n{output}"
        )
    return answer.strip(), sources


# ## Modal Functions
#
# With our application's functionality implemented we can hook it into Modal.
# As said above, we're implementing a web endpoint, `web`, and a CLI command, `cli`.


@app.function()
@web_endpoint(method="GET")
def web(query: str, show_sources: bool = False):
    answer, sources = qanda_langchain(query)
    if show_sources:
        return {
            "answer": answer,
            "sources": sources,
        }
    else:
        return {
            "answer": answer,
        }


@app.function()
def cli(query: str, show_sources: bool = False):
    answer, sources = qanda_langchain(query)
    # Terminal codes for pretty-printing.
    bold, end = "\033[1m", "\033[0m"

    print(f"🦜 {bold}ANSWER:{end}")
    print(answer)
    if show_sources:
        print(f"🔗 {bold}SOURCES:{end}")
        for text in sources:
            print(text)
            print("----")


# ## Test run the CLI
#
# ```bash
# modal run potus_speech_qanda.py --query "What did the president say about Justice Breyer"
# 🦜 ANSWER:
# The president thanked Justice Breyer for his service and mentioned his legacy of excellence. He also nominated Ketanji Brown Jackson to continue in Justice Breyer's legacy.
# ```
#
# To see the text of the sources the model chain used to provide the answer, set the `--show-sources` flag.
#
# ```bash
# modal run potus_speech_qanda.py \
#    --query "How many oil barrels were released from reserves" \
#    --show-sources=True
# ```
#
# ## Test run the web endpoint
#
# Modal makes it trivially easy to ship LangChain chains to the web. We can test drive this app's web endpoint
# by running `modal serve potus_speech_qanda.py` and then hitting the endpoint with `curl`:
#
# ```bash
# curl --get \
#   --data-urlencode "query=What did the president say about Justice Breyer" \
#   https://modal-labs--example-langchain-qanda-web.modal.run
# ```
#
# ```json
# {
#   "answer": "The president thanked Justice Breyer for his service and mentioned his legacy of excellence. He also nominated Ketanji Brown Jackson to continue in Justice Breyer's legacy."
# }
# ```


---

## text generation inference

# # Hosting any LLaMA 3 model with Text Generation Inference (TGI)
#
# In this example, we show how to run an optimized inference server using [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference)
# with performance advantages over standard text generation pipelines including:
# - continuous batching, so multiple generations can take place at the same time on a single container
# - PagedAttention, which applies memory paging to the attention mechanism's key-value cache, increasing throughput
#
# This example deployment, [accessible here](https://modal.chat), can serve LLaMA 3 70B with
# 70 second cold starts, up to 200 tokens/s of throughput, and a per-token latency of 55ms.

# ## Setup
#
# First we import the components we need from `modal`.

import os
import subprocess
from pathlib import Path

from modal import App, Image, Mount, Secret, asgi_app, enter, exit, gpu, method

# Next, we set which model to serve, taking care to specify the GPU configuration required
# to fit the model into VRAM, and the quantization method (`bitsandbytes` or `gptq`) if desired.
# Note that quantization does degrade token generation performance significantly.
#
# Any model supported by TGI can be chosen here.

MODEL_ID = "meta-llama/Meta-Llama-3-70B-Instruct"
MODEL_REVISION = "81ca4500337d94476bda61d84f0c93af67e4495f"
# Add `["--quantize", "gptq"]` for TheBloke GPTQ models.
LAUNCH_FLAGS = [
    "--model-id",
    MODEL_ID,
    "--port",
    "8000",
    "--revision",
    MODEL_REVISION,
]

# ## Define a container image
#
# We want to create a Modal image which has the Huggingface model cache pre-populated.
# The benefit of this is that the container no longer has to re-download the model from Huggingface -
# instead, it will take advantage of Modal's internal filesystem for faster cold starts. On
# the largest 70B model, the 135GB model can be loaded in as little as 70 seconds.
#
# ### Download the weights
# We can use the included utilities to download the model weights (and convert to safetensors, if necessary)
# as part of the image build.
#


def download_model():
    subprocess.run(
        [
            "text-generation-server",
            "download-weights",
            MODEL_ID,
            "--revision",
            MODEL_REVISION,
        ],
    )


# ### Image definition
# We’ll start from a Docker Hub image recommended by TGI, and override the default `ENTRYPOINT` for
# Modal to run its own which enables seamless serverless deployments.
#
# Next we run the download step to pre-populate the image with our model weights.
#
# For this step to work on a [gated model](https://github.com/huggingface/text-generation-inference#using-a-private-or-gated-model)
# such as LLaMA 3, the `HF_TOKEN` environment variable must be set.
#
# After [creating a HuggingFace access token](https://huggingface.co/settings/tokens)
# and accepting the [LLaMA 3 license](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct),
# head to the [secrets page](https://modal.com/secrets) to share it with Modal
#
# Finally, we install the `text-generation` client to interface with TGI's Rust webserver over `localhost`.

app = App(
    "example-tgi-" + MODEL_ID.split("/")[-1]
)  # Note: prior to April 2024, "app" was called "stub"

tgi_image = (
    Image.from_registry("ghcr.io/huggingface/text-generation-inference:1.4")
    .dockerfile_commands("ENTRYPOINT []")
    .run_function(
        download_model,
        secrets=[Secret.from_name("huggingface-secret")],
        timeout=3600,
    )
    .pip_install("text-generation")
)


# ## The model class
#
# The inference function is best represented with Modal's [class syntax](https://modal.com/docs/guide/lifecycle-functions).
# The class syntax is a special representation for a Modal function which splits logic into two parts:
# 1. the `@enter()` function, which runs once per container when it starts up, and
# 2. the `@method()` function, which runs per inference request.
#
# This means the model is loaded into the GPUs, and the backend for TGI is launched just once when each
# container starts, and this state is cached for each subsequent invocation of the function.
# Note that on start-up, we must wait for the Rust webserver to accept connections before considering the
# container ready.
#
# Here, we also
# - specify the secret so the `HUGGING_FACE_HUB_TOKEN` environment variable can be set
# - specify how many A100s we need per container
# - specify that each container is allowed to handle up to 10 inputs (i.e. requests) simultaneously
# - keep idle containers for 10 minutes before spinning down
# - increase the timeout limit


GPU_CONFIG = gpu.H100(count=2)  # 2 H100s


@app.cls(
    secrets=[Secret.from_name("huggingface-secret")],
    gpu=GPU_CONFIG,
    allow_concurrent_inputs=15,
    container_idle_timeout=60 * 10,
    timeout=60 * 60,
    image=tgi_image,
)
class Model:
    @enter()
    def start_server(self):
        import socket
        import time

        from text_generation import AsyncClient

        self.launcher = subprocess.Popen(
            ["text-generation-launcher"] + LAUNCH_FLAGS,
            env={
                **os.environ,
                "HUGGING_FACE_HUB_TOKEN": os.environ["HF_TOKEN"],
            },
        )
        self.client = AsyncClient("http://127.0.0.1:8000", timeout=60)
        self.template = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        # Poll until webserver at 127.0.0.1:8000 accepts connections before running inputs.
        def webserver_ready():
            try:
                socket.create_connection(("127.0.0.1", 8000), timeout=1).close()
                return True
            except (socket.timeout, ConnectionRefusedError):
                # Check if launcher webserving process has exited.
                # If so, a connection can never be made.
                retcode = self.launcher.poll()
                if retcode is not None:
                    raise RuntimeError(
                        f"launcher exited unexpectedly with code {retcode}"
                    )
                return False

        while not webserver_ready():
            time.sleep(1.0)

        print("Webserver ready!")

    @exit()
    def terminate_server(self):
        self.launcher.terminate()

    @method()
    async def generate(self, question: str):
        prompt = self.template.format(user=question)
        result = await self.client.generate(
            prompt, max_new_tokens=1024, stop_sequences=["<|eot_id|>"]
        )

        return result.generated_text

    @method()
    async def generate_stream(self, question: str):
        prompt = self.template.format(user=question)

        async for response in self.client.generate_stream(
            prompt, max_new_tokens=1024, stop_sequences=["<|eot_id|>"]
        ):
            if (
                not response.token.special
                and response.token.text != "<|eot_id|>"
            ):
                yield response.token.text


# ## Run the model
# We define a [`local_entrypoint`](https://modal.com/docs/guide/apps#entrypoints-for-ephemeral-apps) to invoke
# our remote function. You can run this script locally with `modal run text_generation_inference.py`.
@app.local_entrypoint()
def main(prompt: str = None):
    if prompt is None:
        prompt = "Implement a Python function to compute the Fibonacci numbers."
    print(Model().generate.remote(prompt))


# ## Serve the model
# Once we deploy this model with `modal deploy text_generation_inference.py`, we can serve it
# behind an ASGI app front-end. The front-end code (a single file of Alpine.js) is available
# [here](https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-frontend/index.html).
#
# You can try our deployment [here](https://modal.chat).

frontend_path = Path(__file__).parent.parent / "llm-frontend"


@app.function(
    mounts=[Mount.from_local_dir(frontend_path, remote_path="/assets")],
    keep_warm=1,
    allow_concurrent_inputs=10,
    timeout=60 * 10,
)
@asgi_app(label="llama3")
def tgi_app():
    import json

    import fastapi
    import fastapi.staticfiles
    from fastapi.responses import StreamingResponse

    web_app = fastapi.FastAPI()

    @web_app.get("/stats")
    async def stats():
        stats = await Model().generate_stream.get_current_stats.aio()
        return {
            "backlog": stats.backlog,
            "num_total_runners": stats.num_total_runners,
            "model": MODEL_ID,
        }

    @web_app.get("/completion/{question}")
    async def completion(question: str):
        from urllib.parse import unquote

        async def generate():
            async for text in Model().generate_stream.remote_gen.aio(
                unquote(question)
            ):
                yield f"data: {json.dumps(dict(text=text), ensure_ascii=False)}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    web_app.mount(
        "/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True)
    )
    return web_app


# ## Invoke the model from other apps
# Once the model is deployed, we can invoke inference from other apps, sharing the same pool
# of GPU containers with all other apps we might need.
#
# ```
# $ python
# >>> import modal
# >>> f = modal.Function.lookup("example-tgi-Meta-Llama-3-70B-Instruct", "Model.generate")
# >>> f.remote("What is the story about the fox and grapes?")
# 'The story about the fox and grapes ...
# ```


---

## tgi mixtral

# # Hosting Mixtral 8x7B with Text Generation Inference (TGI)
#
# In this example, we show how to run an optimized inference server using [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference)
# with performance advantages over standard text generation pipelines including:
# - continuous batching, so multiple generations can take place at the same time on a single container
# - PagedAttention, which applies memory paging to the attention mechanism's key-value cache, increasing throughput
#
# This example deployment, [accessible here](https://modal-labs--tgi-mixtral.modal.run), can serve Mixtral 8x7B on two 80GB A100s, with
# up to 500 tokens/s of throughput and per-token latency of 78ms.

# ## Setup
#
# First we import the components we need from `modal`.

import os
import subprocess
from pathlib import Path

from modal import App, Image, Mount, Secret, asgi_app, enter, exit, gpu, method

# Next, we set which model to serve, taking care to specify the GPU configuration required
# to fit the model into VRAM, and the quantization method (`bitsandbytes` or `gptq`) if desired.
# Note that quantization does degrade token generation performance significantly.
#
# Any model supported by TGI can be chosen here.

GPU_CONFIG = gpu.A100(memory=40, count=4)
MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
MODEL_REVISION = "f1ca00645f0b1565c7f9a1c863d2be6ebf896b04"
# Add `["--quantize", "gptq"]` for TheBloke GPTQ models.
LAUNCH_FLAGS = [
    "--model-id",
    MODEL_ID,
    "--revision",
    MODEL_REVISION,
    "--port",
    "8000",
]

# ## Define a container image
#
# We want to create a Modal image which has the Hugging Face model cache pre-populated.
# The benefit of this is that the container no longer has to re-download the model from Huggingface -
# instead, it will take advantage of Modal's internal filesystem for faster cold starts.
# The 95GB model can be loaded in as little as 70 seconds.
#
# ### Download the weights
# We can use the included utilities to download the model weights (and convert to safetensors, if necessary)
# as part of the image build.
#
# For this step to work on a [gated model](https://huggingface.co/docs/text-generation-inference/en/basic_tutorials/gated_model_access)
# like Mixtral 8x7B, the `HF_TOKEN` environment variable must be set.
#
# After [creating a HuggingFace access token](https://huggingface.co/settings/tokens)
# and accepting the [terms of use](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1),
# head to the [secrets page](https://modal.com/secrets) to share it with Modal as `huggingface-secret`.


def download_model():
    # the secret name is different for TGI and for transformers
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    subprocess.run(
        [
            "text-generation-server",
            "download-weights",
            MODEL_ID,
            "--revision",
            MODEL_REVISION,
        ]
    )


# ### Image definition
# We’ll start from a Docker Hub image recommended by TGI, and override the default `ENTRYPOINT` for
# Modal to run its own which enables seamless serverless deployments.
#
# Next we run the download step to pre-populate the image with our model weights.
#
# Finally, we install the `text-generation` client to interface with TGI's Rust webserver over `localhost`.

tgi_image = (
    Image.from_registry("ghcr.io/huggingface/text-generation-inference:1.3.3")
    .dockerfile_commands("ENTRYPOINT []")
    .run_function(
        download_model,
        timeout=60 * 20,
        secrets=[Secret.from_name("huggingface-secret")],
    )
    .pip_install("text-generation")
)

app = App(
    "example-tgi-mixtral"
)  # Note: prior to April 2024, "app" was called "stub"


# ## The model class
#
# The inference function is best represented with Modal's [class syntax](https://modal.com/docs/guide/lifecycle-functions).
# The class syntax is a special representation for a Modal function which splits logic into two parts:
# 1. the `@enter()` function, which runs once per container when it starts up, and
# 2. the `@method()` function, which runs per inference request.
#
# This means the model is loaded into the GPUs, and the backend for TGI is launched just once when each
# container starts, and this state is cached for each subsequent invocation of the function.
# Note that on start-up, we must wait for the Rust webserver to accept connections before considering the
# container ready.
#
# Here, we also
# - specify how many A100s we need per container
# - specify that each container is allowed to handle up to 10 inputs (i.e. requests) simultaneously
# - keep idle containers for 10 minutes before spinning down
# - lift the timeout of each request.


@app.cls(
    secrets=[Secret.from_name("huggingface-secret")],
    gpu=GPU_CONFIG,
    allow_concurrent_inputs=10,
    container_idle_timeout=60 * 10,
    timeout=60 * 60,
    image=tgi_image,
)
class Model:
    @enter()
    def start_server(self):
        import socket
        import time

        from text_generation import AsyncClient

        self.launcher = subprocess.Popen(
            ["text-generation-launcher"] + LAUNCH_FLAGS,
            env={
                **os.environ,
                "HUGGING_FACE_HUB_TOKEN": os.environ["HF_TOKEN"],
            },
        )
        self.client = AsyncClient("http://127.0.0.1:8000", timeout=60)
        self.template = "[INST] {user} [/INST]"

        # Poll until webserver at 127.0.0.1:8000 accepts connections before running inputs.
        webserver_ready = False
        while not webserver_ready:
            try:
                socket.create_connection(("127.0.0.1", 8000), timeout=1).close()
                webserver_ready = True
                print("Webserver ready!")
            except (socket.timeout, ConnectionRefusedError):
                # If launcher process exited, a connection can never be made.
                if retcode := self.launcher.poll():
                    raise RuntimeError(f"launcher exited with code {retcode}")
                time.sleep(1.0)

    @exit()
    def terminate_server(self):
        self.launcher.terminate()

    @method()
    async def generate(self, question: str):
        prompt = self.template.format(user=question)
        result = await self.client.generate(prompt, max_new_tokens=1024)

        return result.generated_text

    @method()
    async def generate_stream(self, question: str):
        prompt = self.template.format(user=question)

        async for response in self.client.generate_stream(
            prompt, max_new_tokens=1024
        ):
            if not response.token.special:
                yield response.token.text


# ## Run the model
# We define a [`local_entrypoint`](https://modal.com/docs/guide/apps#entrypoints-for-ephemeral-apps) to invoke
# our remote function. You can run this script locally with `modal run text_generation_inference.py`.
@app.local_entrypoint()
def main():
    print(
        Model().generate.remote(
            "Implement a Python function to compute the Fibonacci numbers."
        )
    )


# ## Serve the model
# Once we deploy this model with `modal deploy text_generation_inference.py`, we can serve it
# behind an ASGI app front-end. The front-end code (a single file of Alpine.js) is available
# [here](https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-frontend/index.html).
#
# You can try our deployment [here](https://modal-labs--tgi-mixtral.modal.run).

frontend_path = Path(__file__).parent.parent / "llm-frontend"


@app.function(
    mounts=[Mount.from_local_dir(frontend_path, remote_path="/assets")],
    keep_warm=1,
    allow_concurrent_inputs=20,
    timeout=60 * 10,
)
@asgi_app()
def tgi_mixtral():
    import json

    import fastapi
    import fastapi.staticfiles
    from fastapi.responses import StreamingResponse

    web_app = fastapi.FastAPI()

    @web_app.get("/stats")
    async def stats():
        stats = await Model().generate_stream.get_current_stats.aio()
        return {
            "backlog": stats.backlog,
            "num_total_runners": stats.num_total_runners,
            "model": MODEL_ID + " (TGI)",
        }

    @web_app.get("/completion/{question}")
    async def completion(question: str):
        from urllib.parse import unquote

        async def generate():
            async for text in Model().generate_stream.remote_gen.aio(
                unquote(question)
            ):
                yield f"data: {json.dumps(dict(text=text), ensure_ascii=False)}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    web_app.mount(
        "/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True)
    )
    return web_app


# ## Invoke the model from other apps
# Once the model is deployed, we can invoke inference from other apps, sharing the same pool
# of GPU containers with all other apps we might need.
#
# ```
# $ python
# >>> import modal
# >>> f = modal.Function.lookup("example-tgi-Mixtral-8x7B-Instruct-v0.1", "Model.generate")
# >>> f.remote("What is the story about the fox and grapes?")
# 'The story about the fox and grapes ...
# ```


---

## trtllm llama

# # Serverless TensorRT-LLM (LLaMA 3 8B)
#
# In this example, we demonstrate how to use the TensorRT-LLM framework to serve Meta's LLaMA 3 8B model
# at a total throughput of roughly 4,500 output tokens per second on a single NVIDIA A100 40GB GPU.
# At [Modal's on-demand rate](https://modal.com/pricing) of ~$4/hr, that's under $0.20 per million tokens --
# on auto-scaling infrastructure and served via a customizable API.
#
# Additional optimizations like speculative sampling and FP8 quantization can further improve throughput.
# For more on the throughput levels that are possible with TensorRT-LLM for different combinations
# of model, hardware, and workload, see the
# [official benchmarks](https://github.com/NVIDIA/TensorRT-LLM/blob/71d8d4d3dc655671f32535d6d2b60cab87f36e87/docs/source/performance.md).
#
# ## Overview
#
# This guide is intended to document two things:
# the general process for building TensorRT-LLM on Modal
# and a specific configuration for serving the LLaMA 3 8B model.
#
# ### Build process
#
# Any given TensorRT-LLM service requires a multi-stage build process,
# starting from model weights and ending with a compiled engine.
# Because that process touches many sharp-edged high-performance components
# across the stack, it can easily go wrong in subtle and hard-to-debug ways
# that are idiosyncratic to specific systems.
# And debugging GPU workloads is expensive!
#
# This example builds an entire service from scratch, from downloading weight tensors
# to responding to requests, and so serves as living, interactive documentation of a TensorRT-LLM
# build process that works on Modal.
#
# ### Engine configuration
#
# TensorRT-LLM is the Lamborghini of inference engines: it achieves seriously
# impressive performance, but only if you tune it carefully.
# We carefully document the choices we made here and point to additional resources
# so you know where and how you might adjust the parameters for your use case.
#
# ## Installing TensorRT-LLM
#
# To run TensorRT-LLM, we must first install it. Easier said than done!
#
# In Modal, we define [container images](https://modal.com/docs/guide/custom-containers) that run our serverless workloads.
# All Modal containers have access to GPU drivers via the underlying host environment,
# but we still need to install the software stack on top of the drivers, from the CUDA runtime up.
#
# We start from the official `nvidia/cuda:12.1.1-devel-ubuntu22.04` image,
# which includes the CUDA runtime & development libraries
# and the environment configuration necessary to run them.

import modal

tensorrt_image = modal.Image.from_registry(
    "nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10"
)

# On top of that, we add some system dependencies of TensorRT-LLM,
# including OpenMPI for distributed communication, some core software like `git`,
# and the `tensorrt_llm` package itself.

tensorrt_image = tensorrt_image.apt_install(
    "openmpi-bin", "libopenmpi-dev", "git", "git-lfs", "wget"
).pip_install(
    "tensorrt_llm==0.10.0.dev2024042300",
    pre=True,
    extra_index_url="https://pypi.nvidia.com",
)

# Note that we're doing this by [method-chaining](https://quanticdev.com/articles/method-chaining/)
# a number of calls to methods on the `modal.Image`. If you're familiar with
# Dockerfiles, you can think of this as a Pythonic interface to instructions like `RUN` and `CMD`.
#
# End-to-end, this step takes five minutes.
# If you're reading this from top to bottom,
# you might want to stop here and execute the example
# with `modal run trtllm_llama.py`
# so that it runs in the background while you read the rest.
#
# ## Downloading the Model
#
# Next, we download the model we want to serve. In this case, we're using the instruction-tuned
# version of Meta's Llama 3 8B model.
# We use the function below to download the model from the Hugging Face Hub.

MODEL_DIR = "/root/model/model_input"
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_REVISION = "7840f95a8c7a781d3f89c4818bf693431ab3119a"  # pin model revisions to prevent unexpected changes!


def download_model():
    import os

    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(MODEL_DIR, exist_ok=True)
    snapshot_download(
        MODEL_ID,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt", "*.bin"],  # using safetensors
        revision=MODEL_REVISION,
    )
    move_cache()


# Just defining that function doesn't actually download the model, though.
# We can run it by adding it to the image's build process with `run_function`.
# The download process has its own dependencies, which we add here.

MINUTES = 60  # seconds
tensorrt_image = (  # update the image by downloading the model we're using
    tensorrt_image.pip_install(  # add utilities for downloading the model
        "hf-transfer==0.1.6",
        "huggingface_hub==0.22.2",
        "requests~=2.31.0",
    )
    .env(  # hf-transfer: faster downloads, but fewer comforts
        {"HF_HUB_ENABLE_HF_TRANSFER": "1"}
    )
    .run_function(  # download the model
        download_model,
        timeout=20 * MINUTES,
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
)

# ## Configuring the model
#
# Now that we have the model downloaded, we need to convert it to a format that TensorRT-LLM can use.
# We use a convenience script provided by the TensorRT-LLM team.
# This script takes a few minutes to run.

GIT_HASH = "71d8d4d3dc655671f32535d6d2b60cab87f36e87"
CHECKPOINT_SCRIPT_URL = f"https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/{GIT_HASH}/examples/llama/convert_checkpoint.py"

# TensorRT-LLM requires that a GPU be present to load the model, even though it isn't used directly during this conversion process.
# We'll use a single A100-40GB GPU for this example, but we have also tested it successfully with A10G, A100-80GB, and H100 GPUs.
#
# The most important feature to track when selecting hardware to run on is GPU RAM:
# larger models, longer sequences, and bigger batches all require more memory,
# We tuned all three to maximize throughput on this example.
#
# The amount of GPU RAM on a single card is a tight constraint for most LLMs:
# RAM is measured in tens of gigabytes and
# models have billions of floating point parameters,
# each consuming one to four bytes of memory.
# The performance cliff if you need to spill to CPU memory is steep,
# so the only solution is to split the model across multiple GPUs.
# This is particularly important when serving larger models (e.g. 70B or 8x22B).

N_GPUS = 1  # Heads up: this example has not yet been tested with multiple GPUs
GPU_CONFIG = modal.gpu.A100(count=N_GPUS)

# This is also the point where we specify the data type for this model.
# We use IEEE 754-compliant half-precision floats, (`float16`), because we found that it resulted in marginally higher throughput,
# but the model is provided in Google's
# [`bfloat16` format](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format).
# On the latest Ada Lovelace chips, you might use `float8` to reduce GPU RAM usage and speed up inference,
# but note that the FP8 format is very new, so expect rough edges.

DTYPE = "float16"

# We put that all together with another invocation of `.run_commands`.

CKPT_DIR = "/root/model/model_ckpt"
tensorrt_image = (  # update the image by converting the model to TensorRT format
    tensorrt_image.run_commands(  # takes ~5 minutes
        [
            f"wget {CHECKPOINT_SCRIPT_URL} -O /root/convert_checkpoint.py",
            f"python /root/convert_checkpoint.py --model_dir={MODEL_DIR} --output_dir={CKPT_DIR}"
            + f" --tp_size={N_GPUS} --dtype={DTYPE}",
        ],
        gpu=GPU_CONFIG,  # GPU must be present to load tensorrt_llm
    )
)

# ## Compiling the engine
#
# TensorRT-LLM achieves its high throughput primarily by compiling the model:
# making concrete choices of CUDA kernels to execute for each operation.
# These kernels are much more specific than `matrix_multiply` or `softmax` --
# they have names like `maxwell_scudnn_winograd_128x128_ldg1_ldg4_tile148t_nt`.
# They are optimized for the specific types and shapes of tensors that the model uses
# and for the specific hardware that the model runs on.
#
# That means we need to know all of that information a priori --
# more like the original TensorFlow, which defined static graphs, than like PyTorch,
# which builds up a graph of kernels dynamically at runtime.
#
# This extra layer of constraint on our LLM service is precisely
# what allows TensorRT-LLM to achieve its high throughput.
#
# So we need to specify things like the maximum batch size and the lengths of inputs and outputs.
# The closer these are to the actual values we'll use in production, the better the throughput we'll get.

MAX_INPUT_LEN, MAX_OUTPUT_LEN = 256, 256
MAX_BATCH_SIZE = (
    128  # better throughput at larger batch sizes, limited by GPU RAM
)
ENGINE_DIR = "/root/model/model_output"

SIZE_ARGS = f"--max_batch_size={MAX_BATCH_SIZE} --max_input_len={MAX_INPUT_LEN} --max_output_len={MAX_OUTPUT_LEN}"

# There are many additional options you can pass to `trtllm-build` to tune the engine for your specific workload.
# You can find the document we used for LLaMA
# [here](https://github.com/NVIDIA/TensorRT-LLM/tree/66ef1df492f7bc9c8eeb01d7e14db01838e3f0bd/examples/llama),
# which you can use to adjust the arguments to fit your workloads,
# e.g. adjusting rotary embeddings and block sizes for longer contexts.
#
# We selected plugins that accelerate two core components of the model: dense matrix multiplication and attention.
# You can read more about the plugin options [here](https://fetch.ai/blog/advancing-llm-optimization).

PLUGIN_ARGS = f"--gemm_plugin={DTYPE} --gpt_attention_plugin={DTYPE}"

# We put all of this together with another invocation of `.run_commands`.

tensorrt_image = (  # update the image by building the TensorRT engine
    tensorrt_image.run_commands(  # takes ~5 minutes
        [
            f"trtllm-build --checkpoint_dir {CKPT_DIR} --output_dir {ENGINE_DIR}"
            + f" --tp_size={N_GPUS} --workers={N_GPUS}"
            + f" {SIZE_ARGS}"
            + f" {PLUGIN_ARGS}"
        ],
        gpu=GPU_CONFIG,  # TRT-LLM compilation is GPU-specific, so make sure this matches production!
    ).env(  # show more log information from the inference engine
        {"TLLM_LOG_LEVEL": "INFO"}
    )
)

# ## Serving inference at thousands of tokens per second
#
# Now that we have the engine compiled, we can serve it with Modal by creating an `App`.

app = modal.App(f"example-trtllm-{MODEL_ID}", image=tensorrt_image)

# Thanks to our custom container runtime system, even this
# large, many gigabyte container boots in seconds.
#
# At container start time, we boot up the engine, which completes in under 30 seconds.
# Container starts are triggered when Modal scales up your infrastructure,
# like the first time you run this code or the first time a request comes in after a period of inactivity.
#
# Container lifecycles in Modal are managed via our `Cls` interface, so we define one below
# to manage the engine and run inference.
# For details, see [this guide](https://modal.com/docs/guide/lifecycle-functions).


@app.cls(
    gpu=GPU_CONFIG,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    container_idle_timeout=10 * MINUTES,
)
class Model:
    @modal.enter()
    def load(self):
        """Loads the TRT-LLM engine and configures our tokenizer.

        The @enter decorator ensures that it runs only once per container, when it starts."""
        import time

        print(
            f"{COLOR['HEADER']}🥶 Cold boot: spinning up TRT-LLM engine{COLOR['ENDC']}"
        )
        self.init_start = time.monotonic_ns()

        import tensorrt_llm
        from tensorrt_llm.runtime import ModelRunner
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        # LLaMA models do not have a padding token, so we use the EOS token
        self.tokenizer.add_special_tokens(
            {"pad_token": self.tokenizer.eos_token}
        )
        # and then we add it from the left, to minimize impact on the output
        self.tokenizer.padding_side = "left"
        self.pad_id = self.tokenizer.pad_token_id
        self.end_id = self.tokenizer.eos_token_id

        runner_kwargs = dict(
            engine_dir=f"{ENGINE_DIR}",
            lora_dir=None,
            rank=tensorrt_llm.mpi_rank(),  # this will need to be adjusted to use multiple GPUs
        )

        self.model = ModelRunner.from_dir(**runner_kwargs)

        self.init_duration_s = (time.monotonic_ns() - self.init_start) / 1e9
        print(
            f"{COLOR['HEADER']}🚀 Cold boot finished in {self.init_duration_s}s{COLOR['ENDC']}"
        )

    @modal.method()
    def generate(self, prompts: list[str], settings=None):
        """Generate responses to a batch of prompts, optionally with custom inference settings."""
        import time

        if settings is None:
            settings = dict(
                temperature=0.1,  # temperature 0 not allowed, so we set top_k to 1 to get the same effect
                top_k=1,
                stop_words_list=None,
                repetition_penalty=1.1,
            )

        settings[
            "max_new_tokens"
        ] = MAX_OUTPUT_LEN  # exceeding this will raise an error
        settings["end_id"] = self.end_id
        settings["pad_id"] = self.pad_id

        num_prompts = len(prompts)

        if num_prompts > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {num_prompts} exceeds maximum of {MAX_BATCH_SIZE}"
            )

        print(
            f"{COLOR['HEADER']}🚀 Generating completions for batch of size {num_prompts}...{COLOR['ENDC']}"
        )
        start = time.monotonic_ns()

        parsed_prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for prompt in prompts
        ]

        print(
            f"{COLOR['HEADER']}Parsed prompts:{COLOR['ENDC']}",
            *parsed_prompts,
            sep="\n\t",
        )

        inputs_t = self.tokenizer(
            parsed_prompts, return_tensors="pt", padding=True, truncation=False
        )["input_ids"]

        print(
            f"{COLOR['HEADER']}Input tensors:{COLOR['ENDC']}", inputs_t[:, :8]
        )

        outputs_t = self.model.generate(inputs_t, **settings)

        outputs_text = self.tokenizer.batch_decode(
            outputs_t[:, 0]
        )  # only one output per input, so we index with 0

        responses = [
            extract_assistant_response(output_text)
            for output_text in outputs_text
        ]
        duration_s = (time.monotonic_ns() - start) / 1e9

        num_tokens = sum(
            map(lambda r: len(self.tokenizer.encode(r)), responses)
        )

        for prompt, response in zip(prompts, responses):
            print(
                f"{COLOR['HEADER']}{COLOR['GREEN']}{prompt}",
                f"\n{COLOR['BLUE']}{response}",
                "\n\n",
                sep=COLOR["ENDC"],
            )
            time.sleep(0.01)  # to avoid log truncation

        print(
            f"{COLOR['HEADER']}{COLOR['GREEN']}Generated {num_tokens} tokens from {MODEL_ID} in {duration_s:.1f} seconds,"
            f" throughput = {num_tokens / duration_s:.0f} tokens/second for batch of size {num_prompts} on {GPU_CONFIG}.{COLOR['ENDC']}"
        )

        return responses


# ## Calling our inference function
#
# Now, how do we actually run the model?
#
# There are two basic methods: from Python via our SDK or from anywhere, by setting up an API.
#
# ### Calling inference from Python
#
# To run our `Model`'s `.generate` method from Python, we just need to call it --
# with `.remote` appended to run it on Modal.
#
# We wrap that logic in a `local_entrypoint` so you can run it from the command line with
# ```bash
# modal run trtllm_llama.py
# ```
#
# For simplicity, we hard-code a batch of 128 questions to ask the model.


@app.local_entrypoint()
def main():
    questions = [
        # Generic assistant questions
        "What are you?",
        "What can you do?",
        # Coding
        "Implement a Python function to compute the Fibonacci numbers.",
        "Write a Rust function that performs binary exponentiation.",
        "How do I allocate memory in C?",
        "What are the differences between Javascript and Python?",
        "How do I find invalid indices in Postgres?",
        "How can you implement a LRU (Least Recently Used) cache in Python?",
        "What approach would you use to detect and prevent race conditions in a multithreaded application?",
        "Can you explain how a decision tree algorithm works in machine learning?",
        "How would you design a simple key-value store database from scratch?",
        "How do you handle deadlock situations in concurrent programming?",
        "What is the logic behind the A* search algorithm, and where is it used?",
        "How can you design an efficient autocomplete system?",
        "What approach would you take to design a secure session management system in a web application?",
        "How would you handle collision in a hash table?",
        "How can you implement a load balancer for a distributed system?",
        "Implement a Python class for a doubly linked list.",
        "Write a Haskell function that generates prime numbers using the Sieve of Eratosthenes.",
        "Develop a simple HTTP server in Rust.",
        # Literate and creative writing
        "What is the fable involving a fox and grapes?",
        "Who does Harry turn into a balloon?",
        "Write a story in the style of James Joyce about a trip to the Australian outback in 2083 to see robots in the beautiful desert.",
        "Write a tale about a time-traveling historian who's determined to witness the most significant events in human history.",
        "Describe a day in the life of a secret agent who's also a full-time parent.",
        "Create a story about a detective who can communicate with animals.",
        "What is the most unusual thing about living in a city floating in the clouds?",
        "In a world where dreams are shared, what happens when a nightmare invades a peaceful dream?",
        "Describe the adventure of a lifetime for a group of friends who found a map leading to a parallel universe.",
        "Tell a story about a musician who discovers that their music has magical powers.",
        "In a world where people age backwards, describe the life of a 5-year-old man.",
        "Create a tale about a painter whose artwork comes to life every night.",
        "What happens when a poet's verses start to predict future events?",
        "Imagine a world where books can talk. How does a librarian handle them?",
        "Tell a story about an astronaut who discovered a planet populated by plants.",
        "Describe the journey of a letter traveling through the most sophisticated postal service ever.",
        "Write a tale about a chef whose food can evoke memories from the eater's past.",
        "Write a poem in the style of Walt Whitman about the modern digital world.",
        "Create a short story about a society where people can only speak in metaphors.",
        "What are the main themes in Dostoevsky's 'Crime and Punishment'?",
        # History and Philosophy
        "What were the major contributing factors to the fall of the Roman Empire?",
        "How did the invention of the printing press revolutionize European society?",
        "What are the effects of quantitative easing?",
        "How did the Greek philosophers influence economic thought in the ancient world?",
        "What were the economic and philosophical factors that led to the fall of the Soviet Union?",
        "How did decolonization in the 20th century change the geopolitical map?",
        "What was the influence of the Khmer Empire on Southeast Asia's history and culture?",
        "What led to the rise and fall of the Mongol Empire?",
        "Discuss the effects of the Industrial Revolution on urban development in 19th century Europe.",
        "How did the Treaty of Versailles contribute to the outbreak of World War II?",
        "What led to the rise and fall of the Mongol Empire?",
        "Discuss the effects of the Industrial Revolution on urban development in 19th century Europe.",
        "How did the Treaty of Versailles contribute to the outbreak of World War II?",
        "Explain the concept of 'tabula rasa' in John Locke's philosophy.",
        "What does Nietzsche mean by 'ressentiment'?",
        "Compare and contrast the early and late works of Ludwig Wittgenstein. Which do you prefer?",
        "How does the trolley problem explore the ethics of decision-making in critical situations?",
        # Thoughtfulness
        "Describe the city of the future, considering advances in technology, environmental changes, and societal shifts.",
        "In a dystopian future where water is the most valuable commodity, how would society function?",
        "If a scientist discovers immortality, how could this impact society, economy, and the environment?",
        "What could be the potential implications of contact with an advanced alien civilization?",
        "Describe how you would mediate a conflict between two roommates about doing the dishes using techniques of non-violent communication.",
        "If you could design a school curriculum for the future, what subjects would you include to prepare students for the next 50 years?",
        "How would society change if teleportation was invented and widely accessible?",
        "Consider a future where artificial intelligence governs countries. What are the potential benefits and pitfalls?",
        # Math
        "What is the product of 9 and 8?",
        "If a train travels 120 kilometers in 2 hours, what is its average speed?",
        "Think through this step by step. If the sequence a_n is defined by a_1 = 3, a_2 = 5, and a_n = a_(n-1) + a_(n-2) for n > 2, find a_6.",
        "Think through this step by step. Calculate the sum of an arithmetic series with first term 3, last term 35, and total terms 11.",
        "Think through this step by step. What is the area of a triangle with vertices at the points (1,2), (3,-4), and (-2,5)?",
        "Think through this step by step. Solve the following system of linear equations: 3x + 2y = 14, 5x - y = 15.",
        # Facts
        "Who was Emperor Norton I, and what was his significance in San Francisco's history?",
        "What is the Voynich manuscript, and why has it perplexed scholars for centuries?",
        "What was Project A119 and what were its objectives?",
        "What is the 'Dyatlov Pass incident' and why does it remain a mystery?",
        "What is the 'Emu War' that took place in Australia in the 1930s?",
        "What is the 'Phantom Time Hypothesis' proposed by Heribert Illig?",
        "Who was the 'Green Children of Woolpit' as per 12th-century English legend?",
        "What are 'zombie stars' in the context of astronomy?",
        "Who were the 'Dog-Headed Saint' and the 'Lion-Faced Saint' in medieval Christian traditions?",
        "What is the story of the 'Globsters', unidentified organic masses washed up on the shores?",
        "Which countries in the European Union use currencies other than the Euro, and what are those currencies?",
        # Multilingual
        "战国时期最重要的人物是谁?",
        "Tuende hatua kwa hatua. Hesabu jumla ya mfululizo wa kihesabu wenye neno la kwanza 2, neno la mwisho 42, na jumla ya maneno 21.",
        "Kannst du die wichtigsten Eigenschaften und Funktionen des NMDA-Rezeptors beschreiben?",
        "¿Cuáles son los principales impactos ambientales de la deforestación en la Amazonía?",
        "Décris la structure et le rôle de la mitochondrie dans une cellule.",
        "Какие были социальные последствия Перестройки в Советском Союзе?",
        # Economics and Business
        "What are the principles of behavioral economics and how do they influence consumer choices?",
        "Discuss the impact of blockchain technology on traditional banking systems.",
        "What are the long-term effects of trade wars on global economic stability?",
        "What is the law of supply and demand?",
        "Explain the concept of inflation and its typical causes.",
        "What is a trade deficit, and why does it matter?",
        "How do interest rates affect consumer spending and saving?",
        "What is GDP and why is it important for measuring economic health?",
        "What is the difference between revenue and profit?",
        "Describe the role of a business plan in startup success.",
        "How does market segmentation benefit a company?",
        "Explain the concept of brand equity.",
        "What are the advantages of franchising a business?",
        "What are Michael Porter's five forces and how do they impact strategy for tech startups?",
        # Science and Technology
        "Discuss the potential impacts of quantum computing on data security.",
        "How could CRISPR technology change the future of medical treatments?",
        "Explain the significance of graphene in the development of future electronics.",
        "How do renewable energy sources compare to fossil fuels in terms of environmental impact?",
        "What are the most promising technologies for carbon capture and storage?",
        "Explain why the sky is blue.",
        "What is the principle behind the operation of a microwave oven?",
        "How does Newton's third law apply to rocket propulsion?",
        "What causes iron to rust?",
        "Describe the process of photosynthesis in simple terms.",
        "What is the role of a catalyst in a chemical reaction?",
        "What is the basic structure of a DNA molecule?",
        "How do vaccines work to protect the body from disease?",
        "Explain the significance of mitosis in cellular reproduction.",
        "What are tectonic plates and how do they affect earthquakes?",
        "How does the greenhouse effect contribute to global warming?",
        "Describe the water cycle and its importance to Earth's climate.",
        "What causes the phases of the Moon?",
        "How do black holes form?",
        "Explain the significance of the Big Bang theory.",
        "What is the function of the CPU in a computer system?",
        "Explain the difference between RAM and ROM.",
        "How does a solid-state drive (SSD) differ from a hard disk drive (HDD)?",
        "What role does the motherboard play in a computer system?",
        "Describe the purpose and function of a GPU.",
        "What is TensorRT? What role does it play in neural network inference?",
    ]

    model = Model()
    model.generate.remote(questions)
    # if you're calling this service from another Python project,
    # use [`Model.lookup`](https://modal.com/docs/reference/modal.Cls#lookup)


# ### Calling inference via an API
#
# We can use `modal.web_endpoint` and `app.function` to turn any Python function into a web API.
#
# This API wrapper doesn't need all the dependencies of the core inference service,
# so we switch images here to a basic Linux image, `debian_slim`, which has everything we need.

web_image = modal.Image.debian_slim(python_version="3.10")

# From there, we can take the same remote generation logic we used in `main`
# and serve it with only a few more lines of code.


@app.function(image=web_image)
@modal.web_endpoint(method="POST")
def generate_web(data: dict):
    return Model.generate.remote(data["prompts"], settings=None)


# To set our function up as a web endpoint, we need to run this file --
# with `modal serve` to create a hot-reloading development server or `modal deploy` to deploy it to production.
#
# ```bash
# modal serve trtllm_llama.py
# ```
#
# You can test the endpoint by sending a POST request with `curl` from another terminal:
#
# ```bash
# curl -X POST url-from-output-of-modal-serve-here \
# -H "Content-Type: application/json" \
# -d '{
#     "prompts": ["Tell me a joke", "Describe a dream you had recently", "Share your favorite childhood memory"]
# }' | python -m json.tool # python for pretty-printing, optional
# ```
#
# And now you have a high-throughput, low-latency, autoscaling API for serving LLaMA 3 8B completions!
#
# ## Footer
#
# The rest of the code in this example is utility code.


COLOR = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "ENDC": "\033[0m",
}


def extract_assistant_response(output_text):
    """Model-specific code to extract model responses.

    See this doc for LLaMA 3: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/."""
    # Split the output text by the assistant header token
    parts = output_text.split("<|start_header_id|>assistant<|end_header_id|>")

    if len(parts) > 1:
        # Join the parts after the first occurrence of the assistant header token
        response = parts[1].split("<|eot_id|>")[0].strip()

        # Remove any remaining special tokens and whitespace
        response = response.replace("<|eot_id|>", "").strip()

        return response
    else:
        return output_text


---

## vllm gemma

# # Fast inference with vLLM (Gemma 7B)
#
# In this example, we show how to run basic LLM inference, using [`vLLM`](https://github.com/vllm-project/vllm)
# to take advantage of [PagedAttention](https://arxiv.org/abs/2309.06180), which speeds up inference on longer sequences with optimized key-value caching.
# You can read more about PagedAttention [here](https://charlesfrye.github.io/programming/2023/11/10/llms-systems.html).
#
# We'll run the [Gemma 7B Instruct](https://huggingface.co/google/gemma-7b-it) large language model.
# Gemma is the weights-available version of Google's Gemini model series.
#
# The "7B" in the name refers to the number of parameters (floating point numbers used to control inference)
# in the model. Applying those 7,000,000,000 numbers onto an input is a lot of work,
# so we'll use a GPU to speed up the process -- specifically, a top-of-the-line [NVIDIA H100](https://modal.com/blog/introducing-h100).
#
# "Instruct" means that this version of Gemma is not simply a statistical model of language,
# but has been fine-tuned to follow instructions -- like ChatGPT or Claude,
# it is a model of an assistant that can understand and follow instructions.
#
# You can expect cold starts in under 30 seconds and well over 1000 tokens/second throughput.
# The larger the batch of prompts, the higher the throughput. For example, with the 64 prompts below,
# we can produce nearly 15k tokens with a latency just over 5 seconds, for a throughput of >2.5k tokens/second.
# That's a lot of text!
#
#
# To run
# [any of the other supported models](https://vllm.readthedocs.io/en/latest/models/supported_models.html),
# just change the model name. You may also need to change engine configuration, like `trust_remote_code`,
# or GPU configuration, in order to run some models.
#
# ## Setup
#
# First we import the components we need from `modal`.

import os
import time

import modal

MODEL_DIR = "/model"
MODEL_NAME = "google/gemma-7b-it"


# ## Define a container image
#
# We want to create a Modal image which has the model weights pre-saved to a directory. The benefit of this
# is that the container no longer has to re-download the model from Hugging Face - instead, it will take
# advantage of Modal's internal filesystem for faster cold starts.
#
# ### Download the weights
# Make sure you have created a [HuggingFace access token](https://huggingface.co/settings/tokens).
# To access the token in a Modal function, we can create a secret on the [secrets page](https://modal.com/secrets).
# Now the token will be available via the environment variable named `HF_TOKEN`. Functions that inject this secret
# will have access to the environment variable.
#
# We can download the model to a particular directory using the HuggingFace utility function `snapshot_download`.
#
# You may need to accept the license agreement from an account associated with that Hugging Face Token
# to download the model.
def download_model_to_image(model_dir, model_name):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(model_dir, exist_ok=True)

    snapshot_download(
        model_name,
        local_dir=model_dir,
        token=os.environ["HF_TOKEN"],
        ignore_patterns=["*.pt", "*.gguf"],  # Using safetensors
    )
    move_cache()


# ### Image definition
# We’ll start from a Docker Hub image by NVIDIA and install `vLLM`.
# Then we’ll use `run_function` to execute `download_model_to_image`
# and save the resulting files to the container image -- that way we don't need
# to redownload the weights every time we change the server's code or start up more instances of the server.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "vllm==0.4.0.post1",
        "torch==2.1.2",
        "transformers==4.39.3",
        "ray==2.10.0",
        "huggingface_hub==0.19.4",
        "hf-transfer==0.1.4",
    )
    # Use the barebones hf-transfer package for maximum download speeds. Varies from 100MB/s to 1.5 GB/s,
    # so download times can vary from under a minute to tens of minutes.
    # If your download slows down or times out, try interrupting and restarting.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_image,
        secrets=[modal.Secret.from_name("huggingface-secret")],
        timeout=60 * 20,
        kwargs={"model_dir": MODEL_DIR, "model_name": MODEL_NAME},
    )
)

app = modal.App(
    f"example-vllm-{MODEL_NAME}", image=image
)  # Note: prior to April 2024, "app" was called "stub"

# Using `image.imports` allows us to have a reference to vLLM in global scope without getting an error when our script executes locally.
with image.imports():
    import vllm

# ## Encapulate the model in a class
#
# The inference function is best represented with Modal's [class syntax](/docs/guide/lifecycle-functions) and the `@enter` decorator.
# This enables us to load the model into memory just once every time a container starts up, and keep it cached
# on the GPU for each subsequent invocation of the function.
#
# The `vLLM` library allows the code to remain quite clean!

GPU_CONFIG = modal.gpu.H100(count=1)


@app.cls(gpu=GPU_CONFIG, secrets=[modal.Secret.from_name("huggingface-secret")])
class Model:
    @modal.enter()
    def load(self):
        self.template = (
            "<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n"
        )

        # Load the model. Tip: Some models, like MPT, may require `trust_remote_code=true`.
        self.llm = vllm.LLM(
            MODEL_DIR,
            enforce_eager=True,  # skip graph capturing for faster cold starts
            tensor_parallel_size=GPU_CONFIG.count,
        )

    @modal.method()
    def generate(self, user_questions):
        prompts = [self.template.format(user=q) for q in user_questions]

        sampling_params = vllm.SamplingParams(
            temperature=0.75,
            top_p=0.99,
            max_tokens=256,
            presence_penalty=1.15,
        )
        start = time.monotonic_ns()
        result = self.llm.generate(prompts, sampling_params)
        duration_s = (time.monotonic_ns() - start) / 1e9
        num_tokens = 0

        COLOR = {
            "HEADER": "\033[95m",
            "BLUE": "\033[94m",
            "GREEN": "\033[92m",
            "RED": "\033[91m",
            "ENDC": "\033[0m",
        }

        for output in result:
            num_tokens += len(output.outputs[0].token_ids)
            print(
                f"{COLOR['HEADER']}{COLOR['GREEN']}{output.prompt}",
                f"\n{COLOR['BLUE']}{output.outputs[0].text}",
                "\n\n",
                sep=COLOR["ENDC"],
            )
            time.sleep(0.01)
        print(
            f"{COLOR['HEADER']}{COLOR['GREEN']}Generated {num_tokens} tokens from {MODEL_NAME} in {duration_s:.1f} seconds,"
            f" throughput = {num_tokens / duration_s:.0f} tokens/second on {GPU_CONFIG}.{COLOR['ENDC']}"
        )

    @modal.exit()
    def stop_engine(self):
        if GPU_CONFIG.count > 1:
            import ray

            ray.shutdown()


# ## Run the model
# We define a [`local_entrypoint`](/docs/guide/apps#entrypoints-for-ephemeral-apps) to call our remote function
# sequentially for a list of inputs. Run it by executing the command `modal run vllm_inference.py`.
#
# The examples below are meant to put the model through its paces, with a variety of questions and prompts.
# We also calculate the throughput and latency we achieve.
@app.local_entrypoint()
def main():
    questions = [
        # Coding questions
        "Implement a Python function to compute the Fibonacci numbers.",
        "Write a Rust function that performs binary exponentiation.",
        "How do I allocate memory in C?",
        "What are the differences between Javascript and Python?",
        "How do I find invalid indices in Postgres?",
        "How can you implement a LRU (Least Recently Used) cache in Python?",
        "What approach would you use to detect and prevent race conditions in a multithreaded application?",
        "Can you explain how a decision tree algorithm works in machine learning?",
        "How would you design a simple key-value store database from scratch?",
        "How do you handle deadlock situations in concurrent programming?",
        "What is the logic behind the A* search algorithm, and where is it used?",
        "How can you design an efficient autocomplete system?",
        "What approach would you take to design a secure session management system in a web application?",
        "How would you handle collision in a hash table?",
        "How can you implement a load balancer for a distributed system?",
        # Literature
        "What is the fable involving a fox and grapes?",
        "Write a story in the style of James Joyce about a trip to the Australian outback in 2083, to see robots in the beautiful desert.",
        "Who does Harry turn into a balloon?",
        "Write a tale about a time-traveling historian who's determined to witness the most significant events in human history.",
        "Describe a day in the life of a secret agent who's also a full-time parent.",
        "Create a story about a detective who can communicate with animals.",
        "What is the most unusual thing about living in a city floating in the clouds?",
        "In a world where dreams are shared, what happens when a nightmare invades a peaceful dream?",
        "Describe the adventure of a lifetime for a group of friends who found a map leading to a parallel universe.",
        "Tell a story about a musician who discovers that their music has magical powers.",
        "In a world where people age backwards, describe the life of a 5-year-old man.",
        "Create a tale about a painter whose artwork comes to life every night.",
        "What happens when a poet's verses start to predict future events?",
        "Imagine a world where books can talk. How does a librarian handle them?",
        "Tell a story about an astronaut who discovered a planet populated by plants.",
        "Describe the journey of a letter traveling through the most sophisticated postal service ever.",
        "Write a tale about a chef whose food can evoke memories from the eater's past.",
        # History
        "What were the major contributing factors to the fall of the Roman Empire?",
        "How did the invention of the printing press revolutionize European society?",
        "What are the effects of quantitative easing?",
        "How did the Greek philosophers influence economic thought in the ancient world?",
        "What were the economic and philosophical factors that led to the fall of the Soviet Union?",
        "How did decolonization in the 20th century change the geopolitical map?",
        "What was the influence of the Khmer Empire on Southeast Asia's history and culture?",
        # Thoughtfulness
        "Describe the city of the future, considering advances in technology, environmental changes, and societal shifts.",
        "In a dystopian future where water is the most valuable commodity, how would society function?",
        "If a scientist discovers immortality, how could this impact society, economy, and the environment?",
        "What could be the potential implications of contact with an advanced alien civilization?",
        "Describe how you would mediate a conflict between two roommates about doing the dishes using techniques of non-violent communication.",
        # Math
        "What is the product of 9 and 8?",
        "If a train travels 120 kilometers in 2 hours, what is its average speed?",
        "Think through this step by step. If the sequence a_n is defined by a_1 = 3, a_2 = 5, and a_n = a_(n-1) + a_(n-2) for n > 2, find a_6.",
        "Think through this step by step. Calculate the sum of an arithmetic series with first term 3, last term 35, and total terms 11.",
        "Think through this step by step. What is the area of a triangle with vertices at the points (1,2), (3,-4), and (-2,5)?",
        "Think through this step by step. Solve the following system of linear equations: 3x + 2y = 14, 5x - y = 15.",
        # Facts
        "Who was Emperor Norton I, and what was his significance in San Francisco's history?",
        "What is the Voynich manuscript, and why has it perplexed scholars for centuries?",
        "What was Project A119 and what were its objectives?",
        "What is the 'Dyatlov Pass incident' and why does it remain a mystery?",
        "What is the 'Emu War' that took place in Australia in the 1930s?",
        "What is the 'Phantom Time Hypothesis' proposed by Heribert Illig?",
        "Who was the 'Green Children of Woolpit' as per 12th-century English legend?",
        "What are 'zombie stars' in the context of astronomy?",
        "Who were the 'Dog-Headed Saint' and the 'Lion-Faced Saint' in medieval Christian traditions?",
        "What is the story of the 'Globsters', unidentified organic masses washed up on the shores?",
        # Multilingual
        "战国时期最重要的人物是谁?",
        "Tuende hatua kwa hatua. Hesabu jumla ya mfululizo wa kihesabu wenye neno la kwanza 2, neno la mwisho 42, na jumla ya maneno 21.",
        "Kannst du die wichtigsten Eigenschaften und Funktionen des NMDA-Rezeptors beschreiben?",
    ]
    model = Model()
    model.generate.remote(questions)


---

## vllm inference

# # Fast inference with vLLM (Mistral 7B)
#
# In this example, we show how to run basic inference, using [`vLLM`](https://github.com/vllm-project/vllm)
# to take advantage of PagedAttention, which speeds up sequential inferences with optimized key-value caching.
#
# `vLLM` also supports a use case as a FastAPI server, which we will explore in a future guide. This example
# walks through setting up an environment that works with `vLLM ` for basic inference.
#
# We are running the [Mistral 7B Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) model here,
# which is version of Mistral's 7B model that hase been fine-tuned to follow instructions.
# You can expect 20 second cold starts and well over 1000 tokens/second. The larger the batch of prompts, the higher the throughput.
# For example, with the 64 prompts below, we can produce 15k tokens in less than 7 seconds, a throughput of over 2k tokens/second.
#
# To run [any of the other supported models](https://vllm.readthedocs.io/en/latest/models/supported_models.html),
# simply replace the model name in the download step.
#
# ## Setup
#
# First, we import the Modal client and define the model that we want to serve.

import os
import time

import modal

MODEL_DIR = "/model"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"


# ## Define a container image
#
# We want to create a Modal image which has the model weights pre-saved to a directory. The benefit of this
# is that the container no longer has to re-download the model from Hugging Face - instead, it will take
# advantage of Modal's internal filesystem for faster cold starts.
#
# ### Download the weights
# We can download the model to a particular directory using the HuggingFace utility function `snapshot_download`.
#
# For this step to work on a [gated model](https://huggingface.co/docs/hub/en/models-gated)
# like Mistral 7B, the `HF_TOKEN` environment variable must be set.
#
# After [creating a HuggingFace access token](https://huggingface.co/settings/tokens)
# and accepting the [terms of use](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1),
# head to the [secrets page](https://modal.com/secrets) to share it with Modal as `huggingface-secret`.
#
# Tip: avoid using global variables in this function.
# Changes to code outside this function will not be detected, and the download step will not re-run.
def download_model_to_image(model_dir, model_name):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(model_dir, exist_ok=True)

    snapshot_download(
        model_name,
        local_dir=model_dir,
        ignore_patterns=["*.pt", "*.bin"],  # Using safetensors
        token=os.environ["HF_TOKEN"],
    )
    move_cache()


# ### Image definition
# We’ll start from Modal's Debian slim image.
# Then we’ll use `run_function` with `download_model_to_image` to write the model into the container image.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "vllm==0.4.0.post1",
        "torch==2.1.2",
        "transformers==4.39.3",
        "ray==2.10.0",
        "hf-transfer==0.1.6",
        "huggingface_hub==0.22.2",
    )
    # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_image,
        timeout=60 * 20,
        kwargs={"model_dir": MODEL_DIR, "model_name": MODEL_NAME},
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
)

app = modal.App(
    "example-vllm-inference", image=image
)  # Note: prior to April 2024, "app" was called "stub"

# Using `image.imports` allows us to have a reference to vLLM in global scope without getting an error when our script executes locally.
with image.imports():
    import vllm

# ## The model class
#
# The inference function is best represented with Modal's [class syntax](https://modal.com/docs/guide/lifecycle-functions),
# using a `load_model` method decorated with `@modal.enter`. This enables us to load the model into memory just once,
# every time a container starts up, and to keep it cached on the GPU for subsequent invocations of the function.
#
# The `vLLM` library allows the code to remain quite clean.

# Hint: try out an H100 if you've got a large model or big batches!
GPU_CONFIG = modal.gpu.A100(count=1)  # 40GB A100 by default


@app.cls(gpu=GPU_CONFIG)
class Model:
    @modal.enter()
    def load_model(self):
        # Tip: models that are not fully implemented by Hugging Face may require `trust_remote_code=true`.
        self.llm = vllm.LLM(MODEL_DIR, tensor_parallel_size=GPU_CONFIG.count)
        self.template = """[INST] <<SYS>>
{system}
<</SYS>>

{user} [/INST]"""

    @modal.method()
    def generate(self, user_questions):
        prompts = [
            self.template.format(system="", user=q) for q in user_questions
        ]

        sampling_params = vllm.SamplingParams(
            temperature=0.75,
            top_p=1,
            max_tokens=256,
            presence_penalty=1.15,
        )
        start = time.monotonic_ns()
        result = self.llm.generate(prompts, sampling_params)
        duration_s = (time.monotonic_ns() - start) / 1e9
        num_tokens = 0

        COLOR = {
            "HEADER": "\033[95m",
            "BLUE": "\033[94m",
            "GREEN": "\033[92m",
            "RED": "\033[91m",
            "ENDC": "\033[0m",
        }

        for output in result:
            num_tokens += len(output.outputs[0].token_ids)
            print(
                f"{COLOR['HEADER']}{COLOR['GREEN']}{output.prompt}",
                f"\n{COLOR['BLUE']}{output.outputs[0].text}",
                "\n\n",
                sep=COLOR["ENDC"],
            )
            time.sleep(0.01)
        print(
            f"{COLOR['HEADER']}{COLOR['GREEN']}Generated {num_tokens} tokens from {MODEL_NAME} in {duration_s:.1f} seconds,"
            f" throughput = {num_tokens / duration_s:.0f} tokens/second on {GPU_CONFIG}.{COLOR['ENDC']}"
        )


# ## Run the model
# We define a [`local_entrypoint`](https://modal.com/docs/guide/apps#entrypoints-for-ephemeral-apps) to call our remote function
# sequentially for a list of inputs. You can run this locally with `modal run vllm_inference.py`.
@app.local_entrypoint()
def main():
    questions = [
        # Coding questions
        "Implement a Python function to compute the Fibonacci numbers.",
        "Write a Rust function that performs binary exponentiation.",
        "How do I allocate memory in C?",
        "What are the differences between Javascript and Python?",
        "How do I find invalid indices in Postgres?",
        "How can you implement a LRU (Least Recently Used) cache in Python?",
        "What approach would you use to detect and prevent race conditions in a multithreaded application?",
        "Can you explain how a decision tree algorithm works in machine learning?",
        "How would you design a simple key-value store database from scratch?",
        "How do you handle deadlock situations in concurrent programming?",
        "What is the logic behind the A* search algorithm, and where is it used?",
        "How can you design an efficient autocomplete system?",
        "What approach would you take to design a secure session management system in a web application?",
        "How would you handle collision in a hash table?",
        "How can you implement a load balancer for a distributed system?",
        # Literature
        "What is the fable involving a fox and grapes?",
        "Write a story in the style of James Joyce about a trip to the Australian outback in 2083, to see robots in the beautiful desert.",
        "Who does Harry turn into a balloon?",
        "Write a tale about a time-traveling historian who's determined to witness the most significant events in human history.",
        "Describe a day in the life of a secret agent who's also a full-time parent.",
        "Create a story about a detective who can communicate with animals.",
        "What is the most unusual thing about living in a city floating in the clouds?",
        "In a world where dreams are shared, what happens when a nightmare invades a peaceful dream?",
        "Describe the adventure of a lifetime for a group of friends who found a map leading to a parallel universe.",
        "Tell a story about a musician who discovers that their music has magical powers.",
        "In a world where people age backwards, describe the life of a 5-year-old man.",
        "Create a tale about a painter whose artwork comes to life every night.",
        "What happens when a poet's verses start to predict future events?",
        "Imagine a world where books can talk. How does a librarian handle them?",
        "Tell a story about an astronaut who discovered a planet populated by plants.",
        "Describe the journey of a letter traveling through the most sophisticated postal service ever.",
        "Write a tale about a chef whose food can evoke memories from the eater's past.",
        # History
        "What were the major contributing factors to the fall of the Roman Empire?",
        "How did the invention of the printing press revolutionize European society?",
        "What are the effects of quantitative easing?",
        "How did the Greek philosophers influence economic thought in the ancient world?",
        "What were the economic and philosophical factors that led to the fall of the Soviet Union?",
        "How did decolonization in the 20th century change the geopolitical map?",
        "What was the influence of the Khmer Empire on Southeast Asia's history and culture?",
        # Thoughtfulness
        "Describe the city of the future, considering advances in technology, environmental changes, and societal shifts.",
        "In a dystopian future where water is the most valuable commodity, how would society function?",
        "If a scientist discovers immortality, how could this impact society, economy, and the environment?",
        "What could be the potential implications of contact with an advanced alien civilization?",
        "Describe how you would mediate a conflict between two roommates about doing the dishes using techniques of non-violent communication.",
        # Math
        "What is the product of 9 and 8?",
        "If a train travels 120 kilometers in 2 hours, what is its average speed?",
        "Think through this step by step. If the sequence a_n is defined by a_1 = 3, a_2 = 5, and a_n = a_(n-1) + a_(n-2) for n > 2, find a_6.",
        "Think through this step by step. Calculate the sum of an arithmetic series with first term 3, last term 35, and total terms 11.",
        "Think through this step by step. What is the area of a triangle with vertices at the points (1,2), (3,-4), and (-2,5)?",
        "Think through this step by step. Solve the following system of linear equations: 3x + 2y = 14, 5x - y = 15.",
        # Facts
        "Who was Emperor Norton I, and what was his significance in San Francisco's history?",
        "What is the Voynich manuscript, and why has it perplexed scholars for centuries?",
        "What was Project A119 and what were its objectives?",
        "What is the 'Dyatlov Pass incident' and why does it remain a mystery?",
        "What is the 'Emu War' that took place in Australia in the 1930s?",
        "What is the 'Phantom Time Hypothesis' proposed by Heribert Illig?",
        "Who was the 'Green Children of Woolpit' as per 12th-century English legend?",
        "What are 'zombie stars' in the context of astronomy?",
        "Who were the 'Dog-Headed Saint' and the 'Lion-Faced Saint' in medieval Christian traditions?",
        "What is the story of the 'Globsters', unidentified organic masses washed up on the shores?",
        # Multilingual
        "战国时期最重要的人物是谁?",
        "Tuende hatua kwa hatua. Hesabu jumla ya mfululizo wa kihesabu wenye neno la kwanza 2, neno la mwisho 42, na jumla ya maneno 21.",
        "Kannst du die wichtigsten Eigenschaften und Funktionen des NMDA-Rezeptors beschreiben?",
    ]
    model = Model()
    model.generate.remote(questions)


---

## vllm mixtral

# # Fast inference with vLLM (Mixtral 8x7B)
#
# In this example, we show how to run basic inference, using [`vLLM`](https://github.com/vllm-project/vllm)
# to take advantage of PagedAttention, which speeds up sequential inferences with optimized key-value caching.
#
# We are running the [Mixtral 8x7B Instruct](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) model here,
# which is a mixture-of-experts model finetuned for conversation.
# You can expect ~3 minute cold starts.
# For a single request, the throughput is over 50 tokens/second.
# The larger the batch of prompts, the higher the throughput (up to hundreds of tokens per second).
#
# ## Setup
#
# First we import the components we need from `modal`.

import os
import time

import modal

MODEL_DIR = "/model"
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
MODEL_REVISION = "1e637f2d7cb0a9d6fb1922f305cb784995190a83"
GPU_CONFIG = modal.gpu.A100(memory=80, count=2)


# ## Define a container image
#
# We want to create a Modal image which has the model weights pre-saved to a directory. The benefit of this
# is that the container no longer has to re-download the model from Huggingface - instead, it will take
# advantage of Modal's internal filesystem for faster cold starts.
#
# ### Download the weights
#
# We can download the model to a particular directory using the HuggingFace utility function `snapshot_download`.
#
# For this step to work on a [gated model](https://huggingface.co/docs/hub/en/models-gated)
# like Mixtral 8x7B, the `HF_TOKEN` environment variable must be set.
#
# After [creating a HuggingFace access token](https://huggingface.co/settings/tokens)
# and accepting the [terms of use](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1),
# head to the [secrets page](https://modal.com/secrets) to share it with Modal as `huggingface-secret`.
#
# Mixtral is beefy, at nearly 100 GB in `safetensors` format, so this can take some time -- at least a few minutes.
def download_model_to_image(model_dir, model_name, model_revision):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(model_dir, exist_ok=True)

    snapshot_download(
        model_name,
        revision=model_revision,
        local_dir=model_dir,
        ignore_patterns=["*.pt", "*.bin"],  # Using safetensors
    )
    move_cache()


# ### Image definition
# We’ll start from a Dockerhub image recommended by `vLLM`, and use
# run_function to run the function defined above to ensure the weights of
# the model are saved within the container image.

vllm_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "vllm==0.4.0.post1",
        "torch==2.1.2",
        "transformers==4.39.3",
        "ray==2.10.0",
        "hf-transfer==0.1.6",
        "huggingface_hub==0.22.2",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_image,
        timeout=60 * 20,
        kwargs={
            "model_dir": MODEL_DIR,
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
        },
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
)

app = modal.App(
    "example-vllm-mixtral"
)  # Note: prior to April 2024, "app" was called "stub"


# ## The model class
#
# The inference function is best represented with Modal's [class syntax](/docs/guide/lifecycle-functions) and the `@enter` decorator.
# This enables us to load the model into memory just once every time a container starts up, and keep it cached
# on the GPU for each subsequent invocation of the function.
#
# The `vLLM` library allows the code to remain quite clean. We do have to patch the multi-GPU setup due to issues with Ray.
@app.cls(
    gpu=GPU_CONFIG,
    timeout=60 * 10,
    container_idle_timeout=60 * 10,
    allow_concurrent_inputs=10,
    image=vllm_image,
)
class Model:
    @modal.enter()
    def start_engine(self):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        print("🥶 cold starting inference")
        start = time.monotonic_ns()

        engine_args = AsyncEngineArgs(
            model=MODEL_DIR,
            tensor_parallel_size=GPU_CONFIG.count,
            gpu_memory_utilization=0.90,
            enforce_eager=False,  # capture the graph for faster inference, but slower cold starts
            disable_log_stats=True,  # disable logging so we can stream tokens
            disable_log_requests=True,
        )
        self.template = "[INST] {user} [/INST]"

        # this can take some time!
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        duration_s = (time.monotonic_ns() - start) / 1e9
        print(f"🏎️ engine started in {duration_s:.0f}s")

    @modal.method()
    async def completion_stream(self, user_question):
        from vllm import SamplingParams
        from vllm.utils import random_uuid

        sampling_params = SamplingParams(
            temperature=0.75,
            max_tokens=128,
            repetition_penalty=1.1,
        )

        request_id = random_uuid()
        result_generator = self.engine.generate(
            self.template.format(user=user_question),
            sampling_params,
            request_id,
        )
        index, num_tokens = 0, 0
        start = time.monotonic_ns()
        async for output in result_generator:
            if (
                output.outputs[0].text
                and "\ufffd" == output.outputs[0].text[-1]
            ):
                continue
            text_delta = output.outputs[0].text[index:]
            index = len(output.outputs[0].text)
            num_tokens = len(output.outputs[0].token_ids)

            yield text_delta
        duration_s = (time.monotonic_ns() - start) / 1e9

        yield (
            f"\n\tGenerated {num_tokens} tokens from {MODEL_NAME} in {duration_s:.1f}s,"
            f" throughput = {num_tokens / duration_s:.0f} tokens/second on {GPU_CONFIG}.\n"
        )

    @modal.exit()
    def stop_engine(self):
        if GPU_CONFIG.count > 1:
            import ray

            ray.shutdown()


# ## Run the model
# We define a [`local_entrypoint`](/docs/guide/apps#entrypoints-for-ephemeral-apps) to call our remote function
# sequentially for a list of inputs. You can run this locally with `modal run -q vllm_mixtral.py`. The `q` flag
# enables the text to stream in your local terminal.
@app.local_entrypoint()
def main():
    questions = [
        "Implement a Python function to compute the Fibonacci numbers.",
        "What is the fable involving a fox and grapes?",
        "What were the major contributing factors to the fall of the Roman Empire?",
        "Describe the city of the future, considering advances in technology, environmental changes, and societal shifts.",
        "What is the product of 9 and 8?",
        "Who was Emperor Norton I, and what was his significance in San Francisco's history?",
    ]
    model = Model()
    for question in questions:
        print("Sending new request:", question, "\n\n")
        for text in model.completion_stream.remote_gen(question):
            print(text, end="", flush=text.endswith("\n"))


# ## Deploy and invoke the model
# Once we deploy this model with `modal deploy vllm_mixtral.py`,
# we can invoke inference from other apps, sharing the same pool
# of GPU containers with all other apps we might need.
#
# ```
# $ python
# >>> import modal
# >>> f = modal.Function.lookup("example-vllm-mixtral", "Model.completion_stream")
# >>> for text in f.remote_gen("What is the story about the fox and grapes?"):
# >>>    print(text, end="", flush=text.endswith("\n"))
# 'The story about the fox and grapes ...
# ```

# ## Coupling a frontend web application
#
# We can stream inference from a FastAPI backend, also deployed on Modal.
#
# You can try our deployment [here](https://modal-labs--vllm-mixtral.modal.run).

from pathlib import Path

from modal import Mount, asgi_app

frontend_path = Path(__file__).parent.parent / "llm-frontend"


@app.function(
    mounts=[Mount.from_local_dir(frontend_path, remote_path="/assets")],
    keep_warm=1,
    allow_concurrent_inputs=20,
    timeout=60 * 10,
)
@asgi_app()
def vllm_mixtral():
    import json

    import fastapi
    import fastapi.staticfiles
    from fastapi.responses import StreamingResponse

    web_app = fastapi.FastAPI()

    @web_app.get("/stats")
    async def stats():
        stats = await Model().completion_stream.get_current_stats.aio()
        return {
            "backlog": stats.backlog,
            "num_total_runners": stats.num_total_runners,
            "model": MODEL_NAME + " (vLLM)",
        }

    @web_app.get("/completion/{question}")
    async def completion(question: str):
        from urllib.parse import unquote

        async def generate():
            async for text in Model().completion_stream.remote_gen.aio(
                unquote(question)
            ):
                yield f"data: {json.dumps(dict(text=text), ensure_ascii=False)}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    web_app.mount(
        "/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True)
    )
    return web_app


---

## instructor generate

# ---
# output-directory: "/tmp/instructor_generate"
# ---
# # Structured Data Extraction using `instructor`
#
# This example demonstrates how to use the `instructor` library to extract structured, schematized data from unstructured text.
#
# Structured output is a powerful but under-appreciated feature of LLMs.
# Structured output allows LLMs and multimodal models to connect to traditional software,
# for example enabling the ingestion of unstructured data like text files into structured databases.
# Applied properly, it makes them an extreme example of the [Robustness Principle](https://en.wikipedia.org/wiki/Robustness_principle)
# Jon Postel formulated for TCP: "Be conservative in what you send, be liberal in what you accept".
#
# The unstructured data used in this example code is the code from the examples in the Modal examples repository --
# including this example's code!
#
# The output includes a JSONL file containing, on each line, the metadata extracted from the code in one example.
# This can be consumed downstream by other software systems, like a database or a dashboard.
#
# ## Environment setup
#
# We set up the environment our code will run in first.
# In Modal, we define environments via [container images](https://modal.com/docs/guide/custom-container),
# much like Docker images, by iteratively chaining together commands.
#
# Here there's just one command, installing instructor and the Python SDK for Anthropic's LLM API.
from pathlib import Path
from typing import Literal, Optional

import modal
from pydantic import BaseModel, Field

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "instructor~=1.0.0", "anthropic~=0.23.1"
)

# This example uses models from Anthropic, so if you want to run it yourself,
# you'll need to set up a Modal [`Secret`](https://modal.com/docs/guide/secrets)
# called `my-anthropic-secret` for your OpenAI API key.

app = modal.App(
    image=image, secrets=[modal.Secret.from_name("my-anthropic-secret")]
)  # Note: prior to April 2024, "app" was called "stub"

# ## Running Modal functions from the command line
#
# We'll run the example by calling `modal run instructor_generate.py` from the command line.
#
# When we invoke `modal run` on a Python file, we run the function
# marked with `@app.local_entrypoint`.
#
# This is the only code that runs locally -- it coordinates
# the activity of the rest of our code, which runs in Modal's cloud.
#
# The logic is fairly simple: collect up the code for our examples,
# and then use `instructor` to extract metadata from them,
# which we then write to a file.
#
# By default, the language model is Claude 3 Haiku, the smallest model
# in the Claude 3 family.  We include the option to run `with_opus`,
# which gives much better results, but it is off by default because
# Opus is also ~60x more expensive, at ~$30 per million tokens.


@app.local_entrypoint()
def main(limit: int = 1, with_opus: bool = False):
    # find all of the examples in the repo
    examples = get_examples()
    # optionally limit the number of examples we process
    if limit == 1:
        examples = [None]  # just run on this example
    else:
        examples = examples[:limit]
    # use Modal to map our extraction function over the examples concurrently
    results = extract_example_metadata.map(
        (  # iterable of file contents
            Path(example.filename).read_text() if example else None
            for example in examples
        ),
        (  # iterable of filenames
            example.stem if example else None for example in examples
        ),
        kwargs={"with_opus": with_opus},
    )

    # save the results to a local file
    results_path = Path("/tmp") / "instructor_generate" / "results.jsonl"
    results_dir = results_path.parent
    if not results_dir.exists():
        results_dir.mkdir(parents=True)

    print(f"writing results to {results_path}")
    with open(results_path, "w") as f:
        for result in results:
            print(result)
            f.write(result + "\n")


# ## Extracting JSON from unstructured text with `instructor` and Pydantic
#
# The real meat of this example is in this section, in the `extract_example_metadata` function and its schemas.
#
# We define a schema for the data we want the LLM to extract, using Pydantic.
# Instructor ensures that the LLM's output matches this schema.
#
# We can use the type system provided by Python and Pydantic to express many useful features
# of the data we want to extract -- ranging from wide-open fields like a `str`ing-valued `summary`
# to constrained fields like `difficulty`, which can only take on value between 1 and 5.


class ExampleMetadataExtraction(BaseModel):
    """Extracted metadata about an example from the Modal examples repo."""

    summary: str = Field(..., description="A brief summary of the example.")
    has_thorough_explanation: bool = Field(
        ...,
        description="The example contains, in the form of inline comments with markdown formatting, a thorough explanation of what the code does.",
    )
    domains: list[
        Literal[
            "artificial_intelligence",
            "machine_learning",
            "data_science",
            "web_serving",
            "parallel_computing",
        ]
    ] = Field(..., description="The")
    difficulty: Literal[1, 2, 3, 4, 5] = Field(
        ...,
        description="The difficulty of the example, from 1 to 5. An example that uses only one or two basic Modal features and is understandable by a professional Python developer familiar with the basics of the relevant domains is a 1, while an example that uses many Modal features and uses advanced Python features like async generator coroutines or metaclasses is a 5.",
    )
    freshness: float = Field(
        ...,
        description="The freshness of the example, from 0 to 1. This is relative to your knowledge cutoff. Examples are less fresh if they use older libraries and tools.",
    )


# That schema describes the data to be extracted by the LLM, but not all data is best extracted by an LLM.
# For example, the filename is easily determined in software.
#
# So we inject that information into the output after the LLM has done its work. That necessitates
# an additional schema, which inherits from the first.


class ExampleMetadata(ExampleMetadataExtraction):
    """Metadata about an example from the Modal examples repo."""

    filename: Optional[str] = Field(
        ..., description="The filename of the example."
    )


# With these schemas in hand, it's straightforward to write the function that extracts the metadata.
# Note that we decorate it with `@app.function` to make it run on Modal.


@app.function(concurrency_limit=5)  # watch those LLM API rate limits!
def extract_example_metadata(
    example_contents: Optional[str] = None,
    filename: Optional[str] = None,
    with_opus=False,
):
    import instructor
    from anthropic import Anthropic

    # if no example is provided, use the contents of this example
    if example_contents is None:
        example_contents = Path(__file__).read_text()
        filename = Path(__file__).name

    client = instructor.from_anthropic(Anthropic())
    model = "claude-3-opus-20240229" if with_opus else "claude-3-haiku-20240307"

    # add the schema as the `response_model` argument in what otherwise looks like a normal LLM API call
    extracted_metadata = client.messages.create(
        model=model,
        temperature=0.0,
        max_tokens=1024,
        response_model=ExampleMetadataExtraction,
        messages=[
            {
                "role": "user",
                "content": f"Extract the metadata for this example.\n\n-----EXAMPLE BEGINS-----{example_contents}-----EXAMPLE ENDS-----\n\n",
            },
        ],
    )

    # inject the filename
    full_metadata = ExampleMetadata(
        **extracted_metadata.dict(), filename=filename
    )

    # return it as JSON
    return full_metadata.model_dump_json()


# ## Addenda
#
# The rest of the code used in this example is not particularly interesting:
# just a utility function to find all of the examples, which we invoke in the `local_entrypoint` above.


def get_examples(silent=True):
    """Find all of the examples using a utility from this repo.

    We use importlib to avoid the need to define the repo as a package."""
    import importlib

    examples_root = Path(__file__).parent.parent.parent
    spec = importlib.util.spec_from_file_location(
        "utils", f"{examples_root}/internal/utils.py"
    )
    example_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example_utils)
    examples = [
        example
        for example in example_utils.get_examples()
        if example.type != 2  # filter out non-code assets
    ]
    return examples


---

## jsonformer generate

# ---
# lambda-test: false
# ---
# # Generate synthetic data with Jsonformer
#
# [Jsonformer](https://github.com/1rgs/jsonformer) is a tool that generates structured synthetic data using LLMs.
# You provide a JSON spec and it generates a JSON object following the spec. It's a
# great tool for developing, benchmarking, and testing applications.


from typing import Any

import modal

# We will be using one of [Databrick's Dolly](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)
# models, choosing for the smallest version with 3B parameters. Feel free to use any of the other models
# available from the [Huggingface Hub Dolly repository](https://huggingface.co/databricks).
MODEL_ID: str = "databricks/dolly-v2-3b"
CACHE_PATH: str = "/root/cache"


# ## Build image and cache model
#
# We'll download models from the Huggingface Hub and store them in our image.
# This skips the downloading of models during inference and reduces cold boot
# times.
def download_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, use_cache=True, device_map="auto"
    )
    model.save_pretrained(CACHE_PATH, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, use_fast=True, use_cache=True
    )
    tokenizer.save_pretrained(CACHE_PATH, safe_serialization=True)


# Define our image; install dependencies.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "jsonformer==0.9.0",
        "transformers",
        "torch",
        "accelerate",
        "safetensors",
    )
    .run_function(download_model)
)
app = modal.App(
    "example-jsonformer"
)  # Note: prior to April 2024, "app" was called "stub"


# ## Generate examples
#
# The generate function takes two arguments `prompt` and `json_schema`, where
# `prompt` is used to describe the domain of your data (for example, "plants")
# and the schema contains the JSON schema you want to populate.
@app.function(gpu=modal.gpu.A10G(), image=image)
def generate(prompt: str, json_schema: dict[str, Any]) -> dict[str, Any]:
    from jsonformer import Jsonformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        CACHE_PATH, use_cache=True, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, use_fast=True, use_cache=True, device_map="auto"
    )

    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt)
    generated_data = jsonformer()

    return generated_data


# Add Modal entrypoint for invoking your script, and done!
@app.local_entrypoint()
def main():
    prompt = "Generate random plant information based on the following schema:"
    json_schema = {
        "type": "object",
        "properties": {
            "height_cm": {"type": "number"},
            "bearing_fruit": {"type": "boolean"},
            "classification": {
                "type": "object",
                "properties": {
                    "species": {"type": "string"},
                    "kingdom": {"type": "string"},
                    "family": {"type": "string"},
                    "genus": {"type": "string"},
                },
            },
        },
    }

    result = generate.remote(prompt, json_schema)
    print(result)


---

## outlines generate

# # Enforcing JSON outputs on LLMs

# [Outlines](https://github.com/outlines-dev/outlines) is a tool that lets you control the generation of language models to make their output more predictable.

# This includes things like:

# - Reducing the completion to a choice between multiple possibilities
# - Type constraints
# - Efficient regex-structured generation
# - Efficient JSON generation following a Pydantic model
# - Efficient JSON generation following a JSON schema

# Outlines is considered an alternative to tools like [JSONFormer](https://github.com/1rgs/jsonformer), and can be used on top of a variety of LLMs, including:

# - OpenAI models
# - Transformers models
# - Llama
# - Mamba

# In this guide, we will show how you can use Outlines to enforce a JSON schema on the output of Mistral-7B.

# ## Build image
#
#  First, you'll want to build an image and install the relevant Python dependencies:
# `outlines` and a Hugging Face inference stack.

from modal import App, Image, Secret, gpu

app = App(
    name="outlines-app"
)  # Note: prior to April 2024, "app" was called "stub"

outlines_image = Image.debian_slim(python_version="3.11").pip_install(
    "outlines==0.0.34",
    "transformers==4.38.2",
    "datasets==2.18.0",
    "accelerate==0.27.2",
)

# ## Download the model
#
# Next, we download the Mistral-7B model from Hugging Face.
# We do this as part of the definition of our Modal image so that
# we don't need to download it every time our inference function is run.
#
# For this step to work on a [gated model](https://huggingface.co/docs/hub/en/models-gated)
# like Mistral 7B, the `HF_TOKEN` environment variable must be set.
#
# After [creating a HuggingFace access token](https://huggingface.co/settings/tokens)
# and accepting the [terms of use](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1),
# head to the [secrets page](https://modal.com/secrets) to share it with Modal as `huggingface-secret`.


def import_model():
    import outlines

    outlines.models.transformers("mistralai/Mistral-7B-v0.1")


outlines_image = outlines_image.run_function(
    import_model,
    secrets=[Secret.from_name("huggingface-secret")],
)


# ## Define the schema

# Next, we define the schema that we want to enforce on the output of Mistral-7B. This schema is for a character description, and includes a name, age, armor, weapon, and strength.

schema = """{
    "title": "Character",
    "type": "object",
    "properties": {
        "name": {
            "title": "Name",
            "maxLength": 10,
            "type": "string"
        },
        "age": {
            "title": "Age",
            "type": "integer"
        },
        "armor": {"$ref": "#/definitions/Armor"},
        "weapon": {"$ref": "#/definitions/Weapon"},
        "strength": {
            "title": "Strength",
            "type": "integer"
        }
    },
    "required": ["name", "age", "armor", "weapon", "strength"],
    "definitions": {
        "Armor": {
            "title": "Armor",
            "description": "An enumeration.",
            "enum": ["leather", "chainmail", "plate"],
            "type": "string"
        },
        "Weapon": {
            "title": "Weapon",
            "description": "An enumeration.",
            "enum": ["sword", "axe", "mace", "spear", "bow", "crossbow"],
            "type": "string"
        }
    }
}"""

# ## Define the function

# Next, we define the generation function.
# We use the `@app.function` decorator to tell Modal to run this function on the app we defined above.
# Note that we import `outlines` from inside the Modal function. This is because the `outlines` package exists in the container, but not necessarily locally.

# We specify that we want to use the Mistral-7B model, and then ask for a character, and we'll receive structured data with the right schema.


@app.function(image=outlines_image, gpu=gpu.A100(memory=80))
def generate(
    prompt: str = "Amiri, a 53 year old warrior woman with a sword and leather armor.",
):
    import outlines

    model = outlines.models.transformers(
        "mistralai/Mistral-7B-v0.1", device="cuda"
    )

    generator = outlines.generate.json(model, schema)
    character = generator(
        f"Give me a character description. Describe {prompt}."
    )

    print(character)


# ## Define the entrypoint

# Finally, we define the entrypoint that will connect our local computer
# to the functions above, that run on Modal, and we are done!
#
# When you run this script with `modal run`, you should see something like this printed out:
#
#  `{'name': 'Amiri', 'age': 53, 'armor': 'leather', 'weapon': 'sword', 'strength': 10}`


@app.local_entrypoint()
def main(
    prompt: str = "Amiri, a 53 year old warrior woman with a sword and leather armor.",
):
    generate.remote(prompt)


---

## webcam

# ---
# cmd: ["modal", "serve", "06_gpu_and_ml/obj_detection_webcam/webcam.py"]
# deploy: true
# ---
# # Machine learning model inference endpoint that uses the webcam
#
# This example creates a web endpoint that uses a Huggingface model for object detection.
#
# The web endpoint takes an image from their webcam, and sends it to a Modal web endpoint.
# The Modal web endpoint in turn calls a Modal function that runs the actual model.
#
# If you run this, it will look something like this:
#
# ![webcam](./webcam.png)
#
# ## Live demo
#
# [Take a look at the deployed app](https://modal-labs-example-webcam-object-detection-fastapi-app.modal.run/).
#
# A couple of caveats:
# * This is not optimized for latency: every prediction takes about 1s, and
#   there's an additional overhead on the first prediction since the containers
#   have to be started and the model initialized.
# * This doesn't work on iPhone unfortunately due to some issues with HTML5
#   webcam components
#
# ## Code
#
# Starting with imports:

import base64
import io
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles
from modal import App, Image, Mount, asgi_app, build, enter, method

# We need to install [transformers](https://github.com/huggingface/transformers)
# which is a package Huggingface uses for all their models, but also
# [Pillow](https://python-pillow.org/) which lets us work with images from Python,
# and a system font for drawing.
#
# This example uses the `facebook/detr-resnet-50` pre-trained model, which is downloaded
# once at image build time using the `@build` hook and saved into the image. 'Baking'
# models into the `modal.Image` at build time provided the fastest cold start.

model_repo_id = "facebook/detr-resnet-50"


app = App(
    "example-webcam-object-detection"
)  # Note: prior to April 2024, "app" was called "stub"
image = (
    Image.debian_slim()
    .pip_install(
        "huggingface-hub==0.16.4",
        "Pillow",
        "timm",
        "transformers",
    )
    .apt_install("fonts-freefont-ttf")
)


# ## Prediction function
#
# The object detection function has a few different features worth mentioning:
#
# * There's a container initialization step in the method decorated with `@enter()`,
#   which runs on every container start. This lets us load the model only once per
#   container, so that it's reused for subsequent function calls.
# * Above we stored the model in the container image. This lets us download the model only
#   when the image is (re)built, and not everytime the function is called.
# * We're running it on multiple CPUs for extra performance
#
# Note that the function takes an image and returns a new image.
# The input image is from the webcam
# The output image is an image with all the bounding boxes and labels on them,
# with an alpha channel so that most of the image is transparent so that the
# web interface can render it on top of the webcam view.


with image.imports():
    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image, ImageColor, ImageDraw, ImageFont
    from transformers import DetrForObjectDetection, DetrImageProcessor


@app.cls(
    cpu=4,
    image=image,
)
class ObjectDetection:
    @build()
    def download_model(self):
        snapshot_download(repo_id=model_repo_id, cache_dir="/cache")

    @enter()
    def load_model(self):
        self.feature_extractor = DetrImageProcessor.from_pretrained(
            model_repo_id,
            cache_dir="/cache",
        )
        self.model = DetrForObjectDetection.from_pretrained(
            model_repo_id,
            cache_dir="/cache",
        )

    @method()
    def detect(self, img_data_in):
        # Based on https://huggingface.co/spaces/nateraw/detr-object-detection/blob/main/app.py
        # Read png from input
        image = Image.open(io.BytesIO(img_data_in)).convert("RGB")

        # Make prediction
        inputs = self.feature_extractor(image, return_tensors="pt")
        outputs = self.model(**inputs)
        img_size = torch.tensor([tuple(reversed(image.size))])
        processed_outputs = (
            self.feature_extractor.post_process_object_detection(
                outputs=outputs,
                target_sizes=img_size,
                threshold=0,
            )
        )
        output_dict = processed_outputs[0]

        # Grab boxes
        keep = output_dict["scores"] > 0.7
        boxes = output_dict["boxes"][keep].tolist()
        scores = output_dict["scores"][keep].tolist()
        labels = output_dict["labels"][keep].tolist()

        # Plot bounding boxes
        colors = list(ImageColor.colormap.values())
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/freefont/FreeMono.ttf", 18
        )
        output_image = Image.new("RGBA", (image.width, image.height))
        output_image_draw = ImageDraw.Draw(output_image)
        for _score, box, label in zip(scores, boxes, labels):
            color = colors[label % len(colors)]
            text = self.model.config.id2label[label]
            box = tuple(map(int, box))
            output_image_draw.rectangle(box, outline=color)
            output_image_draw.text(
                box[:2], text, font=font, fill=color, width=3
            )

        # Return PNG as bytes
        with io.BytesIO() as output_buf:
            output_image.save(output_buf, format="PNG")
            return output_buf.getvalue()


# ## Defining the web interface
#
# To keep things clean, we define the web endpoints separate from the prediction
# function. This will introduce a tiny bit of extra latency (every web request
# triggers a Modal function call which will call another Modal function) but in
# practice the overhead is much smaller than the overhead of running the prediction
# function etc.
#
# We also serve a static html page which contains some tiny bit of Javascript to
# capture the webcam feed and send it to Modal.

web_app = FastAPI()
static_path = Path(__file__).with_name("webcam").resolve()


# The endpoint for the prediction function takes an image as a
# [data URI](https://en.wikipedia.org/wiki/Data_URI_scheme)
# and returns another image, also as a data URI:


@web_app.post("/predict")
async def predict(request: Request):
    # Takes a webcam image as a datauri, returns a bounding box image as a datauri
    body = await request.body()
    img_data_in = base64.b64decode(body.split(b",")[1])  # read data-uri
    img_data_out = ObjectDetection().detect.remote(img_data_in)
    output_data = b"data:image/png;base64," + base64.b64encode(img_data_out)
    return Response(content=output_data)


# ## Exposing the web server
#
# Let's take the Fast API app and expose it to Modal.


@app.function(
    mounts=[Mount.from_local_dir(static_path, remote_path="/assets")],
)
@asgi_app()
def fastapi_app():
    web_app.mount("/", StaticFiles(directory="/assets", html=True))
    return web_app


# ## Running this locally
#
# You can run this as an ephemeral app, by running
#
# ```shell
# modal serve webcam.py
# ```


---

## main

# Fine-tuning the OpenAI Whisper model on Modal for improved
# transcription performance on the Hindi language.
#
# Based on the work done in https://huggingface.co/blog/fine-tune-whisper.

import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Union

import modal

from .config import DataTrainingArguments, ModelArguments, app_config
from .logs import get_logger, setup_logging

try:
    from transformers import HfArgumentParser, Seq2SeqTrainingArguments
except ModuleNotFoundError:
    exit(
        "The 'transformers' library is required to run both locally and in Modal."
    )


persistent_volume = modal.Volume.from_name(
    app_config.persistent_vol_name,
    create_if_missing=True,
)
image = modal.Image.debian_slim().pip_install_from_requirements(
    "requirements.txt"
)
app = modal.App(
    name=app_config.app_name,
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)  # Note: prior to April 2024, "app" was called "stub"

logger = get_logger(__name__)


@app.function(
    gpu="A10G",
    volumes={app_config.model_dir: persistent_volume},
    # 12hrs
    timeout=12 * 60 * 60,
    # For occasional connection error to 'cdn-lfs.huggingface.co'
    retries=1,
)
def train(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: Seq2SeqTrainingArguments,
):
    import datasets
    import evaluate
    import torch
    from datasets import DatasetDict, load_dataset
    from transformers import (
        AutoConfig,
        AutoFeatureExtractor,
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
        AutoTokenizer,
        Seq2SeqTrainer,
    )
    from transformers.trainer_utils import get_last_checkpoint, is_main_process

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        """
        Data collator that will dynamically pad the inputs received.
        Args:
            processor ([`WhisperProcessor`])
                The processor used for processing the data.
            decoder_start_token_id (`int`)
                The begin-of-sentence of the decoder.
            forward_attention_mask (`bool`)
                Whether to return attention_mask.
        """

        processor: Any
        decoder_start_token_id: int
        forward_attention_mask: bool

        def __call__(
            self, features: list[dict[str, Union[list[int], torch.Tensor]]]
        ) -> dict[str, torch.Tensor]:
            # split inputs and labels since they have to be of different lengths and need
            # different padding methods
            model_input_name = self.processor.model_input_names[0]
            input_features = [
                {model_input_name: feature[model_input_name]}
                for feature in features
            ]
            label_features = [
                {"input_ids": feature["labels"]} for feature in features
            ]

            batch = self.processor.feature_extractor.pad(
                input_features, return_tensors="pt"
            )

            if self.forward_attention_mask:
                batch["attention_mask"] = torch.LongTensor(
                    [feature["attention_mask"] for feature in features]
                )

            labels_batch = self.processor.tokenizer.pad(
                label_features, return_tensors="pt"
            )

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )

            # if bos token is appended in previous tokenization step,
            # cut bos token here as it's append later anyways
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels

            return batch

    logger.info("Starting training run")
    logger.info(
        f"Finetuned model will be persisted to '{training_args.output_dir}'"
    )
    setup_logging(
        logger=logger,
        log_level=training_args.get_process_log_level(),
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} "
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    logger.info(
        "3. Detecting last checkpoint and eventually continue from last checkpoint"
    )
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if (
            last_checkpoint is None
            and len(os.listdir(training_args.output_dir)) > 0
        ):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None
            and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    logger.info("4. Load datasets")
    raw_datasets = DatasetDict()
    raw_datasets["train"] = load_dataset(
        "mozilla-foundation/common_voice_11_0",
        "hi",
        split="train+validation",
        use_auth_token=os.environ["HF_TOKEN"],
    )
    raw_datasets["eval"] = load_dataset(
        "mozilla-foundation/common_voice_11_0",
        "hi",
        split="test",
        use_auth_token=os.environ["HF_TOKEN"],
    )

    # Most ASR datasets only provide input audio samples (audio) and
    # the corresponding transcribed text (sentence).
    # Common Voice contains additional metadata information,
    # such as accent and locale, which we can disregard for ASR.
    # Keeping the training function as general as possible,
    # we only consider the input audio and transcribed text for fine-tuning,
    # discarding the additional metadata information:
    raw_datasets = raw_datasets.remove_columns(
        [
            "accent",
            "age",
            "client_id",
            "down_votes",
            "gender",
            "locale",
            "path",
            "segment",
            "up_votes",
        ]
    )

    logger.info("5. Load pretrained model, tokenizer, and feature extractor")
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    config = AutoConfig.from_pretrained(
        (
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path
        ),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=os.environ["HF_TOKEN"],
    )

    config.update(
        {
            "forced_decoder_ids": model_args.forced_decoder_ids,
            "suppress_tokens": model_args.suppress_tokens,
        }
    )
    # SpecAugment for whisper models
    config.update({"apply_spec_augment": model_args.apply_spec_augment})

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        (
            model_args.feature_extractor_name
            if model_args.feature_extractor_name
            else model_args.model_name_or_path
        ),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        (
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path
        ),
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined"
        )

    if model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    if model_args.freeze_encoder:
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False

    if data_args.language is not None:
        # We only need to set the task id when the language is specified (i.e. in a multilingual setting)
        tokenizer.set_prefix_tokens(
            language=data_args.language, task=data_args.task
        )

    logger.info("6. Resample speech dataset if necessary")
    dataset_sampling_rate = (
        next(iter(raw_datasets.values()))
        .features[data_args.audio_column_name]
        .sampling_rate
    )
    if dataset_sampling_rate != feature_extractor.sampling_rate:
        logger.info("Resampling necessary")
        raw_datasets = raw_datasets.cast_column(
            data_args.audio_column_name,
            datasets.features.Audio(
                sampling_rate=feature_extractor.sampling_rate
            ),
        )

    logger.info("7. Preprocessing the datasets.")
    # We need to read the audio files as arrays and tokenize the targets.
    max_input_length = (
        data_args.max_duration_in_seconds * feature_extractor.sampling_rate
    )
    min_input_length = (
        data_args.min_duration_in_seconds * feature_extractor.sampling_rate
    )
    audio_column_name = data_args.audio_column_name
    num_workers = data_args.preprocessing_num_workers
    text_column_name = data_args.text_column_name
    model_input_name = feature_extractor.model_input_names[0]
    do_lower_case = data_args.do_lower_case
    # if SpecAugment is used for whisper models, return attention_mask to guide the mask along time axis
    forward_attention_mask = (
        getattr(config, "model_type", None) == "whisper"
        and getattr(config, "apply_spec_augment", False)
        and getattr(config, "mask_time_prob", 0) > 0
    )

    if data_args.max_train_samples is not None:
        raw_datasets["train"] = raw_datasets["train"].select(
            range(data_args.max_train_samples)
        )

    if data_args.max_eval_samples is not None:
        raw_datasets["eval"] = raw_datasets["eval"].select(
            range(data_args.max_eval_samples)
        )

    def prepare_dataset(batch):
        # process audio
        sample = batch[audio_column_name]
        inputs = feature_extractor(
            sample["array"],
            sampling_rate=sample["sampling_rate"],
            return_attention_mask=forward_attention_mask,
        )
        # process audio length
        batch[model_input_name] = inputs.get(model_input_name)[0]
        batch["input_length"] = len(sample["array"])
        if forward_attention_mask:
            batch["attention_mask"] = inputs.get("attention_mask")[0]

        # process targets
        input_str = (
            batch[text_column_name].lower()
            if do_lower_case
            else batch[text_column_name]
        )
        batch["labels"] = tokenizer(input_str).input_ids
        return batch

    with training_args.main_process_first(desc="dataset map pre-processing"):
        vectorized_datasets = raw_datasets.map(
            prepare_dataset,
            remove_columns=next(iter(raw_datasets.values())).column_names,
            num_proc=data_args.preprocessing_num_workers,
            desc="preprocess train dataset",
        )

    # filter data that is shorter than min_input_length or longer than
    # max_input_length
    def is_audio_in_length_range(length):
        return length > min_input_length and length < max_input_length

    vectorized_datasets = vectorized_datasets.filter(
        is_audio_in_length_range,
        num_proc=num_workers,
        input_columns=["input_length"],
    )

    # for large datasets it is advised to run the preprocessing on a
    # single machine first with `args.preprocessing_only` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step `args.preprocessing_only` can then be set to `False` to load the
    # cached dataset
    if data_args.preprocessing_only:
        cache = {k: v.cache_files for k, v in vectorized_datasets.items()}
        logger.info(f"Data preprocessing finished. Files cached at {cache}.")
        return

    logger.info("8. Loading WER Metric")
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions

        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(
            pred.label_ids, skip_special_tokens=True
        )

        wer = metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    logger.info("9. Create a single speech processor")
    # make sure all processes wait until data is saved
    with training_args.main_process_first():
        # only the main process saves them
        if is_main_process(training_args.local_rank):
            logger.info("saving feature extractor, tokenizer and config")
            feature_extractor.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            config.save_pretrained(training_args.output_dir)

    processor = AutoProcessor.from_pretrained(training_args.output_dir)

    logger.info("10. Constructing data collator")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        forward_attention_mask=forward_attention_mask,
    )

    logger.info("11. Initializing Trainer class")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=(
            vectorized_datasets["train"] if training_args.do_train else None
        ),
        eval_dataset=(
            vectorized_datasets["eval"] if training_args.do_eval else None
        ),
        tokenizer=feature_extractor,
        data_collator=data_collator,
        compute_metrics=(
            compute_metrics if training_args.predict_with_generate else None
        ),
    )

    logger.info("12. Running training")
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            logger.info("Restoring from previous training checkpoint")
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        logger.info("Saving model")
        trainer.save_model()  # Saves the feature extractor too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(vectorized_datasets["train"])
        )
        metrics["train_samples"] = min(
            max_train_samples, len(vectorized_datasets["train"])
        )
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        persistent_volume.commit()

    logger.info("13. Running evaluation")
    results = {}  # type: ignore
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            metric_key_prefix="eval",
            max_length=training_args.generation_max_length,
            num_beams=training_args.generation_num_beams,
        )
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(vectorized_datasets["eval"])
        )
        metrics["eval_samples"] = min(
            max_eval_samples, len(vectorized_datasets["eval"])
        )

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info("14. Write training stats")
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "automatic-speech-recognition",
    }
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs[
                "dataset"
            ] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    logger.info("Training run complete!")
    return results


def main() -> int:
    with app.run(detach=True):
        run_id = app.app_id
        output_dir = str(pathlib.Path(app_config.model_dir, run_id))
        args = sys.argv[1:] + [f"--output_dir={str(output_dir)}"]
        # Modal's @app.local_entrypoint() uses tiangolo/typer, which doesn't support
        # building CLI interfaces from dataclasses. https://github.com/tiangolo/typer/issues/154
        parser = HfArgumentParser(
            (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
        )
        (
            model_args,
            data_args,
            training_args,
        ) = parser.parse_args_into_dataclasses(args)

        logger.info("Starting training")
        result = train.remote(model_args, data_args, training_args)
        logger.info(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


---

## config

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModalAppConfig:
    app_name = "example-whisper-fine-tune"
    persistent_vol_name = "example-whisper-fine-tune-vol"
    dataset = "mozilla-foundation/common_voice_11_0"
    cache_dir = "/cache"
    model_dir = "/models"


app_config = ModalAppConfig()


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which models/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    feature_extractor_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "feature extractor name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=app_config.cache_dir,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    freeze_feature_encoder: bool = field(
        default=True,
        metadata={
            "help": "Whether to freeze the feature encoder layers of the model."
        },
    )
    freeze_encoder: bool = field(
        default=False,
        metadata={
            "help": "Whether to freeze the entire encoder of the seq2seq model."
        },
    )
    forced_decoder_ids: list[list[int]] = field(
        default=None,
        metadata={
            "help": (
                "A list of pairs of integers which indicates a mapping from generation indices to token indices "
                "that will be forced before sampling. For example, [[0, 123]] means the first generated token "
                "will always be a token of index 123."
            )
        },
    )
    suppress_tokens: list[int] = field(
        default=None,
        metadata={
            "help": "A list of tokens that will be suppressed at generation."
        },
    )
    apply_spec_augment: bool = field(
        default=False,
        metadata={
            "help": "Whether to apply *SpecAugment* data augmentation to the input features. This is currently only relevant for Wav2Vec2, HuBERT, WavLM and Whisper models."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        default=None,
        metadata={
            "help": "The name of the dataset to use (via the datasets library)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the full texts (for summarization)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of processes to use for the preprocessing."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={
            "help": "The name of the dataset column containing the audio data. Defaults to 'audio'"
        },
    )
    text_column_name: str = field(
        default="sentence",
        metadata={
            "help": "The name of the dataset column containing the text data. Defaults to 'sentence'"
        },
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={
            "help": (
                "Truncate audio files that are longer than `max_duration_in_seconds` seconds to"
                " 'max_duration_in_seconds`"
            )
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0,
        metadata={
            "help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"
        },
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only do data preprocessing and skip training. This is especially useful when data"
                " preprocessing errors out in distributed training due to timeout. In this case, one should run the"
                " preprocessing in a non-distributed setup with `preprocessing_only=True` so that the cached datasets"
                " can consequently be loaded in distributed training"
            )
        },
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="test",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    do_lower_case: bool = field(
        default=True,
        metadata={"help": "Whether the target text should be lower cased."},
    )
    language: str = field(
        default=None,
        metadata={
            "help": (
                "Language for multilingual fine-tuning. This argument should be set for multilingual fine-tuning "
                "only. For English speech recognition, it should be set to `None`."
            )
        },
    )
    task: str = field(
        default="transcribe",
        metadata={
            "help": "Task, either `transcribe` for speech recognition or `translate` for speech translation."
        },
    )


---

## end to end check

"""
A full fine-tuning run on GPUs takes multiple hours, but we
want to be able to validate changes quickly while coding.

This module contains an end-to-end test that runs only 1 step of training,
before testing that the partially trained model can be serialized, saved to
persistent storage, and then downloaded locally for inference.
"""

import pathlib

import modal
from transformers import Seq2SeqTrainingArguments

from .__main__ import app, train
from .config import DataTrainingArguments, ModelArguments, app_config
from .logs import get_logger
from .transcribe import whisper_transcribe_audio

test_volume = modal.NetworkFileSystem.from_name(
    "example-whisper-fine-tune-test-vol", create_if_missing=True
)

logger = get_logger(__name__)

# Test the `main.train` function by passing in test-specific configuration
# that does only a minimal amount of training steps and saves the model
# to the temporary (ie. ephemeral) network file system disk.
#
# This remote function should take only ~1 min to run.


@app.function(network_file_systems={app_config.model_dir: test_volume})
def test_finetune_one_step_and_save_to_vol(run_id: str):
    output_dir = pathlib.Path(app_config.model_dir, run_id)
    test_model_args = ModelArguments(
        model_name_or_path="openai/whisper-small",
        freeze_feature_encoder=False,
    )
    test_data_args = DataTrainingArguments(
        preprocessing_num_workers=16,
        max_train_samples=5,
        max_eval_samples=5,
    )

    train(
        model_args=test_model_args,
        data_args=test_data_args,
        training_args=Seq2SeqTrainingArguments(
            do_train=True,
            output_dir=output_dir,
            num_train_epochs=1.0,
            learning_rate=3e-4,
            warmup_steps=0,
            max_steps=1,
        ),
    )


# Test model serialization and persistence by starting a new remote
# function that reads back the model files from the temporary network file system disk
# and does a single sentence of translation.
#
# When doing full training runs, the saved model will be loaded in the same way
# but from a *persisted* network file system, which keeps data around even after the Modal
# ephemeral app that ran the training has stopped.


@app.function(network_file_systems={app_config.model_dir: test_volume})
def test_download_and_tryout_model(run_id: str):
    from datasets import Audio, load_dataset
    from evaluate import load

    lang, lang_short = (
        "french",
        "fr",
    )  # the language doesn't matter for this test.
    model_dir = pathlib.Path(app_config.model_dir, run_id)

    # load streaming dataset and read first audio sample
    ds = load_dataset(
        app_config.dataset,
        lang_short,
        split="test",
        streaming=True,
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    test_row = next(iter(ds))
    input_speech = test_row["audio"]

    predicted_transcription = whisper_transcribe_audio(
        model_dir=model_dir,
        language=lang,
        data=input_speech["array"],
        sample_rate_hz=input_speech["sampling_rate"],
    )
    expected_transcription = test_row["sentence"]
    wer = load("wer")
    wer_score = wer.compute(
        predictions=[predicted_transcription],
        references=[expected_transcription],
    )
    logger.info(
        f"{expected_transcription=}\n{predicted_transcription=}\n"
        f"Word Error Rate (WER): {wer_score}"
    )
    assert (
        wer_score < 1.0
    ), f"Even without finetuning, a WER score of {wer_score} is far too high."


# This simple entrypoint function just starts an ephemeral app run and calls
# the two test functions in sequence.
#
# Any runtime errors or assertion errors will fail the app and exit non-zero.


def run_test() -> int:
    with app.run():
        test_finetune_one_step_and_save_to_vol.remote(run_id=app.app_id)
        test_download_and_tryout_model.remote(run_id=app.app_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_test())


---

## logs

import logging


def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(levelname)s: %(asctime)s: %(name)s  %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def setup_logging(*, logger: logging.Logger, log_level: int) -> None:
    import datasets
    import transformers

    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


---

## transcribe

import os
import pathlib
import sys
from typing import TYPE_CHECKING

import modal
from modal.cli.volume import FileType

from .config import app_config
from .logs import get_logger

if TYPE_CHECKING:
    from numpy import ndarray

logger = get_logger(__name__)


def download_model_locally(run_id: str) -> pathlib.Path:
    """
    Download a finetuned model locally.

    NOTE: These models were trained on GPU and require torch.distributed installed locally.
    """
    logger.info(f"Saving finetuning run {run_id} model locally")
    vol = modal.NetworkFileSystem.lookup(app_config.persistent_vol_name)
    for entry in vol.listdir(f"{run_id}/**"):
        p = pathlib.Path(f".{app_config.model_dir}", entry.path)

        if entry.type == FileType.DIRECTORY:
            p.mkdir(parents=True, exist_ok=True)
        elif entry.type == FileType.FILE:
            logger.info(f"Downloading {entry.path} to {p}")
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "wb") as f:
                for chunk in vol.read_file(entry.path):
                    f.write(chunk)
        else:
            logger.warning(
                f"Skipping unknown entry '{p}' with unknown filetype"
            )
    return pathlib.Path(f".{app_config.model_dir}", run_id)


def whisper_transcribe_local_file(
    model_dir: os.PathLike,
    language: str,
    filepath: os.PathLike,
    sample_rate_hz: int,
) -> str:
    """Convenience function for transcribing a single local audio file with a Whisper model already saved to disk."""
    from datasets import Audio, Dataset

    audio_dataset = Dataset.from_dict({"audio": [str(filepath)]}).cast_column(
        "audio", Audio(sampling_rate=sample_rate_hz)
    )
    row = next(iter(audio_dataset))
    return whisper_transcribe_audio(
        model_dir,
        language,
        data=row["audio"]["array"],
        sample_rate_hz=row["audio"]["sampling_rate"],
    )


def whisper_transcribe_audio(
    model_dir: os.PathLike,
    language: str,
    data: "ndarray",
    sample_rate_hz: int,
) -> str:
    """Transcribes a single audio sample with a Whisper model, for demonstration purposes."""
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    # load model and processor
    processor = WhisperProcessor.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language, task="transcribe"
    )
    input_features = processor(
        data,
        sampling_rate=sample_rate_hz,
        return_tensors="pt",
    ).input_features

    # generate token ids
    predicted_ids = model.generate(
        input_features, forced_decoder_ids=forced_decoder_ids
    )
    # decode token ids to text
    predicted_transcription = processor.batch_decode(
        predicted_ids, skip_special_tokens=True
    )[0]
    return predicted_transcription


if __name__ == "__main__":
    download_model_locally(run_id=sys.argv[1])


---

## init



---

## api

import asyncio
import json
import time
from typing import List, NamedTuple

from fastapi import FastAPI, Request

from . import config
from .main import (
    get_episode_metadata_path,
    get_transcript_path,
    in_progress,
    populate_podcast_metadata,
    process_episode,
    search_podcast,
)
from .podcast import coalesce_short_transcript_segments

logger = config.get_logger(__name__)
web_app = FastAPI()

# A transcription taking > 10 minutes should be exceedingly rare.
MAX_JOB_AGE_SECS = 10 * 60


class InProgressJob(NamedTuple):
    call_id: str
    start_time: int


@web_app.get("/api/episode/{podcast_id}/{episode_guid_hash}")
async def get_episode(podcast_id: str, episode_guid_hash: str):
    episode_metadata_path = get_episode_metadata_path(
        podcast_id, episode_guid_hash
    )
    transcription_path = get_transcript_path(episode_guid_hash)

    with open(episode_metadata_path, "r") as f:
        metadata = json.load(f)

    if not transcription_path.exists():
        return dict(metadata=metadata)

    with open(transcription_path, "r") as f:
        data = json.load(f)

    return dict(
        metadata=metadata,
        segments=coalesce_short_transcript_segments(data["segments"]),
    )


@web_app.get("/api/podcast/{podcast_id}")
async def get_podcast(podcast_id: str):
    pod_metadata_path = (
        config.PODCAST_METADATA_DIR / podcast_id / "metadata.json"
    )
    previously_stored = True
    if not pod_metadata_path.exists():
        previously_stored = False
        # Don't run this Modal function in a separate container in the cloud, because then
        # we'd be exposed to a race condition with the NFS if we don't wait for the write
        # to propogate.
        raw_populate_podcast_metadata = populate_podcast_metadata.get_raw_f()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, raw_populate_podcast_metadata, podcast_id
        )

    with open(pod_metadata_path, "r") as f:
        pod_metadata = json.load(f)

    episodes = []
    for file in (config.PODCAST_METADATA_DIR / podcast_id).iterdir():
        if file == pod_metadata_path:
            continue

        with open(file, "r") as f:
            ep = json.load(f)
            ep["transcribed"] = get_transcript_path(ep["guid_hash"]).exists()
            episodes.append(ep)

    episodes.sort(key=lambda ep: ep.get("publish_date"), reverse=True)

    # Refresh possibly stale data asynchronously.
    if previously_stored:
        populate_podcast_metadata.spawn(podcast_id)
    return dict(pod_metadata=pod_metadata, episodes=episodes)


@web_app.post("/api/podcasts")
async def podcasts_endpoint(request: Request):
    import dataclasses

    form = await request.form()
    name = form["podcast"]
    podcasts_response = []
    for pod in search_podcast.remote(name):
        podcasts_response.append(dataclasses.asdict(pod))
    return podcasts_response


@web_app.post("/api/transcribe")
async def transcribe_job(podcast_id: str, episode_id: str):
    now = int(time.time())
    try:
        inprogress_job = in_progress[episode_id]
        # NB: runtime type check is to handle present of old `str` values that didn't expire.
        if (
            isinstance(inprogress_job, InProgressJob)
            and (now - inprogress_job.start_time) < MAX_JOB_AGE_SECS
        ):
            existing_call_id = inprogress_job.call_id
            logger.info(
                f"Found existing, unexpired call ID {existing_call_id} for episode {episode_id}"
            )
            return {"call_id": existing_call_id}
    except KeyError:
        pass

    call = process_episode.spawn(podcast_id, episode_id)
    in_progress[episode_id] = InProgressJob(
        call_id=call.object_id, start_time=now
    )

    return {"call_id": call.object_id}


@web_app.get("/api/status/{call_id}")
async def poll_status(call_id: str):
    from modal.call_graph import InputInfo, InputStatus
    from modal.functions import FunctionCall

    function_call = FunctionCall.from_id(call_id)
    graph: List[InputInfo] = function_call.get_call_graph()

    try:
        function_call.get(timeout=0.1)
    except TimeoutError:
        pass
    except Exception as exc:
        if exc.args:
            inner_exc = exc.args[0]
            if "HTTPError 403" in inner_exc:
                return dict(error="permission denied on podcast audio download")
        return dict(error="unknown job processing error")

    try:
        map_root = graph[0].children[0].children[0]
    except IndexError:
        return dict(finished=False)

    assert map_root.function_name == "transcribe_episode"

    leaves = map_root.children
    tasks = len(set([leaf.task_id for leaf in leaves]))
    done_segments = len(
        [leaf for leaf in leaves if leaf.status == InputStatus.SUCCESS]
    )
    total_segments = len(leaves)
    finished = map_root.status == InputStatus.SUCCESS

    return dict(
        finished=finished,
        total_segments=total_segments,
        tasks=tasks,
        done_segments=done_segments,
    )


---

## config

import dataclasses
import logging
import pathlib


@dataclasses.dataclass
class ModelSpec:
    name: str
    params: str
    relative_speed: int  # Higher is faster


def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(levelname)s: %(asctime)s: %(name)s  %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


CACHE_DIR = "/cache"
# Where downloaded podcasts are stored, by guid hash.
# Mostly .mp3 files 50-100MiB.
RAW_AUDIO_DIR = pathlib.Path(CACHE_DIR, "raw_audio")
# Stores metadata of individual podcast episodes as JSON.
PODCAST_METADATA_DIR = pathlib.Path(CACHE_DIR, "podcast_metadata")
# Completed episode transcriptions. Stored as flat files with
# files structured as '{guid_hash}-{model_slug}.json'.
TRANSCRIPTIONS_DIR = pathlib.Path(CACHE_DIR, "transcriptions")
# Searching indexing files, refreshed by scheduled functions.
SEARCH_DIR = pathlib.Path(CACHE_DIR, "search")
# Location of modal checkpoint.
MODEL_DIR = pathlib.Path(CACHE_DIR, "model")
# Location of web frontend assets.
ASSETS_PATH = pathlib.Path(__file__).parent / "frontend" / "dist"

transcripts_per_podcast_limit = 2

supported_whisper_models = {
    "tiny.en": ModelSpec(name="tiny.en", params="39M", relative_speed=32),
    # Takes around 3-10 minutes to transcribe a podcast, depending on length.
    "base.en": ModelSpec(name="base.en", params="74M", relative_speed=16),
    "small.en": ModelSpec(name="small.en", params="244M", relative_speed=6),
    "medium.en": ModelSpec(name="medium.en", params="769M", relative_speed=2),
    # Very slow. Will take around 45 mins to 1.5 hours to transcribe.
    "large": ModelSpec(name="large", params="1550M", relative_speed=1),
}

DEFAULT_MODEL = supported_whisper_models["base.en"]


---

## main

"""
whisper-pod-transcriber uses OpenAI's Whisper modal to do speech-to-text transcription
of podcasts.
"""

import dataclasses
import datetime
import json
import pathlib
from typing import Iterator, Tuple

from modal import (
    App,
    Dict,
    Image,
    Mount,
    NetworkFileSystem,
    Secret,
    asgi_app,
)

from . import config, podcast, search

logger = config.get_logger(__name__)
volume = NetworkFileSystem.from_name(
    "dataset-cache-vol", create_if_missing=True
)

app_image = (
    Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "git+https://github.com/openai/whisper.git",
        "dacite",
        "jiwer",
        "ffmpeg-python",
        "gql[all]~=3.0.0a5",
        "python-multipart~=0.0.9",
        "pandas",
        "loguru==0.6.0",
        "torchaudio==2.1.0",
    )
    .apt_install("ffmpeg")
    .pip_install("ffmpeg-python")
)
search_image = Image.debian_slim(python_version="3.10").pip_install(
    "scikit-learn~=1.3.0",
    "tqdm~=4.46.0",
    "numpy~=1.23.3",
    "dacite",
)

app = App(
    "whisper-pod-transcriber",
    image=app_image,
    secrets=[Secret.from_name("podchaser")],
)  # Note: prior to April 2024, "app" was called "stub"

in_progress = Dict.from_name(
    "pod-transcriber-in-progress", create_if_missing=True
)


def utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def get_episode_metadata_path(podcast_id: str, guid_hash: str) -> pathlib.Path:
    return config.PODCAST_METADATA_DIR / podcast_id / f"{guid_hash}.json"


def get_transcript_path(guid_hash: str) -> pathlib.Path:
    return config.TRANSCRIPTIONS_DIR / f"{guid_hash}.json"


@app.function(network_file_systems={config.CACHE_DIR: volume})
def populate_podcast_metadata(podcast_id: str):
    from gql import gql

    metadata_dir = config.PODCAST_METADATA_DIR / podcast_id
    metadata_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = config.PODCAST_METADATA_DIR / podcast_id / "metadata.json"
    pod_metadata: podcast.PodcastMetadata = podcast.fetch_podcast(
        gql, podcast_id
    )

    with open(metadata_path, "w") as f:
        json.dump(dataclasses.asdict(pod_metadata), f)

    episodes = fetch_episodes.remote(
        show_name=pod_metadata.title, podcast_id=podcast_id
    )

    for ep in episodes:
        metadata_path = get_episode_metadata_path(podcast_id, ep.guid_hash)
        with open(metadata_path, "w") as f:
            json.dump(dataclasses.asdict(ep), f)

    logger.info(f"Populated metadata for {pod_metadata.title}")


@app.function(
    mounts=[Mount.from_local_dir(config.ASSETS_PATH, remote_path="/assets")],
    network_file_systems={config.CACHE_DIR: volume},
    keep_warm=2,
)
@asgi_app()
def fastapi_app():
    import fastapi.staticfiles

    from .api import web_app

    web_app.mount(
        "/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True)
    )

    return web_app


@app.function(
    image=app_image,
)
def search_podcast(name):
    from gql import gql

    logger.info(f"Searching for '{name}'")
    client = podcast.create_podchaser_client()
    podcasts_raw = podcast.search_podcast_name(
        gql, client, name, max_results=10
    )
    logger.info(f"Found {len(podcasts_raw)} results for '{name}'")
    return [
        podcast.PodcastMetadata(
            id=pod["id"],
            title=pod["title"],
            description=pod["description"],
            html_description=pod["htmlDescription"],
            language=pod["language"],
            web_url=pod["webUrl"],
        )
        for pod in podcasts_raw
    ]


@app.function(
    image=search_image,
    network_file_systems={config.CACHE_DIR: volume},
    timeout=(400 * 60),
)
def refresh_index():
    import dataclasses
    from collections import defaultdict

    import dacite

    logger.info(f"Running scheduled index refresh at {utc_now()}")
    config.SEARCH_DIR.mkdir(parents=True, exist_ok=True)

    episodes = defaultdict(list)
    guid_hash_to_episodes = {}

    for pod_dir in config.PODCAST_METADATA_DIR.iterdir():
        if not pod_dir.is_dir():
            continue

        for filepath in pod_dir.iterdir():
            if filepath.name == "metadata.json":
                continue

            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
            except json.decoder.JSONDecodeError:
                logger.warning(
                    f"Removing corrupt JSON metadata file: {filepath}."
                )
                filepath.unlink()

            ep = dacite.from_dict(data_class=podcast.EpisodeMetadata, data=data)
            episodes[ep.podcast_title].append(ep)
            guid_hash_to_episodes[ep.guid_hash] = ep

    logger.info(f"Loaded {len(guid_hash_to_episodes)} podcast episodes.")

    transcripts = {}
    if config.TRANSCRIPTIONS_DIR.exists():
        for file in config.TRANSCRIPTIONS_DIR.iterdir():
            with open(file, "r") as f:
                data = json.load(f)
                guid_hash = file.stem.split("-")[0]
                transcripts[guid_hash] = data

    # Important: These have to be the same length and have same episode order.
    # i-th element of indexed_episodes is the episode indexed by the i-th element
    # of search_records
    indexed_episodes = []
    search_records = []
    for key, value in transcripts.items():
        idxd_episode = guid_hash_to_episodes.get(key)
        if idxd_episode:
            search_records.append(
                search.SearchRecord(
                    title=idxd_episode.title,
                    text=value["text"],
                )
            )
            # Prepare records for JSON serialization
            indexed_episodes.append(dataclasses.asdict(idxd_episode))

    logger.info(
        f"Matched {len(search_records)} transcripts to episode records."
    )

    filepath = config.SEARCH_DIR / "all.json"
    logger.info(f"writing {filepath}")
    with open(filepath, "w") as f:
        json.dump(indexed_episodes, f)

    logger.info(
        "calc feature vectors for all transcripts, keeping track of similar podcasts"
    )
    X, v = search.calculate_tfidf_features(search_records)
    sim_svm = search.calculate_similarity_with_svm(X)
    filepath = config.SEARCH_DIR / "sim_tfidf_svm.json"
    logger.info(f"writing {filepath}")
    with open(filepath, "w") as f:
        json.dump(sim_svm, f)

    logger.info("calculate the search index to support search")
    search_dict = search.build_search_index(search_records, v)
    filepath = config.SEARCH_DIR / "search.json"
    logger.info(f"writing {filepath}")
    with open(filepath, "w") as f:
        json.dump(search_dict, f)


def split_silences(
    path: str, min_segment_length: float = 30.0, min_silence_length: float = 1.0
) -> Iterator[Tuple[float, float]]:
    """Split audio file into contiguous chunks using the ffmpeg `silencedetect` filter.
    Yields tuples (start, end) of each chunk in seconds."""

    import re

    import ffmpeg

    silence_end_re = re.compile(
        r" silence_end: (?P<end>[0-9]+(\.?[0-9]*)) \| silence_duration: (?P<dur>[0-9]+(\.?[0-9]*))"
    )

    metadata = ffmpeg.probe(path)
    duration = float(metadata["format"]["duration"])

    reader = (
        ffmpeg.input(str(path))
        .filter("silencedetect", n="-10dB", d=min_silence_length)
        .output("pipe:", format="null")
        .run_async(pipe_stderr=True)
    )

    cur_start = 0.0
    num_segments = 0

    while True:
        line = reader.stderr.readline().decode("utf-8")
        if not line:
            break
        match = silence_end_re.search(line)
        if match:
            silence_end, silence_dur = match.group("end"), match.group("dur")
            split_at = float(silence_end) - (float(silence_dur) / 2)

            if (split_at - cur_start) < min_segment_length:
                continue

            yield cur_start, split_at
            cur_start = split_at
            num_segments += 1

    # silencedetect can place the silence end *after* the end of the full audio segment.
    # Such segments definitions are negative length and invalid.
    if duration > cur_start:
        yield cur_start, duration
        num_segments += 1
    logger.info(f"Split {path} into {num_segments} segments")


@app.function(
    image=app_image,
    network_file_systems={config.CACHE_DIR: volume},
    cpu=2,
    timeout=400,
)
def transcribe_segment(
    start: float,
    end: float,
    audio_filepath: pathlib.Path,
    model: config.ModelSpec,
):
    import tempfile
    import time

    import ffmpeg
    import torch
    import whisper

    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
        (
            ffmpeg.input(str(audio_filepath))
            .filter("atrim", start=start, end=end)
            .output(f.name)
            .overwrite_output()
            .run(quiet=True)
        )

        use_gpu = torch.cuda.is_available()
        device = "cuda" if use_gpu else "cpu"
        model = whisper.load_model(
            model.name, device=device, download_root=config.MODEL_DIR
        )
        result = model.transcribe(f.name, language="en", fp16=use_gpu)  # type: ignore

    logger.info(
        f"Transcribed segment {start:.2f} to {end:.2f} ({end - start:.2f}s duration) in {time.time() - t0:.2f} seconds."
    )

    # Add back offsets.
    for segment in result["segments"]:
        segment["start"] += start
        segment["end"] += start

    return result


@app.function(
    image=app_image,
    network_file_systems={config.CACHE_DIR: volume},
    timeout=900,
)
def transcribe_episode(
    audio_filepath: pathlib.Path,
    result_path: pathlib.Path,
    model: config.ModelSpec,
):
    segment_gen = split_silences(str(audio_filepath))

    output_text = ""
    output_segments = []
    for result in transcribe_segment.starmap(
        segment_gen, kwargs=dict(audio_filepath=audio_filepath, model=model)
    ):
        output_text += result["text"]
        output_segments += result["segments"]

    result = {
        "text": output_text,
        "segments": output_segments,
        "language": "en",
    }

    logger.info(f"Writing openai/whisper transcription to {result_path}")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=4)


@app.function(
    image=app_image,
    network_file_systems={config.CACHE_DIR: volume},
    timeout=900,
)
def process_episode(podcast_id: str, episode_id: str):
    import dacite
    import whisper

    try:
        # pre-download the model to the cache path, because the _download fn is not
        # thread-safe.
        model = config.DEFAULT_MODEL
        whisper._download(whisper._MODELS[model.name], config.MODEL_DIR, False)

        config.RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        config.TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)

        metadata_path = get_episode_metadata_path(podcast_id, episode_id)
        with open(metadata_path, "r") as f:
            data = json.load(f)
            episode = dacite.from_dict(
                data_class=podcast.EpisodeMetadata, data=data
            )

        destination_path = config.RAW_AUDIO_DIR / episode_id
        podcast.store_original_audio(
            url=episode.original_download_link,
            destination=destination_path,
        )

        logger.info(
            f"Using the {model.name} model which has {model.params} parameters."
        )
        logger.info(f"Wrote episode metadata to {metadata_path}")

        transcription_path = get_transcript_path(episode.guid_hash)
        if transcription_path.exists():
            logger.info(
                f"Transcription already exists for '{episode.title}' with ID {episode.guid_hash}."
            )
            logger.info("Skipping transcription.")
        else:
            transcribe_episode.remote(
                audio_filepath=destination_path,
                result_path=transcription_path,
                model=model,
            )
    finally:
        del in_progress[episode_id]

    return episode


@app.function(
    image=app_image,
    network_file_systems={config.CACHE_DIR: volume},
)
def fetch_episodes(show_name: str, podcast_id: str, max_episodes=100):
    import hashlib

    from gql import gql

    client = podcast.create_podchaser_client()
    episodes_raw = podcast.fetch_episodes_data(
        gql, client, podcast_id, max_episodes=max_episodes
    )
    logger.info(f"Retrieved {len(episodes_raw)} raw episodes")
    episodes = [
        podcast.EpisodeMetadata(
            podcast_id=podcast_id,
            podcast_title=show_name,
            title=ep["title"],
            publish_date=ep["airDate"],
            description=ep["description"],
            episode_url=ep["url"],
            html_description=ep["htmlDescription"],
            guid=ep["guid"],
            guid_hash=hashlib.md5(ep["guid"].encode("utf-8")).hexdigest(),
            original_download_link=ep["audioUrl"],
        )
        for ep in episodes_raw
        if "guid" in ep and ep["guid"] is not None
    ]
    no_guid_count = len(episodes) - len(episodes_raw)
    logger.info(f"{no_guid_count} episodes had no GUID and couldn't be used.")
    return episodes


@app.local_entrypoint()
def search_entrypoint(name: str):
    # To search for a podcast, run:
    # modal run app.main --name "search string"
    for pod in search_podcast.remote(name):
        print(pod)


---

## podcast

import dataclasses
import os
import pathlib
import urllib.request
from typing import NamedTuple, Optional, TypedDict, Union

from . import config

logger = config.get_logger(__name__)
Segment = TypedDict("Segment", {"text": str, "start": float, "end": float})


@dataclasses.dataclass
class EpisodeMetadata:
    # Unique ID of podcast this episode is associated with.
    podcast_id: Union[str, int]
    # Title of podcast this episode is associated with.
    podcast_title: Optional[str]
    title: str
    # The publish date of the episode as specified by the publisher
    publish_date: str
    # Plaintext description of episode. nb: has whitespace issues so not suitable in UI.
    description: str
    # HTML markup description. Suitable for display in UI.
    html_description: str
    # The unique identifier of this episode within the context of the podcast
    guid: str
    # Hash the guid into something appropriate for filenames.
    guid_hash: str
    # Link to episode on Podchaser website.
    episode_url: Optional[str]
    # Link to audio file for episode. Typically an .mp3 file.
    original_download_link: str


@dataclasses.dataclass
class PodcastMetadata:
    # Unique ID for a podcast
    id: str
    # Title of podcast, eg. 'The Joe Rogan Experience'.
    title: str
    # Plaintext description of episode. nb: has whitespace issues so not suitable in UI.
    description: str
    html_description: str
    # Link to podcast on Podchaser website.
    web_url: str
    # Used to detect non-English podcasts.
    language: Optional[str] = None


class DownloadResult(NamedTuple):
    data: bytes
    # Helpful to store and transmit when uploading to cloud bucket.
    content_type: str


def download_podcast_file(url: str) -> DownloadResult:
    req = urllib.request.Request(
        url,
        data=None,
        # Set a user agent to avoid 403 response from some podcast audio servers.
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36"
        },
    )
    with urllib.request.urlopen(req) as response:
        return DownloadResult(
            data=response.read(),
            content_type=response.headers["content-type"],
        )


def create_podchaser_client():
    """
    Use's Podchaser's graphql API to get an new access token and instantiate
    a graphql client with it.
    """
    from gql import Client, gql
    from gql.transport.aiohttp import AIOHTTPTransport

    transport = AIOHTTPTransport(url="https://api.podchaser.com/graphql")
    client = Client(transport=transport, fetch_schema_from_transport=True)
    podchaser_client_id = os.environ.get("PODCHASER_CLIENT_ID")
    podchaser_client_secret = os.environ.get("PODCHASER_CLIENT_SECRET")

    if not podchaser_client_id or not podchaser_client_secret:
        exit(
            "Must provide both PODCHASER_CLIENT_ID and PODCHASER_CLIENT_SECRET as environment vars."
        )

    query = gql(
        """
        mutation {{
            requestAccessToken(
                input: {{
                    grant_type: CLIENT_CREDENTIALS
                    client_id: "{client_id}"
                    client_secret: "{client_secret}"
                }}
            ) {{
                access_token
                token_type
            }}
        }}
    """.format(
            client_id=podchaser_client_id,
            client_secret=podchaser_client_secret,
        )
    )

    result = client.execute(query)

    access_token = result["requestAccessToken"]["access_token"]
    transport = AIOHTTPTransport(
        url="https://api.podchaser.com/graphql",
        headers={"Authorization": f"Bearer {access_token}"},
    )

    return Client(transport=transport, fetch_schema_from_transport=True)


def search_podcast_name(gql, client, name, max_results=5) -> list[dict]:
    """
    Search for a podcast by name/title. eg. 'Joe Rogan Experience' or 'Serial'.

    This method does not paginate queries because 100s of search results is not
    useful in this application.
    """
    if max_results > 100:
        raise ValueError(
            f"A maximum of 100 results is supported, but {max_results} results were requested."
        )
    current_page = 0
    max_episodes_per_request = max_results
    search_podcast_name_query = gql(
        """
        query {{
            podcasts(searchTerm: "{name}", first: {max_episodes_per_request}, page: {current_page}) {{
                paginatorInfo {{
                    currentPage,
                    hasMorePages,
                    lastPage,
                }},
                data {{
                    id,
                    title,
                    description,
                    language,
                    htmlDescription,
                    webUrl,
                }}
            }}
        }}
        """.format(
            name=name,
            max_episodes_per_request=max_episodes_per_request,
            current_page=current_page,
        )
    )
    logger.info(f"Querying Podchaser for podcasts matching query '{name}'.")
    result = client.execute(search_podcast_name_query)
    podcasts_in_page = result["podcasts"]["data"]
    return podcasts_in_page


def fetch_episodes_data(
    gql, client, podcast_id, max_episodes=100
) -> list[dict]:
    """
    Use the Podchaser API to grab a podcast's episodes.
    """
    max_episodes_per_request = 100  # Max allowed by API
    episodes = []
    has_more_pages = True
    current_page = 0
    while has_more_pages:
        list_episodes_query = gql(
            """
            query getPodList {{
                podcast(identifier: {{id: "{id}", type: PODCHASER}}) {{
                    episodes(first: {max_episodes_per_request}, page: {current_page}) {{
                        paginatorInfo {{
                          count
                          currentPage
                          firstItem
                          hasMorePages
                          lastItem
                          lastPage
                          perPage
                          total
                        }}
                        data {{
                          id
                          title
                          airDate
                          audioUrl
                          description
                          htmlDescription
                          guid
                          url
                        }}
                    }}
                }}
            }}
        """.format(
                id=podcast_id,
                max_episodes_per_request=max_episodes_per_request,
                current_page=current_page,
            )
        )

        logger.info(f"Fetching {max_episodes_per_request} episodes from API.")
        result = client.execute(list_episodes_query)
        has_more_pages = result["podcast"]["episodes"]["paginatorInfo"][
            "hasMorePages"
        ]
        episodes_in_page = result["podcast"]["episodes"]["data"]
        episodes.extend(episodes_in_page)
        current_page += 1
        if len(episodes) >= max_episodes:
            break
    return episodes


def fetch_podcast_data(gql, client, podcast_id) -> dict:
    podcast_metadata_query = gql(
        """
        query {{
            podcast(identifier: {{id: "{podcast_id}", type: PODCHASER}}) {{
                id,
                title,
                description,
                htmlDescription,
                webUrl,
            }}
        }}
        """.format(
            podcast_id=podcast_id,
        )
    )
    logger.info(f"Querying Podchaser for podcast with ID {podcast_id}.")
    result = client.execute(podcast_metadata_query)
    return result["podcast"]


def fetch_podcast(gql, podcast_id: str) -> PodcastMetadata:
    client = create_podchaser_client()
    data = fetch_podcast_data(gql=gql, client=client, podcast_id=podcast_id)
    return PodcastMetadata(
        id=data["id"],
        title=data["title"],
        description=data["description"],
        html_description=data["htmlDescription"],
        web_url=data["webUrl"],
    )


def sizeof_fmt(num, suffix="B") -> str:
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)


def store_original_audio(
    url: str, destination: pathlib.Path, overwrite: bool = False
) -> None:
    if destination.exists():
        if overwrite:
            logger.info(
                f"Audio file exists at {destination} but overwrite option is specified."
            )
        else:
            logger.info(
                f"Audio file exists at {destination}, skipping download."
            )
            return

    podcast_download_result = download_podcast_file(url=url)
    humanized_bytes_str = sizeof_fmt(num=len(podcast_download_result.data))
    logger.info(f"Downloaded {humanized_bytes_str} episode from URL.")
    with open(destination, "wb") as f:
        f.write(podcast_download_result.data)
    logger.info(f"Stored audio episode at {destination}.")


def coalesce_short_transcript_segments(
    segments: list[Segment],
) -> list[Segment]:
    """
    Some extracted transcript segments from openai/whisper are really short, like even just one word.
    This function accepts a minimum segment length and combines short segments until the minimum is reached.
    """
    minimum_transcript_len = 200  # About 2 sentences.
    previous = None
    long_enough_segments = []
    for current in segments:
        if previous is None:
            previous = current
        elif len(previous["text"]) < minimum_transcript_len:
            previous = _merge_segments(left=previous, right=current)
        else:
            long_enough_segments.append(previous)
            previous = current
    if previous:
        long_enough_segments.append(previous)
    return long_enough_segments


def _merge_segments(left: Segment, right: Segment) -> Segment:
    return {
        "text": left["text"] + " " + right["text"],
        "start": left["start"],
        "end": right["end"],
    }


---

## search

import dataclasses
import json
import pathlib
from typing import Any

from . import podcast


@dataclasses.dataclass
class SearchRecord:
    title: str
    text: str


def search_transcripts(
    search_dict_path: pathlib.Path,
    query: str,
    items: list[podcast.EpisodeMetadata],
):
    query_parts = query.lower().strip().split()
    print(f"loading search dictionary from {search_dict_path}")
    with open(search_dict_path, "r") as f:
        search_dict = json.load(f)

    n = len(items)
    scores = []
    for i, sd in enumerate(search_dict):
        score = sum(sd.get(q, 0) for q in query_parts)
        if score == 0:
            continue  # no match whatsoever, don't include
        score += (
            1.0 * (n - i) / n
        )  # give a small boost to more recent episodes (low index)
        scores.append((score, items[i]))
    # Sort descending, best scores first.
    scores.sort(reverse=True, key=lambda x: x[0])
    return scores


def calculate_tfidf_features(
    records: list[SearchRecord],
    max_features: int = 5000,
    max_df: float = 1.0,
    min_df: int = 3,
):
    """
    Compute tfidf features with scikit learn.
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    v = TfidfVectorizer(
        input="content",
        encoding="utf-8",
        decode_error="replace",
        strip_accents="unicode",
        lowercase=True,
        analyzer="word",
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z0-9_-]+\b",
        ngram_range=(1, 1),
        max_features=max_features,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
        max_df=max_df,
        min_df=min_df,
    )
    corpus = [(a.title + ". " + a.text) for a in records]
    X = v.fit_transform(corpus)
    X = np.asarray(X.astype(np.float32).todense())
    print("tfidf calculated array of shape ", X.shape)
    return X, v


def calculate_sim_dot_product(X, ntake=40):
    """
    Take `X` (N,D) features and for each index return closest `ntake` indices via dot product.
    """
    from numpy import np

    S = np.dot(X, X.T)
    IX = np.argsort(S, axis=1)[
        :, : -ntake - 1 : -1
    ]  # take last ntake sorted backwards
    return IX.tolist()


def calculate_similarity_with_svm(X, ntake=40):
    """
    Take X (N,D) features and for each index return closest `ntake` indices using exemplar SVM.
    """
    import numpy as np
    import sklearn.svm
    from tqdm import tqdm

    n, d = X.shape
    ntake = min(ntake, n)  # Cannot take more than is available
    IX = np.zeros((n, ntake), dtype=np.int64)
    print(f"training {n} svms for each paper...")
    for i in tqdm(range(n)):
        # set all examples as negative except this one
        y = np.zeros(X.shape[0], dtype=np.float32)
        y[i] = 1
        # train an SVM
        clf = sklearn.svm.LinearSVC(
            class_weight="balanced",
            verbose=False,
            max_iter=10000,
            tol=1e-4,
            C=0.1,
        )
        clf.fit(X, y)
        s = clf.decision_function(X)
        ix = np.argsort(s)[
            : -ntake - 1 : -1
        ]  # take last ntake sorted backwards
        IX[i] = ix
    return IX.tolist()


def build_search_index(records: list[SearchRecord], v):
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    # construct a reverse index for supporting search
    vocab = v.vocabulary_
    idf = v.idf_
    punc = "'!\"#$%&'()*+,./:;<=>?@[\\]^_`{|}~'"  # removed hyphen from string.punctuation
    trans_table = {ord(c): None for c in punc}

    def makedict(s, forceidf=None):
        words = set(s.lower().translate(trans_table).strip().split())
        words = set(
            w for w in words if len(w) > 1 and (w not in ENGLISH_STOP_WORDS)
        )
        idfd = {}
        for w in words:
            if forceidf is None:
                if w in vocab:
                    idfval = idf[vocab[w]]  # we have a computed idf for this
                else:
                    idfval = (
                        1.0  # some word we don't know; assume idf 1.0 (low)
                    )
            else:
                idfval = forceidf
            idfd[w] = idfval
        return idfd

    def merge_dicts(dict_list: list[dict]):
        m: dict[str, Any] = {}
        for d in dict_list:
            for key, val in d.items():
                m[key] = m.get(key, 0) + val
        return m

    search_dict = []
    for p in records:
        dict_title = makedict(p.title, forceidf=10)
        dict_summary = makedict(p.text)
        qdict = merge_dicts([dict_title, dict_summary])
        search_dict.append(qdict)

    return search_dict


---

## transcribe check

import pathlib

from . import config, podcast
from .main import (
    app,
    app_image,
    split_silences,
    transcribe_episode,
    transcribe_segment,
    volume,
)

logger = config.get_logger(__name__)


def _transcribe_serially(
    audio_path: pathlib.Path, offset: int = 0
) -> list[tuple[float, float]]:
    model = config.DEFAULT_MODEL
    segment_gen = split_silences(str(audio_path))
    failed_segments = []
    for i, (start, end) in enumerate(segment_gen):
        if i < offset:
            continue
        logger.info(f"Attempting transcription of ({start}, {end})...")
        try:
            transcribe_segment(
                start=start, end=end, audio_filepath=audio_path, model=model
            )
        except Exception as exc:
            logger.info(f"Transcription failed for ({start}, {end}).")
            print(exc)
            failed_segments.append((start, end))
    logger.info(f"{len(failed_segments)} failed to transcribe.")
    return failed_segments


@app.function(
    image=app_image,
    network_file_systems={config.CACHE_DIR: volume},
    timeout=1000,
)
def test_transcribe_handles_dangling_segment():
    """
    Some podcast episodes have an empty, dangling audio segment after being split on silences.
    This test runs transcription on such an episode to check that we haven't broken transcription
    on episodes like this.

    If the transcription does fail, individual segments are checked to pull out the problem segments
    for further debugging.
    ```
    libpostproc    55.  7.100 / 55.  7.100
    [mp3 @ 0x557b828bb380] Format mp3 detected only with low score of 24, misdetection possible!
    [mp3 @ 0x557b828bb380] Failed to read frame size: Could not seek to 1026.
    /tmp/tmpuyr2iwce.mp3: Invalid argument
    ```
    """
    import ffmpeg

    # Stripped down podcast episode metadata for an episode which fails to transcribe @ commit e7093414.
    problem_episode = {
        "guid_hash": "b5b3005075fce663b3646f88a41b2b32",
        "podcast_id": "217829",
        "episode_url": "https://www.podchaser.com/podcasts/super-data-science-217829/episodes/sds-503-deep-reinforcement-lea-98045099",
        "original_download_link": "http://www.podtrac.com/pts/redirect.mp3/feeds.soundcloud.com/stream/1120216126-superdatascience-sds-503-deep-reinforcement-learning-for-robotics.mp3",
    }
    audio_path = pathlib.Path(
        config.CACHE_DIR, "test", f"{problem_episode['guid_hash']}.tmp.mp3"
    )
    audio_path.parent.mkdir(exist_ok=True)
    podcast.store_original_audio(
        url=problem_episode["original_download_link"],
        destination=audio_path,
    )

    model = config.DEFAULT_MODEL

    try:
        result_path = pathlib.Path(
            config.CACHE_DIR,
            "test",
            f"{problem_episode['guid_hash']}.transcription.json",
        )
        transcribe_episode(
            audio_filepath=audio_path,
            result_path=result_path,
            model=model,
        )
    except Exception as exc:
        print(exc)
        logger.error(
            "Transcription failed. Proceeding to checks of individual segments."
        )
    else:
        return  # Transcription worked fine.

    failed_segments = _transcribe_serially(audio_path, offset=107)
    # Checking the 1st is probably sufficient to discover bug.
    problem_segment = failed_segments[0]
    start = problem_segment[0]
    end = problem_segment[1]
    logger.info(f"Problem segment time range is ({start}, {end})")
    try:
        transcribe_segment(
            start=start, end=end, audio_filepath=audio_path, model=model
        )
    except Exception:
        logger.info(
            "Writing the problem segment to the network file system for further debugging."
        )
        bad_segment_path = pathlib.Path(
            config.CACHE_DIR,
            "test",
            f"{problem_episode['guid_hash']}.badsegment.mp3",
        )
        with open(bad_segment_path, "wb") as f:
            (
                ffmpeg.input(str(audio_path))
                .filter("atrim", start=start, end=end)
                .output(f.name)
                .overwrite_output()
                .run(quiet=True)
            )
        raise


if __name__ == "__main__":
    with app.run():
        test_transcribe_handles_dangling_segment()


---

## main

# ---
# runtimes: ["runc", "gvisor"]
# ---
import asyncio
import io
import logging
import pathlib
import re
import tempfile
import time
from typing import Iterator

import modal
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

image = (
    modal.Image.debian_slim()
    .apt_install("git", "ffmpeg")
    .pip_install(
        "https://github.com/openai/whisper/archive/v20230314.tar.gz",
        "ffmpeg-python",
        "pytube @ git+https://github.com/felipeucelli/pytube",
    )
)
app = modal.App(
    name="example-whisper-streaming", image=image
)  # Note: prior to April 2024, "app" was called "stub"
web_app = FastAPI()
CHARLIE_CHAPLIN_DICTATOR_SPEECH_URL = (
    "https://www.youtube.com/watch?v=J7GY1Xg6X20"
)


def load_audio(data: bytes, start=None, end=None, sr: int = 16000):
    import ffmpeg
    import numpy as np

    try:
        fp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        fp.write(data)
        fp.close()
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        if start is None and end is None:
            out, _ = (
                ffmpeg.input(fp.name, threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .run(
                    cmd=["ffmpeg", "-nostdin"],
                    capture_stdout=True,
                    capture_stderr=True,
                )
            )
        else:
            out, _ = (
                ffmpeg.input(fp.name, threads=0)
                .filter("atrim", start=start, end=end)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .run(
                    cmd=["ffmpeg", "-nostdin"],
                    capture_stdout=True,
                    capture_stderr=True,
                )
            )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def split_silences(
    path: str, min_segment_length: float = 30.0, min_silence_length: float = 0.8
) -> Iterator[tuple[float, float]]:
    """
    Split audio file into contiguous chunks using the ffmpeg `silencedetect` filter.
    Yields tuples (start, end) of each chunk in seconds.

    Parameters
    ----------
    path: str
        path to the audio file on disk.
    min_segment_length : float
        The minimum acceptable length for an audio segment in seconds. Lower values
        allow for more splitting and increased parallelizing, but decrease transcription
        accuracy. Whisper models expect to transcribe in 30 second segments, so this is the
        default minimum.
    min_silence_length : float
        Minimum silence to detect and split on, in seconds. Lower values are more likely to split
        audio in middle of phrases and degrade transcription accuracy.
    """
    import ffmpeg

    silence_end_re = re.compile(
        r" silence_end: (?P<end>[0-9]+(\.?[0-9]*)) \| silence_duration: (?P<dur>[0-9]+(\.?[0-9]*))"
    )

    metadata = ffmpeg.probe(path)
    duration = float(metadata["format"]["duration"])

    reader = (
        ffmpeg.input(str(path))
        .filter("silencedetect", n="-10dB", d=min_silence_length)
        .output("pipe:", format="null")
        .run_async(pipe_stderr=True)
    )

    cur_start = 0.0
    num_segments = 0

    while True:
        line = reader.stderr.readline().decode("utf-8")
        if not line:
            break
        match = silence_end_re.search(line)
        if match:
            silence_end, silence_dur = match.group("end"), match.group("dur")
            split_at = float(silence_end) - (float(silence_dur) / 2)

            if (split_at - cur_start) < min_segment_length:
                continue

            yield cur_start, split_at
            cur_start = split_at
            num_segments += 1

    # silencedetect can place the silence end *after* the end of the full audio segment.
    # Such segments definitions are negative length and invalid.
    if duration > cur_start and (duration - cur_start) > min_segment_length:
        yield cur_start, duration
        num_segments += 1
    print(f"Split {path} into {num_segments} segments")


@app.function()
def download_mp3_from_youtube(youtube_url: str) -> bytes:
    from pytube import YouTube

    logging.getLogger("pytube").setLevel(logging.INFO)
    yt = YouTube(youtube_url)
    video = yt.streams.filter(only_audio=True).first()
    buffer = io.BytesIO()
    video.stream_to_buffer(buffer)
    buffer.seek(0)
    return buffer.read()


@app.function(cpu=2)
def transcribe_segment(
    start: float,
    end: float,
    audio_data: bytes,
    model: str,
):
    import torch
    import whisper

    print(
        f"Transcribing segment {start:.2f} to {end:.2f} ({end - start:.2f}s duration)"
    )

    t0 = time.time()
    use_gpu = torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    model = whisper.load_model(model, device=device)
    np_array = load_audio(audio_data, start=start, end=end)
    result = model.transcribe(np_array, language="en", fp16=use_gpu)  # type: ignore
    print(
        f"Transcribed segment {start:.2f} to {end:.2f} ({end - start:.2f}s duration) in {time.time() - t0:.2f} seconds."
    )

    # Add back offsets.
    for segment in result["segments"]:
        segment["start"] += start
        segment["end"] += start

    return result


async def stream_whisper(audio_data: bytes):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(audio_data)
        f.flush()
        segment_gen = split_silences(f.name)

    async for result in transcribe_segment.starmap(
        segment_gen, kwargs=dict(audio_data=audio_data, model="base.en")
    ):
        # Must cooperatively yield here otherwise `StreamingResponse` will not iteratively return stream parts.
        # see: https://github.com/python/asyncio/issues/284#issuecomment-154162668
        await asyncio.sleep(0)
        yield result["text"]


@web_app.get("/transcribe")
async def transcribe(url: str):
    """
    Usage:

    ```sh
    curl --no-buffer \
        https://modal-labs--example-whisper-streaming-web.modal.run/transcribe?url=https://www.youtube.com/watch?v=s_LncVnecLA"
    ```

    This endpoint will stream back the Youtube's audio transcription as it makes progress.

    Some example Youtube videos for inspiration:

    1. Churchill's 'We shall never surrender' speech - https://www.youtube.com/watch?v=s_LncVnecLA
    2. Charlie Chaplin's final speech from The Great Dictator - https://www.youtube.com/watch?v=J7GY1Xg6X20
    """
    import pytube.exceptions

    print(f"downloading {url}")
    try:
        audio_data = download_mp3_from_youtube.remote(url)
    except pytube.exceptions.RegexMatchError:
        raise HTTPException(
            status_code=422, detail=f"Could not process url {url}"
        )
    print(f"streaming transcription of {url} audio to client...")
    return StreamingResponse(
        stream_whisper(audio_data), media_type="text/event-stream"
    )


@app.function()
@modal.asgi_app()
def web():
    return web_app


@app.function()
async def transcribe_cli(data: bytes, suffix: str):
    async for result in stream_whisper(data):
        print(result)


@app.local_entrypoint()
def main(path: str = CHARLIE_CHAPLIN_DICTATOR_SPEECH_URL):
    if path.startswith("https"):
        data = download_mp3_from_youtube.remote(path)
        suffix = ".mp3"
    else:
        filepath = pathlib.Path(path)
        data = filepath.read_bytes()
        suffix = filepath.suffix
    transcribe_cli.remote(
        data,
        suffix=suffix,
    )


---

## init



---

## app

"""
Contains only definitions of Modal objects, to be imported
from other modules.
"""

import modal

image = modal.Image.debian_slim(
    python_version="3.10"
).pip_install(
    "datasets~=2.7.1",
    "dill==0.3.4",  # pinned b/c of https://github.com/uqfoundation/dill/issues/481
    "evaluate~=0.3.0",
    "loguru~=0.6.0",
    "pyarrow~=10.0.1",
    "scikit-learn~=1.1.3",  # Required by evaluate pkg.
    "torch~=1.13.0",
    "transformers~=4.24.0",
)

app = modal.App(
    name="example-spam-detect-llm", image=image
)  # Note: prior to April 2024, "app" was called "stub"
# Used to store datasets, trained models, model metadata, config.
volume = modal.Volume.from_name(
    "example-spam-detect-vol", create_if_missing=True
)


---

## config

import enum
import logging
import pathlib
import sys

VOLUME_DIR: str = "/cache"
MODEL_STORE_DIR = pathlib.Path(VOLUME_DIR, "models")
MODEL_REGISTRY_FILENAME: str = "registry.json"
DATA_DIR = pathlib.Path(VOLUME_DIR, "data")

SERVING_MODEL_ID: str = (
    "sha256.12E5065BE4C3F7D2F79B7A0FD203380869F6E308DCBB4B8C9579FFAE6F32B837"
)


class ModelType(str, enum.Enum):
    BAD_WORDS = "BAD_WORDS"
    LLM = "LLM"
    NAIVE_BAYES = "NAIVE_BAYES"


def get_logger():
    try:
        from loguru import logger

        logger.remove()
        logger.add(sys.stderr, colorize=True, level="INFO")
    except ModuleNotFoundError:
        logger = logging.getLogger()
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(levelname)s: %(asctime)s: %(name)s  %(message)s"
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = (
            False  # Prevent the modal client from double-logging.
        )
    return logger


---

## dataset

"""
Module for the fetching, pre-processing, and loading of spam classification datasets.
Currently only provides access to the ENRON email dataset.
"""

import csv
import json
import pathlib
import shutil
import tempfile
import urllib.request
import zipfile
from typing import NamedTuple

# TODO:
# This dataset only produces ~50,000 examples.
# Other links to the dataset claim ~500,000 examples, eg. https://www.kaggle.com/wcukierski/enron-email-dataset
# which links to https://www.cs.cmu.edu/~./enron/.
enron_dataset_url = "https://github.com/MWiechmann/enron_spam_data/raw/master/enron_spam_data.zip"


class Example(NamedTuple):
    email: str
    spam: bool


RawEnronDataset = list[Example]
CleanEnronDataset = dict[str, Example]


def dataset_path(base: pathlib.Path) -> pathlib.Path:
    return base / "raw" / "enron" / "all.json"


def deserialize_dataset(dataset_path: pathlib.Path) -> RawEnronDataset:
    with open(dataset_path, "r") as f:
        items = json.load(f)
    return [Example(email=item[0], spam=bool(item[1])) for item in items]


def _download_and_extract_dataset(destination_root_path: pathlib.Path, logger):
    logger.info("Downloading raw enron dataset.")
    destination_path = destination_root_path / "enron.zip"
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        enron_dataset_url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36"
        },
    )
    with urllib.request.urlopen(req) as response, open(
        destination_path, "wb"
    ) as out_file:
        shutil.copyfileobj(response, out_file)

    with zipfile.ZipFile(destination_path, "r") as zip_ref:
        logger.info("Extracting zip with contents: ")
        zip_ref.printdir()
        zip_ref.extractall(destination_root_path)

    return destination_root_path / "enron_spam_data.csv"


def fix_nulls(f):
    for line in f:
        yield line.replace("\0", "")


def download(logger, base: pathlib.Path) -> None:
    dest = dataset_path(base)
    dest.parent.mkdir(exist_ok=True, parents=True)
    tmp_path = pathlib.Path(tempfile.TemporaryDirectory().name)
    dataset_csv_path = _download_and_extract_dataset(
        destination_root_path=tmp_path, logger=logger
    )
    ds: list[Example] = []
    spam_count = 0
    with open(dataset_csv_path, "r") as csvfile:
        csv.field_size_limit(100_000_000)
        reader = csv.DictReader(fix_nulls(csvfile), delimiter=",")
        for row in reader:
            is_spam = row["Spam/Ham"] == "spam"
            if is_spam:
                spam_count += 1
            ex = Example(
                email=row["Subject"] + " " + row["Message"],
                spam=is_spam,
            )
            ds.append(ex)

    spam_percentage = round((spam_count / len(ds)) * 100, ndigits=4)
    logger.info(
        f"writing processed raw dataset to file. dataset contains {len(ds)} examples and is {spam_percentage}% spam"
    )
    with open(dest, "w") as f:
        json.dump(ds, f, indent=4)


---

## model registry

"""
Defines minimal data structures and command-line interface (CLI) commands for a model registry.
The CLI commands are operationally useful, used to inspect prior trained models and promote the
most promising models to production serving.
"""

import json
from typing import Callable, NamedTuple, Optional

from . import config
from .app import app, volume


class Prediction(NamedTuple):
    spam: bool
    score: float


SpamClassifier = Callable[[str], Prediction]


class TrainMetrics(NamedTuple):
    # human-readable identifier for the dataset used in training.
    dataset_id: str
    # How many examples in the evaluation subset.
    eval_set_size: int
    # (TP + TN) / (TP + TN + FP + FN)
    accuracy: Optional[float] = None
    # TP / (TP + FP)
    precision: Optional[float] = None
    # TP / (TP + FN)
    recall: Optional[float] = None


class ModelMetadata(NamedTuple):
    impl_name: str
    save_date: str  # UTC+ISO8601 formatted.
    git_commit_hash: str
    metrics: Optional[TrainMetrics] = None

    def serialize(self) -> dict:
        d = self._asdict()
        if d["metrics"]:
            d["metrics"] = d["metrics"]._asdict()
        return d

    @classmethod
    def from_dict(cls, m: dict) -> "ModelMetadata":
        if "metrics" not in m or m["metrics"] is None:
            metrics = None
        else:
            metrics = TrainMetrics(
                dataset_id=m["metrics"]["dataset_id"],
                eval_set_size=m["metrics"]["eval_set_size"],
                accuracy=m["metrics"]["accuracy"],
                precision=m["metrics"]["precision"],
                recall=m["metrics"]["recall"],
            )
        return cls(
            impl_name=m["impl_name"],
            save_date=m["save_date"],
            git_commit_hash=m["git_commit_hash"],
            metrics=metrics,
        )


@app.function(volumes={config.VOLUME_DIR: volume})
def _list_models() -> dict[str, ModelMetadata]:
    registry_filepath = config.MODEL_STORE_DIR / config.MODEL_REGISTRY_FILENAME
    with open(registry_filepath, "r") as f:
        registry_data = json.load(f)
    return {
        m_id: ModelMetadata.from_dict(m) for m_id, m in registry_data.items()
    }


@app.function(volumes={config.VOLUME_DIR: volume})
def delete_model(
    # sha256 hashtag of model. eg 'sha256.1234567890abcd'
    model_id: str,
    # Don't actually delete, just show deletion plan.
    dry_run: bool = True,
) -> None:
    """Remove a model from registry and storage."""
    pass


@app.local_entrypoint()
def list_models() -> None:
    """Show all models in registry."""
    with app.run():
        models = _list_models.remote()
    newest_to_oldest = sorted(
        [(key, value) for key, value in models.items()],
        key=lambda item: item[1].save_date,
        reverse=True,
    )
    for model_id, metadata in newest_to_oldest:
        print(
            f"\033[96m {model_id} \033[0m{metadata.impl_name}\033[93m {metadata.save_date} \033[0m"
        )


if __name__ == "__main__":
    print("USAGE: modal run spam_detect.model_registry [FUNCTION]")
    raise SystemExit(1)


---

## model storage

"""
The model storage module contains functions for the serialization, and
disk-based storage of the email spam models defined within models.py.
"""

import datetime
import hashlib
import io
import json
import pathlib
import pickle
import random
import string
import subprocess
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
)

from . import config, dataset
from .model_registry import ModelMetadata, SpamClassifier, TrainMetrics

logger = config.get_logger()

Dataset = Iterable[dataset.Example]
TrainingFunc = Callable[[Dataset], Any]
ModelBuilder = Callable[[Dataset, Optional[TrainingFunc]], SpamClassifier]


Sha256Hash = str
ModelRegistryMetadata = Dict[Sha256Hash, ModelMetadata]


def get_git_revision_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"])
        .decode("ascii")
        .strip()
    )


def serialize_model(
    model_func: SpamClassifier,
) -> bytes:
    try:
        from datasets.utils.py_utils import Pickler
    except ModuleNotFoundError:
        from pickle import Pickler  # type: ignore

    def dumps(obj, **kwds):
        file = io.BytesIO()
        Pickler(file, **kwds).dump(obj)
        return file.getvalue()

    return dumps(model_func)


def create_hashtag_from_dir(dir: pathlib.Path) -> str:
    dgst = hashlib.sha256()
    for f in dir.glob("**/*"):
        dgst.update(f.name.encode())
        dgst.update(f.read_bytes())
    return f"sha256.{dgst.hexdigest().upper()}"


def create_hashtag_from_bytes(b: bytes) -> str:
    hash_base = hashlib.sha256(b).hexdigest().upper()
    return f"sha256.{hash_base}"


def store_huggingface_model(
    trainer: Any,
    train_metrics: TrainMetrics,
    model_name: str,
    model_destination_root: pathlib.Path,
    git_commit_hash: str,
) -> str:
    """
    Accepts a Hugginface model that implements `save_model()` and stores it in model
    registry and persistent filesystem.
    """
    tmp_dirname = "".join(
        random.choices(string.ascii_uppercase + string.digits, k=20)
    )
    model_save_path = model_destination_root / tmp_dirname
    trainer.save_model(output_dir=model_save_path)
    model_hashtag = create_hashtag_from_dir(model_save_path)
    model_save_path.rename(model_destination_root / model_hashtag)

    logger.info(f"serialized model's hash is {model_hashtag}")

    model_registry_metadata = load_model_registry_metadata(
        model_registry_root=model_destination_root,
    )

    model_dest_path = model_destination_root / model_hashtag
    if model_dest_path.is_file():
        logger.warning(
            (
                f"model {model_hashtag} already exists. No need to save again. "
                "consider caching model training to save compute cycles."
            )
        )

    logger.info(
        f"updating models registry metadata to include information about {model_hashtag}"
    )
    metadata = ModelMetadata(
        impl_name=model_name,
        save_date=datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        git_commit_hash=git_commit_hash,
    )
    store_model_registry_metadata(
        model_registry_metadata=model_registry_metadata,
        sha256_hash=model_hashtag,
        metadata=metadata,
        destination_root=model_destination_root,
    )
    logger.info("📦 done! Model stored.")
    return model_hashtag


def store_pickleable_model(
    *,
    classifier_func: SpamClassifier,
    metrics: TrainMetrics,
    model_destination_root: pathlib.Path,
    current_git_commit_hash: str,
) -> str:
    """
    Stores a pickle-able model in registry and persistent filesystem.
    The `pickle` process only works on a single classifier Python object,
    and should only be used for simple, pure-Python classifiers.
    """
    logger.info("storing spam model to model registry using pickling.")

    serialized_model = serialize_model(classifier_func)
    ser_clssfr_hash = create_hashtag_from_bytes(serialized_model)

    logger.info(f"serialized model's hash is {ser_clssfr_hash}")

    model_registry_metadata = load_model_registry_metadata(
        model_registry_root=model_destination_root,
    )

    model_dest_path = model_destination_root / ser_clssfr_hash
    if model_dest_path.is_file():
        logger.warning(
            (
                f"model {ser_clssfr_hash} already exists. No need to save again. "
                "consider caching model training to save compute cycles."
            )
        )
    else:
        logger.info(f"saving model to file at '{model_dest_path}'")
        model_dest_path.write_bytes(serialized_model)

    logger.info(
        f"updating models registry metadata to include information about {ser_clssfr_hash}"
    )
    metadata = ModelMetadata(
        impl_name=model_name_from_function(classifier_func),
        save_date=datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        git_commit_hash=current_git_commit_hash,
        metrics=metrics,
    )
    store_model_registry_metadata(
        model_registry_metadata=model_registry_metadata,
        sha256_hash=ser_clssfr_hash,
        metadata=metadata,
        destination_root=model_destination_root,
    )
    logger.info("📦 done! Model stored.")
    return ser_clssfr_hash


def model_name_from_function(model_func: SpamClassifier) -> str:
    # NOTE: This may be buggy, and create name clashes or ambiguity.
    return model_func.__qualname__


def load_model_registry_metadata(
    *,
    model_registry_root: pathlib.Path,
):
    model_registry_metadata_filepath = (
        model_registry_root / config.MODEL_REGISTRY_FILENAME
    )
    if not model_registry_metadata_filepath.exists():
        # Create registry metadata file on first save of a model.
        model_registry_metadata_filepath.write_text("{}")

    with open(model_registry_metadata_filepath, "r") as model_registry_f:
        data = json.load(model_registry_f)
    model_registry_metadata: ModelRegistryMetadata = {
        key: ModelMetadata(
            impl_name=value["impl_name"],
            save_date=value["save_date"],
            git_commit_hash=value["git_commit_hash"],
        )
        for key, value in data.items()
    }
    return model_registry_metadata


def retrieve_model_registry_metadata(
    *,
    model_registry_metadata: ModelRegistryMetadata,
    sha256_hash: str,
) -> Optional[ModelMetadata]:
    return model_registry_metadata.get(sha256_hash)


def store_model_registry_metadata(
    *,
    model_registry_metadata: ModelRegistryMetadata,
    sha256_hash: str,
    metadata: ModelMetadata,
    destination_root: pathlib.Path,
) -> None:
    existing_metadata = retrieve_model_registry_metadata(
        model_registry_metadata=model_registry_metadata,
        sha256_hash=sha256_hash,
    )
    if existing_metadata is not None:
        logger.debug("classifier with matching hash found in registry.")
        # compare new metadata with old to detect registry corruption or
        # strange renaming.
        if metadata.impl_name != existing_metadata.impl_name:
            raise RuntimeError(
                "Existing classifier with identical sha256 hash to current classifier found "
                "with conflicting metadata. "
                "Something has gone wrong."
            )
    model_registry_metadata_dict = {
        key: value._asdict() for key, value in model_registry_metadata.items()
    }
    # NOTE: Potentially overwrites with new metadata.
    model_registry_metadata_dict[sha256_hash] = metadata.serialize()
    with open(
        destination_root / config.MODEL_REGISTRY_FILENAME, "w"
    ) as model_registry_f:
        json.dump(model_registry_metadata_dict, model_registry_f, indent=4)


def load_pickle_serialized_model(
    *,
    sha256_hash: str,
    destination_root: pathlib.Path,
) -> SpamClassifier:
    def check_integrity(*, expected_hash: str, actual_hash: str) -> None:
        if not expected_hash == actual_hash:
            err_msg = f"Shasum integrity check failure. Expected '{expected_hash}' but got '{actual_hash}'"
            raise ValueError(err_msg)

    expected_prefix = "sha256."
    if not sha256_hash.startswith(expected_prefix):
        raise ValueError(
            f"model sha256 hashes are expected to start with the prefix '{expected_prefix}"
        )

    model_path = destination_root / sha256_hash
    with open(model_path, "rb") as f:
        model_bytes = f.read()

    hash_base = hashlib.sha256(model_bytes).hexdigest().upper()
    filestored_model_hashtag = f"sha256.{hash_base}"

    check_integrity(
        expected_hash=sha256_hash,
        actual_hash=filestored_model_hashtag,
    )
    return pickle.loads(model_bytes)


---

## models

"""

The core model interface is `SpamModel`, which must be implemented by all
trainable and serveable spam-detection models in the module.

Current model implementations are:

* BadWords (a baseline heuristic classifier)
* LLM (a fine-tuned BERT language classifier)
* NaiveBayes
"""

import json
import math
import pathlib
import re
from collections import defaultdict
from typing import (
    Optional,
    Protocol,
    cast,
)

from . import config, model_storage
from .dataset import Example
from .model_registry import (
    ModelMetadata,
    Prediction,
    SpamClassifier,
    TrainMetrics,
)

Dataset = list[Example]


def load_model(model_id: str):
    registry_filepath = config.MODEL_STORE_DIR / config.MODEL_REGISTRY_FILENAME
    with open(registry_filepath, "r") as f:
        registry_data = json.load(f)
    if model_id not in registry_data:
        raise ValueError(f"{model_id} not contained in registry.")

    metadata = ModelMetadata.from_dict(registry_data[model_id])
    m: SpamModel
    if metadata.impl_name == "bert-base-cased":
        m = LLM()
    elif "NaiveBayes" in metadata.impl_name:
        m = NaiveBayes()
    else:
        raise ValueError(f"Loading '{metadata.impl_name}' not yet supported.")

    classifier = m.load(
        sha256_digest=model_id,
        model_registry_root=config.MODEL_STORE_DIR,
    )
    return classifier, metadata


def tokenize(text: str) -> set[str]:
    text = text.lower()
    all_words = re.findall("[a-z0-9]+", text)  # extract the words
    return set(all_words)


class SpamModel(Protocol):
    """The training and storage interface that all spam-classification models must implement."""

    def train(self, dataset: Dataset) -> tuple[SpamClassifier, TrainMetrics]:
        ...

    def load(
        self, sha256_digest: str, model_registry_root: pathlib.Path
    ) -> SpamClassifier:
        ...

    def save(
        self,
        fn: SpamClassifier,
        metrics: TrainMetrics,
        model_registry_root: pathlib.Path,
        git_commit_hash: str,
    ) -> str:
        ...


def construct_huggingface_dataset(dataset: Dataset, label2id: dict[str, int]):
    import datasets
    import pyarrow as pa

    emails = pa.array((ex.email for ex in dataset), type=pa.string())
    labels = pa.array(
        (label2id["SPAM"] if ex.spam else label2id["HAM"] for ex in dataset),
        type=pa.uint8(),
    )
    pa_table = pa.table([emails, labels], names=["text", "labels"])
    return datasets.Dataset(pa_table).train_test_split(test_size=0.1)


class LLMSpamClassifier:
    """SpamClassifier that wraps a fine-tuned Huggingface BERT transformer model."""

    def __init__(self, tokenizer, model) -> None:
        self.tokenizer = tokenizer
        self.model = model

    def __call__(self, email: str) -> Prediction:
        """Ensures this class-based classifier can be used just like a function-based classifer."""
        import torch

        inputs = self.tokenizer(email, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_class_id = logits.argmax().item()
        spam_id = self.model.config.label2id["SPAM"]
        spam_score = logits[0][spam_id]
        predicted_label: str = self.model.config.id2label[predicted_class_id]
        return Prediction(
            spam=bool(predicted_label == "SPAM"),
            score=spam_score,
        )


def train_llm_classifier(
    dataset: Dataset, dry_run: bool = False
) -> tuple[LLMSpamClassifier, TrainMetrics]:
    import evaluate
    import numpy as np
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    logger = config.get_logger()

    id2label = {0: "HAM", 1: "SPAM"}
    label2id = {"HAM": 0, "SPAM": 1}
    huggingface_dataset = construct_huggingface_dataset(dataset, label2id)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True
        )

    tokenized_datasets = huggingface_dataset.map(
        tokenize_function, batched=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(output_dir="test_trainer")

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="test_trainer", evaluation_strategy="epoch"
    )

    small_train_dataset = (
        tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    )
    small_eval_dataset = (
        tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    logger.opt(colors=True).info("<light-yellow>training</light-yellow> 🏋️")

    if not dry_run:
        trainer.train()
        logger.opt(colors=True).info(
            "<light-green>✔️ training done!</light-green>"
        )
    else:
        logger.info(f"{dry_run=}, so skipping training step.")

    metrics = TrainMetrics(
        dataset_id="enron",
        eval_set_size=-1,
        accuracy=0.0,
    )
    return trainer, metrics


class LLM(SpamModel):
    """
    A large-language model (LLM) fine-tuned for the SPAM/HAM text classification problem.

    Uses huggingface/transformers library.
    """

    model_name = "bert-base-cased"

    def train(self, dataset: Dataset) -> tuple[SpamClassifier, TrainMetrics]:
        from transformers import AutoTokenizer

        trainer, metrics = train_llm_classifier(dataset=dataset)
        model = trainer.model
        tokenizer = AutoTokenizer.from_pretrained(LLM.model_name)
        return (
            LLMSpamClassifier(
                tokenizer=tokenizer,
                model=model,
            ),
            metrics,
        )

    def load(
        self, sha256_digest: str, model_registry_root: pathlib.Path
    ) -> SpamClassifier:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        # TODO: refactor to use model_storage module for loading.
        model_path = model_registry_root / sha256_digest
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(LLM.model_name)
        return LLMSpamClassifier(
            tokenizer=tokenizer,
            model=model,
        )

    def save(
        self,
        fn: SpamClassifier,
        metrics: TrainMetrics,
        model_registry_root: pathlib.Path,
        git_commit_hash: str,
    ) -> str:
        from transformers import Trainer

        llm_fn = cast(LLMSpamClassifier, fn)
        trainer = Trainer(model=llm_fn.model)
        return model_storage.store_huggingface_model(
            trainer=trainer,
            train_metrics=metrics,
            model_name=LLM.model_name,
            model_destination_root=model_registry_root,
            git_commit_hash=git_commit_hash,
        )


class BadWords(SpamModel):
    """
    An extremely rudimentary heuritistic model. If a trained model
    can't beat this something is very wrong.
    """

    def train(self, dataset: Dataset) -> tuple[SpamClassifier, TrainMetrics]:
        def bad_words_spam_classifier(email: str) -> Prediction:
            tokens = email.split(" ")
            tokens_set = set(tokens)
            # NB: using a set here makes pickle serialization non-deterministic.
            bad_words = [
                "click",  # http://www.paulgraham.com/spam.html
                "sex",
                "xxx",
                "nigerian",
                "teens",
            ]
            max_bad_words = 2
            bad_words_count = 0
            for word in bad_words:
                if word in tokens_set:
                    bad_words_count += 1
            return (
                Prediction(score=1.0, spam=True)
                if bad_words_count > max_bad_words
                else Prediction(score=0.0, spam=False)
            )

        accuracy, precision = self._calc_metrics(
            classifier=bad_words_spam_classifier, dataset=dataset
        )
        metrics = TrainMetrics(
            dataset_id="enron",
            eval_set_size=0,
            accuracy=accuracy,
            precision=precision,
        )
        return bad_words_spam_classifier, metrics

    def load(
        self, sha256_digest: str, model_registry_root: pathlib.Path
    ) -> SpamClassifier:
        return model_storage.load_pickle_serialized_model(
            sha256_hash=sha256_digest,
            destination_root=model_registry_root,
        )

    def save(
        self,
        fn: SpamClassifier,
        metrics: TrainMetrics,
        model_registry_root: pathlib.Path,
        git_commit_hash: str,
    ) -> str:
        return model_storage.store_pickleable_model(
            classifier_func=fn,
            metrics=metrics,
            model_destination_root=model_registry_root,
            current_git_commit_hash=git_commit_hash,
        )

    def _calc_metrics(
        self, classifier: SpamClassifier, dataset: Dataset
    ) -> tuple[float, float]:
        if len(dataset) == 0:
            raise ValueError("Evaluation dataset cannot be empty.")
        tp, tn, fp, fn = 0, 0, 0, 0
        for example in dataset:
            pred = classifier(example.email)
            if pred.spam and example.spam:
                tp += 1
            elif pred.spam and not example.spam:
                fp += 1
            elif not pred.spam and not example.spam:
                tn += 1
            else:
                fn += 1
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        print(f"Summary: {tp=} {fp=} {tn=} {fn=}")
        return accuracy, precision


class NaiveBayes(SpamModel):
    """
    The classic Naive-Bayes classifier. Implementation drawn from the
    *Data Science From Scratch* book: github.com/joelgrus/data-science-from-scratch.
    """

    def __init__(
        self,
        k: float = 0.5,
        decision_boundary: Optional[float] = None,
        test_set_size: float = 0.05,
    ) -> None:
        self.k = k
        self.decision_boundary = decision_boundary
        self.classify_fn: Optional[SpamClassifier] = None
        self.test_set_size = test_set_size

    def train(self, dataset: Dataset) -> tuple[SpamClassifier, TrainMetrics]:
        k = self.k
        dataset_tokens: set[str] = set()
        token_spam_counts: dict[str, int] = defaultdict(int)
        token_ham_counts: dict[str, int] = defaultdict(int)
        spam_messages = ham_messages = 0
        test_samples = int(len(dataset) * self.test_set_size)
        if test_samples > 0:
            train_set = dataset[:-test_samples]
            test_set = dataset[-test_samples:]
        else:
            train_set = dataset
            test_set = []

        for example in train_set:
            if example.spam:
                spam_messages += 1
            else:
                ham_messages += 1

            # Increment word counts
            for token in tokenize(example.email):
                dataset_tokens.add(token)
                if example.spam:
                    token_spam_counts[token] += 1
                else:
                    token_ham_counts[token] += 1

        print("finished building word count dicts")

        def predict_prob(email: str) -> float:
            email_tokens = tokenize(email)
            log_prob_if_spam = log_prob_if_ham = 0.0

            # Iterate through each word in our vocabulary
            for token in dataset_tokens:
                spam = token_spam_counts[token]
                ham = token_ham_counts[token]

                prob_if_spam = (spam + k) / (spam_messages + 2 * k)
                prob_if_ham = (ham + k) / (ham_messages + 2 * k)
                # If *token* appears in the message,
                # add the log probability of seeing it
                if token in email_tokens:
                    log_prob_if_spam += math.log(prob_if_spam)
                    log_prob_if_ham += math.log(prob_if_ham)
                # Otherwise add the log probability of _not_ seeing it,
                # which is log(1 - probability of seeing it)
                else:
                    log_prob_if_spam += math.log(1.0 - prob_if_spam)
                    log_prob_if_ham += math.log(1.0 - prob_if_ham)
            prob_if_spam = math.exp(log_prob_if_spam)
            prob_if_ham = math.exp(log_prob_if_ham)
            score = (
                prob_if_spam / (prob_if_spam + prob_if_ham)
                if prob_if_spam
                else 0.0
            )
            return score

        def make_classifier(
            prob_fn, decision_boundary: float
        ) -> SpamClassifier:
            def inner(email: str):
                score = prob_fn(email)
                return Prediction(
                    spam=score > decision_boundary,
                    score=score,
                )

            return inner

        if self.decision_boundary:
            decision_boundary, precision, recall = (
                self.decision_boundary,
                None,
                None,
            )
        else:
            print("setting decision boundary for binary classifier")
            decision_boundary, precision, recall = self._set_decision_boundary(
                prob_fn=predict_prob,
                test_dataset=test_set,
            )

        metrics = TrainMetrics(
            dataset_id="enron",
            eval_set_size=len(test_set),
            accuracy=None,
            precision=precision,
            recall=recall,
        )
        print("making classifier")
        return make_classifier(predict_prob, decision_boundary), metrics

    def load(
        self, sha256_digest: str, model_registry_root: pathlib.Path
    ) -> SpamClassifier:
        return model_storage.load_pickle_serialized_model(
            sha256_hash=sha256_digest,
            destination_root=model_registry_root,
        )

    def save(
        self,
        fn: SpamClassifier,
        metrics: TrainMetrics,
        model_registry_root: pathlib.Path,
        git_commit_hash: str,
    ) -> str:
        return model_storage.store_pickleable_model(
            classifier_func=fn,
            metrics=metrics,
            model_destination_root=model_registry_root,
            current_git_commit_hash=git_commit_hash,
        )

    def _set_decision_boundary(
        self, prob_fn, test_dataset
    ) -> tuple[float, float, float]:
        import numpy as np
        from sklearn.metrics import precision_recall_curve

        print(
            f"Using {len(test_dataset)} test dataset examples to set decision boundary.\n"
            "Warning: this is a slow operation that takes ~15 minutes."
        )

        minimum_acceptable_precision = (
            0.98  # ie. 2 in a 100 legit emails get marked as spam.
        )
        y_true = np.array([1 if ex.spam else 0 for ex in test_dataset])
        # scores are rounded because curve calculation time scales quickly in dim U, where U is number of unique scores.
        # NB: The precision-recall curve calculation is extremely slow on N ~10k+
        y_scores = np.array(
            [round(prob_fn(ex.email), ndigits=2) for ex in test_dataset]
        )
        # TODO: Optimize this very slow process.
        precisions, recalls, thresholds = precision_recall_curve(
            y_true, y_scores
        )
        for p, r, thres in zip(precisions, recalls, thresholds):
            print(
                "Using threshold={} as decision boundary, we reach precision={} and recall={}".format(
                    thres, p, r
                )
            )
            if p >= minimum_acceptable_precision:
                print(
                    f"Reached {minimum_acceptable_precision=} at threshold {thres}. Setting that as boundary."
                )
                break
        return thres, p, r


---

## serving

"""
Defines a serverless web API to expose trained models
"""

from typing import Optional

import modal
from fastapi import FastAPI, Header
from pydantic import BaseModel

from . import config, models
from .app import app, volume

web_app = FastAPI()


class ModelInput(BaseModel):
    text: str


class ModelMetdata(BaseModel):
    model_name: str
    model_id: str


class ModelOutput(BaseModel):
    spam: bool
    score: float
    metadata: ModelMetdata


# TODO(Jonathon): This will acquire a GPU even when `model_id` doesn't
# require it, which is inefficient. Find an elegant way to make the GPU optional.
@app.cls(gpu="A10G", volumes={config.VOLUME_DIR: volume})
class Model:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        classifier, metadata = models.load_model(model_id=self.model_id)
        self.classifier = classifier
        self.metadata = metadata

    @modal.method()
    def generate(self, text: str) -> ModelOutput:
        prediction = self.classifier(text)
        return ModelOutput(
            spam=prediction.spam,
            score=prediction.score,
            metadata=ModelMetdata(
                model_name=self.metadata.impl_name,
                model_id=self.model_id,
            ),
        )


@web_app.get("/api/v1/models")
async def handle_list_models():
    """
    Show details of actively serving models.
    """
    _, metadata = models.load_model(config.SERVING_MODEL_ID)
    return {config.SERVING_MODEL_ID: metadata.serialize()}


@web_app.post("/api/v1/classify")
async def handle_classification(
    input_: ModelInput, model_id: Optional[str] = Header(None)
):
    r"""
    Classify a body of text as spam or ham.

    eg.

    ```bash
    curl -X POST https://modal-labs--example-spam-detect-llm-web.modal.run/api/v1/classify \
    -H 'Content-Type: application/json' \
    -H 'Model-Id: sha256.12E5065BE4C3F7D2F79B7A0FD203380869F6E308DCBB4B8C9579FFAE6F32B837' \
    -d '{"text": "hello world"}'
    ```
    """
    model_id = model_id or config.SERVING_MODEL_ID
    print(model_id)
    model = Model(model_id)
    return model.generate.remote(input_.text)


@app.function()
@modal.asgi_app()
def web():
    return web_app


if __name__ == "__main__":
    app.serve()


---

## train

# # A Plan for Spam, 20 Years On: LLM vs. Naive Bayes
#
# This example trains multiple models (LLM, Naive Bayes) to perform
# spam classification on the ENRON email dataset. This is a return to Paul Graham's
# well-known 2002 post, A Plan For Spam (http://www.paulgraham.com/spam.html).
#
# Graham's original post focused on the Naive Bayes model. Here we pit that model against
# a current state-of-the-art large-language-model (LLM). Both models are trained on the dataset
# and served via a model API (serving.py).
#
# This module, train.py, is the model training entrypoint, providing functions to do CPU/GPU training
# before saving to disk. The other significant modules are as follows:
#
# * models.py — contains the core `SpamModel` interface and three implementing model classes, including `LLMSpamClassifier`.
# * serving.py — a minimal FastAPI model serving API, loading models by ID from a Modal persistent volume.
# * model_registry.py — defines minimal data structures and CLI commands for a model registry stored on Modal.
# * model_storage.py — functions concerned withn serializing and deserializing (ie. loading) the trained ML models.
#

import pathlib
import random
import subprocess
from datetime import timedelta

import modal

from . import config, dataset, models
from .app import app, volume


def fetch_git_commit_hash(allow_dirty: bool) -> str:
    # Ensure git state is clean so that the git commit hash accurately reflects
    # the configuration of the training run.
    #
    # Ignoring dirty git state when kicking off a training run means accepting
    # unreproducible model training outcomes.
    if not allow_dirty:
        if (
            subprocess.run(
                ("git", "diff-index", "--quiet", "--cached", "HEAD", "--")
            ).returncode
            != 0
        ):
            breakpoint()
            raise RuntimeError(
                "Dirty git status. Repository has staged but not yet committed changes.\n"
                "Commit these changes or remove them to get a clean git state."
            )
        elif subprocess.run(("git", "diff-files", "--quiet")).returncode != 0:
            raise RuntimeError(
                "Dirty git status. Repository has changes that could be staged.\n"
                "Commit these changes or add them to .gitignore."
            )
        res = subprocess.run(
            ("git", "ls-files", "--exclude-standard", "--others"),
            capture_output=True,
        )
        if res.returncode != 0:
            raise RuntimeError(
                f"Could not check `git` for untracked files. {res.stderr.decode()}"
            )
        if res.stdout:
            raise RuntimeError(
                "Dirty git status. Repository has untracked files.\n"
                "Remove these files, commit them, or add them to .gitignore."
            )
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
    )
    return result.stdout.decode().strip()


@app.function(volumes={config.VOLUME_DIR: volume})
def init_volume():
    config.MODEL_STORE_DIR.mkdir(parents=True, exist_ok=True)
    volume.commit()  # Persist changes


@app.function(
    timeout=int(timedelta(minutes=8).total_seconds()),
    volumes={config.VOLUME_DIR: volume},
)
def prep_dataset():
    logger = config.get_logger()
    datasets_path = config.DATA_DIR
    datasets_path.mkdir(parents=True, exist_ok=True)
    dataset.download(base=datasets_path, logger=logger)
    volume.commit()  # Persist changes


@app.function(
    volumes={config.VOLUME_DIR: volume},
    secrets=[modal.Secret.from_dict({"PYTHONHASHSEED": "10"})],
    timeout=int(timedelta(minutes=30).total_seconds()),
)
def train(
    model: models.SpamModel, dataset_path: pathlib.Path, git_commit_hash: str
):
    logger = config.get_logger()
    enron_dataset = dataset.deserialize_dataset(dataset_path)
    random.shuffle(enron_dataset)
    classifier, metrics = model.train(enron_dataset)
    model_id = model.save(
        fn=classifier,
        metrics=metrics,
        model_registry_root=config.MODEL_STORE_DIR,
        git_commit_hash=git_commit_hash,
    )
    volume.commit()  # Persist changes
    logger.info(f"saved model to model store. {model_id=}")
    # Reload the model
    logger.info("🔁 testing reload of model")
    classifier = model.load(
        sha256_digest=model_id,
        model_registry_root=config.MODEL_STORE_DIR,
    )
    is_spam = classifier("fake email!")
    print(f"classification: {is_spam=}")


@app.function(
    volumes={config.VOLUME_DIR: volume},
    secrets=[modal.Secret.from_dict({"PYTHONHASHSEED": "10"})],
    timeout=int(timedelta(minutes=30).total_seconds()),
    gpu=modal.gpu.T4(),
)
def train_gpu(
    model: models.SpamModel, dataset_path: pathlib.Path, git_commit_hash: str
):
    logger = config.get_logger()
    enron_dataset = dataset.deserialize_dataset(dataset_path)
    random.shuffle(enron_dataset)
    classifier, metrics = model.train(enron_dataset)
    model_id = model.save(
        fn=classifier,
        metrics=metrics,
        model_registry_root=config.MODEL_STORE_DIR,
        git_commit_hash=git_commit_hash,
    )
    volume.commit()  # Persist changes
    logger.info(f"saved model to model store. {model_id=}")


@app.function(
    secrets=[modal.Secret.from_dict({"PYTHONHASHSEED": "10"})],
    timeout=int(timedelta(minutes=30).total_seconds()),
)
def main(git_commit_hash: str, model_type=config.ModelType.BAD_WORDS):
    logger = config.get_logger()
    logger.opt(colors=True).info(
        "Ready to detect <fg #9dc100><b>SPAM</b></fg #9dc100> from <fg #ffb6c1><b>HAM</b></fg #ffb6c1>?"
    )
    dataset_path = dataset.dataset_path(config.DATA_DIR)

    logger.info(
        f"💪 training a {model_type} model at git commit {git_commit_hash[:8]}"
    )
    if model_type == config.ModelType.NAIVE_BAYES:
        train.remote(
            model=models.NaiveBayes(),
            dataset_path=dataset_path,
            git_commit_hash=git_commit_hash,
        )
    elif model_type == config.ModelType.LLM:
        train_gpu.remote(
            model=models.LLM(),
            dataset_path=dataset_path,
            git_commit_hash=git_commit_hash,
        )
    elif model_type == config.ModelType.BAD_WORDS:
        train.remote(
            model=models.BadWords(),
            dataset_path=dataset_path,
            git_commit_hash=git_commit_hash,
        )
    else:
        raise ValueError(f"Unknown model type '{model_type}'")


# Pass in the string representation of a supported `ModelType` to train
# a model of that type on the dataset.
#
# Example:
#
# ```
# modal run spam_detect.train::app.train_model --model-type "BAD_WORDS"
# ```


@app.local_entrypoint()
def train_model(model_type: str):
    model_type_val = config.ModelType(model_type)
    # All training runs are versioned against git repository state.
    git_commit_hash: str = fetch_git_commit_hash(allow_dirty=False)
    init_volume.remote()
    main.remote(
        git_commit_hash=git_commit_hash,
        model_type=model_type_val,
    )


if __name__ == "__main__":
    with app.run():
        train_model(model_type="NAIVE_BAYES")


---

## model storage test

import pytest
from spam_detect import model_storage, models


def dummy_classifier(email: str) -> models.Prediction:
    _ = email
    return models.Prediction(spam=False, score=0.86)


def test_hashtag_from_bytes():
    b = b"1234"
    expected = "sha256.03AC674216F3E15C761EE1A5E255F067953623C8B388B4459E13F978D7C846F4"
    assert model_storage.create_hashtag_from_bytes(b) == expected


def test_hashtag_from_dir(tmp_path):
    dir = tmp_path / "dir1"
    dir.mkdir()
    contents = {
        "one": b"1234",
        "two": b"5678",
        "three": b"hello world",
    }
    for filename, b in contents.items():
        p = dir / filename
        p.write_bytes(b)

    hashtag = model_storage.create_hashtag_from_dir(dir)
    # If we add file, the hash changes
    p = dir / "four"
    p.write_bytes(b"I change the hash")
    hashtag_2 = model_storage.create_hashtag_from_dir(dir)
    assert hashtag != hashtag_2
    # Renaming a file changes the hash
    p = dir / "one"
    p.rename(dir / "one.2")
    hashtag_3 = model_storage.create_hashtag_from_dir(dir)
    assert hashtag != hashtag_3
    assert hashtag_2 != hashtag_3


def test_load_model_success(tmp_path):
    tmp_classifier_digest = model_storage.store_pickleable_model(
        classifier_func=dummy_classifier,
        metrics=None,
        model_destination_root=tmp_path,
        current_git_commit_hash="TEST-NOT-REALLY-A-COMMIT-HASH",
    )

    loaded_dummy_classifier = model_storage.load_pickle_serialized_model(
        sha256_hash=tmp_classifier_digest,
        destination_root=tmp_path,
    )
    test_email = "test email: doesn't matter what contents"
    assert dummy_classifier(test_email) == loaded_dummy_classifier(test_email)


def test_load_model_corrupted_data(tmp_path):
    dummy_classifier_b: bytes = model_storage.serialize_model(dummy_classifier)
    # intentionally write model to non-content-addressed location. bogus path.
    bogus_hash = "bogus_pocus"
    bogus_path = tmp_path / bogus_hash
    with open(bogus_path, "wb") as f:
        f.write(dummy_classifier_b)

    with pytest.raises(ValueError):
        _ = model_storage.load_pickle_serialized_model(
            sha256_hash=bogus_hash,
            destination_root=tmp_path,
        )


---

## models test

import math

from spam_detect import models
from spam_detect.dataset import Example


def test_prob_calculation():
    dataset = [
        Example(email="spam rules", spam=True),
        Example(email="ham rules", spam=False),
        Example(email="hello ham", spam=False),
    ]

    classify_func, _ = models.NaiveBayes(
        decision_boundary=0.5, test_set_size=0.0
    ).train(dataset)
    email = "hello spam"
    probs_if_ham = [
        (1 + 0.5) / (2 + 2 * 0.5),  # "hello" (present)
        (0 + 0.5) / (2 + 2 * 0.5),  # "spam" (present)
        1 - (2 + 0.5) / (2 + 2 * 0.5),  # "ham" (not present)
        1 - (1 + 0.5) / (2 + 2 * 0.5),  # "rules" (not present)
    ]
    probs_if_spam = [
        (0 + 0.5) / (1 + 2 * 0.5),  # "hello" (present)
        (1 + 0.5) / (1 + 2 * 0.5),  # "spam" (present)
        1 - (0 + 0.5) / (1 + 2 * 0.5),  # "ham" (not present)
        1 - (1 + 0.5) / (1 + 2 * 0.5),  # "rules" (not present)
    ]
    p_if_spam = math.exp(sum(math.log(p) for p in probs_if_spam))
    p_if_ham = math.exp(sum(math.log(p) for p in probs_if_ham))

    # Should be about 0.83
    actual = classify_func(email).score
    expected = p_if_spam / (p_if_spam + p_if_ham)
    residual = abs(actual - expected)
    assert residual <= 0.001


---

## a1111 webui

# ---
# lambda-test: false
# ---
# # Stable Diffusion (A1111)
#
# This example runs the popular [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
# project on Modal, without modification. We just port the environment setup to a Modal container image
# and wrap the launch script with a `@web_server` decorator, and we're ready to go.
#
# You can run a temporary A1111 server with `modal serve a1111_webui.py` or deploy it permanently with `modal deploy a1111_webui.py`.

import subprocess

from modal import App, Image, web_server

PORT = 8000

# First, we define the image A1111 will run in.
# This takes a few steps because A1111 usually install its dependencies on launch via a script.
# The process may take a few minutes the first time, but subsequent image builds should only take a few seconds.

a1111_image = (
    Image.debian_slim(python_version="3.11")
    .apt_install(
        "wget",
        "git",
        "libgl1",
        "libglib2.0-0",
        "google-perftools",  # For tcmalloc
    )
    .env({"LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"})
    .run_commands(
        "git clone --depth 1 --branch v1.7.0 https://github.com/AUTOMATIC1111/stable-diffusion-webui /webui",
        "python -m venv /webui/venv",
        "cd /webui && . venv/bin/activate && "
        + "python -c 'from modules import launch_utils; launch_utils.prepare_environment()' --xformers",
        gpu="a10g",
    )
    .run_commands(
        "cd /webui && . venv/bin/activate && "
        + "python -c 'from modules import shared_init, initialize; shared_init.initialize(); initialize.initialize()'",
        gpu="a10g",
    )
)

app = App(
    "example-a1111-webui", image=a1111_image
)  # Note: prior to April 2024, "app" was called "stub"

# After defining the custom container image, we start the server with `accelerate launch`. This
# function is also where you would configure hardware resources, CPU/memory, and timeouts.
#
# If you want to run it with an A100 or H100 GPU, just change `gpu="a10g"` to `gpu="a100"` or `gpu="h100"`.
#
# Startup of the web server should finish in under one to three minutes.


@app.function(
    gpu="a10g",
    cpu=2,
    memory=1024,
    timeout=3600,
    # Allows 100 concurrent requests per container.
    allow_concurrent_inputs=100,
    # Keep at least one instance of the server running.
    keep_warm=1,
)
@web_server(port=PORT, startup_timeout=180)
def run():
    START_COMMAND = f"""
cd /webui && \
. venv/bin/activate && \
accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision=fp16 \
    --dynamo_backend=inductor \
    --num_cpu_threads_per_process=6 \
    /webui/launch.py \
        --skip-prepare-environment \
        --no-gradio-queue \
        --listen \
        --port {PORT}
"""
    subprocess.Popen(START_COMMAND, shell=True)


---

## playground

# ---
# output-directory: "/tmp/playground-2-5"
# args: ["--prompt", "A cinematic shot of a baby raccoon wearing an intricate Italian priest robe."]
# ---

from pathlib import Path

import fastapi.staticfiles
import modal
from fastapi import FastAPI, Request, Response
from fastapi.templating import Jinja2Templates

app = modal.App(
    "playground-2-5"
)  # Note: prior to April 2024, "app" was called "stub"

DIFFUSERS_GIT_SHA = "2e31a759b5bd8ca2b288b5c61709636a96c4bae9"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        f"git+https://github.com/huggingface/diffusers.git@{DIFFUSERS_GIT_SHA}",
        "transformers~=4.38.1",
        "accelerate==0.27.2",
        "safetensors==0.4.2",
    )
)


with image.imports():
    import io

    import torch
    from diffusers import DiffusionPipeline


@app.cls(image=image, gpu="H100")
class Model:
    @modal.build()
    @modal.enter()
    def load_weights(self):
        self.pipe = DiffusionPipeline.from_pretrained(
            "playgroundai/playground-v2.5-1024px-aesthetic",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")

        # # Optional: Use DPM++ 2M Karras scheduler for crisper fine details
        # from diffusers import EDMDPMSolverMultistepScheduler
        # pipe.scheduler = EDMDPMSolverMultistepScheduler()

    def _inference(self, prompt, n_steps=24, high_noise_frac=0.8):
        image = self.pipe(
            prompt,
            negative_prompt="disfigured, ugly, deformed",
            num_inference_steps=50,
            guidance_scale=3,
        ).images[0]

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")

        return buffer

    @modal.method()
    def inference(self, prompt, n_steps=24, high_noise_frac=0.8):
        return self._inference(
            prompt, n_steps=n_steps, high_noise_frac=high_noise_frac
        ).getvalue()

    @modal.web_endpoint()
    def web_inference(self, prompt, n_steps=24, high_noise_frac=0.8):
        return Response(
            content=self._inference(
                prompt, n_steps=n_steps, high_noise_frac=high_noise_frac
            ).getvalue(),
            media_type="image/jpeg",
        )


frontend_path = Path(__file__).parent / "frontend"

web_image = modal.Image.debian_slim().pip_install("jinja2")


@app.function(
    image=web_image,
    mounts=[modal.Mount.from_local_dir(frontend_path, remote_path="/assets")],
    allow_concurrent_inputs=20,
)
@modal.asgi_app()
def ui():
    web_app = FastAPI()
    templates = Jinja2Templates(directory="/assets")

    @web_app.get("/")
    async def read_root(request: Request):
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "inference_url": Model.web_inference.web_url,
                "model_name": "Playground 2.5",
                "default_prompt": "Astronaut in the ocean, cold color palette, muted colors, detailed, 8k",
            },
        )

    web_app.mount(
        "/static",
        fastapi.staticfiles.StaticFiles(directory="/assets"),
        name="static",
    )

    return web_app


@app.local_entrypoint()
def main(prompt: str):
    image_bytes = Model().inference.remote(prompt)

    dir = Path("/tmp/playground-2-5")
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    output_path = dir / "output.png"
    print(f"Saving it to {output_path}")
    with open(output_path, "wb") as f:
        f.write(image_bytes)


---

## stable diffusion cli

# ---
# output-directory: "/tmp/stable-diffusion"
# args: ["--prompt", "A 1600s oil painting of the New York City skyline"]
# runtimes: ["runc", "gvisor"]
# ---
# # Stable Diffusion CLI
#
# This example shows Stable Diffusion 1.5 with a number of optimizations
# that makes it run faster on Modal. The example takes about 10s to cold start
# and about 1.0s per image generated.
#
# To use the XL 1.0 model, see the example posted [here](/docs/examples/stable_diffusion_xl).
#
# For instance, here are 9 images produced by the prompt
# `A 1600s oil painting of the New York City skyline`
#
# ![stable diffusion montage](./stable_diffusion_montage.png)
#
# As mentioned, we use a few optimizations to run this faster:
#
# * Use [run_function](/docs/reference/modal.Image#run_function) to download the model while building the container image
# * Use a [container lifecycle method](https://modal.com/docs/guide/lifecycle-functions) to initialize the model on container startup
# * Use A10G GPUs
# * Use 16 bit floating point math


# ## Basic setup
from __future__ import annotations

import io
import time
from pathlib import Path

from modal import App, Image, build, enter, method

# All Modal programs need a [`App`](/docs/reference/modal.App) — an object that acts as a recipe for
# the application. Let's give it a friendly name.

app = App(
    "stable-diffusion-cli"
)  # Note: prior to April 2024, "app" was called "stub"

# ## Model dependencies
#
# Your model will be running remotely inside a container. We will be installing
# all the model dependencies in the next step. We will also be "baking the model"
# into the image by running a Python function as a part of building the image.
# This lets us start containers much faster, since all the data that's needed is
# already inside the image.

model_id = "runwayml/stable-diffusion-v1-5"

image = Image.debian_slim(python_version="3.10").pip_install(
    "accelerate==0.29.2",
    "diffusers==0.15.1",
    "ftfy==6.2.0",
    "safetensors==0.4.2",
    "torch==2.2.2",
    "torchvision",
    "transformers~=4.25.1",
    "triton~=2.2.0",
    "xformers==0.0.25post1",
)

with image.imports():
    import diffusers
    import torch


# ## Using container lifecycle methods
#
# Modal lets you implement code that runs every time a container starts. This
# can be a huge optimization when you're calling a function multiple times,
# since Modal reuses the same containers when possible.
#
# The way to implement this is to turn the Modal function into a method on a
# class that also has lifecycle methods (decorated with `@enter()` and/or `@exit()`).
#
# We have also have applied a few model optimizations to make the model run
# faster. On an A10G, the model takes about 6.5s to load into memory, and then
# 1.6s per generation on average. On a T4, it takes 13s to load and 3.7s per
# generation. Other optimizations are also available [here](https://huggingface.co/docs/diffusers/optimization/fp16#memory-and-speed).

# This is our Modal function. The function runs through the `StableDiffusionPipeline` pipeline.
# It sends the PIL image back to our CLI where we save the resulting image in a local file.


@app.cls(image=image, gpu="A10G")
class StableDiffusion:
    @build()
    @enter()
    def initialize(self):
        scheduler = diffusers.DPMSolverMultistepScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            solver_order=2,
            prediction_type="epsilon",
            thresholding=False,
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            denoise_final=True,  # important if steps are <= 10
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        self.pipe = diffusers.StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        self.pipe.enable_xformers_memory_efficient_attention()

    @method()
    def run_inference(
        self, prompt: str, steps: int = 20, batch_size: int = 4
    ) -> list[bytes]:
        with torch.inference_mode():
            with torch.autocast("cuda"):
                images = self.pipe(
                    [prompt] * batch_size,
                    num_inference_steps=steps,
                    guidance_scale=7.0,
                ).images

        # Convert to PNG bytes
        image_output = []
        for image in images:
            with io.BytesIO() as buf:
                image.save(buf, format="PNG")
                image_output.append(buf.getvalue())
        return image_output


# This is the command we'll use to generate images. It takes a `prompt`,
# `samples` (the number of images you want to generate), `steps` which
# configures the number of inference steps the model will make, and `batch_size`
# which determines how many images to generate for a given prompt.


@app.local_entrypoint()
def entrypoint(
    prompt: str = "A 1600s oil painting of the New York City skyline",
    samples: int = 5,
    steps: int = 10,
    batch_size: int = 1,
):
    print(
        f"prompt => {prompt}, steps => {steps}, samples => {samples}, batch_size => {batch_size}"
    )

    dir = Path("/tmp/stable-diffusion")
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    sd = StableDiffusion()
    for i in range(samples):
        t0 = time.time()
        images = sd.run_inference.remote(prompt, steps, batch_size)
        total_time = time.time() - t0
        print(
            f"Sample {i} took {total_time:.3f}s ({(total_time)/len(images):.3f}s / image)."
        )
        for j, image_bytes in enumerate(images):
            output_path = dir / f"output_{j}_{i}.png"
            print(f"Saving it to {output_path}")
            with open(output_path, "wb") as f:
                f.write(image_bytes)


# And this is our entrypoint; where the CLI is invoked. Explore CLI options
# with: `modal run stable_diffusion_cli.py --help`
#
# ## Performance
#
# This example can generate pictures in about a second, with startup time of about 10s for the first picture.
#
# See distribution of latencies below. This data was gathered by running 500 requests in sequence (meaning only
# the first request incurs a cold start). As you can see, the 90th percentile is 1.2s and the 99th percentile is 2.30s.
#
# ![latencies](./stable_diffusion_latencies.png)


---

## stable diffusion xl

# ---
# output-directory: "/tmp/stable-diffusion-xl"
# args: ["--prompt", "An astronaut riding a green horse"]
# runtimes: ["runc", "gvisor"]
# ---
# # Stable Diffusion XL 1.0
#
# This example is similar to the [Stable Diffusion CLI](/docs/examples/stable_diffusion_cli)
# example, but it generates images from the larger SDXL 1.0 model. Specifically, it runs the
# first set of steps with the base model, followed by the refiner model.
#
# [Try out the live demo here!](https://modal-labs--stable-diffusion-xl-app.modal.run/) The first
# generation may include a cold-start, which takes around 20 seconds. The inference speed depends on the GPU
# and step count (for reference, an A100 runs 40 steps in 8 seconds).

# ## Basic setup

import io
from pathlib import Path

from modal import (
    App,
    Image,
    Mount,
    asgi_app,
    build,
    enter,
    gpu,
    method,
    web_endpoint,
)

# ## Define a container image
#
# To take advantage of Modal's blazing fast cold-start times, we'll need to download our model weights
# inside our container image with a download function. We ignore binaries, ONNX weights and 32-bit weights.
#
# Tip: avoid using global variables in this function to ensure the download step detects model changes and
# triggers a rebuild.


sdxl_image = (
    Image.debian_slim(python_version="3.10")
    .apt_install(
        "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1"
    )
    .pip_install(
        "diffusers==0.26.3",
        "invisible_watermark==0.2.0",
        "transformers~=4.38.2",
        "accelerate==0.27.2",
        "safetensors==0.4.2",
    )
)

app = App(
    "stable-diffusion-xl"
)  # Note: prior to April 2024, "app" was called "stub"

with sdxl_image.imports():
    import torch
    from diffusers import DiffusionPipeline
    from fastapi import Response

# ## Load model and run inference
#
# The container lifecycle [`@enter` decorator](https://modal.com/docs/guide/lifecycle-functions#container-lifecycle-beta)
# loads the model at startup. Then, we evaluate it in the `run_inference` function.
#
# To avoid excessive cold-starts, we set the idle timeout to 240 seconds, meaning once a GPU has loaded the model it will stay
# online for 4 minutes before spinning down. This can be adjusted for cost/experience trade-offs.


@app.cls(gpu=gpu.A10G(), container_idle_timeout=240, image=sdxl_image)
class Model:
    @build()
    def build(self):
        from huggingface_hub import snapshot_download

        ignore = [
            "*.bin",
            "*.onnx_data",
            "*/diffusion_pytorch_model.safetensors",
        ]
        snapshot_download(
            "stabilityai/stable-diffusion-xl-base-1.0", ignore_patterns=ignore
        )
        snapshot_download(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            ignore_patterns=ignore,
        )

    @enter()
    def enter(self):
        load_options = dict(
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            device_map="auto",
        )

        # Load base model
        self.base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", **load_options
        )

        # Load refiner model
        self.refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.base.text_encoder_2,
            vae=self.base.vae,
            **load_options,
        )

        # Compiling the model graph is JIT so this will increase inference time for the first run
        # but speed up subsequent runs. Uncomment to enable.
        # self.base.unet = torch.compile(self.base.unet, mode="reduce-overhead", fullgraph=True)
        # self.refiner.unet = torch.compile(self.refiner.unet, mode="reduce-overhead", fullgraph=True)

    def _inference(self, prompt, n_steps=24, high_noise_frac=0.8):
        negative_prompt = "disfigured, ugly, deformed"
        image = self.base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images
        image = self.refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=image,
        ).images[0]

        byte_stream = io.BytesIO()
        image.save(byte_stream, format="JPEG")

        return byte_stream

    @method()
    def inference(self, prompt, n_steps=24, high_noise_frac=0.8):
        return self._inference(
            prompt, n_steps=n_steps, high_noise_frac=high_noise_frac
        ).getvalue()

    @web_endpoint()
    def web_inference(self, prompt, n_steps=24, high_noise_frac=0.8):
        return Response(
            content=self._inference(
                prompt, n_steps=n_steps, high_noise_frac=high_noise_frac
            ).getvalue(),
            media_type="image/jpeg",
        )


# And this is our entrypoint; where the CLI is invoked. Explore CLI options
# with: `modal run stable_diffusion_xl.py --help


@app.local_entrypoint()
def main(prompt: str = "Unicorns and leprechauns sign a peace treaty"):
    image_bytes = Model().inference.remote(prompt)

    dir = Path("/tmp/stable-diffusion-xl")
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    output_path = dir / "output.png"
    print(f"Saving it to {output_path}")
    with open(output_path, "wb") as f:
        f.write(image_bytes)


# ## A user interface
#
# Here we ship a simple web application that exposes a front-end (written in Alpine.js) for
# our backend deployment.
#
# The Model class will serve multiple users from a its own shared pool of warm GPU containers automatically.
#
# We can deploy this with `modal deploy stable_diffusion_xl.py`.

frontend_path = Path(__file__).parent / "frontend"

web_image = Image.debian_slim().pip_install("jinja2")


@app.function(
    image=web_image,
    mounts=[Mount.from_local_dir(frontend_path, remote_path="/assets")],
    allow_concurrent_inputs=20,
)
@asgi_app()
def ui():
    import fastapi.staticfiles
    from fastapi import FastAPI, Request
    from fastapi.templating import Jinja2Templates

    web_app = FastAPI()
    templates = Jinja2Templates(directory="/assets")

    @web_app.get("/")
    async def read_root(request: Request):
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "inference_url": Model.web_inference.web_url,
                "model_name": "Stable Diffusion XL",
                "default_prompt": "A cinematic shot of a baby raccoon wearing an intricate italian priest robe.",
            },
        )

    web_app.mount(
        "/static",
        fastapi.staticfiles.StaticFiles(directory="/assets"),
        name="static",
    )

    return web_app


---

## stable diffusion xl lightning

from pathlib import Path

import modal

app = modal.App(
    "stable-diffusion-xl-lightning"
)  # Note: prior to April 2024, "app" was called "stub"

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "diffusers==0.26.3", "transformers~=4.37.2", "accelerate==0.27.2"
)

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_unet.safetensors"


with image.imports():
    import io

    import torch
    from diffusers import (
        EulerDiscreteScheduler,
        StableDiffusionXLPipeline,
        UNet2DConditionModel,
    )
    from fastapi import Response
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file


@app.cls(image=image, gpu="a100")
class Model:
    @modal.build()
    @modal.enter()
    def load_weights(self):
        unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(
            "cuda", torch.float16
        )
        unet.load_state_dict(
            load_file(hf_hub_download(repo, ckpt), device="cuda")
        )
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            base, unet=unet, torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")

        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config, timestep_spacing="trailing"
        )

    def _inference(self, prompt, n_steps=4):
        negative_prompt = "disfigured, ugly, deformed"
        image = self.pipe(
            prompt=prompt,
            guidance_scale=0,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
        ).images[0]

        byte_stream = io.BytesIO()
        image.save(byte_stream, format="JPEG")

        return byte_stream

    @modal.method()
    def inference(self, prompt, n_steps=4):
        return self._inference(
            prompt,
            n_steps=n_steps,
        ).getvalue()

    @modal.web_endpoint()
    def web_inference(self, prompt, n_steps=4):
        return Response(
            content=self._inference(
                prompt,
                n_steps=n_steps,
            ).getvalue(),
            media_type="image/jpeg",
        )


# And this is our entrypoint; where the CLI is invoked. Run this example
# with: `modal run stable_diffusion_xl_lightning.py --prompt "An astronaut riding a green horse"`


@app.local_entrypoint()
def main(
    prompt: str = "in the style of Dali, a surrealist painting of a weasel in a tuxedo riding a bicycle in the rain",
):
    image_bytes = Model().inference.remote(prompt)

    dir = Path("/tmp/stable-diffusion-xl-lightning")
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    output_path = dir / "output.png"
    print(f"Saving it to {output_path}")
    with open(output_path, "wb") as f:
        f.write(image_bytes)


# ## A user interface
#
# Here we ship a simple web application that exposes a front-end (written in Alpine.js) for
# our backend deployment.
#
# The Model class will serve multiple users from a its own shared pool of warm GPU containers automatically.
#
# We can deploy this with `modal deploy stable_diffusion_xl_lightning.py`.

frontend_path = Path(__file__).parent / "frontend"

web_image = modal.Image.debian_slim().pip_install("jinja2")


@app.function(
    image=web_image,
    mounts=[modal.Mount.from_local_dir(frontend_path, remote_path="/assets")],
    allow_concurrent_inputs=20,
)
@modal.asgi_app()
def ui():
    import fastapi.staticfiles
    from fastapi import FastAPI, Request
    from fastapi.templating import Jinja2Templates

    web_app = FastAPI()
    templates = Jinja2Templates(directory="/assets")

    @web_app.get("/")
    async def read_root(request: Request):
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "inference_url": Model.web_inference.web_url,
                "model_name": "Stable Diffusion XL Lightning",
                "default_prompt": "A cinematic shot of a baby raccoon wearing an intricate Italian priest robe.",
            },
        )

    web_app.mount(
        "/static",
        fastapi.staticfiles.StaticFiles(directory="/assets"),
        name="static",
    )

    return web_app


---

## stable diffusion xl turbo

# ---
# output-directory: "/tmp/stable-diffusion-xl-turbo"
# args: []
# runtimes: ["runc", "gvisor"]
# ---
# # Stable Diffusion XL Turbo Image-to-image
#
# This example is similar to the [Stable Diffusion XL](/docs/examples/stable_diffusion_xl)
# example, but it's a distilled model trained for real-time synthesis and is image-to-image. Learn more about it [here](https://stability.ai/news/stability-ai-sdxl-turbo).
#
# Input prompt:
# `dog wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k`
#
# Input             |  Output
# :-------------------------:|:-------------------------:
# ![](./stable_diffusion_turbo_input.png)  |  ![](./stable_diffusion_turbo_output.png)

# ## Basic setup

from io import BytesIO
from pathlib import Path

from modal import App, Image, build, enter, gpu, method

# ## Define a container image


image = Image.debian_slim().pip_install(
    "Pillow~=10.1.0",
    "diffusers~=0.24.0",
    "transformers~=4.35.2",  # This is needed for `import torch`
    "accelerate~=0.25.0",  # Allows `device_map="auto"``, which allows computation of optimized device_map
    "safetensors~=0.4.1",  # Enables safetensor format as opposed to using unsafe pickle format
)

app = App(
    "stable-diffusion-xl-turbo", image=image
)  # Note: prior to April 2024, "app" was called "stub"

with image.imports():
    import torch
    from diffusers import AutoPipelineForImage2Image
    from diffusers.utils import load_image
    from huggingface_hub import snapshot_download
    from PIL import Image


# ## Load model and run inference
#
# The container lifecycle [`@enter` decorator](https://modal.com/docs/guide/lifecycle-functions#container-lifecycle-beta)
# loads the model at startup. Then, we evaluate it in the `inference` function.
#
# To avoid excessive cold-starts, we set the idle timeout to 240 seconds, meaning once a GPU has loaded the model it will stay
# online for 4 minutes before spinning down. This can be adjusted for cost/experience trade-offs.


@app.cls(gpu=gpu.A10G(), container_idle_timeout=240)
class Model:
    @build()
    def download_models(self):
        # Ignore files that we don't need to speed up download time.
        ignore = [
            "*.bin",
            "*.onnx_data",
            "*/diffusion_pytorch_model.safetensors",
        ]

        snapshot_download("stabilityai/sdxl-turbo", ignore_patterns=ignore)

    @enter()
    def enter(self):
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            device_map="auto",
        )

    @method()
    def inference(self, image_bytes, prompt):
        init_image = load_image(Image.open(BytesIO(image_bytes))).resize(
            (512, 512)
        )
        num_inference_steps = 4
        strength = 0.9
        # "When using SDXL-Turbo for image-to-image generation, make sure that num_inference_steps * strength is larger or equal to 1"
        # See: https://huggingface.co/stabilityai/sdxl-turbo
        assert num_inference_steps * strength >= 1

        image = self.pipe(
            prompt,
            image=init_image,
            num_inference_steps=num_inference_steps,
            strength=strength,
            guidance_scale=0.0,
        ).images[0]

        byte_stream = BytesIO()
        image.save(byte_stream, format="PNG")
        image_bytes = byte_stream.getvalue()

        return image_bytes


DEFAULT_IMAGE_PATH = Path(__file__).parent / "demo_images/dog.png"


@app.local_entrypoint()
def main(
    image_path=DEFAULT_IMAGE_PATH,
    prompt="dog wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k",
):
    with open(image_path, "rb") as image_file:
        input_image_bytes = image_file.read()
        output_image_bytes = Model().inference.remote(input_image_bytes, prompt)

    dir = Path("/tmp/stable-diffusion-xl-turbo")
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    output_path = dir / "output.png"
    print(f"Saving it to {output_path}")
    with open(output_path, "wb") as f:
        f.write(output_image_bytes)


# ## Running the model
#
# We can run the model with different parameters using the following command,
# ```
# modal run stable_diffusion_xl_turbo.py --prompt="harry potter, glasses, wizard" --image-path="dog.png"
# ```


---

## stable video diffusion

# ---
# cmd: ["modal", "serve", "06_gpu_and_ml/stable_diffusion/stable_video_diffusion.py"]
# ---
import os
import sys

import modal

app = modal.App(
    name="example-stable-video-diffusion-streamlit"
)  # Note: prior to April 2024, "app" was called "stub"
q = modal.Queue.from_name(
    "stable-video-diffusion-streamlit", create_if_missing=True
)

session_timeout = 15 * 60


def download_model():
    # Needed because all paths are relative :/
    os.chdir("/sgm")
    sys.path.append("/sgm")

    from huggingface_hub import snapshot_download
    from omegaconf import OmegaConf
    from scripts.demo.streamlit_helpers import load_model_from_config
    from scripts.demo.video_sampling import VERSION2SPECS

    snapshot_download(
        "stabilityai/stable-video-diffusion-img2vid",
        local_dir="checkpoints/",
        local_dir_use_symlinks=False,
    )

    spec = VERSION2SPECS["svd"]
    config = OmegaConf.load(spec["config"])
    load_model_from_config(config, spec["ckpt"])


svd_image = (
    # The generative-models repo hardcodes `tokenizers==0.12.1`, for which there is no
    # pre-built python 3.11 wheel.
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .run_commands(
        "git clone https://github.com/Stability-AI/generative-models.git /sgm"
    )
    .workdir("/sgm")
    .pip_install(".")
    .pip_install(
        "torch==2.0.1+cu118",
        "torchvision==0.15.2+cu118",
        "torchaudio==2.0.2+cu118",
        extra_index_url="https://download.pytorch.org/whl/cu118",
    )
    .run_commands("pip install -r requirements/pt2.txt")
    .apt_install("ffmpeg", "libsm6", "libxext6")  # for CV2
    .pip_install("safetensors")
    .run_function(download_model, gpu="any")
)


@app.function(image=svd_image, timeout=session_timeout, gpu="A100")
def run_streamlit(publish_url: bool = False):
    from streamlit.web.bootstrap import load_config_options, run

    # TODO: figure out better way to do this with streamlit.
    os.chdir("/sgm")
    sys.path.append("/sgm")

    # Run the server. This function will not return until the server is shut down.
    with modal.forward(8501) as tunnel:
        # Reload Streamlit config with information about Modal tunnel address.
        if publish_url:
            q.put(tunnel.url)
        load_config_options(
            {"browser.serverAddress": tunnel.host, "browser.serverPort": 443}
        )
        run(
            main_script_path="/sgm/scripts/demo/video_sampling.py",
            is_hello=False,
            args=["--timeout", str(session_timeout)],
            flag_options={},
        )


@app.function()
@modal.web_endpoint(method="GET", label="svd")
def share():
    from fastapi.responses import RedirectResponse

    run_streamlit.spawn(publish_url=True)
    url = q.get()
    return RedirectResponse(url, status_code=303)


---

## tensorflow tutorial

# ---
# args: ["--just-run"]
# runtimes: ["runc", "gvisor"]
# ---
# # TensorFlow tutorial
#
# This is essentially a version of the
# [image classification example in the TensorFlow documentation](https://www.tensorflow.org/tutorials/images/classification)
# running inside Modal on a GPU.
# If you run this script, it will also create an TensorBoard URL you can go to to watch the model train and review the results:
#
# ![tensorboard](./tensorboard.png)
#
# ## Setting up the dependencies
#
# Configuring a system to properly run GPU-accelerated TensorFlow can be challenging.
# Luckily, Modal makes it easy to stand on the shoulders of giants and
# [use a pre-built Docker container image](https://modal.com/docs/guide/custom-containers#use-an-existing-container-image-with-from_registry) from a registry like Docker Hub.
# We recommend TensorFlow's [official base Docker container images](https://hub.docker.com/r/tensorflow/tensorflow), which come with `tensorflow` and its matching CUDA libraries already installed.
#
# If you want to install TensorFlow some other way, check out [their docs](https://www.tensorflow.org/install) for options and instructions.
# GPU-enabled containers on Modal will always have NVIDIA drivers available, but you will need to add higher-level tools like CUDA and cuDNN yourself.
# See the [Modal guide on customizing environments](https://modal.com/docs/guide/custom-container) for options we support.

import time

from modal import App, Image, NetworkFileSystem, wsgi_app

dockerhub_image = Image.from_registry(
    "tensorflow/tensorflow:2.12.0-gpu",
).pip_install("protobuf==3.20.*")

app = App(
    "example-tensorflow-tutorial", image=dockerhub_image
)  # Note: prior to April 2024, "app" was called "stub"

# ## Logging data to TensorBoard
#
# Training ML models takes time. Just as we need to monitor long-running systems like databases or web servers for issues,
# we also need to monitor the training process of our ML models. TensorBoard is a tool that comes with TensorFlow that helps you visualize
# the state of your ML model training. It is packaged as a web server.
#
# We want to run the web server for TensorBoard at the same time as we are training the TensorFlow model.
# The easiest way to do this is to set up a shared filesystem between the training and the web server.

fs = NetworkFileSystem.from_name("tensorflow-tutorial", create_if_missing=True)
logdir = "/tensorboard"

# ## Training function
#
# This is basically the same code as [the official example](https://www.tensorflow.org/tutorials/images/classification) from the TensorFlow docs.
# A few Modal-specific things are worth pointing out:
#
# * We set up the shared storage with TensorBoard in the arguments to `app.function`
# * We also annotate this function with `gpu="T4"` to make sure it runs on a GPU
# * We put all the TensorFlow imports inside the function body.
#   This makes it possible to run this example even if you don't have TensorFlow installed on your local computer -- a key benefit of Modal!
#
# You may notice some warnings in the logs about certain CPU performance optimizations (NUMA awareness and AVX/SSE instruction set support) not being available.
# While these optimizations can be important for some workloads, especially if you are running ML models on a CPU, they are not critical for most cases.


@app.function(network_file_systems={logdir: fs}, gpu="T4", timeout=600)
def train():
    import pathlib

    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential

    # load raw data from storage
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file(
        "flower_photos.tar", origin=dataset_url, extract=True
    )
    data_dir = pathlib.Path(data_dir).with_suffix("")

    # construct Keras datasets from raw data
    batch_size = 32
    img_height = img_width = 180

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    class_names = train_ds.class_names
    train_ds = (
        train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)  # type: ignore
    )
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)  # type: ignore
    num_classes = len(class_names)

    model = Sequential(
        [
            layers.Rescaling(1.0 / 255, input_shape=(img_height, img_width, 3)),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.summary()

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logdir,
        histogram_freq=1,
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=[tensorboard_callback],
    )


# ## Running TensorBoard
#
# TensorBoard is compatible with a Python web server standard called [WSGI](https://www.fullstackpython.com/wsgi-servers.html),
# the same standard used by [Flask](https://flask.palletsprojects.com/).
# Modal [speaks WSGI too](https://modal.com/docs/guide/webhooks#wsgi), so it's straightforward to run TensorBoard in a Modal app.
#
# The WSGI app isn't exposed directly through the TensorBoard library, but we can build it
# the same way it's built internally --
# [see the TensorBoard source code for details](https://github.com/tensorflow/tensorboard/blob/0c5523f4b27046e1ca7064dd75347a5ee6cc7f79/tensorboard/program.py#L466-L476).
#
# Note that the TensorBoard server runs in a different container.
# This container shares the same log directory containing the logs from the training.
# The server does not need GPU support.
# Note that this server will be exposed to the public internet!


@app.function(network_file_systems={logdir: fs})
@wsgi_app()
def tensorboard_app():
    import tensorboard

    board = tensorboard.program.TensorBoard()
    board.configure(logdir=logdir)
    (data_provider, deprecated_multiplexer) = board._make_data_provider()
    wsgi_app = tensorboard.backend.application.TensorBoardWSGIApp(
        board.flags,
        board.plugin_loaders,
        data_provider,
        board.assets_zip_provider,
        deprecated_multiplexer,
    )  # Note: prior to April 2024, "app" was called "stub"
    return wsgi_app


# ## Local entrypoint code
#
# Let's kick everything off.
# Everything runs in an ephemeral "app" that gets destroyed once it's done.
# In order to keep the TensorBoard web server running, we sleep in an infinite loop
# until the user hits ctrl-c.
#
# The script will take a few minutes to run, although each epoch is quite fast since it runs on a GPU.
# The first time you run it, it might have to build the image, which can take an additional few minutes.


@app.local_entrypoint()
def main(just_run: bool = False):
    train.remote()
    if not just_run:
        print(
            "Training is done, but the app is still running TensorBoard until you hit ctrl-c."
        )
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Terminating app")


---

## pokemon naming test

from text_to_pokemon import pokemon_naming


def test_prompt_2_name_basic_matching():
    test_candidates = {
        "sleepmon",
        "bulbasaur",
        "bulbasaur",
        "foobar",
    }
    assert (
        pokemon_naming.prompt_2_name(
            prompt="sleepy monkey",
            candidates=test_candidates,
        )
        == "sleepmon"
    )
    assert (
        pokemon_naming.prompt_2_name(
            prompt="monkey asleep",
            candidates=test_candidates,
        )
        == "sleepmon"
    )
    # TODO(erikbern): reenable this. See #151 also.
    # assert (
    #     pokemon_naming.prompt_2_name(
    #         prompt="garlic bulb",
    #         candidates=test_candidates,
    #     )
    #     == "bulbasaur"
    # )
    assert (
        pokemon_naming.prompt_2_name(
            prompt="f",
            candidates=test_candidates,
        )
        == "foobar"
    )


---

## init



---

## api

from fastapi import FastAPI

from .main import create_pokemon_cards

web_app = FastAPI()


@web_app.get("/api/status/{call_id}")
async def poll_status(call_id: str):
    from modal.functions import FunctionCall

    function_call = FunctionCall.from_id(call_id)
    try:
        result = function_call.get(timeout=0.1)
        return dict(
            finished=True,
            cards=result,
        )
    except TimeoutError:
        return dict(finished=False)
    except Exception:
        return dict(error="unknown job processing error")


@web_app.get("/api/create")
async def create_pokemon_job(prompt: str):
    call = create_pokemon_cards.spawn(prompt)
    return {"call_id": call.object_id}


---

## config

import pathlib
import time

from modal import App, Image, Volume

CACHE_DIR = "/cache"
MODEL_CACHE = pathlib.Path("/models")
# Where generated Pokémon images are stored, by hash of prompt.
POKEMON_IMGS = pathlib.Path(CACHE_DIR, "generated_samples")
# Where human-generated and ML-generated new Pokémon names are stored.
POKEMON_NAMES = pathlib.Path(CACHE_DIR, "pokemon_names")
# Where fully compose Pokémon card output images are stored, by hash of prompt.
FINAL_IMGS = pathlib.Path(CACHE_DIR, "final_cards")
# Location of web frontend assets.
ASSETS_PATH = pathlib.Path(__file__).parent / "frontend" / "dist"
# Card composite component images
CARD_PART_IMGS = pathlib.Path(CACHE_DIR, "card_parts")
# Sometimes the NSFW checker is confused by the Pokémon images.
# You can disable it at your own risk.
DISABLE_SAFETY = True


POKEMON_CARDS = [
    {
        "id": "swshp-SWSH039",
        "name": "Pikachu",
        "supertype": "Pokémon",
        "subtypes": ["Basic"],
        "number": "SWSH039",
        "rarity": "Rare Holo Galaxy",
        "images": {
            "small": "https://images.pokemontcg.io/swshp/SWSH039.png",
            "large": "https://images.pokemontcg.io/swshp/SWSH039_hires.png",
        },
        "colors": [[246, 207, 87], [242, 186, 14], [210, 180, 140]],
    },
    {
        "id": "sm35-1",
        "name": "Bulbasaur",
        "supertype": "Pok\u00e9mon",
        "subtypes": ["Basic"],
        "number": "1",
        "rarity": "Common",
        "images": {
            "small": "https://images.pokemontcg.io/sm35/1.png",
            "large": "https://images.pokemontcg.io/sm35/1_hires.png",
        },
        "colors": [[221, 221, 64], [164, 199, 63], [131, 184, 156]],
    },
    {
        "id": "sm10-33",
        "name": "Squirtle",
        "supertype": "Pok\u00e9mon",
        "subtypes": ["Basic"],
        "number": "33",
        "rarity": "Common",
        "images": {
            "small": "https://images.pokemontcg.io/sm10/33.png",
            "large": "https://images.pokemontcg.io/sm10/33_hires.png",
        },
        "colors": [[87, 186, 227], [253, 224, 100], [191, 225, 240]],
    },
    {
        "id": "sm115-7",
        "name": "Charmander",
        "supertype": "Pok\u00e9mon",
        "subtypes": ["Basic"],
        "number": "7",
        "rarity": "Common",
        "images": {
            "small": "https://images.pokemontcg.io/sm115/7.png",
            "large": "https://images.pokemontcg.io/sm115/7_hires.png",
        },
        "colors": [[235, 131, 68], [235, 88, 52], [252, 222, 98]],
    },
    {
        "id": "swsh45-35",
        "name": "Morpeko",
        "supertype": "Pok\u00e9mon",
        "subtypes": ["Basic"],
        "number": "35",
        "rarity": "Common",
        "images": {
            "small": "https://images.pokemontcg.io/swsh45/35.png",
            "large": "https://images.pokemontcg.io/swsh45/35_hires.png",
        },
        "colors": [[252, 220, 55], [202, 167, 99], [238, 236, 194]],
    },
    {
        "id": "swsh9-120",
        "name": "Bidoof",
        "supertype": "Pok\u00e9mon",
        "subtypes": ["Basic"],
        "number": "120",
        "rarity": "Common",
        "images": {
            "small": "https://images.pokemontcg.io/swsh9/120.png",
            "large": "https://images.pokemontcg.io/swsh9/120_hires.png",
        },
        "colors": [[236, 231, 223], [249, 224, 101], [190, 154, 113]],
    },
    {
        "id": "sm8-142",
        "name": "Dedenne",
        "supertype": "Pok\u00e9mon",
        "subtypes": ["Basic"],
        "number": "142",
        "rarity": "Uncommon",
        "images": {
            "small": "https://images.pokemontcg.io/sm8/142.png",
            "large": "https://images.pokemontcg.io/sm8/142_hires.png",
        },
        "colors": [[216, 77, 140], [226, 118, 169], [252, 223, 100]],
    },
    {
        "id": "pgo-24",
        "name": "Articuno",
        "supertype": "Pok\u00e9mon",
        "subtypes": ["Basic"],
        "number": "24",
        "rarity": "Rare Holo",
        "images": {
            "small": "https://images.pokemontcg.io/pgo/24.png",
            "large": "https://images.pokemontcg.io/pgo/24_hires.png",
        },
        "colors": [[90, 184, 225], [254, 225, 99], [186, 220, 237]],
    },
    {
        "id": "pgo-29",
        "name": "Zapdos",
        "supertype": "Pok\u00e9mon",
        "subtypes": ["Basic"],
        "number": "29",
        "rarity": "Rare Holo",
        "images": {
            "small": "https://images.pokemontcg.io/pgo/29.png",
            "large": "https://images.pokemontcg.io/pgo/29_hires.png",
        },
        "colors": [[253, 220, 56], [121, 173, 202], [224, 175, 69]],
    },
    {
        "id": "pgo-12",
        "name": "Moltres",
        "supertype": "Pok\u00e9mon",
        "subtypes": ["Basic"],
        "number": "12",
        "rarity": "Rare Holo",
        "images": {
            "small": "https://images.pokemontcg.io/pgo/12.png",
            "large": "https://images.pokemontcg.io/pgo/12_hires.png",
        },
        "colors": [[238, 131, 72], [236, 89, 59], [253, 222, 98]],
    },
]


def load_stable_diffusion_pokemon_model():
    import torch
    from diffusers import StableDiffusionPipeline

    model_id = "lambdalabs/sd-pokemon-diffusers"
    cache_dir = MODEL_CACHE / model_id
    if cache_dir.exists():
        print(f"Using diskcached model for '{model_id}'")
        local_files_only = True
        load_action = "loading"
    else:
        print(f"No diskcached model found for '{model_id}'")
        cache_dir.mkdir(parents=True, exist_ok=True)
        local_files_only = False
        load_action = "downloading"
    load_start_time = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    print(
        f"finished {load_action} model, took {time.time() - load_start_time:.3f}s."
    )

    if DISABLE_SAFETY:

        def null_safety(images, **kwargs):
            return images, False

        pipe.safety_checker = null_safety
    return pipe


volume = Volume.from_name("txt-to-pokemon-cache-vol", create_if_missing=True)
image = (
    Image.debian_slim()
    .pip_install(
        "accelerate",
        "colorgram.py",
        "diffusers~=0.11.1",
        "ftfy",
        "torch",
        "transformers",
        "scipy",
    )
    .run_function(load_stable_diffusion_pokemon_model)
)
app = App(
    name="example-text-to-pokemon", image=image
)  # Note: prior to April 2024, "app" was called "stub"


---

## inpaint

"""
Inpainting removes unwanted parts of an image. The module has
inpainting functionality to remove the Pokémon name that appears on the 'base' card,
eg. Articuno, so that it can be replaced with a new, made up name for the model generated
Pokémon character.

This code is partly based on code from github.com/Sanster/lama-cleaner/.
"""

import io

import modal

cv_image = (
    modal.Image.debian_slim()
    .pip_install(
        "opencv-python~=4.6.0.66",
        "Pillow~=9.3.0",
        "numpy~=1.23.5",
    )
    .run_commands(
        "apt-get update",
        # Required to install libs such as libGL.so.1
        "apt-get install ffmpeg libsm6 libxext6 --yes",
    )
)

# Configs for opencv inpainting
# opencv document https://docs.opencv.org/4.6.0/d7/d8b/group__photo__inpaint.html#gga8002a65f5a3328fbf15df81b842d3c3ca05e763003a805e6c11c673a9f4ba7d07
cv2_flag: str = "INPAINT_NS"
cv2_radius: int = 4


# From lama-cleaner
def load_img(img_bytes, gray: bool = False):
    import cv2
    import numpy as np

    alpha_channel = None
    nparr = np.frombuffer(img_bytes, np.uint8)
    if gray:
        np_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    else:
        np_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if len(np_img.shape) == 3 and np_img.shape[2] == 4:
            alpha_channel = np_img[:, :, -1]
            np_img = cv2.cvtColor(np_img, cv2.COLOR_BGRA2RGB)
        else:
            np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)

    return np_img, alpha_channel


def numpy_to_bytes(image_numpy, ext: str) -> bytes:
    import cv2

    data = cv2.imencode(
        f".{ext}",
        image_numpy,
        [
            int(cv2.IMWRITE_JPEG_QUALITY),
            100,
            int(cv2.IMWRITE_PNG_COMPRESSION),
            0,
        ],
    )[1]
    image_bytes = data.tobytes()
    return image_bytes


def new_pokemon_name(
    card_image: bytes, pokemon_name: str = "Randomon"
) -> bytes:
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    # 1. Paint out the existing name.

    flag_map = {
        "INPAINT_NS": cv2.INPAINT_NS,
        "INPAINT_TELEA": cv2.INPAINT_TELEA,
    }
    img, alpha_channel = load_img(card_image)

    pokecard_name_top_left_crnr = (139, 43)
    pokecard_name_size = (225, 45)  # (width, height)

    mask_im = Image.new("L", img.shape[:2][::-1], 0)
    draw = ImageDraw.Draw(mask_im)
    (left, upper, right, lower) = (
        pokecard_name_top_left_crnr[0],
        pokecard_name_top_left_crnr[1],
        pokecard_name_top_left_crnr[0] + pokecard_name_size[0],
        pokecard_name_top_left_crnr[1] + pokecard_name_size[1],
    )
    draw.rectangle((left, upper, right, lower), fill=255)
    mask_im = mask_im.convert("L")
    with io.BytesIO() as buf:
        mask_im.save(buf, format="PNG")
        mask_img_bytes = buf.getvalue()
        mask, _ = load_img(mask_img_bytes)

    assert (
        img.shape[:2] == mask.shape[:2]
    ), "shapes of base image and mask must match"

    # "No GPU is required, and for simple backgrounds, the results may even be better than AI models."
    cur_res = cv2.inpaint(
        img[:, :, ::-1],
        mask[:, :, 0],  # Slicing ensures we get 1 channel not 3.
        inpaintRadius=cv2_radius,
        flags=flag_map[cv2_flag],
    )

    # 2. Insert the new name!

    # make a blank image for the text, initialized to transparent text color
    txt = Image.new("RGBA", cur_res.shape[:2][::-1], (255, 255, 255, 0))
    # Dejavu is only font installed on Debian-slim images.
    # TODO: Get the real Pokémon card font. (This Dejavu is pretty close though)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    fnt = ImageFont.truetype(font_path, size=40)
    fnt.fontmode = "L"  # antialiasing
    # get a drawing context
    d = ImageDraw.Draw(txt)

    # draw text, full opacity
    # -3 is done to put text at right line height position
    text_position = (
        pokecard_name_top_left_crnr[0],
        pokecard_name_top_left_crnr[1] - 5,
    )
    # Note that the text is a *little* transparent. This looks closer to the original
    # text. Full opacity is too flat.
    d.text(text_position, pokemon_name, font=fnt, fill=(20, 20, 20, 230))

    # https://stackoverflow.com/a/45646235/4885590
    cur_res_correct_color = cv2.cvtColor(cur_res, cv2.COLOR_BGR2RGB)
    cur_res_image = Image.fromarray(cur_res_correct_color).convert("RGBA")
    out = Image.alpha_composite(cur_res_image, txt)

    with io.BytesIO() as buf:
        out.save(buf, format="PNG")
        return buf.getvalue()


---

## main

import base64
import dataclasses
import hashlib
import io
import pathlib
import random
import re
import time
import urllib.request
from datetime import timedelta

from modal import Mount, asgi_app, enter, method

from . import config, inpaint, ops, pokemon_naming
from .config import app, volume


@dataclasses.dataclass(frozen=True)
class PokemonCardResponseItem:
    name: str
    bar: int
    b64_encoded_image: str
    mime: str = "image/png"
    rarity: str = "Common"


def _choose_rarity() -> str:
    val = random.random()
    if val < 0.65:
        return "Common"
    elif val < 0.80:
        return "Uncommon"
    elif val < 0.95:
        return "Rare Holo"
    return random.choice(
        ["Rare Holo Galaxy", "Rare Holo V", "Rare Ultra", "Rare Rainbow Alt"]
    )


def log_prompt(prompt: str) -> str:
    max_len = 100
    return f"{prompt[:max_len]}…" if len(prompt) > max_len else prompt


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    from PIL import Image

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def image_to_byte_array(image) -> bytes:
    with io.BytesIO() as buf:
        image.save(buf, format="PNG")
        return buf.getvalue()


@app.cls(gpu="A10G", volumes={config.CACHE_DIR: volume}, keep_warm=1)
class Model:
    @enter()
    def load_model(self):
        import threading

        if not pokemon_naming.rnn_names_output_path.exists():
            threading.Thread(target=ops.generate_pokemon_names.remote).start()
        self.pipe = config.load_stable_diffusion_pokemon_model().to("cuda")

    @method()
    def text_to_pokemon(self, prompt: str) -> list[bytes]:
        from torch import autocast

        n_samples = 4
        print(
            f"Generating {n_samples} Pokémon samples for the prompt: '{log_prompt(prompt)}'"
        )
        with autocast("cuda"):
            images = self.pipe(n_samples * [prompt], guidance_scale=10).images
        return [image_to_byte_array(image=img) for img in images]


def normalize_prompt(p: str) -> str:
    return re.sub("[^a-z0-9- ]", "", p.lower())


@app.function(volumes={config.CACHE_DIR: volume})
def diskcached_text_to_pokemon(prompt: str) -> list[bytes]:
    start_time = time.monotonic()
    cached = False

    norm_prompt = normalize_prompt(prompt)
    norm_prompt_digest = hashlib.sha256(norm_prompt.encode()).hexdigest()

    config.POKEMON_IMGS.mkdir(parents=True, exist_ok=True)

    prompt_samples_dir = config.POKEMON_IMGS / norm_prompt_digest
    if prompt_samples_dir.exists():
        print("Cached! — using prompt samples prepared earlier.")
        cached = True
        samples_data = []
        for sample_file in prompt_samples_dir.iterdir():
            with open(sample_file, "rb") as f:
                samples_data.append(f.read())
    else:
        # 1. Create images (expensive)
        model = Model()
        samples_data = model.text_to_pokemon.remote(prompt=norm_prompt)
        # 2. Save them (for later run to be cached)
        # Allow prior existence of dir because multiple concurrent requests for same prompt
        # can race each other.
        prompt_samples_dir.mkdir(exist_ok=True)
        for i, image_bytes in enumerate(samples_data):
            dest_path = prompt_samples_dir / f"{i}.png"
            with open(dest_path, "wb") as f:
                f.write(image_bytes)
            print(f"✔️ Saved a Pokémon sample to {dest_path}.")
        volume.commit()
    total_duration_secs = timedelta(
        seconds=time.monotonic() - start_time
    ).total_seconds()
    print(
        f"[{cached=}] took {total_duration_secs} secs to create {len(samples_data)} samples for '{log_prompt(norm_prompt)}'."
    )
    return samples_data


@app.function(
    mounts=[
        Mount.from_local_dir(
            local_path=config.ASSETS_PATH, remote_path="/assets"
        )
    ],
)
@asgi_app()
def fastapi_app():
    import fastapi.staticfiles

    from .api import web_app

    web_app.mount(
        "/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True)
    )

    return web_app


@app.function(
    image=inpaint.cv_image,
    volumes={config.CACHE_DIR: volume},
    interactive=False,
)
def inpaint_new_pokemon_name(card_image: bytes, prompt: str) -> bytes:
    """
    Pick a name for the generated Pokémon character based on the prompt,
    and replace the base card's Pokémon name with it.

    Without this, created cards look a bit weird, as the generated Pokémon
    will have names like 'Articuno', 'Bidoof', and 'Pikachu'.
    """
    candidates = pokemon_naming.load_names(
        include_model_generated=True,
        include_human_generated=True,
    )
    best_name = pokemon_naming.prompt_2_name(prompt, candidates)
    return inpaint.new_pokemon_name(card_image, best_name.capitalize())


def composite_pokemon_card(
    base: io.BytesIO, character_img: io.BytesIO, prompt: str
) -> bytes:
    """Constructs a new, unique Pokémon card image from existing and model-generated components."""
    from PIL import Image, ImageDraw, ImageFilter

    pokecard_window_top_right_crnr = (61, 99)
    pokecard_window_size = (618, 383)  # (width, height)

    base_i = Image.open(base)
    character_i = Image.open(character_img)

    # Fit Pokémon character image to size of base card's character illustration window.
    character_i.thumbnail(
        size=(pokecard_window_size[0], pokecard_window_size[0])
    )
    (left, upper, right, lower) = (
        0,
        0,
        pokecard_window_size[0],
        pokecard_window_size[1],
    )
    cropped_character_i = character_i.crop((left, upper, right, lower))

    # Soften edges of paste
    mask_im = Image.new("L", cropped_character_i.size, 0)
    draw = ImageDraw.Draw(mask_im)
    edge_depth = 3
    draw.rectangle(
        (
            left + edge_depth,
            upper + edge_depth,
            right - edge_depth,
            lower - edge_depth,
        ),
        fill=255,
    )
    mask_im_blur = mask_im.filter(ImageFilter.GaussianBlur(20))

    back_im = base_i.copy()
    back_im.paste(
        cropped_character_i, pokecard_window_top_right_crnr, mask_im_blur
    )

    # If a (manually uploaded) mini Modal logo exists, paste that discreetly onto the image too :)
    mini_modal_logo = config.CARD_PART_IMGS / "mini-modal-logo.png"
    if mini_modal_logo.exists():
        logo_img = Image.open(str(mini_modal_logo))
        mini_logo_top_right_crnr = (220, 935)
        back_im.paste(logo_img, mini_logo_top_right_crnr)
    else:
        print(
            f"WARN: Mini-Modal logo not found at {mini_modal_logo}, so not compositing that image part."
        )

    print("Replacing Pokémon card name")
    return inpaint_new_pokemon_name.remote(
        card_image=image_to_byte_array(back_im), prompt=prompt
    )


def color_dist(
    one: tuple[float, float, float], two: tuple[float, float, float]
) -> float:
    """
    A decent but not great RGB color distance function. Range of distance result is [0.0, 3.0].
    """
    import numpy as np

    fst = np.array([[x / 255.0 for x in one]])
    snd = np.array([[x / 255.0 for x in two]])
    rm = 0.5 * (fst[:, 0] + snd[:, 0])
    drgb = (fst - snd) ** 2
    t = np.array([2 + rm, 4 + 0 * rm, 3 - rm]).T
    delta_e = np.sqrt(np.sum(t * drgb, 1))
    return delta_e


@app.function(volumes={config.CACHE_DIR: volume})
def create_composite_card(i: int, sample: bytes, prompt: str) -> bytes:
    """
    Takes a single Pokémon sample and creates a Pokémon card image for it.
    .starmap over this function to boost performance.
    """
    print(f"Determining base card for generated sample {i}.")
    closest_card = closest_pokecard_by_color(
        sample=sample, cards=config.POKEMON_CARDS
    )
    base_card_url = closest_card["images"]["large"]
    print(f"Closest base card for sample {i} is '{closest_card['name']}'")
    req = urllib.request.Request(
        base_card_url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36"
        },
    )
    base_bytes = urllib.request.urlopen(req).read()
    print(f"Compositing generated sample {i} onto a Pokémon card.")
    return composite_pokemon_card(
        base=io.BytesIO(base_bytes),
        character_img=io.BytesIO(sample),
        prompt=prompt,
    )


@app.function(volumes={config.CACHE_DIR: volume})
def create_pokemon_cards(prompt: str) -> list[dict]:
    norm_prompt = normalize_prompt(prompt)
    print(f"Creating for prompt '{norm_prompt}'")
    norm_prompt_digest = hashlib.sha256(norm_prompt.encode()).hexdigest()
    config.FINAL_IMGS.mkdir(parents=True, exist_ok=True)
    final_cards_dir = config.FINAL_IMGS / norm_prompt_digest

    if final_cards_dir.exists():
        print(
            "Cached! - prompt has had cards composed before, returning previous Pokémon card results."
        )
        cards_data = [
            card_file.read_bytes() for card_file in final_cards_dir.iterdir()
        ]
    else:
        print("No existing final card outputs for prompts. Proceeding...")
        # Produce the Pokémon character samples with the StableDiffusion model.
        samples_data = diskcached_text_to_pokemon.remote(prompt)
        print(f"Compositing {len(samples_data)} samples onto cards...")
        cards_data = list(
            create_composite_card.starmap(
                (i, sample, norm_prompt)
                for (i, sample) in enumerate(samples_data)
            )
        )
        print(
            f"Persisting {len(cards_data)} results for later disk-cache retrieval."
        )
        final_cards_dir.mkdir()
        for i, c_data in enumerate(cards_data):
            c_path = final_cards_dir / f"{i}.png"
            c_path.write_bytes(c_data)

    # Return Pokémon cards to client as base64-encoded images with metadata.
    cards = []
    for i, image_bytes in enumerate(cards_data):
        encoded_image_string = base64.b64encode(image_bytes).decode("ascii")
        cards.append(
            PokemonCardResponseItem(
                name=str(i),
                bar=i,
                b64_encoded_image=encoded_image_string,
                rarity=_choose_rarity(),
            )
        )

    print(f"✔️ Returning {len(cards)} Pokémon samples.")
    return [dataclasses.asdict(card) for card in cards]


def closest_pokecard_by_color(sample: bytes, cards):
    """
    Takes the list of POKEMON_CARDS and returns the item that's closest
    in color to the provided model-generate sample image.
    """
    import colorgram

    sample_colors = colorgram.extract(io.BytesIO(sample), 3)  # Top 3 colors
    sample_rgb_colors = [color.rgb for color in sample_colors]

    min_distance = None
    closest_card = None
    for card in cards:
        dominant_color = card["colors"][0]
        d = color_dist(
            one=dominant_color,
            two=sample_rgb_colors[0],
        )
        if min_distance is None or d < min_distance:
            closest_card = card
            min_distance = d
    return closest_card


@app.local_entrypoint()
def run_local(prompt: str):
    images_data = diskcached_text_to_pokemon.remote(prompt)

    now = int(time.time())
    for i, image_bytes in enumerate(images_data):
        dest_path = pathlib.Path(".", f"{now}_{i}.png")
        with open(dest_path, "wb") as f:
            f.write(image_bytes)
        print(f"✔️ Saved a Pokémon sample to {dest_path}.")


---

## ops

"""
Operational tools and scripts. These are run manually by an engineer to facilitate
the development and maintenance of the application.

eg. python -m text_to_pokemon.ops reset-diskcache
"""

import argparse
import io
import json
import sys
import urllib.request

from . import config
from .config import app, volume
from .pokemon_naming import (
    fetch_pokemon_names,
    generate_names,
    rnn_image,
    rnn_names_output_path,
    train_rnn,
)


@app.function(volumes={config.CACHE_DIR: volume})
def reset_diskcache(dry_run=True) -> None:
    """
    Delete all Pokémon character samples and cards from disk cache.
    Useful when a changes are made to character or card generation process
    and you want create cache misses so the changes so up for users.
    """
    if config.POKEMON_IMGS.exists():
        files = [f for f in config.POKEMON_IMGS.glob("**/*") if f.is_file()]
        i = 0
        for i, filepath in enumerate(files):
            if not dry_run:
                filepath.unlink()
        if files and dry_run:
            print(
                f"🏜 dry-run: would have deleted {i+1} Pokémon character samples"
            )
        elif files:
            print(f"deleted {i+1} Pokémon character samples")
        else:
            print("No Pokémon character samples to delete")

        if not dry_run:
            dirs = [f for f in config.POKEMON_IMGS.glob("**/*") if f.is_dir()]
            for d in dirs:
                d.rmdir()

    if config.FINAL_IMGS.exists():
        files = [f for f in config.FINAL_IMGS.glob("**/*") if f.is_file()]
        i = 0
        for i, filepath in enumerate(files):
            if not dry_run:
                filepath.unlink()

        if files and dry_run:
            print(f"🏜 dry-run: would have deleted {i+1} Pokémon card images")
        elif files:
            print(f"deleted {i+1} Pokémon card images")
        else:
            print("No Pokémon character card images to delete")

        if not dry_run:
            dirs = [f for f in config.FINAL_IMGS.glob("**/*") if f.is_dir()]
            for d in dirs:
                d.rmdir()

    volume.commit()


@app.function()
def extract_colors(num=3) -> None:
    """
    Extracts the colors for all Pokémon cards contained in `config` module
    and updates the card config with color data.

    Copy-paste this function's output back into the `config` module.
    """
    import colorgram

    for card in config.POKEMON_CARDS:
        print(f"Processing {card['name']}")
        req = urllib.request.Request(
            card["images"]["large"],  # type: ignore
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36"
            },
        )
        image_bytes = urllib.request.urlopen(req).read()
        colors = colorgram.extract(io.BytesIO(image_bytes), num)
        card["colors"] = [list(color.rgb) for color in colors]

    print(json.dumps(config.POKEMON_CARDS, indent=4))


@app.function(
    image=rnn_image,
    volumes={config.CACHE_DIR: volume},
    timeout=15 * 60,
)
def generate_pokemon_names():
    """
    Use a text-generation ML model to create new Pokémon names
    and persist them in a volume for later use in the card creation
    process.
    """
    desired_generations = 100
    poke_names = fetch_pokemon_names()
    # Hyphenated Pokémon names, eg. Hakamo-o, don't play mix with RNN model.
    training_names = [n for n in poke_names if "-" not in n]
    max_sequence_len = max([len(name) for name in training_names])
    model = train_rnn(
        training_names=training_names,
        max_sequence_len=max_sequence_len,
    )

    model_path = config.MODEL_CACHE / "poke_gen_model.h5"
    print(f"Storing model at '{model_path}'")
    model.save(model_path)

    print(f"Generating {desired_generations} new names.")
    new_names = generate_names(
        model=model,
        training_names=set(training_names),
        num=desired_generations,
        max_sequence_len=max_sequence_len,
    )

    print(
        f"Storing {desired_generations} generated names. eg. '{new_names[0]}'"
    )
    output_path = rnn_names_output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(new_names))


def main() -> int:
    parser = argparse.ArgumentParser(prog="text-to-pokemon-ops")
    sub_parsers = parser.add_subparsers(dest="subcommand")
    sub_parsers.add_parser(
        "extract-colors", help="Extract colors for all Pokémon base cards."
    )
    sub_parsers.add_parser(
        "gen-pokemon-names", help="Generate new Pokémon names."
    )
    parser_reset_diskcache = sub_parsers.add_parser(
        "reset-diskcache",
        help="Delete all cached Pokémon card parts from volume.",
    )
    parser_reset_diskcache.add_argument(
        "--nodry-run",
        action="store_true",
        default=False,
        help="Actually delete files from volume.",
    )

    args = parser.parse_args()
    if args.subcommand == "gen-pokemon-names":
        with app.run():
            generate_pokemon_names.remote()
    elif args.subcommand == "extract-colors":
        with app.run():
            extract_colors.remote()
    elif args.subcommand == "reset-diskcache":
        with app.run():
            reset_diskcache.remote(dry_run=not args.nodry_run)
    elif args.subcommand is None:
        parser.print_help(sys.stderr)
    else:
        raise AssertionError(
            f"Unimplemented subcommand '{args.subcommand}' was invoked."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


---

## pokemon naming

"""
Our AI-generated Pokémon characters need their own names!
"""

import dataclasses
import json
import time
import urllib.request
from typing import Any

import modal

from . import config

rnn_image = modal.Image.debian_slim().pip_install(
    "keras",
    "pandas",
    "numpy",
    "tensorflow",
)

# Longer names don't fit on Pokémon card
MAX_NAME_LEN = 14
# Discard names too short to make sense
MIN_NAME_LEN = 4

rnn_names_output_path = config.POKEMON_NAMES / "rnn.txt"


@dataclasses.dataclass
class TrainingDataset:
    X: Any  # numpy arr
    Y: Any  # numpy arr
    num_unique_chars: int


def load_names(
    include_model_generated: bool,
    include_human_generated: bool,
) -> set[str]:
    names = set()
    if include_model_generated:
        if rnn_names_output_path.exists():
            model_names = set(rnn_names_output_path.read_text().split("\n"))
            names.update(model_names)
        else:
            print(
                f"Model generated names at `{rnn_names_output_path}` are not ready, skipping"
            )
    if include_human_generated:
        names.update(FANDOM_NAMES)
        names.update(PREFILL_PROMPT_NAMES)
    return names


def prompt_2_name(prompt: str, candidates: set[str]) -> str:
    if not prompt:
        raise ValueError("`prompt` argument cannot be empty")
    return max(
        (cand for cand in candidates),
        key=lambda c: len(lcs(prompt, c)),
    )


def lcs(one: str, two: str) -> str:
    matrix = [["" for x in range(len(two))] for x in range(len(one))]
    for i in range(len(one)):
        for j in range(len(two)):
            if one[i] == two[j]:
                if i == 0 or j == 0:
                    matrix[i][j] = one[i]
                else:
                    matrix[i][j] = matrix[i - 1][j - 1] + one[i]
            else:
                matrix[i][j] = max(matrix[i - 1][j], matrix[i][j - 1], key=len)

    longest = matrix[-1][-1]
    return longest


def generate_names(
    model,
    training_names: set[str],
    num: int,
    max_sequence_len: int,
):
    """Accepts training dataset and trained model, and generates `num` new Pokémon names."""
    import numpy as np

    concat_names = "\n".join(training_names).lower()
    # Start sequence generation from end of the input sequence
    sequence = concat_names[-(max_sequence_len - 1) :] + "\n"

    new_names: set[str] = set()
    chars = sorted(list(set(concat_names)))
    num_chars = len(chars)

    # Build translation dictionaries
    char2idx = {c: i for i, c in enumerate(chars)}  # a -> 0
    idx2char = {i: c for i, c in enumerate(chars)}  # 0 -> a

    while len(new_names) < num:
        # Vectorize sequence for prediction
        x = np.zeros((1, max_sequence_len, num_chars))
        for i, char in enumerate(sequence):
            x[0, i, char2idx[char]] = 1

        # Sample next char from predicted probabilities
        probs = model.predict(x, verbose=0)[0]
        probs /= probs.sum()
        next_idx = np.random.choice(len(probs), p=probs)
        next_char = idx2char[next_idx]
        sequence = sequence[1:] + next_char

        # Newline means we have a new name
        if next_char == "\n":
            gen_name = [name for name in sequence.split("\n")][1]

            # Never start name with two identical chars
            if len(gen_name) > 2 and gen_name[0] == gen_name[1]:
                gen_name = gen_name[1:]

            if len(gen_name) > MAX_NAME_LEN:
                continue
            elif len(gen_name) >= MIN_NAME_LEN:
                # Only allow new and unique names
                if gen_name not in training_names and gen_name not in new_names:
                    new_names.add(gen_name)

            if len(new_names) % 10 == 0:
                print("generated {} new names".format(len(new_names)))
    return list(new_names)


def prep_dataset(
    training_names: list[str], max_sequence_len: int
) -> TrainingDataset:
    import numpy as np

    step_length = (
        1  # The step length we take to get our samples from our corpus
    )
    # Make it all to a long string
    concat_names = "\n".join(training_names).lower()

    chars = sorted(list(set(concat_names)))
    num_chars = len(chars)

    # Build translation dictionary, 'a' -> 0
    char2idx = dict((c, i) for i, c in enumerate(chars))

    # Use longest name length as our sequence window
    max_sequence_len = max([len(name) for name in training_names])

    print(f"Total chars: {num_chars}")
    print("Corpus length:", len(concat_names))
    print("Number of names: ", len(training_names))
    print("Longest name: ", max_sequence_len)

    sequences = []
    next_chars = []

    # Loop over our data and extract pairs of sequances and next chars
    for i in range(0, len(concat_names) - max_sequence_len, step_length):
        sequences.append(concat_names[i : i + max_sequence_len])
        next_chars.append(concat_names[i + max_sequence_len])

    num_sequences = len(sequences)

    print("Number of sequences:", num_sequences)
    print("First 10 sequences and next chars:")
    for i in range(10):
        print(
            "X=[{}]   y=[{}]".replace("\n", " ")
            .format(sequences[i], next_chars[i])
            .replace("\n", " ")
        )

    X = np.zeros((num_sequences, max_sequence_len, num_chars), dtype=np.bool)
    Y = np.zeros((num_sequences, num_chars), dtype=np.bool)

    for i, sequence in enumerate(sequences):
        for j, char in enumerate(sequence):
            X[i, j, char2idx[char]] = 1
        Y[i, char2idx[next_chars[i]]] = 1

    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    return TrainingDataset(
        X=X,
        Y=Y,
        num_unique_chars=num_chars,
    )


def train_rnn(
    training_names: list[str],
    max_sequence_len: int,
):
    from keras.layers import LSTM, Dense
    from keras.models import Sequential
    from keras.optimizers import RMSprop

    epochs = 100  # Number of times we train on our full data
    batch_size = 32  # Data samples in each training step
    latent_dim = 64  # Size of our LSTM
    dropout_rate = 0.2  # Regularization with dropout
    verbosity = 1  # Print result for each epoch

    dataset = prep_dataset(training_names, max_sequence_len)

    input_shape = (
        max_sequence_len,
        dataset.num_unique_chars,
    )
    model = Sequential()
    model.add(
        LSTM(
            latent_dim, input_shape=input_shape, recurrent_dropout=dropout_rate
        )
    )
    model.add(Dense(units=dataset.num_unique_chars, activation="softmax"))

    optimizer = RMSprop(learning_rate=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer)

    model.summary()

    start = time.time()
    print("Training for {} epochs".format(epochs))
    model.fit(
        dataset.X,
        dataset.Y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbosity,
    )
    print(f"Finished training - time elapsed: {(time.time() - start)} seconds")
    return model


def fetch_pokemon_names() -> list[str]:
    """
    Source training data by getting all Pokémon names from the pokeapi.co API.
    There are 1008 Pokémon as of early December 2022.
    """
    get_all_url = "https://pokeapi.co/api/v2/pokemon?limit=1500"  # Set limit > than total number of Pokémon.
    req = urllib.request.Request(
        get_all_url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/35.0.1916.47 Safari/537.36"
            )
        },
    )
    response = urllib.request.urlopen(req)
    data = json.load(response)

    pokemon_names = [item["name"] for item in data["results"]]
    print(f"Fetched {len(pokemon_names)} Pokémon names")
    return pokemon_names


# Hand-writing good Pokémon names for the prefill prompts defined in the frontend.
PREFILL_PROMPT_NAMES: set[str] = {
    "abrahamad",  # Abraham Linclon
    "jordasaur",  # Air Jordans
    "rattlebub",  # A Happy Baby With A Rattle
    "bananapeel",  # Banana in Pajamas
    "cheeseclaw",  # Crab Made of Cheese
    "Trumpistan",  # Donald Trump
    "duckhoof",  # Duck sized horse
    "elephhix",  # Elephant With Six Legs
    "frodomon",  # Frodo Baggins
    "goldsealy",  # Golden Seal
    "homerimpson",  # Homer Simpson
    "hoofduck",  # Horse sized duck
    "iphoneuous",  # IPhone 7 Device
    "jokerclown",  # Joker Evil
    "kingkongmon",  # King Kong
    "popandafu",  # Kung Fu Panda
    "limamonk",  # Lima Monkey
    "marvin",  # Marvin The Paranoid Robot
    "nocturas",  # Nocturnal Animal
    "buddhismo",  # Old Buddhist Monk in Orange Robes
    "pdp-11",  # PDP-11 Computer
    "coupleous",  # Power Couple
    "questsight",  # Question Mark With Eyes
    "roomba",  # Roomba
    "ragesound",  # Rage Against The Machine
    "metalflight",  # Snake With Metal Wings
    "armorgator",  # Suit of Armor Alligator
    "stevejobs",  # Steve Jobs
    "devilmon",  # The Devil
    "fearmon",  # The Fear
    "uranus",  # Uranus The Planet
    "vladmarx",  # Vladimir Lenin
    "willycat",  # Willy Wonka Cat
    "xenomorphmon",  # Xenomorph Alien
    "yoyoma",  # Yoyo Toy
    "zoroblade",  # Zoro The Masked Bandit
}

FANDOM_NAMES: set[str] = {
    "azelfuel",
    "billiaze",
    "bronzera",
    "camyke",
    "cocodunt",
    "cocomut",
    "colirus",
    "cysting",
    "eleafant",
    "elephfern",
    "eleplant",
    "eloha",
    "elopun",
    "gladiatron",
    "golerno",
    "ivoany",
    "oliosa",
    "pachygerm",
    "palmtrunk",
    "pinealf",
    "rute",
    "scorbit",
    "scrash",
    "sproutrunk",
    "stampyro",
    "taphromet",
    "tephracorna",
    "troot",
    "tropiphant",
    "truncoco",
    "trute",
    "vectol",
    "virachnid",
    "virachnus",
}


---

## vision model training

# ---
# deploy: true
# lambda-test: false
# ---
#
# # FastAI model training with Weights & Biases and Gradio
#
# This example trains a vision model to 98-99% accuracy on the CIFAR-10 dataset,
# and then makes this trained model shareable with others using the [Gradio.app](https://gradio.app/)
# web interface framework (Huggingface's competitor to Streamlit).
#
# Combining GPU-accelerated Modal Functions, a network file system for caching, and Modal
# webhooks for the model demo, we have a simple, productive, and cost-effective
# pathway to building and deploying ML in the cloud!
#
# ![Gradio.app image classifer interface](./gradio-image-classify.png)
#
# ## Setting up the dependencies
#
# Our GPU training is done with PyTorch which bundles its CUDA dependencies, so
# we can start with a slim Debian OS image and install a handful of `pip` packages into it.

import dataclasses
import os
import pathlib
import sys
from typing import List, Optional, Tuple

from fastapi import FastAPI
from modal import (
    App,
    Image,
    Mount,
    Secret,
    Volume,
    asgi_app,
    enter,
    method,
)

web_app = FastAPI()
assets_path = pathlib.Path(__file__).parent / "vision_model_training" / "assets"
app = App(
    name="example-fastai-wandb-gradio-cifar10-demo"
)  # Note: prior to April 2024, "app" was called "stub"
image = Image.debian_slim(python_version="3.10").pip_install(
    "fastai~=2.7.9",
    "gradio~=3.6.0",
    "httpx~=0.23.0",
    # When using pip PyTorch is not automatically installed by fastai.
    "torch~=1.12.1",
    "torchvision~=0.13.1",
    "wandb~=0.13.4",
)

# A persisted volume will store trained model artefacts across Modal app runs.
# This is crucial as training runs are separate from the Gradio.app we run as a webhook.

volume = Volume.from_name("cifar10-training-vol", create_if_missing=True)

FASTAI_HOME = "/fastai_home"
MODEL_CACHE = pathlib.Path(FASTAI_HOME, "models")
USE_GPU = os.environ.get("MODAL_GPU")
MODEL_EXPORT_PATH = pathlib.Path(MODEL_CACHE, "model-exports", "inference.pkl")
os.environ[
    "FASTAI_HOME"
] = FASTAI_HOME  # Ensure fastai saves data into persistent volume path.

# ## Config
#
# Training config gets its own dataclass to avoid smattering special/magic values throughout code.


@dataclasses.dataclass
class WandBConfig:
    project: str = "fast-cifar10"
    entity: Optional[str] = None


@dataclasses.dataclass
class Config:
    epochs: int = 10
    img_dims: Tuple[int, int] = (32, 224)
    gpu: str = USE_GPU
    wandb: WandBConfig = dataclasses.field(default_factory=WandBConfig)


# ## Get CIFAR-10 dataset
#
# The `fastai` framework famously requires very little code to get things done,
# so our downloading function is very short and simple. The CIFAR-10 dataset is
# also not large, about 150MB, so we don't bother persisting it in a network file system
# and just download and unpack it to ephemeral disk.


def download_dataset():
    from fastai.data.external import URLs, untar_data

    path = untar_data(URLs.CIFAR)
    print(f"Finished downloading and unpacking CIFAR-10 dataset to {path}.")
    return path


# ## Training a vision model with FastAI
#
# To address the CIFAR-10 image classification problem, we use the high-level fastAI framework
# to train a Deep Residual Network (https://arxiv.org/pdf/1512.03385.pdf) with 18-layers, called `resnet18`.
#
# We don't train the model from scratch. A transfer learning approach is sufficient to reach 98-99%
# accuracy on the classification task. The main tweak we make is to adjust the image size of the CIFAR-10
# examples to be 224x224, which was the image size the `resnet18` model was originally trained on.
#
# Just to demonstrate the affect of the image upscaling on classification performance, we still train on
# the original size 32x32 images.
#
# ### Tracking with Weights & Biases
#
# ![weights & biases UI](./wandb-ui.png)
#
# Weights & Biases (W&B) is a product that provides out-of-the-box model training observability. With a few
# lines of code and an account, we gain a dashboard will key metrics such as training loss, accuracy, and GPU
# utilization.
#
# If you want to run this example without setting up Weights & Biases, just remove the
# `secrets=[Secret.from_name("wandb")]` line from the Function decorator below; this will disable Weights & Biases
# functionality.
#
# ### Detaching our training run
#
# Fine-tuning the base ResNet model takes about 30-40 minutes on a GPU. To avoid
# needing to keep our terminal active, we can run training as a 'detached run'.
#
# `MODAL_GPU=any modal run --detach vision_model_training.py::app.train`
#


@app.function(
    image=image,
    gpu=USE_GPU,
    volumes={str(MODEL_CACHE): volume},
    secrets=[Secret.from_name("my-wandb-secret")],
    timeout=2700,  # 45 minutes
)
def train():
    import wandb
    from fastai.callback.wandb import WandbCallback
    from fastai.data.transforms import parent_label
    from fastai.metrics import accuracy
    from fastai.vision.all import Resize, models, vision_learner
    from fastai.vision.data import (
        CategoryBlock,
        DataBlock,
        ImageBlock,
        TensorCategory,
        get_image_files,
    )

    config: Config = Config()

    print("Downloading dataset")
    dataset_path = download_dataset()

    wandb_enabled = bool(os.environ.get("WANDB_API_KEY"))
    if wandb_enabled:
        wandb.init(
            id=app.app_id,
            project=config.wandb.project,
            entity=config.wandb.entity,
        )
        callbacks = WandbCallback()
    else:
        callbacks = None

    for dim in config.img_dims:
        print(f"Training on {dim}x{dim} size images.")
        dblock = DataBlock(
            blocks=(ImageBlock(), CategoryBlock()),
            get_items=get_image_files,
            get_y=parent_label,
            item_tfms=Resize(dim),
        )

        dls = dblock.dataloaders(dataset_path, bs=64)

        learn = vision_learner(
            dls, models.resnet18, metrics=accuracy, cbs=callbacks
        ).to_fp16()
        learn.fine_tune(config.epochs, freeze_epochs=3)
        learn.save(f"cifar10_{dim}")

        # run on test set
        test_files = get_image_files(dataset_path / "test")
        label = TensorCategory(
            [dls.vocab.o2i[parent_label(f)] for f in test_files]
        )

        test_set = learn.dls.test_dl(test_files)
        pred = learn.get_preds(0, test_set)
        acc = accuracy(pred[0], label).item()
        print(f"{dim}x{dim}, test accuracy={acc}")

    # 🐝 Close wandb run
    if wandb_enabled:
        wandb.finish()

    print("Exporting model for later inference.")
    MODEL_EXPORT_PATH.parent.mkdir(exist_ok=True)
    learn.remove_cbs(
        WandbCallback
    )  # Added W&B callback is not compatible with inference.
    learn.export(MODEL_EXPORT_PATH)
    volume.commit()


# ## Trained model plumbing
#
# To load a trained model into the demo app, we write a small
# amount of harness code that loads the saved model from persistent
# disk once on container start.
#
# The model's predict function accepts an image as bytes or a numpy array.


@app.cls(
    image=image,
    volumes={str(MODEL_CACHE): volume},
)
class ClassifierModel:
    @enter()
    def load_model(self):
        from fastai.learner import load_learner

        self.model = load_learner(MODEL_EXPORT_PATH)

    @method()
    def predict(self, image) -> str:
        prediction = self.model.predict(image)
        classification = prediction[0]
        return classification


@app.function(
    image=image,
)
def classify_url(image_url: str) -> None:
    """Utility function for command-line classification runs."""
    import httpx

    r = httpx.get(image_url)
    if r.status_code != 200:
        raise RuntimeError(f"Could not download '{image_url}'")

    classifier = ClassifierModel()
    label = classifier.predict.remote(image=r.content)
    print(f"Classification: {label}")


# ## Wrap the trained model in Gradio's web UI
#
# Gradio.app makes it super easy to expose a model's functionality
# in an intuitive web interface.
#
# This model is an image classifier, so we set up an interface that
# accepts an image via drag-and-drop and uses the trained model to
# output a classification label.
#
# Remember, this model was trained on tiny CIFAR-10 images so it's
# going to perform best against similarly simple and scaled-down images.


def create_demo_examples() -> List[str]:
    # NB: Don't download these images to a network FS as it doesn't play well with Gradio.
    import httpx

    example_imgs = {
        "lion.jpg": "https://i0.wp.com/lioncenter.umn.edu/wp-content/uploads/2018/10/cropped-DSC4884_Lion_Hildur-1.jpg",
        "mouse.jpg": "https://static-s.aa-cdn.net/img/ios/1077591533/18f74754ae55ee78e96e04d14e8bff35?v=1",
        "plane.jpg": "https://x-plane.hu/L-410/img/about/2.png",
        "modal.jpg": "https://pbs.twimg.com/profile_images/1567270019075031040/Hnrebn0M_400x400.jpg",
    }
    available_examples = []
    for dest, url in example_imgs.items():
        filepath = pathlib.Path(dest)
        r = httpx.get(url)
        if r.status_code != 200:
            print(f"Could not download '{url}'", file=sys.stderr)
            continue

        with open(filepath, "wb") as f:
            f.write(r.content)
        available_examples.append(str(filepath))
    return available_examples


@app.function(
    image=image,
    volumes={str(MODEL_CACHE): volume},
    mounts=[Mount.from_local_dir(assets_path, remote_path="/assets")],
)
@asgi_app()
def fastapi_app():
    import gradio as gr
    from gradio.routes import mount_gradio_app

    classifier = ClassifierModel()
    interface = gr.Interface(
        fn=classifier.predict.remote,
        inputs=gr.Image(shape=(224, 224)),
        outputs="label",
        examples=create_demo_examples(),
        css="/assets/index.css",
    )
    return mount_gradio_app(
        app=web_app,
        blocks=interface,
        path="/",
    )


## Running this
#
# To run training as an ephemeral app:
#
# ```shell
# modal run vision_model_training.py::app.train
# ```
#
# To test the model on an image, run:
#
# ```shell
# modal run vision_model_training.py::app.classify_url --image-url <url>
# ```
#
# To run the Gradio server, run:
#
# ```shell
# modal serve vision_model_training.py
# ```
#
# This ML app is already deployed on Modal and you can try it out at https://modal-labs-example-fastai-wandb-gradio-cifar10-demo-fastapi-app.modal.run.


---

## badges

# ---
# cmd: ["modal", "serve", "07_web_endpoints/badges.py"]
# ---
# # Serve a dynamic SVG badge

# In this example, we use Modal's [webhook](/docs/guide/webhooks) capability to host a dynamic SVG badge that shows
# you the current # of downloads for a Python package.
#
# First let's start off by creating a Modal app, and defining an image with the Python packages we're going to be using:

from modal import App, Image, web_endpoint

app = App(
    "example-web-badges",
    image=Image.debian_slim().pip_install("pybadges", "pypistats"),
)  # Note: prior to April 2024, "app" was called "stub"

# ## Defining the web endpoint
#
# In addition to using `@app.function()` to decorate our function, we use the
# `@modal.web_endpoint` decorator ([learn more](/docs/guide/webhooks#web_endpoint)), which instructs Modal
# to create a REST endpoint that serves this function. Note that the default method is `GET`, but this
# can be overridden using the `method` argument.


@app.function()
@web_endpoint()
async def package_downloads(package_name: str):
    import json

    import pypistats
    from fastapi import Response
    from pybadges import badge

    stats = json.loads(pypistats.recent(package_name, format="json"))
    svg = badge(
        left_text=f"{package_name} downloads",
        right_text=str(stats["data"]["last_month"]),
        right_color="blue",
    )

    return Response(content=svg, media_type="image/svg+xml")


# In this function, we use `pypistats` to query the most recent stats for our package, and then
# use that as the text for a SVG badge, rendered using `pybadges`.
# Since Modal web endpoints are FastAPI functions under the hood, we return this SVG wrapped in a FastAPI response with the correct media type.
# Also note that FastAPI automatically interprets `package_name` as a [query param](https://fastapi.tiangolo.com/tutorial/query-params/).

# ## Running and deploying
#
# We can now run an ephemeral app on the command line using:
#
# ```shell
# modal serve badges.py
# ```
#
# This will create a short-lived web url that exists until you terminate the script.
# It will also hot-reload the code if you make changes to it.
#
# If you want to create a persistent URL, you have to deploy the script.
# To deploy using the Modal CLI by running `modal deploy badges.py`,
#
# Either way, as soon as we run this command, Modal gives us the link to our brand new
# web endpoint in the output:
#
# ![web badge deployment](./badges_deploy.png)
#
# We can now visit the link using a web browser, using a `package_name` of our choice in the URL query params.
# For example:
# - `https://YOUR_SUBDOMAIN.modal.run/?package_name=synchronicity`
# - `https://YOUR_SUBDOMAIN.modal.run/?package_name=torch`


---

## basic web

# ---
# cmd: ["modal", "serve", "07_web_endpoints/basic_web.py"]
# ---
import modal
from modal import enter, web_endpoint

app = modal.App(
    name="example-lifecycle-web"
)  # Note: prior to April 2024, "app" was called "stub"

# Hello world!
#
# This is as simple as it gets. A GET endpoint which
# returns a string.


@app.function()
@web_endpoint()
def hello():
    return "Hello world!"


# Lifecycle-based.
#
# Web endpoints can be methods on a [lifecycle class](/docs/guide/lifecycle-functions#container-lifecycle-functions-and-parameters).
# This example will only set the `val` instance variable once, on container startup.
# But note that they don't need the [`modal.method`](/docs/reference/modal.method#modalmethod) decorator.


@app.cls()
class WebApp:
    @enter()
    def startup(self):
        print("🏁 Startup up!")
        self.val = "Hello world"

    @web_endpoint()
    def web(self):
        return {"message": self.val}


---

## chatbot spa

# ---
# args: ["--message", "what's up?"]
# ---
"""Single-page application that lets you talk to a transformer chatbot.

This is a complex example demonstrating an end-to-end web application backed by
serverless web handlers and GPUs. The user visits a single-page application,
written using Solid.js. This interface makes API requests that are handled by a
Modal function running on the GPU.

The weights of the model are saved in the image, so they don't need to be
downloaded again while the app is running.

Chat history tensors are saved in a `modal.Dict` distributed dictionary.
"""

import uuid
from pathlib import Path
from typing import Optional, Tuple

import fastapi
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from modal import App, Dict, Image, Mount, asgi_app

assets_path = Path(__file__).parent / "chatbot_spa"
app = App(
    "example-chatbot-spa"
)  # Note: prior to April 2024, "app" was called "stub"
chat_histories = Dict.from_name(
    "example-chatbot-spa-history", create_if_missing=True
)


def load_tokenizer_and_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/DialoGPT-large",
        device_map="auto",
    )
    return tokenizer, model


gpu_image = (
    Image.debian_slim()
    .pip_install("torch", find_links="https://download.pytorch.org/whl/cu116")
    .pip_install("transformers~=4.31", "accelerate")
    .run_function(load_tokenizer_and_model)
)


with gpu_image.imports():
    import torch

    tokenizer, model = load_tokenizer_and_model()


@app.function(mounts=[Mount.from_local_dir(assets_path, remote_path="/assets")])
@asgi_app()
def transformer():
    app = fastapi.FastAPI()

    @app.post("/chat")
    def chat(body: dict = fastapi.Body(...)):
        message = body["message"]
        chat_id = body.get("id")
        id, response = generate_response.remote(message, chat_id)
        return JSONResponse({"id": id, "response": response})

    app.mount("/", StaticFiles(directory="/assets", html=True))
    return app


@app.function(gpu="any", image=gpu_image)
def generate_response(
    message: str, id: Optional[str] = None
) -> Tuple[str, str]:
    new_input_ids = tokenizer.encode(
        message + tokenizer.eos_token, return_tensors="pt"
    ).to("cuda")
    if id is not None:
        chat_history = chat_histories[id]
        bot_input_ids = torch.cat([chat_history, new_input_ids], dim=-1)
    else:
        id = str(uuid.uuid4())
        bot_input_ids = new_input_ids

    chat_history = model.generate(
        bot_input_ids, max_length=1250, pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(
        chat_history[:, bot_input_ids.shape[-1] :][0], skip_special_tokens=True
    )

    chat_histories[id] = chat_history
    return id, response


@app.local_entrypoint()
def test_response(message: str):
    _, response = generate_response.remote(message)
    print(response)


---

## count faces

# ---
# lambda-test: false
# ---

import os

import modal

app = modal.App(
    "example-count-faces"
)  # Note: prior to April 2024, "app" was called "stub"


open_cv_image = (
    modal.Image.debian_slim()
    .apt_install("python3-opencv")
    .pip_install("opencv-python", "numpy")
)


@app.function(image=open_cv_image)
def count_faces(image_bytes):
    import cv2
    import numpy as np

    # Example borrowed from https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(
        os.path.join(
            cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
        )
    )
    # Read the input image
    np_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return len(faces)


if __name__ == "__main__":
    # Code below could have been put in a different file, but keeping it in one place for cohesion
    import sanic

    app = sanic.Sanic("web_worker_example")

    @app.get("/")
    def index(request):
        return sanic.html(
            """
<html>
<form action="/process" method="post" enctype="multipart/form-data">
    <input type="file" name="file" id="file" />
    <input type="submit" />
</form>
</html>
    """
        )

    @app.post("/process")
    async def process(request: sanic.Request):
        input_file = request.files["file"][0]
        async with app.run():  # type: ignore
            num_faces = await count_faces.remote(input_file.body)

        return sanic.json({"faces": num_faces})

    app.run(auto_reload=True, debug=True)


---

## fastapi app

# ---
# lambda-test: false
# ---

from typing import Optional

from fastapi import FastAPI, Header
from modal import App, Image, asgi_app, web_endpoint
from pydantic import BaseModel

web_app = FastAPI()
app = App(
    "example-fastapi-app"
)  # Note: prior to April 2024, "app" was called "stub"
image = Image.debian_slim()


class Item(BaseModel):
    name: str


@web_app.get("/")
async def handle_root(user_agent: Optional[str] = Header(None)):
    print(f"GET /     - received user_agent={user_agent}")
    return "Hello World"


@web_app.post("/foo")
async def handle_foo(item: Item, user_agent: Optional[str] = Header(None)):
    print(
        f"POST /foo - received user_agent={user_agent}, item.name={item.name}"
    )
    return item


@app.function(image=image)
@asgi_app()
def fastapi_app():
    return web_app


@app.function()
@web_endpoint(method="POST")
def f(item: Item):
    return "Hello " + item.name


if __name__ == "__main__":
    app.deploy("webapp")


---

## flask app

# ---
# lambda-test: false
# ---

from modal import App, Image, wsgi_app

app = App(
    "example-web-flask",
    image=Image.debian_slim().pip_install("flask"),
)  # Note: prior to April 2024, "app" was called "stub"


@app.function()
@wsgi_app()
def flask_app():
    from flask import Flask, request

    web_app = Flask(__name__)

    @web_app.get("/")
    def home():
        return "Hello Flask World!"

    @web_app.post("/foo")
    def foo():
        return request.json

    return web_app


---

## flask streaming

# ---
# lambda-test: false
# ---

import modal
from modal import wsgi_app

app = modal.App(
    "example-web-flask-stream",
    image=modal.Image.debian_slim().pip_install("flask"),
)  # Note: prior to April 2024, "app" was called "stub"


@app.function()
def generate_rows():
    """
    This creates a large CSV file, about 10MB, which will be streaming downloaded
    by a web client.
    """
    for i in range(10_000):
        line = ",".join(str((j + i) * i) for j in range(128))
        yield f"{line}\n"


@app.function()
@wsgi_app()
def flask_app():
    from flask import Flask

    web_app = Flask(__name__)

    # These web handlers follow the example from
    # https://flask.palletsprojects.com/en/2.2.x/patterns/streaming/

    @web_app.route("/")
    def generate_large_csv():
        # Run the function locally in the web app's container.
        return generate_rows.local(), {"Content-Type": "text/csv"}

    @web_app.route("/remote")
    def generate_large_csv_in_container():
        # Run the function remotely in a separate container,
        # which will stream back results to the web app container,
        # which will stream back to the web client.
        #
        # This is less efficient, but demonstrates how web serving
        # containers can be separated from and cooperate with other
        # containers.
        return generate_rows.remote(), {"Content-Type": "text/csv"}

    return web_app


---

## streaming

# ---
# cmd: ["modal", "serve", "07_web_endpoints/streaming.py"]
# deploy: true
# ---
import asyncio
import time

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from modal import App, asgi_app, web_endpoint

app = App(
    "example-fastapi-streaming"
)  # Note: prior to April 2024, "app" was called "stub"

web_app = FastAPI()

# This asynchronous generator function simulates
# progressively returning data to the client. The `asyncio.sleep`
# is not necessary, but makes it easier to see the iterative behavior
# of the response.


async def fake_video_streamer():
    for i in range(10):
        yield f"frame {i}: hello world!".encode()
        await asyncio.sleep(1.0)


# ASGI app with streaming handler.
#
# This `fastapi_app` also uses the fake video streamer async generator,
# passing it directly into `StreamingResponse`.


@web_app.get("/")
async def main():
    return StreamingResponse(
        fake_video_streamer(), media_type="text/event-stream"
    )


@app.function()
@asgi_app()
def fastapi_app():
    return web_app


# This `hook` web endpoint Modal function calls *another* Modal function,
# and it just works!


@app.function()
def sync_fake_video_streamer():
    for i in range(10):
        yield f"frame {i}: some data\n".encode()
        time.sleep(1)


@app.function()
@web_endpoint()
def hook():
    return StreamingResponse(
        sync_fake_video_streamer.remote_gen(), media_type="text/event-stream"
    )


# This `mapped` web endpoint Modal function does a parallel `.map` on a simple
# Modal function. Using `.starmap` also would work in the same fashion.


@app.function()
def map_me(i):
    time.sleep(i)  # stagger the results for demo purposes
    return f"hello from {i}\n"


@app.function()
@web_endpoint()
def mapped():
    return StreamingResponse(
        map_me.map(range(10)), media_type="text/event-stream"
    )


# A collection of basic examples of a webhook streaming response.
#
#
# ```
# modal serve streaming.py
# ```
#
# To try out the webhook, ensure that your client is not buffering the server response
# until it gets newline (\n) characters. By default browsers and `curl` are buffering,
# though modern browsers should respect the "text/event-stream" content type header being set.
#
# ```shell
# curl --no-buffer https://modal-labs--example-fastapi-streaming-fastapi-app.modal.run
# curl --no-buffer https://modal-labs--example-fastapi-streaming-hook.modal.run
# curl --no-buffer https://modal-labs--example-fastapi-streaming-mapped.modal.run
# ````


---

## generators async

import modal

app = modal.App(
    "example-generators-async"
)  # Note: prior to April 2024, "app" was called "stub"


@app.function()
def f(i):
    for j in range(i):
        yield j


@app.local_entrypoint()
async def run_async():
    async for r in f.remote_gen.aio(10):
        print(r)


---

## hello world async

# # Async functions
#
# Modal natively supports async/await syntax using asyncio.

# First, let's import some global stuff.

import sys

import modal

app = modal.App(
    "example-hello-world-async"
)  # Note: prior to April 2024, "app" was called "stub"


# ## Defining a function
#
# Now, let's define a function. The wrapped function can be synchronous or
# asynchronous, but calling it in either context will still work.
# Let's stick to a normal synchronous function


@app.function()
def f(i):
    if i % 2 == 0:
        print("hello", i)
    else:
        print("world", i, file=sys.stderr)

    return i * i


# ## Running the app with asyncio
#
# Let's make the main entrypoint asynchronous. In async contexts, we should
# call the function using `await` or iterate over the map using `async for`.
# Otherwise we would block the event loop while our call is being run.


@app.local_entrypoint()
async def run_async():
    # Call the function using .remote.aio() in order to run it asynchronously
    print(await f.remote.aio(1000))

    # Parallel map.
    total = 0
    # Call .map asynchronously using using f.map.aio(...)
    async for ret in f.map.aio(range(20)):
        total += ret

    print(total)


---

## parallel execution

import time

import modal

app = modal.App(
    "example-parallel"
)  # Note: prior to April 2024, "app" was called "stub"


@app.function()
def step1(word):
    time.sleep(2)
    print("step1 done")
    return word


@app.function()
def step2(number):
    time.sleep(1)
    print("step2 done")
    if number == 0:
        raise ValueError("custom error")
    return number


@app.local_entrypoint()
def main():
    # Start running a function and return a handle to its result.
    word_call = step1.spawn("foo")
    number_call = step2.spawn(2)

    # Print "foofoo" after 2 seconds.
    print(word_call.get() * number_call.get())

    # Alternatively, use `modal.functions.gather(...)` as a convenience wrapper,
    # which returns an error if either call fails.
    results = modal.functions.gather(step1.spawn("bar"), step2.spawn(4))
    assert results == ["bar", 4]

    # Raise exception after 2 seconds.
    try:
        modal.functions.gather(step1.spawn("bar"), step2.spawn(0))
    except ValueError as exc:
        assert str(exc) == "custom error"


---

## poll delayed result

# ---
# lambda-test: false
# ---
import fastapi
from modal import App, Image, asgi_app
from modal.functions import FunctionCall
from starlette.responses import HTMLResponse, RedirectResponse

app = App("example-poll")  # Note: prior to April 2024, "app" was called "stub"

web_app = fastapi.FastAPI()


@app.function(image=Image.debian_slim().pip_install("primefac"))
def factor_number(number):
    import primefac

    return list(primefac.primefac(number))  # could take a long time


@web_app.get("/")
async def index():
    return HTMLResponse(
        """
    <form method="get" action="/factors">
        Enter a number: <input name="number" />
        <input type="submit" value="Factorize!"/>
    </form>
    """
    )


@web_app.get("/factors")
async def web_submit(request: fastapi.Request, number: int):
    call = factor_number.spawn(
        number
    )  # returns a FunctionCall without waiting for result
    polling_url = request.url.replace(
        path="/result", query=f"function_id={call.object_id}"
    )
    return RedirectResponse(polling_url)


@web_app.get("/result")
async def web_poll(function_id: str):
    function_call = FunctionCall.from_id(function_id)
    try:
        result = function_call.get(timeout=0)
    except TimeoutError:
        result = "not ready"

    return result


@app.function()
@asgi_app()
def fastapi_app():
    return web_app


---

## doc jobs

# ---
# deploy: true
# ---
#
# # Document OCR job queue
#
# This tutorial shows you how to use Modal as an infinitely scalable job queue
# that can service async tasks from a web app. For the purpose of this tutorial,
# we've also built a [React + FastAPI web app on Modal](/docs/examples/doc_ocr_webapp)
# that works together with it, but note that you don't need a web app running on Modal
# to use this pattern. You can submit async tasks to Modal from any Python
# application (for example, a regular Django app running on Kubernetes).
#
# Our job queue will handle a single task: running OCR transcription for images.
# We'll make use of a pre-trained Document Understanding model using the
# [donut](https://github.com/clovaai/donut) package to accomplish this. Try
# it out for yourself [here](https://modal-labs-example-doc-ocr-webapp-wrapper.modal.run/).
#
# ![receipt parser frontend](./receipt_parser_frontend_2.jpg)

# ## Define an App
#
# Let's first import `modal` and define a [`App`](/docs/reference/modal.App). Later, we'll use the name provided
# for our `App` to find it from our web app, and submit tasks to it.

import urllib.request

import modal

app = modal.App(
    "example-doc-ocr-jobs"
)  # Note: prior to April 2024, "app" was called "stub"

# ## Model cache
#
# `donut` downloads the weights for pre-trained models to a local directory, if those weights don't already exist.
# To decrease start-up time, we want this download to happen just once, even across separate function invocations.
# To accomplish this, we use the [`Image.run_function`](/docs/reference/modal.Image#run_function) method, which allows
# us to run some code at image build time to save the model weights into the image.

CACHE_PATH = "/root/model_cache"
MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"


def download_model_weights() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=MODEL_NAME, cache_dir=CACHE_PATH)


image = (
    modal.Image.debian_slim(python_version="3.9")
    .pip_install(
        "donut-python==1.0.7",
        "huggingface-hub==0.16.4",
        "transformers==4.21.3",
        "timm==0.5.4",
    )
    .run_function(download_model_weights)
)

# ## Handler function
#
# Now let's define our handler function. Using the [@app.function()](https://modal.com/docs/reference/modal.App#function)
# decorator, we set up a Modal [Function](/docs/reference/modal.Function) that uses GPUs,
# runs on a [custom container image](/docs/guide/custom-container),
# and automatically [retries](/docs/guide/retries#function-retries) failures up to 3 times.


@app.function(
    gpu="any",
    image=image,
    retries=3,
)
def parse_receipt(image: bytes):
    import io

    import torch
    from donut import DonutModel
    from PIL import Image

    # Use donut fine-tuned on an OCR dataset.
    task_prompt = "<s_cord-v2>"
    pretrained_model = DonutModel.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_PATH,
    )

    # Initialize model.
    pretrained_model.half()
    device = torch.device("cuda")
    pretrained_model.to(device)

    # Run inference.
    input_img = Image.open(io.BytesIO(image))
    output = pretrained_model.inference(image=input_img, prompt=task_prompt)[
        "predictions"
    ][0]
    print("Result: ", output)

    return output


# ## Deploy
#
# Now that we have a function, we can publish it by deploying the app:
#
# ```shell
# modal deploy doc_ocr_jobs.py
# ```
#
# Once it's published, we can [look up](/docs/guide/trigger-deployed-functions) this function from another
# Python process and submit tasks to it:
#
# ```python
# fn = modal.Function.lookup("example-doc-ocr-jobs", "parse_receipt")
# fn.spawn(my_image)
# ```
#
# Modal will auto-scale to handle all the tasks queued, and
# then scale back down to 0 when there's no work left. To see how you could use this from a Python web
# app, take a look at the [receipt parser frontend](/docs/examples/doc_ocr_webapp)
# tutorial.

# ## Run manually
#
# We can also trigger `parse_receipt` manually for easier debugging:
# `modal run doc_ocr_jobs::app.main`
# To try it out, you can find some
# example receipts [here](https://drive.google.com/drive/folders/1S2D1gXd4YIft4a5wDtW99jfl38e85ouW).


@app.local_entrypoint()
def main():
    from pathlib import Path

    receipt_filename = Path(__file__).parent / "receipt.png"
    if receipt_filename.exists():
        with open(receipt_filename, "rb") as f:
            image = f.read()
    else:
        image = urllib.request.urlopen(
            "https://nwlc.org/wp-content/uploads/2022/01/Brandys-walmart-receipt-8.webp"
        ).read()
    print(parse_receipt.remote(image))


---

## doc webapp

# ---
# deploy: true
# lambda-test: false
# ---
#
# # Document OCR web app
#
# This tutorial shows you how to use Modal to deploy a fully serverless
# [React](https://reactjs.org/) + [FastAPI](https://fastapi.tiangolo.com/) application.
# We're going to build a simple "Receipt Parser" web app that submits OCR transcription
# tasks to a separate Modal app defined in the [Job Queue
# tutorial](/docs/examples/doc_ocr_jobs), polls until the task is completed, and displays
# the results. Try it out for yourself
# [here](https://modal-labs-example-doc-ocr-webapp-wrapper.modal.run/).
#
# ![receipt parser frontend](./receipt_parser_frontend.jpg)

# ## Basic setup
#
# Let's get the imports out of the way and define a [`App`](/docs/reference/modal.App).

from pathlib import Path

import fastapi
import fastapi.staticfiles
from modal import App, Function, Mount, asgi_app

app = App(
    "example-doc-ocr-webapp"
)  # Note: prior to April 2024, "app" was called "stub"

# Modal works with any [ASGI](/docs/guide/webhooks#serving-asgi-and-wsgi-apps) or
# [WSGI](/docs/guide/webhooks#wsgi) web framework. Here, we choose to use [FastAPI](https://fastapi.tiangolo.com/).

web_app = fastapi.FastAPI()

# ## Define endpoints
#
# We need two endpoints: one to accept an image and submit it to the Modal job queue,
# and another to poll for the results of the job.
#
# In `parse`, we're going to submit tasks to the function defined in the [Job
# Queue tutorial](/docs/examples/doc_ocr_jobs), so we import it first using
# [`Function.lookup`](/docs/reference/modal.Function#lookup).
#
# We call [`.spawn()`](/docs/reference/modal.Function#spawn) on the function handle
# we imported above, to kick off our function without blocking on the results. `spawn` returns
# a unique ID for the function call, that we can use later to poll for its result.


@web_app.post("/parse")
async def parse(request: fastapi.Request):
    parse_receipt = Function.lookup("example-doc-ocr-jobs", "parse_receipt")

    form = await request.form()
    receipt = await form["receipt"].read()  # type: ignore
    call = parse_receipt.spawn(receipt)
    return {"call_id": call.object_id}


# `/result` uses the provided `call_id` to instantiate a `modal.FunctionCall` object, and attempt
# to get its result. If the call hasn't finished yet, we return a `202` status code, which indicates
# that the server is still working on the job.


@web_app.get("/result/{call_id}")
async def poll_results(call_id: str):
    from modal.functions import FunctionCall

    function_call = FunctionCall.from_id(call_id)
    try:
        result = function_call.get(timeout=0)
    except TimeoutError:
        return fastapi.responses.JSONResponse(content="", status_code=202)

    return result


# Finally, we mount the static files for our front-end. We've made [a simple React
# app](https://github.com/modal-labs/modal-examples/tree/main/09_job_queues/doc_ocr_frontend)
# that hits the two endpoints defined above. To package these files with our app, first
# we get the local assets path, and then create a modal [`Mount`](/docs/guide/local-data#mounting-directories)
# that mounts this directory at `/assets` inside our container. Then, we instruct FastAPI to [serve
# this static file directory](https://fastapi.tiangolo.com/tutorial/static-files/) at our root path.

assets_path = Path(__file__).parent / "doc_ocr_frontend"


@app.function(mounts=[Mount.from_local_dir(assets_path, remote_path="/assets")])
@asgi_app()
def wrapper():
    web_app.mount(
        "/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True)
    )

    return web_app


# ## Running
#
# You can run this as an ephemeral app, by running the command
#
# ```shell
# modal serve doc_ocr_webapp.py
# ```
#
# ## Deploy
#
# That's all! To deploy your application, run
#
# ```shell
# modal deploy doc_ocr_webapp.py
# ```
#
# If successful, this will print a URL for your app, that you can navigate to from
# your browser 🎉 .
#
# ![receipt parser processed](./receipt_parser_frontend_2.jpg)
#
# ## Developing
#
# If desired, instead of deploying, we can [serve](/docs/guide/webhooks#developing-with-modal-serve)
# our app ephemerally. In this case, Modal watches all the mounted files, and updates
# the app if anything changes.


---

## algolia indexer

# ---
# deploy: true
# ---
# # Algolia docsearch crawler
#
# This tutorial shows you how to use Modal to run the [Algolia docsearch
# crawler](https://docsearch.algolia.com/docs/legacy/run-your-own/) to index your
# website and make it searchable. This is not just example code - we run the same
# code in production to power search on this page (`Ctrl+K` to try it out!).

# ## Basic setup
#
# Let's get the imports out of the way.

import json
import os
import subprocess

from modal import App, Image, Secret, web_endpoint

# Modal lets you [use and extend existing Docker images](/docs/guide/custom-container#use-an-existing-container-image-with-from_registry),
# as long as they have `python` and `pip` available. We'll use the official crawler image built by Algolia, with a small
# adjustment: since this image has `python` symlinked to `python3.6` and Modal is not compatible with Python 3.6, we
# install Python 3.11 and symlink that as the `python` executable instead.

algolia_image = Image.from_registry(
    "algolia/docsearch-scraper:v1.16.0",
    add_python="3.11",
    setup_dockerfile_commands=["ENTRYPOINT []"],
)

app = App(
    "example-algolia-indexer"
)  # Note: prior to April 2024, "app" was called "stub"

# ## Configure the crawler
#
# Now, let's configure the crawler with the website we want to index, and which
# CSS selectors we want to scrape. Complete documentation for crawler configuration is available
# [here](https://docsearch.algolia.com/docs/legacy/config-file).

CONFIG = {
    "index_name": "modal_docs",
    "start_urls": [
        {"url": "https://modal.com/docs/guide", "page_rank": 2},
        {"url": "https://modal.com/docs/examples", "page_rank": 1},
        {"url": "https://modal.com/docs/reference", "page_rank": 1},
    ],
    "selectors": {
        "lvl0": {
            "selector": ".sidebar .active",
            "default_value": "Documentation",
            "global": True,
        },
        "lvl1": "article h1",
        "lvl2": "article h2",
        "lvl3": "article h3",
        "lvl4": "article h4",
        "text": "article p,article ol,article ul,article pre",
    },
}

# ## Create an API key
#
# If you don't already have one, sign up for an account on [Algolia](https://www.algolia.com/). Set up
# a project and create an API key with `write` access to your index, and with the ACL permissions
# `addObject`, `editSettings` and `deleteIndex`. Now, create a secret on the Modal [Secrets](/secrets)
# page with the `API_KEY` and `APPLICATION_ID` you just created. You can name this anything you want,
# we named it `algolia-secret`.

# ## The actual function
#
# We want to trigger our crawler from our CI/CD pipeline, so we're serving it as a
# [web endpoint](/docs/guide/webhooks#web_endpoint) that can be triggered by a `GET` request during deploy.
# You could also consider running the crawler on a [schedule](/docs/guide/cron).
#
# The Algolia crawler is written for Python 3.6 and needs to run in the `pipenv` created for it,
# so we're invoking it using a subprocess.


@app.function(
    image=algolia_image,
    secrets=[Secret.from_name("algolia-secret")],
)
def crawl():
    # Installed with a 3.6 venv; Python 3.6 is unsupported by Modal, so use a subprocess instead.
    subprocess.run(
        ["pipenv", "run", "python", "-m", "src.index"],
        env={**os.environ, "CONFIG": json.dumps(CONFIG)},
    )


# We want to be able to trigger this function through a webhook.


@app.function()
@web_endpoint()
def crawl_webhook():
    crawl.remote()
    return "Finished indexing docs"


# ## Deploy the indexer
#
# That's all the code we need! To deploy your application, run
#
# ```shell
# modal deploy algolia_indexer.py
# ```
#
# If successful, this will print a URL for your new webhook, that you can hit using
# `curl` or a browser. Logs from webhook invocations can be found from the [apps](/apps)
# page.
#
# The indexed contents can be found at https://www.algolia.com/apps/APP_ID/explorer/browse/, for your
# APP_ID. Once you're happy with the results, you can [set up the `docsearch` package with your
# website](https://docsearch.algolia.com/docs/docsearch-v3/), and create a search component that uses this index.

# ## Entrypoint for development
#
# To make it easier to test this, we also have an entrypoint for when you run
# `modal run algolia_indexer.py`


@app.local_entrypoint()
def run():
    crawl.remote()


---

## cloud bucket mount loras

# ---
# output-directory: "/tmp/stable-diffusion-xl"
# runtimes: ["runc", "gvisor"]
# ---
# # LoRAs Galore: Create a LoRA Playground with Modal, Gradio, and S3
#
# This example shows how to mount an S3 bucket in a Modal app using [`CloudBucketMount`](https://modal.com/docs/reference/modal.CloudBucketMount).
# We will download a bunch of LoRA adapters from the [HuggingFace Hub](https://huggingface.co/models) into our S3 bucket
# then read from that bucket, on the fly, when doing inference.
#
# By default, we use the [IKEA instructions LoRA](https://huggingface.co/ostris/ikea-instructions-lora-sdxl) as an example,
# which produces the following image when prompted to generate "IKEA instructions for building a GPU rig for deep learning":
#
# ![IKEA instructions for building a GPU rig for deep learning](./ikea-instructions-for-building-a-gpu-rig-for-deep-learning.png)
#
# By the end of this example, we've deployed a "playground" app where anyone with a browser can try
# out these custom models. That's the power of Modal: custom, autoscaling AI applications, deployed in seconds.
# You can try out our deployment [here](https://modal-labs--loras-galore-app.modal.run).
#
# ## Basic setup
#

import io
import os
from pathlib import Path
from typing import Optional

from modal import (
    App,
    CloudBucketMount,  # the star of the show
    Image,
    Secret,
    asgi_app,
    build,
    enter,
    method,
)

# You will need to have an S3 bucket and AWS credentials to run this example. Refer to the documentation
# for the detailed [IAM permissions](https://modal.com/docs/guide/cloud-bucket-mounts#iam-permissions) those credentials will need.
#
# After you are done creating a bucket and configuring IAM settings,
# you now need to create a [Modal Secret](https://modal.com/docs/guide/secrets). Navigate to the "Secrets" tab and
# click on the AWS card, then fill in the fields with the AWS key and secret created
# previously. Name the Secret `s3-bucket-secret`.

bucket_secret = Secret.from_name("s3-bucket-secret")

MOUNT_PATH: Path = Path("/mnt/bucket")
LORAS_PATH: Path = MOUNT_PATH / "loras/v5"

# Modal runs serverless functions inside containers.
# The environments those functions run in are defined by
# the container `Image`. The line below constructs an image
# with the dependencies we need -- no need to install them locally.

image = Image.debian_slim().pip_install(
    "huggingface_hub==0.21.4",
    "transformers==4.38.2",
    "diffusers==0.26.3",
    "peft==0.9.0",
    "accelerate==0.27.2",
)

with image.imports():
    # we import these dependencies only inside the container
    import diffusers
    import huggingface_hub
    import torch

# We attach the S3 bucket to all the Modal functions in this app by mounting it on the filesystem they see,
# passing a `CloudBucketMount` to the `volumes` dictionary argument. We can read and write to this mounted bucket
# (almost) as if it were a local directory.
app = App(
    "loras-galore",
    image=image,
    volumes={
        MOUNT_PATH: CloudBucketMount(
            "modal-s3mount-test-bucket",
            secret=bucket_secret,
        )  # Note: prior to April 2024, "app" was called "stub"
    },
)


# ## Acquiring LoRA weights
#
# `search_loras()` will use the Hub API to search for LoRAs. We limit LoRAs
# to a maximum size to avoid downloading very large model weights.
# We went with 800 MiB, but feel free to adapt to what works best for you.
@app.function()
def search_loras(limit: int, max_model_size: int = 1024 * 1024 * 1024):
    api = huggingface_hub.HfApi()

    model_ids: list[str] = []
    for model in api.list_models(
        tags=["lora", "base_model:stabilityai/stable-diffusion-xl-base-1.0"],
        library="diffusers",
        sort="downloads",  # sort by most downloaded
    ):
        try:
            model_size = 0
            for file in api.list_files_info(model.id):
                model_size += file.size

        except huggingface_hub.utils.GatedRepoError:
            print(f"gated model ({model.id}); skipping")
            continue

        # Skip models that are larger than file limit.
        if model_size > max_model_size:
            print(f"model {model.id} is too large; skipping")
            continue

        model_ids.append(model.id)
        if len(model_ids) >= limit:
            return model_ids

    return model_ids


# We want to take the LoRA weights we found and move them from Hugging Face onto S3,
# where they'll be accessible, at short latency and high throughput, for our Modal functions.
# Downloading files in this mount will automatically upload files to S3.
# To speed things up, we will run this function in parallel using Modal's
# [`map`](https://modal.com/docs/reference/modal.Function#map).
@app.function()
def download_lora(repository_id: str) -> Optional[str]:
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    # CloudBucketMounts will report 0 bytes of available space leading to many
    # unnecessary warnings, so we patch the method that emits those warnings.
    from huggingface_hub import file_download

    file_download._check_disk_space = lambda x, y: False

    repository_path = LORAS_PATH / repository_id
    try:
        # skip models we've already downloaded
        if not repository_path.exists():
            huggingface_hub.snapshot_download(
                repository_id,
                local_dir=repository_path.as_posix().replace(".", "_"),
                allow_patterns=["*.safetensors"],
            )
        downloaded_lora = len(list(repository_path.rglob("*.safetensors"))) > 0
    except OSError:
        downloaded_lora = False
    except FileNotFoundError:
        downloaded_lora = False
    if downloaded_lora:
        return repository_id
    else:
        return None


# ## Inference with LoRAs
#
# We define a `StableDiffusionLoRA` class to organize our inference code.
# We load Stable Diffusion XL 1.0 as a base model, then, when doing inference,
# we load whichever LoRA the user specifies from the S3 bucket.
# For more on the decorators we use on the methods below to speed up building and booting,
# check out the [container lifecycle hooks guide](https://modal.com/docs/guide/lifecycle-hooks).
@app.cls(gpu="a10g")  # A10G GPUs are great for inference
class StableDiffusionLoRA:
    pipe_id = "stabilityai/stable-diffusion-xl-base-1.0"

    @build()  # when we setup our image, we download the base model
    def build(self):
        diffusers.DiffusionPipeline.from_pretrained(
            self.pipe_id, torch_dtype=torch.float16
        )

    @enter()  # when a new container starts, we load the base model into the GPU
    def load(self):
        self.pipe = diffusers.DiffusionPipeline.from_pretrained(
            self.pipe_id, torch_dtype=torch.float16
        ).to("cuda")

    @method()  # at inference time, we pull in the LoRA weights and pass the final model the prompt
    def run_inference_with_lora(
        self, lora_id: str, prompt: str, seed: int = 8888
    ) -> bytes:
        for file in (LORAS_PATH / lora_id).rglob("*.safetensors"):
            self.pipe.load_lora_weights(lora_id, weight_name=file.name)
            break

        lora_scale = 0.9
        image = self.pipe(
            prompt,
            num_inference_steps=10,
            cross_attention_kwargs={"scale": lora_scale},
            generator=torch.manual_seed(seed),
        ).images[0]

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        return buffer.getvalue()


# ## Try it locally!
#
# To use our inference code from our local command line, we add a `local_entrypoint` to our `app`.
# Run it using `modal run cloud_bucket_mount_loras.py`, and pass `--help`
# to see the available options.
#
# The inference code will run on our machines, but the results will be available on yours.
@app.local_entrypoint()
def main(
    limit: int = 100,
    example_lora: str = "ostris/ikea-instructions-lora-sdxl",
    prompt: str = "IKEA instructions for building a GPU rig for deep learning",
    seed: int = 8888,
):
    # Download LoRAs in parallel.
    lora_model_ids = [example_lora]
    lora_model_ids += search_loras.remote(limit)

    downloaded_loras = []
    for model in download_lora.map(lora_model_ids):
        if model:
            downloaded_loras.append(model)

    print(f"downloaded {len(downloaded_loras)} loras => {downloaded_loras}")

    # Run inference using one of the downloaded LoRAs.
    byte_stream = StableDiffusionLoRA().run_inference_with_lora.remote(
        example_lora, prompt, seed
    )
    dir = Path("/tmp/stable-diffusion-xl")
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    output_path = dir / f"{as_slug(prompt.lower())}.png"
    print(f"Saving it to {output_path}")
    with open(output_path, "wb") as f:
        f.write(byte_stream)


# ## LoRA Exploradora: A hosted Gradio interface
#
# Command line tools are cool, but we can do better!
# With the Gradio library by Hugging Face, we can create a simple web interface
# around our Python inference function, then use Modal to host it for anyone to try out.
#
# To set up your own, run `modal deploy cloud_bucket_mount_loras.py` and navigate to the URL it prints out.
# If you're playing with the code, use `modal serve` instead to see changes live.

from fastapi import FastAPI

web_app = FastAPI()
web_image = Image.debian_slim().pip_install("gradio~=3.50.2", "pillow~=10.2.0")


@app.function(image=web_image, keep_warm=1, container_idle_timeout=60 * 20)
@asgi_app()
def ui():
    """A simple Gradio interface around our LoRA inference."""
    import io

    import gradio as gr
    from gradio.routes import mount_gradio_app
    from PIL import Image

    # determine with loras are available
    lora_ids = [
        f"{lora_dir.parent.stem}/{lora_dir.stem}"
        for lora_dir in LORAS_PATH.glob("*/*")
    ]

    # pick one to be default, set a default prompt
    default_lora_id = (
        "ostris/ikea-instructions-lora-sdxl"
        if "ostris/ikea-instructions-lora-sdxl" in lora_ids
        else lora_ids[0]
    )
    default_prompt = (
        "IKEA instructions for building a GPU rig for deep learning"
        if default_lora_id == "ostris/ikea-instructions-lora-sdxl"
        else "text"
    )

    # the simple path to making an app on Gradio is an Interface: a UI wrapped around a function.
    def go(lora_id: str, prompt: str, seed: int) -> Image:
        return Image.open(
            io.BytesIO(
                StableDiffusionLoRA().run_inference_with_lora.remote(
                    lora_id, prompt, seed
                )
            ),
        )

    iface = gr.Interface(
        go,
        inputs=[  # the inputs to go/our inference function
            gr.Dropdown(
                choices=lora_ids, value=default_lora_id, label="👉 LoRA ID"
            ),
            gr.Textbox(default_prompt, label="🎨 Prompt"),
            gr.Number(value=8888, label="🎲 Random Seed"),
        ],
        outputs=gr.Image(label="Generated Image"),
        # some extra bits to make it look nicer
        title="LoRAs Galore",
        description="# Try out some of the top custom SDXL models!"
        "\n\nPick a LoRA finetune of SDXL from the dropdown, then prompt it to generate an image."
        "\n\nCheck out [the code on GitHub](https://github.com/modal-labs/modal-examples/blob/main/10_integrations/cloud_bucket_mount_loras.py)"
        " if you want to create your own version or just see how it works."
        "\n\nPowered by [Modal](https://modal.com) 🚀",
        theme="soft",
        allow_flagging="never",
    )

    return mount_gradio_app(app=web_app, blocks=iface, path="/")


def as_slug(name):
    """Converts a string, e.g. a prompt, into something we can use as a filename."""
    import re

    s = str(name).strip().replace(" ", "-")
    s = re.sub(r"(?u)[^-\w.]", "", s)
    return s


---

## covid datasette

# ---
# deploy: true
# runtimes: ["runc", "gvisor"]
# ---
# # Publish interactive datasets with Datasette
#
# ![Datasette user interface](./covid_datasette_ui.png)
#
# This example shows how to serve a Datasette application on Modal. The published dataset
# is COVID-19 case data from Johns Hopkins University which is refreshed daily.
# Try it out for yourself at [modal-labs-example-covid-datasette-app.modal.run/covid-19](https://modal-labs-example-covid-datasette-app.modal.run/covid-19/johns_hopkins_csse_daily_reports).
#
# Some Modal features it uses:
# * Volumes: a persisted volume lets us store and grow the published dataset over time.
# * Scheduled functions: the underlying dataset is refreshed daily, so we schedule a function to run daily.
# * Web endpoints: exposes the Datasette application for web browser interaction and API requests.
#
# ## Basic setup
#
# Let's get started writing code.
# For the Modal container image we need a few Python packages,
# including `GitPython`, which we'll use to download the dataset.

import asyncio
import pathlib
import shutil
import subprocess
from datetime import datetime
from urllib.request import urlretrieve

from modal import App, Image, Period, Volume, asgi_app

app = App(
    "example-covid-datasette"
)  # Note: prior to April 2024, "app" was called "stub"
datasette_image = (
    Image.debian_slim()
    .pip_install("datasette~=0.63.2", "sqlite-utils")
    .apt_install("unzip")
)

# ## Persistent dataset storage
#
# To separate database creation and maintenance from serving, we'll need the underlying
# database file to be stored persistently. To achieve this we use a [`Volume`](/docs/guide/volumes).

volume = Volume.from_name(
    "example-covid-datasette-cache-vol", create_if_missing=True
)

VOLUME_DIR = "/cache-vol"
REPORTS_DIR = pathlib.Path(VOLUME_DIR, "COVID-19")
DB_PATH = pathlib.Path(VOLUME_DIR, "covid-19.db")

# ## Getting a dataset
#
# Johns Hopkins has been publishing up-to-date COVID-19 pandemic data on GitHub since early February 2020, and
# as of late September 2022 daily reporting is still rolling in. Their dataset is what this example will use to
# show off Modal and Datasette's capabilities.
#
# The full git repository size for the dataset is over 6GB, but we only need to shallow clone around 300MB.


@app.function(
    image=datasette_image,
    volumes={VOLUME_DIR: volume},
    retries=2,
)
def download_dataset(cache=True):
    if REPORTS_DIR.exists() and cache:
        print(f"Dataset already present and {cache=}. Skipping download.")
        return
    elif REPORTS_DIR.exists():
        print("Cleaning dataset before re-downloading...")
        shutil.rmtree(REPORTS_DIR)

    print("Downloading dataset...")
    urlretrieve(
        "https://github.com/CSSEGISandData/COVID-19/archive/refs/heads/master.zip",
        "/tmp/covid-19.zip",
    )

    print("Unpacking archive...")
    prefix = "COVID-19-master/csse_covid_19_data/csse_covid_19_daily_reports"
    subprocess.run(
        f"unzip /tmp/covid-19.zip {prefix}/* -d {REPORTS_DIR}", shell=True
    )
    subprocess.run(f"mv {REPORTS_DIR / prefix}/* {REPORTS_DIR}", shell=True)

    print("Committing the volume...")
    volume.commit()

    print("Finished downloading dataset.")


# ## Data munging
#
# This dataset is no swamp, but a bit of data cleaning is still in order. The following two
# functions read a handful of `.csv` files and clean the data, before inserting it into
# SQLite.


def load_daily_reports():
    daily_reports = list(REPORTS_DIR.glob("*.csv"))
    if not daily_reports:
        raise RuntimeError(
            f"Could not find any daily reports in {REPORTS_DIR}."
        )
    for filepath in daily_reports:
        yield from load_report(filepath)


def load_report(filepath):
    import csv

    mm, dd, yyyy = filepath.stem.split("-")
    with filepath.open() as fp:
        for row in csv.DictReader(fp):
            province_or_state = (
                row.get("\ufeffProvince/State")
                or row.get("Province/State")
                or row.get("Province_State")
                or None
            )
            country_or_region = row.get("Country_Region") or row.get(
                "Country/Region"
            )
            yield {
                "day": f"{yyyy}-{mm}-{dd}",
                "country_or_region": (
                    country_or_region.strip() if country_or_region else None
                ),
                "province_or_state": (
                    province_or_state.strip() if province_or_state else None
                ),
                "confirmed": int(float(row["Confirmed"] or 0)),
                "deaths": int(float(row["Deaths"] or 0)),
                "recovered": int(float(row["Recovered"] or 0)),
                "active": int(row["Active"]) if row.get("Active") else None,
                "last_update": row.get("Last Update")
                or row.get("Last_Update")
                or None,
            }


# ## Inserting into SQLite
#
# With the CSV processing out of the way, we're ready to create an SQLite DB and feed data into it.
# Importantly, the `prep_db` function mounts the same volume used by `download_dataset()`, and
# rows are batch inserted with progress logged after each batch, as the full COVID-19 has millions
# of rows and does take some time to be fully inserted.
#
# A more sophisticated implementation would only load new data instead of performing a full refresh,
# but we're keeping things simple for this example!


def chunks(it, size):
    import itertools

    return iter(lambda: tuple(itertools.islice(it, size)), ())


@app.function(
    image=datasette_image,
    volumes={VOLUME_DIR: volume},
    timeout=900,
)
def prep_db():
    import sqlite_utils

    volume.reload()
    print("Loading daily reports...")
    records = load_daily_reports()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite_utils.Database(DB_PATH)
    table = db["johns_hopkins_csse_daily_reports"]

    batch_size = 100_000
    for i, batch in enumerate(chunks(records, size=batch_size)):
        truncate = True if i == 0 else False
        table.insert_all(batch, batch_size=batch_size, truncate=truncate)
        print(f"Inserted {len(batch)} rows into DB.")

    table.create_index(["day"], if_not_exists=True)
    table.create_index(["province_or_state"], if_not_exists=True)
    table.create_index(["country_or_region"], if_not_exists=True)

    print("Syncing DB with volume.")
    volume.commit()
    db.close()


# ## Keep it fresh
#
# Johns Hopkins commits new data to the dataset repository every day, so we set up
# a [scheduled](/docs/guide/cron) function to automatically refresh the database
# every 24 hours.


@app.function(schedule=Period(hours=24), timeout=1000)
def refresh_db():
    print(f"Running scheduled refresh at {datetime.now()}")
    download_dataset.remote(cache=False)
    prep_db.remote()
    volume.commit()
    print("Volume changes committed.")


# ## Web endpoint
#
# Hooking up the SQLite database to a Modal webhook is as simple as it gets.
# The Modal `@asgi_app` decorator wraps a few lines of code: one `import` and a few
# lines to instantiate the `Datasette` instance and return its app server.


@app.function(
    image=datasette_image,
    volumes={VOLUME_DIR: volume},
    allow_concurrent_inputs=16,
)
@asgi_app()
def ui():
    from datasette.app import Datasette

    ds = Datasette(files=[DB_PATH], settings={"sql_time_limit_ms": 10000})
    asyncio.run(ds.invoke_startup())
    return ds.app()


# ## Publishing to the web
#
# Run this script using `modal run covid_datasette.py` and it will create the database.
#
# You can then use `modal serve covid_datasette.py` to create a short-lived web URL
# that exists until you terminate the script.
#
# When publishing the interactive Datasette app you'll want to create a persistent URL.
# Just run `modal deploy covid_datasette.py`.


@app.local_entrypoint()
def run():
    print("Downloading COVID-19 dataset...")
    download_dataset.remote()
    print("Prepping SQLite DB...")
    prep_db.remote()


# You can explore the data at the [deployed web endpoint](https://modal-labs--example-covid-datasette-app.modal.run/covid-19).


---

## dbt duckdb

# ---
# cmd: ["modal", "run", "10_integrations/dbt/dbt_duckdb.py::run", "--command", "run"]
# ---
#
# This example contains a minimal but capable cloud data warehouse.
# It's comprised of the following:
#
# - [DuckDB](https://duckdb.org) as the warehouse's OLAP database engine
# - AWS S3 as the data storage provider
# - [DBT](https://docs.getdbt.com/docs/introduction) as the data transformation tool
#
# Meet your new cloud data warehouse.

from pathlib import Path

import modal

# ## Bucket name configuration
#
# The only thing in the source code that you must update is the S3 bucket name.
# AWS S3 bucket names are globally unique, and the one in this source is used by Modal.
#
# Update the `BUCKET_NAME` variable below and also any references to the original value
# within `sample_proj_duckdb_s3/models/`. The AWS IAM policy below also includes the bucket
# name and that must be updated.

BUCKET_NAME = "modal-example-dbt-duckdb-s3"
LOCAL_DBT_PROJECT = Path(__file__).parent / "sample_proj_duckdb_s3"
PROJ_PATH = "/root/dbt"
PROFILES_PATH = "/root/dbt_profile"
TARGET_PATH = "/root/target"
dbt_image = (
    modal.Image.debian_slim()
    .pip_install(
        "boto3",
        "dbt-duckdb>=1.5.1",
        "pandas",
        "pyarrow",
    )
    .env(
        {
            "DBT_PROJECT_DIR": PROJ_PATH,
            "DBT_PROFILES_DIR": PROFILES_PATH,
            "DBT_TARGET_PATH": TARGET_PATH,
        }
    )
)
app = modal.App(
    name="example-dbt-duckdb-s3", image=dbt_image
)  # Note: prior to April 2024, "app" was called "stub"

# ## DBT Configuration
#
# Most of the DBT code and configuration is taken directly from the
# https://github.com/dbt-labs/jaffle_shop demo and modified to support
# using dbt-duckdb with an S3 bucket.
#
# The DBT profiles.yml configuration is taken from
# https://github.com/jwills/dbt-duckdb#configuring-your-profile.
#
# Here we mount all this local code and configuration into the Modal function
# so that it will be available when we run DBT in the Modal cloud.

dbt_project = modal.Mount.from_local_dir(
    LOCAL_DBT_PROJECT, remote_path=PROJ_PATH
)
dbt_profiles = modal.Mount.from_local_file(
    local_path=LOCAL_DBT_PROJECT / "profiles.yml",
    remote_path=Path(PROFILES_PATH, "profiles.yml"),
)
dbt_target = modal.NetworkFileSystem.from_name(
    "dbt-target", create_if_missing=True
)
# Create this secret using the "AWS" template at https://modal.com/secrets/create.
# Be sure that the AWS user you provide credentials for has permission to
# create S3 buckets and read/write data from them.
#
# The policy required for this example is the following.
# Not that you *must* update the bucket name listed in the policy to your
# own bucket name.
#
# ```json
# {
#     "Statement": [
#         {
#             "Action": "s3:*",
#             "Effect": "Allow",
#             "Resource": [
#                 "arn:aws:s3:::modal-example-dbt-duckdb-s3/*",
#                 "arn:aws:s3:::modal-example-dbt-duckdb-s3"
#             ],
#             "Sid": "duckdbs3access"
#         }
#     ],
#     "Version": "2012-10-17"
# }
# ```
#
# Below we will use this user in a Modal function to create an S3 bucket and
# populate it with .parquet data.
s3_secret = modal.Secret.from_name("modal-examples-aws-user")

# ## Seed data
#
# In order to provide source data for DBT to ingest and transform,
# we have this `create_source_data` function which creates an AWS S3 bucket and
# populates it with .parquet files based of CSV data in the seeds/ directory.
#
# This is not the typical way that seeds/ data is used, but it is fine for this
# demonstration example. See https://docs.getdbt.com/docs/build/seeds for more info.


@app.function(
    mounts=[dbt_project],
    secrets=[s3_secret],
)
def create_source_data():
    import boto3
    import pandas as pd

    s3_client = boto3.client("s3")
    s3_client.create_bucket(Bucket=BUCKET_NAME)

    for seed_csv_path in Path(PROJ_PATH, "seeds").glob("*.csv"):
        print(f"found seed file {seed_csv_path}")
        name = seed_csv_path.stem
        df = pd.read_csv(seed_csv_path)
        parquet_filename = f"{name}.parquet"
        df.to_parquet(parquet_filename)

        object_key = f"sources/{parquet_filename}"
        print(f"uploading {object_key=} to S3 bucket '{BUCKET_NAME}'")
        s3_client.upload_file(parquet_filename, BUCKET_NAME, object_key)


# This `daily_build` function runs on a schedule to keep the DuckDB data warehouse
# up-to-date. Currently, the source data for this warehouse is static, so the updates
# don't really update anything, just re-build. But this example could be extended
# to have sources which continually provide new data across time.


@app.function(
    schedule=modal.Period(days=1),
    secrets=[s3_secret],
    mounts=[dbt_project, dbt_profiles],
    network_file_systems={TARGET_PATH: dbt_target},
)
def daily_build() -> None:
    run("build")


# `modal run dbt_duckdb.py::run --command run`
#
# A successful run will log something a lot like the following:
#
# ```
# 03:41:04  Running with dbt=1.5.0
# 03:41:05  Found 5 models, 8 tests, 0 snapshots, 0 analyses, 313 macros, 0 operations, 3 seed files, 3 sources, 0 exposures, 0 metrics, 0 groups
# 03:41:05
# 03:41:06  Concurrency: 1 threads (target='modal')
# 03:41:06
# 03:41:06  1 of 5 START sql table model main.stg_customers ................................ [RUN]
# 03:41:06  1 of 5 OK created sql table model main.stg_customers ........................... [OK in 0.45s]
# 03:41:06  2 of 5 START sql table model main.stg_orders ................................... [RUN]
# 03:41:06  2 of 5 OK created sql table model main.stg_orders .............................. [OK in 0.34s]
# 03:41:06  3 of 5 START sql table model main.stg_payments ................................. [RUN]
# 03:41:07  3 of 5 OK created sql table model main.stg_payments ............................ [OK in 0.36s]
# 03:41:07  4 of 5 START sql external model main.customers ................................. [RUN]
# 03:41:07  4 of 5 OK created sql external model main.customers ............................ [OK in 0.72s]
# 03:41:07  5 of 5 START sql table model main.orders ....................................... [RUN]
# 03:41:08  5 of 5 OK created sql table model main.orders .................................. [OK in 0.22s]
# 03:41:08
# 03:41:08  Finished running 4 table models, 1 external model in 0 hours 0 minutes and 3.15 seconds (3.15s).
# 03:41:08  Completed successfully
# 03:41:08
# 03:41:08  Done. PASS=5 WARN=0 ERROR=0 SKIP=0 TOTAL=5
# ```
#


@app.function(
    secrets=[s3_secret],
    mounts=[dbt_project, dbt_profiles],
    network_file_systems={TARGET_PATH: dbt_target},
)
def run(command: str) -> None:
    from dbt.cli.main import dbtRunner

    res = dbtRunner().invoke([command])
    if res.exception:
        print(res.exception)


# Look for the "'materialized='external'" DBT config in the SQL templates
# to see how `dbt-duckdb` is able to write back the transformed data to AWS S3!
#
# After running the 'run' command and seeing it succeed, check what's contained
# under the bucket's `out/` key prefix. You'll see that DBT has run the transformations
# defined in `sample_proj_duckdb_s3/models/` and produced output .parquet files.


---

## cli

import click
from modal import lookup

from .modal_functions import main_app, sync_app


@click.group(name="Kedro-Modal")
def commands():
    """Kedro plugin for running kedro pipelines on Modal"""
    pass


@commands.group(
    name="modal", context_settings=dict(help_option_names=["-h", "--help"])
)
def modal_group():
    """Interact with Kedro pipelines run on Modal"""


@modal_group.command(help="Run kedro project on Modal")
@click.pass_obj
def run(metadata):
    app, remote_project_mount_path, remote_data_path = main_app(
        metadata.project_path, metadata.project_name, metadata.package_name
    )
    with app.run() as app:
        app.sync_data(
            remote_project_mount_path / "data", remote_data_path, reset=False
        )
        app.run_kedro(remote_project_mount_path, remote_data_path)


@modal_group.command(help="Run kedro project on Modal")
@click.pass_obj
def debug(metadata):
    app, remote_project_mount_path, remote_data_path = main_app(
        metadata.project_path, metadata.project_name, metadata.package_name
    )
    app.interactive_shell()


@modal_group.command(
    help="Deploy kedro project to Modal, scheduling it to run daily"
)
@click.pass_obj
def deploy(metadata):
    app, remote_project_mount_point, remote_data_path = main_app(
        metadata.project_path, metadata.project_name, metadata.package_name
    )
    name = f"kedro.{metadata.project_name}"
    app.deploy(name)
    sync_data = lookup(name, "sync_data")  # use the deployed function
    sync_data(remote_project_mount_point / "data", remote_data_path)


@modal_group.command(
    short_help="Sync the local data directory to Modal",
    help="Sync the local data directory to Modal, overwriting any existing data there",
)
@click.pass_obj
def reset(metadata):
    app, source_path, destination_path = sync_app(
        metadata.project_path, metadata.project_name
    )
    with app.run() as app:
        app.sync_data(source_path, destination_path, reset=True)


---

## modal functions

import os
import shutil
import subprocess
import warnings
from pathlib import Path

from modal import App, Image, Mount, NetworkFileSystem, create_package_mounts

package_mounts = create_package_mounts(["kedro_modal"])


def run_kedro(project_path: Path, data_path: Path):
    shutil.rmtree(project_path / "data")  # replace project mounted data dir
    (project_path / "data").symlink_to(data_path)
    os.chdir(project_path)
    subprocess.call(["kedro", "run"])


def sync_data(source: Path, destination: Path, reset: bool = False):
    """Sync a local data directory *to* a network file system"""

    # TODO: only sync raw data - no intermediates etc?
    if destination.exists() and reset:
        shutil.rmtree(destination)
    if not destination.exists():
        shutil.copytree(source, destination)


def non_hidden_files(project_path: Path):
    def condition(path):
        rel = Path(path).relative_to(project_path)
        return not any(
            part != ".gitkeep" and part.startswith(".") for part in rel.parts
        )

    return condition


def main_app(project_path, project_name, package_name) -> App:
    requirements_txt = project_path / "src" / "requirements.txt"

    image = Image.debian_slim()
    if requirements_txt.exists():
        image = image.pip_install_from_requirements(requirements_txt)
    else:
        warnings.warn(
            "No requirements.txt in kedro src dir - attaching no dependencies"
        )
        image = image.pip_install("kedro")

    remote_project_mount_point = Path(f"/kedro-project/{package_name}")
    kedro_proj_mount = Mount(
        remote_dir=remote_project_mount_point,
        local_dir=project_path,
        condition=non_hidden_files(project_path),
    )
    app = App(
        f"kedro-run.{project_name}",
        image=image,
        mounts=[kedro_proj_mount] + package_mounts,
    )  # Note: prior to April 2024, "app" was called "stub"
    volume_name = f"kedro.{project_name}.storage"
    data_volume = NetworkFileSystem.from_name(volume_name, create_if_true=True)

    app.function(network_file_systems={"/kedro-storage": data_volume})(
        run_kedro
    )
    app.function(network_file_systems={"/kedro-storage": data_volume})(
        sync_data
    )
    remote_data_path = Path("/kedro-storage/data")
    return app, remote_project_mount_point, remote_data_path


def sync_app(project_path, project_name):
    # slimmer sync app that only mounts the data dir in order to upload raw data
    app = App(
        f"kedro-data-sync.{project_name}"
    )  # Note: prior to April 2024, "app" was called "stub"
    volume_name = f"kedro.{project_name}.storage"
    data_volume = NetworkFileSystem().persist(volume_name)

    remote_source_path = Path("/source-data")
    source_mount = Mount(
        remote_dir=remote_source_path,
        local_dir=project_path / "data",
        condition=non_hidden_files(project_path),
    )
    app.function(
        mounts=[source_mount] + package_mounts,
        network_file_systems={"/kedro-storage": data_volume},
    )(sync_data)
    remote_destination_path = Path("/kedro-storage/data")
    return app, remote_source_path, remote_destination_path


---

## multion news agent

# ---
# lambda-test: false
# ---
# # MultiOn: Twitter News Agent

# In this example, we use Modal to deploy a cron job that periodically checks for AI news everyday and tweets it on Twitter using the MultiOn Agent API.

# ## Import and define the app
#
# Let's start off with imports, and defining a Modal app.

import os

import modal

app = modal.App(
    "multion-news-tweet-agent"
)  # Note: prior to April 2024, "app" was called "stub"

# ## Searching for AI News
#


# Let's also define an image that has the `multion` package installed, so we can query the API.

multion_image = modal.Image.debian_slim().pip_install("multion")

# We can now define our main entrypoint, that uses [MultiOn](https://www.multion.ai/) to scrape AI news everyday and post it on our twitter account. We specify a [schedule](/docs/guide/cron) in the function decorator, which
# means that our function will run automatically at the given interval.

# ## Set up MultiOn
#
# [MultiOn](https://multion.ai/) is a next-gen Web Action Agent that can take actions on behalf of the user. You can watch it in action here: [Youtube demo](https://www.youtube.com/watch?v=Rm67ry6bogw).
#
# The MultiOn API enables building the next level of web automation & custom AI agents capable of performing complex actions on the internet with just a few lines of code.
#
# To get started, first create an account with [MultiOn](https://app.multion.ai/), install the [MultiOn chrome extension](https://chrome.google.com/webstore/detail/ddmjhdbknfidiopmbaceghhhbgbpenmm) and login to your Twitter account in your browser.
# To use the API create a [MultiOn API Key](https://app.multion.ai/api-keys) and store it as a modal secret on [the dashboard](https://modal.com/secrets)


@app.function(
    image=multion_image, secrets=[modal.Secret.from_name("MULTION_API_KEY")]
)
def news_tweet_agent():
    # Import MultiOn
    import multion

    # Login to MultiOn using the API key
    multion.login(use_api=True, multion_api_key=os.environ["MULTION_API_KEY"])

    # Enable the Agent to run locally
    multion.set_remote(False)

    params = {
        "url": "https://www.multion.ai",
        "cmd": "Go to twitter (im already signed in). Search for the last tweets i made (check the last 10 tweets). Remember them so then you can go a search for super interesting AI news. Search the news on up to 3 different sources. If you see that the source has not really interesting AI news or i already made a tweet about that, then go to a different one. When you finish the research, go and make a few small and interesting AI tweets with the info you gathered. Make sure the tweet is small but informative and interesting for AI enthusiasts. Don't do more than 5 tweets",
        "maxSteps": 100,
    }

    response = multion.browse(params)

    print(f"MultiOn response: {response}")


# ## Test running
#
# We can now test run our scheduled function as follows: `modal run multion_news_agent.py.py::app.news_tweet_agent`

# ## Defining the schedule and deploying
#
# Let's define a function that will be called by Modal every day.


@app.function(schedule=modal.Cron("0 9 * * *"))
def run_daily():
    news_tweet_agent.remote()


# In order to deploy this as a persistent cron job, you can run `modal deploy multion_news_agent.py`.

# Once the job is deployed, visit the [apps page](/apps) page to see
# its execution history, logs and other stats.


---

## pyjulia

import modal

image = image = (
    modal.Image.debian_slim()
    # Install Julia 1.10
    .apt_install("wget", "ca-certificates")
    .run_commands(
        "wget -nv https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.0-linux-x86_64.tar.gz",
        "tar -xf julia-1.10.0-linux-x86_64.tar.gz",
        "cp -r julia-1.10.0 /opt/",
        "ln -s /opt/julia-1.10.0/bin/julia /usr/local/bin/julia",
    )
    # Install PyJulia bindings
    .pip_install("julia")
    .run_commands('python -c "import julia; julia.install()"')
)
app = modal.App(
    "example-pyjulia", image=image
)  # Note: prior to April 2024, "app" was called "stub"


@app.function()
def julia_subprocess():
    """Run the Julia interpreter as a subprocess."""
    import subprocess

    print("-> Calling Julia as a subprocess...")
    subprocess.run('julia -e "println(2 + 3)"', shell=True)


@app.function()
def julia_matrix_determinant():
    """Compute the determinant of a random matrix with PyJulia."""
    from julia.Base import rand
    from julia.LinearAlgebra import det

    print("-> Calling Julia using PyJulia...")
    print(det(rand(5, 5)))
    print(det(rand(10, 10)))


@app.local_entrypoint()
def run():
    julia_subprocess.remote()
    julia_matrix_determinant.remote()


---

## s3 bucket mount

# ---
# output-directory: "/tmp/s3_bucket_mount"
# ---
# # Analyze NYC yellow taxi data with DuckDB on Parquet files from S3
#
# This example shows how to use Modal for a classic data science task: loading table-structured data into cloud stores,
# analyzing it, and plotting the results.
#
# In particular, we'll load public NYC taxi ride data into S3 as Parquet files,
# then run SQL queries on it with DuckDB.
#
# We'll mount the S3 bucket in a Modal app with [`CloudBucketMount`](https://modal.com/docs/reference/modal.CloudBucketMount).
# We will write to and then read from that bucket, in each case using
# Modal's [parallel execution features](https://modal.com/docs/guide/scale) to handle many files at once.
#
# ## Basic setup
#
# You will need to have an S3 bucket and AWS credentials to run this example. Refer to the documentation
# for the exact [IAM permissions](https://modal.com/docs/guide/cloud-bucket-mounts#iam-permissions) your credentials will need.
#
# After you are done creating a bucket and configuring IAM settings,
# you now need to create a [`Secret`](https://modal.com/docs/guide/secrets) to share
# the relevant AWS credentials with your Modal apps. Navigate to the "Secrets" tab and
# click on the AWS card, then fill in the fields with your credentials.
# Name the secret `s3-bucket-secret`.

from datetime import datetime
from pathlib import Path

from modal import App, CloudBucketMount, Image, Secret

image = Image.debian_slim().pip_install(
    "requests==2.31.0", "duckdb==0.10.0", "matplotlib==3.8.3"
)
app = App(image=image)  # Note: prior to April 2024, "app" was called "stub"

MOUNT_PATH: Path = Path("/bucket")
YELLOW_TAXI_DATA_PATH: Path = MOUNT_PATH / "yellow_taxi"


# The dependencies installed above are not available locally. The following block instructs Modal
# to only import them inside the container.
with image.imports():
    import duckdb
    import requests


# ## Download New York City's taxi data
#
# NYC makes data about taxi rides publicly available. The city's [Taxi & Limousine Commission (TLC)](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
# publishes files in the Parquet format. Files are organized by year and month.
#
# We are going to download all available files and store them in an S3 bucket. We do this by
# attaching a `modal.CloudBucketMount` with the S3 bucket name and its respective credentials.
# The files in the bucket will then be available at `MOUNT_PATH`.
#
# As we'll see below, this operation can be massively sped up by running it in parallel on Modal.
@app.function(
    volumes={
        MOUNT_PATH: CloudBucketMount(
            "modal-s3mount-test-bucket",
            secret=Secret.from_name("s3-bucket-secret"),
        )
    },
)
def download_data(year: int, month: int) -> str:
    filename = f"yellow_tripdata_{year}-{month:02d}.parquet"
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{filename}"
    s3_path = MOUNT_PATH / filename
    # Skip downloading if file exists.
    if not s3_path.exists():
        if not YELLOW_TAXI_DATA_PATH.exists():
            YELLOW_TAXI_DATA_PATH.mkdir(parents=True, exist_ok=True)
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                print(f"downloading => {s3_path}")
                # It looks like we writing locally, but this is actually writing to S3!
                with open(s3_path, "wb") as file:
                    for chunk in r.iter_content(chunk_size=8192):
                        file.write(chunk)

    return s3_path.as_posix()


# ## Analyze data with DuckDB
#
# [DuckDB](https://duckdb.org/) is an analytical database with rich support for Parquet files.
# It is also very fast. Below, we define a Modal Function that aggregates yellow taxi trips
# within a month (each file contains all the rides from a specific month).
@app.function(
    volumes={
        MOUNT_PATH: CloudBucketMount(
            "modal-s3mount-test-bucket",
            secret=Secret.from_name("s3-bucket-secret"),
        )
    },
)
def aggregate_data(path: str) -> list[tuple[datetime, int]]:
    print(f"processing => {path}")

    # Parse file.
    year_month_part = path.split("yellow_tripdata_")[1]
    year, month = year_month_part.split("-")
    month = month.replace(".parquet", "")

    # Make DuckDB query using in-memory storage.
    con = duckdb.connect(database=":memory:")
    q = """
    with sub as (
        select tpep_pickup_datetime::date d, count(1) c
        from read_parquet(?)
        group by 1
    )
    select d, c from sub
    where date_part('year', d) = ?  -- filter out garbage
    and date_part('month', d) = ?   -- same
    """
    con.execute(q, (path, year, month))
    return list(con.fetchall())


# ## Plot daily taxi rides
#
# Finally, we want to plot our results.
# The plot created shows the number of yellow taxi rides per day in NYC.
# This function runs remotely, on Modal, so we don't need to install plotting libraries locally.
@app.function()
def plot(dataset) -> bytes:
    import io

    import matplotlib.pyplot as plt

    # Sorting data by date
    dataset.sort(key=lambda x: x[0])

    # Unpacking dates and values
    dates, values = zip(*dataset)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(dates, values)
    plt.title("Number of NYC yellow taxi trips by weekday, 2018-2023")
    plt.ylabel("Number of daily trips")
    plt.grid(True)
    plt.tight_layout()

    # Saving plot as raw bytes to send back
    buf = io.BytesIO()

    plt.savefig(buf, format="png")

    buf.seek(0)

    return buf.getvalue()


# ## Run everything
#
# The `@app.local_entrypoint()` defines what happens when we run our Modal program locally.
# We invoke it from the CLI by calling `modal run s3_bucket_mount.py`.
# We first call `download_data()` and `starmap` (named because it's kind of like `map(*args)`)
# on tuples of inputs `(year, month)`. This will download, in parallel,
# all yellow taxi data files into our locally mounted S3 bucket and return a list of
# Parquet file paths. Then, we call `aggregate_data()` with `map` on that list. These files are
# also read from our S3 bucket. So one function writes files to S3 and the other
# reads files from S3 in; both run across many files in parallel.
#
# Finally, we call `plot` to generate the following figure:
#
# ![Number of NYC yellow taxi trips by weekday, 2018-2023](./nyc_yellow_taxi_trips_s3_mount.png)
#
# This program should run in less than 30 seconds.
@app.local_entrypoint()
def main():
    # List of tuples[year, month].
    inputs = [
        (year, month) for year in range(2018, 2023) for month in range(1, 13)
    ]

    # List of file paths in S3.
    parquet_files: list[str] = []
    for path in download_data.starmap(inputs):
        print(f"done => {path}")
        parquet_files.append(path)

    # List of datetimes and number of yellow taxi trips.
    dataset = []
    for r in aggregate_data.map(parquet_files):
        dataset += r

    dir = Path("/tmp") / "s3_bucket_mount"
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    figure = plot.remote(dataset)
    path = dir / "nyc_yellow_taxi_trips_s3_mount.png"
    with open(path, "wb") as file:
        print(f"Saving figure to {path}")
        file.write(figure)


---

## app

# ---
# lambda-test: false
# ---
# ## Demo Streamlit application.
#
# This application is the example from https://docs.streamlit.io/library/get-started/create-an-app.
#
# Streamlit is designed to run its apps as Python scripts, not functions, so we separate the Streamlit
# code into this module, away from the Modal application code.


def main():
    import numpy as np
    import pandas as pd
    import streamlit as st

    st.title("Uber pickups in NYC!")

    DATE_COLUMN = "date/time"
    DATA_URL = (
        "https://s3-us-west-2.amazonaws.com/"
        "streamlit-demo-data/uber-raw-data-sep14.csv.gz"
    )

    @st.cache_data
    def load_data(nrows):
        data = pd.read_csv(DATA_URL, nrows=nrows)

        def lowercase(x):
            return str(x).lower()

        data.rename(lowercase, axis="columns", inplace=True)
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
        return data

    data_load_state = st.text("Loading data...")
    data = load_data(10000)
    data_load_state.text("Done! (using st.cache_data)")

    if st.checkbox("Show raw data"):
        st.subheader("Raw data")
        st.write(data)

    st.subheader("Number of pickups by hour")
    hist_values = np.histogram(
        data[DATE_COLUMN].dt.hour, bins=24, range=(0, 24)
    )[0]
    st.bar_chart(hist_values)

    # Some number in the range 0-23
    hour_to_filter = st.slider("hour", 0, 23, 17)
    filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

    st.subheader("Map of all pickups at %s:00" % hour_to_filter)
    st.map(filtered_data)


if __name__ == "__main__":
    main()


---

## serve streamlit

# ---
# lambda-test: false
# cmd: ["modal", "serve", "10_integrations/streamlit/serve_streamlit.py"]
# ---
#
# # Run and share Streamlit apps
#
# This example shows you how to run a Streamlit app with `modal serve`, and then deploy it as a serverless web app.
#
# ![example streamlit app](./streamlit.png)
#
# This example is structured as two files:
#
# 1. This module, which defines the Modal objects (name the script `serve_streamlit.py` locally).
# 2. `app.py`, which is any Streamlit script to be mounted into the Modal
# function ([download script](https://github.com/modal-labs/modal-examples/blob/main/10_integrations/streamlit/app.py)).

import shlex
import subprocess
from pathlib import Path

import modal

# ## Define container dependencies
#
# The `app.py` script imports three third-party packages, so we include these in the example's
# image definition.

image = modal.Image.debian_slim().pip_install("streamlit", "numpy", "pandas")

app = modal.App(
    name="example-modal-streamlit", image=image
)  # Note: prior to April 2024, "app" was called "stub"

# ## Mounting the `app.py` script
#
# We can just mount the `app.py` script inside the container at a pre-defined path using a Modal
# [`Mount`](https://modal.com/docs/guide/local-data#mounting-directories).

streamlit_script_local_path = Path(__file__).parent / "app.py"
streamlit_script_remote_path = Path("/root/app.py")

if not streamlit_script_local_path.exists():
    raise RuntimeError(
        "app.py not found! Place the script with your streamlit app in the same directory."
    )

streamlit_script_mount = modal.Mount.from_local_file(
    streamlit_script_local_path,
    streamlit_script_remote_path,
)

# ## Spawning the Streamlit server
#
# Inside the container, we will run the Streamlit server in a background subprocess using
# `subprocess.Popen`. We also expose port 8000 using the `@web_server` decorator.


@app.function(
    allow_concurrent_inputs=100,
    mounts=[streamlit_script_mount],
)
@modal.web_server(8000)
def run():
    target = shlex.quote(str(streamlit_script_remote_path))
    cmd = f"streamlit run {target} --server.port 8000 --server.enableCORS=false --server.enableXsrfProtection=false"
    subprocess.Popen(cmd, shell=True)


# ## Iterate and Deploy
#
# While you're iterating on your screamlit app, you can run it "ephemerally" with `modal serve`. This will
# run a local process that watches your files and updates the app if anything changes.
#
# ```shell
# modal serve serve_streamlit.py
# ```
#
# Once you're happy with your changes, you can deploy your application with
#
# ```shell
# modal deploy serve_streamlit.py
# ```
#
# If successful, this will print a URL for your app, that you can navigate to from
# your browser 🎉 .


---

## modal tailscale

# ---
# lambda-test: false
# ---

# # Add Modal Apps to Tailscale
#
# This example demonstrates how to integrate Modal with Tailscale (https://tailscale.com).
# It outlines the steps to configure Modal containers so that they join the Tailscale network.

# We use a custom entrypoint to automatically add containers to a Tailscale network (tailnet).
# This configuration enables the containers to interact with one another and with
# additional applications within the same tailnet.


import modal

# Install Tailscale and copy custom entrypoint script ([entrypoint.sh](https://github.com/modal-labs/modal-examples/blob/main/10_integrations/tailscale/entrypoint.sh)). The script must be
# executable.
image = (
    modal.Image.debian_slim()
    .apt_install("curl")
    .run_commands("curl -fsSL https://tailscale.com/install.sh | sh")
    .pip_install("requests[socks]")
    .copy_local_file("./entrypoint.sh", "/root/entrypoint.sh")
    .dockerfile_commands(
        "RUN chmod a+x /root/entrypoint.sh",
        'ENTRYPOINT ["/root/entrypoint.sh"]',
    )
)
app = modal.App(
    image=image
)  # Note: prior to April 2024, "app" was called "stub"


# Run your function adding a Tailscale secret. It expects an environment variable
# named `TAILSCALE_AUTHKEY`. We suggest creating a [reusable and ephemeral key](https://tailscale.com/kb/1111/ephemeral-nodes).
@app.function(
    secrets=[
        modal.Secret.from_name("tailscale-auth"),
        modal.Secret.from_dict(
            {
                "ALL_PROXY": "socks5://localhost:1080/",
                "HTTP_PROXY": "http://localhost:1080/",
                "http_proxy": "http://localhost:1080/",
            }
        ),
    ],
)
def connect_to_raspberrypi():
    import requests

    # Connect to other machines in your tailnet.
    resp = requests.get("http://raspberrypi:5000")
    print(resp.content)


# Run this script with `modal run modal_tailscale.py`. You will see Tailscale logs
# when the container start indicating that you were able to login successfully and
# that the proxies (SOCKS5 and HTTP) have created been successfully. You will also
# be able to see Modal containers in your Tailscale dashboard in the "Machines" tab.
# Every new container launched will show up as a new "machine". Containers are
# individually addressable using their Tailscale name or IP address.


---

## webscraper

# ---
# runtimes: ["runc", "gvisor"]
# ---
import os

import modal

app = modal.App(
    "example-linkscraper"
)  # Note: prior to April 2024, "app" was called "stub"


playwright_image = modal.Image.debian_slim(
    python_version="3.10"
).run_commands(  # Doesn't work with 3.11 yet
    "apt-get update",
    "apt-get install -y software-properties-common",
    "apt-add-repository non-free",
    "apt-add-repository contrib",
    "pip install playwright==1.42.0",
    "playwright install-deps chromium",
    "playwright install chromium",
)


@app.function(image=playwright_image)
async def get_links(url: str) -> set[str]:
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url)
        links = await page.eval_on_selector_all(
            "a[href]", "elements => elements.map(element => element.href)"
        )
        await browser.close()

    return set(links)


slack_sdk_image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "slack-sdk==3.27.1"
)


@app.function(
    image=slack_sdk_image,
    secrets=[modal.Secret.from_name("scraper-slack-secret")],
)
def bot_token_msg(channel, message):
    import slack_sdk
    from slack_sdk.http_retry.builtin_handlers import RateLimitErrorRetryHandler

    client = slack_sdk.WebClient(token=os.environ["SLACK_BOT_TOKEN"])
    rate_limit_handler = RateLimitErrorRetryHandler(max_retry_count=3)
    client.retry_handlers.append(rate_limit_handler)

    print(f"Posting {message} to #{channel}")
    client.chat_postMessage(channel=channel, text=message)


@app.function()
def scrape():
    links_of_interest = ["http://modal.com"]

    for links in get_links.map(links_of_interest):
        for link in links:
            bot_token_msg.remote("scraped-links", link)


@app.function(schedule=modal.Period(days=1))
def daily_scrape():
    scrape.remote()


@app.local_entrypoint()
def run():
    scrape.remote()


---

## jupyter inside modal

# ---
# args: ["--timeout", 10]
# ---

# ## Overview
#
# Quick snippet showing how to connect to a Jupyter notebook server running inside a Modal container,
# especially useful for exploring the contents of Modal Volumes.
# This uses [Modal Tunnels](https://modal.com/docs/guide/tunnels#tunnels-beta)
# to create a tunnel between the running Jupyter instance and the internet.
#
# If you want to your Jupyter notebook to run _locally_ and execute remote Modal Functions in certain cells, see the `basic.ipynb` example :)

import os
import subprocess
import time

import modal

app = modal.App(
    image=modal.Image.debian_slim().pip_install(
        "jupyter", "bing-image-downloader~=1.1.2"
    )  # Note: prior to April 2024, "app" was called "stub"
)
volume = modal.Volume.from_name(
    "modal-examples-jupyter-inside-modal-data", create_if_missing=True
)

CACHE_DIR = "/root/cache"
JUPYTER_TOKEN = "1234"  # Change me to something non-guessable!


@app.function(volumes={CACHE_DIR: volume})
def seed_volume():
    # Bing it!
    from bing_image_downloader import downloader

    # This will save into the Modal volume and allow you view the images
    # from within Jupyter at a path like `/root/cache/modal labs/Image_1.png`.
    downloader.download(
        query="modal labs",
        limit=10,
        output_dir=CACHE_DIR,
        force_replace=False,
        timeout=60,
        verbose=True,
    )
    volume.commit()


# This is all that's needed to create a long-lived Jupyter server process in Modal
# that you can access in your Browser through a secure network tunnel.
# This can be useful when you want to interactively engage with Volume contents
# without having to download it to your host computer.


@app.function(concurrency_limit=1, volumes={CACHE_DIR: volume}, timeout=1_500)
def run_jupyter(timeout: int):
    jupyter_port = 8888
    with modal.forward(jupyter_port) as tunnel:
        jupyter_process = subprocess.Popen(
            [
                "jupyter",
                "notebook",
                "--no-browser",
                "--allow-root",
                "--ip=0.0.0.0",
                f"--port={jupyter_port}",
                "--NotebookApp.allow_origin='*'",
                "--NotebookApp.allow_remote_access=1",
            ],
            env={**os.environ, "JUPYTER_TOKEN": JUPYTER_TOKEN},
        )

        print(f"Jupyter available at => {tunnel.url}")

        try:
            end_time = time.time() + timeout
            while time.time() < end_time:
                time.sleep(5)
            print(f"Reached end of {timeout} second timeout period. Exiting...")
        except KeyboardInterrupt:
            print("Exiting...")
        finally:
            jupyter_process.kill()


@app.local_entrypoint()
def main(timeout: int = 10_000):
    # Write some images to a volume, for demonstration purposes.
    seed_volume.remote()
    # Run the Jupyter Notebook server
    run_jupyter.remote(timeout=timeout)


# Doing `modal run jupyter_inside_modal.py` will run a Modal app which starts
# the Juypter server at an address like https://u35iiiyqp5klbs.r3.modal.host.
# Visit this address in your browser, and enter the security token
# you set for `JUPYTER_TOKEN`.


---

## deploy

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple, Optional

from utils import ExampleType, get_examples


class DeployError(NamedTuple):
    stdout: str
    stderr: str
    code: int


def deploy(
    deployable: bool,
    module_with_app: Path,
    dry_run: bool,
    filter_pttrn: Optional[str],
) -> Optional[DeployError]:
    if filter_pttrn and not re.match(filter_pttrn, module_with_app.name):
        return None

    if not deployable:
        print(f"⏩ skipping: '{module_with_app.name}' is not marked for deploy")
        return None

    deploy_command = f"modal deploy {module_with_app.name}"
    if dry_run:
        print(f"🌵  dry-run: '{module_with_app.name}' would have deployed")
    else:
        print(f"⛴ deploying: '{module_with_app.name}' ...")
        r = subprocess.run(
            shlex.split(deploy_command),
            cwd=module_with_app.parent,
            capture_output=True,
        )
        if r.returncode != 0:
            print(
                f"⚠️ deployment failed: '{module_with_app.name}'",
                file=sys.stderr,
            )
            print(r.stderr)
            return DeployError(
                stdout=r.stdout, stderr=r.stderr, code=r.returncode
            )
        else:
            print(f"✔️ deployed '{module_with_app.name}")
    return None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deploy Modal example programs to our Modal organization.",
        add_help=True,
    )
    parser.add_argument(
        "--dry-run",
        default=True,
        action="store_true",
        help="show what apps be deployed without deploying them.",
    )
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    parser.add_argument(
        "--filter",
        default=None,
        help="Filter which apps are deployed with basic pattern matching. eg. 'cron' matches 'say_hello_cron.py'.",
    )
    arguments = parser.parse_args()

    if arguments.dry_run:
        print(
            "INFO: dry-run is active. Intended deployments will be displayed to console."
        )

    example_modules = (
        ex for ex in get_examples() if ex.type == ExampleType.MODULE
    )
    filter_pttrn = (
        (r".*" + arguments.filter + r".*") if arguments.filter else None
    )
    results = [
        deploy(
            deployable=bool(ex_mod.metadata.get("deploy")),
            module_with_app=Path(ex_mod.filename),
            dry_run=arguments.dry_run,
            filter_pttrn=filter_pttrn,
        )
        for ex_mod in example_modules
    ]

    failures = [r for r in results if r]
    if any(failures):
        print(f"ERROR: {len(failures)} deployment failures.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


---

## examples test

import importlib
import json
import pathlib
import sys

import pytest
from utils import (
    EXAMPLES_ROOT,
    ExampleType,
    get_examples,
    get_examples_json,
    render_example_md,
)

examples = [ex for ex in get_examples() if ex.type == ExampleType.MODULE]
example_ids = [ex.module for ex in examples]


@pytest.fixture(autouse=False)
def add_root_to_syspath(monkeypatch):
    sys.path.append(str(EXAMPLES_ROOT))
    yield
    sys.path.pop()


@pytest.mark.parametrize("example", examples, ids=example_ids)
def test_filename(example):
    assert not example.repo_filename.startswith("/")
    assert pathlib.Path(example.repo_filename).exists()


@pytest.mark.parametrize("example", examples, ids=example_ids)
def test_import(example, add_root_to_syspath):
    importlib.import_module(example.module)


@pytest.mark.parametrize("example", examples, ids=example_ids)
def test_render(example):
    md = render_example_md(example)
    assert isinstance(md, str)
    assert len(md) > 0


def test_json():
    data = get_examples_json()
    examples = json.loads(data)
    assert isinstance(examples, list)
    assert len(examples) > 0


---

## typecheck

"""
MyPy type-checking script.
Unvalidated, incorrect type-hints are worse than no type-hints!
"""

import concurrent
import os
import pathlib
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import mypy.api


def fetch_git_repo_root() -> pathlib.Path:
    return pathlib.Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        .decode("ascii")
        .strip()
    )


def run_mypy(pkg: str, config_file: pathlib.Path) -> list[str]:
    args = [
        pkg,
        "--no-incremental",
        "--namespace-packages",
        "--config-file",
        str(config_file),
    ]
    result = mypy.api.run(args)
    return result[0].splitlines()


def extract_errors(output: list[str]) -> list[str]:
    if len(output) > 0 and "success" in output[0].lower():
        print(output[0], file=sys.stderr)
        return []
    return [l for l in output if "error" in l]


def main() -> int:
    repo_root = fetch_git_repo_root()
    config_file = repo_root / "pyproject.toml"
    errors = []

    # Type-check scripts
    topic_dirs = sorted(
        [d for d in repo_root.iterdir() if d.name[:2].isdigit()]
    )

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_path = {}
        for topic_dir in topic_dirs:
            for pth in topic_dir.iterdir():
                if not (pth.is_file() and pth.name.endswith(".py")):
                    continue
                elif "__pycache__" in pth.parts:
                    continue
                else:
                    print(f"⌛️ spawning mypy on '{pth}'", file=sys.stderr)
                    future = executor.submit(
                        run_mypy, pkg=str(pth), config_file=config_file
                    )
                    future_to_path[future] = pth

        for future in concurrent.futures.as_completed(
            future_to_path, timeout=60
        ):
            pth = future_to_path[future]
            try:
                output = future.result()
                topic_errors = extract_errors(output)
                if topic_errors:
                    print(f"\nfound {len(topic_errors)} errors in '{pth}'")
                    print("\n".join(topic_errors))
                    errors.extend(topic_errors)
            except Exception as exc:
                print(f"Error on file {pth}: {exc}")
                errors.append(exc)

    # Type-check packages
    # Getting mypy running successfully with a monorepo of heterogenous packaging structures
    # is a bit fiddly, so we expect top-level packages to opt-in to type-checking by placing a
    # `py.typed` file inside themselves. https://peps.python.org/pep-0561/
    for py_typed in repo_root.glob("**/py.typed"):
        if "site-packages" in py_typed.parts:
            continue
        toplevel_pkg = py_typed.parent
        print(f"⌛️ running mypy on '{toplevel_pkg}'", file=sys.stderr)
        package_errors = extract_errors(
            run_mypy(
                pkg=str(toplevel_pkg),
                config_file=config_file,
            )
        )
        if package_errors:
            print(
                f"found {len(package_errors)} errors in '{toplevel_pkg}'",
                file=sys.stderr,
            )
            print("\n".join(package_errors))
            errors.extend(package_errors)

    if errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


---

## utils

import json
import re
import warnings
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional

from pydantic import BaseModel

EXAMPLES_ROOT = Path(__file__).parent.parent


with warnings.catch_warnings():
    # This triggers some dumb warning in jupyter_core
    warnings.simplefilter("ignore")
    import jupytext
    import jupytext.config


class ExampleType(int, Enum):
    MODULE = 1
    ASSET = 2


class Example(BaseModel):
    type: ExampleType
    filename: str  # absolute filepath to example file
    module: Optional[
        str
    ] = None  # python import path, or none if file is not a py module.
    # TODO(erikbern): don't think the module is used (by docs or monitors)?
    metadata: Optional[dict] = None
    repo_filename: str  # git repo relative filepath
    cli_args: Optional[list] = None  # Full command line args to run it
    stem: Optional[str] = None  # stem of path


_RE_NEWLINE = re.compile(r"\r?\n")
_RE_FRONTMATTER = re.compile(r"^---$", re.MULTILINE)
_RE_CODEBLOCK = re.compile(r"\s*```[^`]+```\s*", re.MULTILINE)


def render_example_md(example: Example) -> str:
    """Render a Python code example to Markdown documentation format."""

    with open(example.filename) as f:
        content = f.read()

    lines = _RE_NEWLINE.split(content)
    markdown: list[str] = []
    code: list[str] = []
    for line in lines:
        if line == "#" or line.startswith("# "):
            if code:
                markdown.extend(["```python", *code, "```", ""])
                code = []
            markdown.append(line[2:])
        else:
            if markdown and markdown[-1]:
                markdown.append("")
            if code or line:
                code.append(line)

    if code:
        markdown.extend(["```python", *code, "```", ""])

    text = "\n".join(markdown)
    if _RE_FRONTMATTER.match(text):
        # Strip out frontmatter from text.
        if match := _RE_FRONTMATTER.search(text, 4):
            text = text[match.end() + 1 :]

    if match := _RE_CODEBLOCK.match(text):
        filename = Path(example.filename).name
        if match.end() == len(text):
            # Special case: The entire page is a single big code block.
            text = f"""# Example ({filename})

This is the source code for **{example.module}**.
{text}"""

    return text


def gather_example_files(
    parents: list[str], subdir: Path, ignored: list[str], recurse: bool
) -> Iterator[Example]:
    config = jupytext.config.JupytextConfiguration(
        root_level_metadata_as_raw_cell=False
    )

    for filename in sorted(list(subdir.iterdir())):
        if filename.is_dir() and recurse:
            # Gather two-subdirectories deep, but no further.
            yield from gather_example_files(
                parents + [str(subdir.stem)], filename, ignored, recurse=False
            )
        else:
            filename_abs: str = str(filename.resolve())
            ext: str = filename.suffix
            if parents:
                repo_filename: str = (
                    f"{'/'.join(parents)}/{subdir.name}/{filename.name}"
                )
            else:
                repo_filename: str = f"{subdir.name}/{filename.name}"

            if ext == ".py" and filename.stem != "__init__":
                if parents:
                    parent_mods = ".".join(parents)
                    module = f"{parent_mods}.{subdir.stem}.{filename.stem}"
                else:
                    module = f"{subdir.stem}.{filename.stem}"
                data = jupytext.read(open(filename_abs), config=config)
                metadata = data["metadata"]["jupytext"].get(
                    "root_level_metadata", {}
                )
                cmd = metadata.get("cmd", ["modal", "run", repo_filename])
                args = metadata.get("args", [])
                yield Example(
                    type=ExampleType.MODULE,
                    filename=filename_abs,
                    metadata=metadata,
                    module=module,
                    repo_filename=repo_filename,
                    cli_args=(cmd + args),
                    stem=Path(filename_abs).stem,
                )
            elif ext in [".png", ".jpeg", ".jpg", ".gif", ".mp4"]:
                yield Example(
                    type=ExampleType.ASSET,
                    filename=filename_abs,
                    repo_filename=repo_filename,
                )
            else:
                ignored.append(str(filename))


def get_examples() -> Iterator[Example]:
    """Yield all Python module files and asset files relevant to building modal.com/docs."""
    if not EXAMPLES_ROOT.exists():
        raise Exception(
            f"Can't find directory {EXAMPLES_ROOT}. You might need to clone the modal-examples repo there."
        )

    ignored = []
    for subdir in sorted(
        p
        for p in EXAMPLES_ROOT.iterdir()
        if p.is_dir()
        and not p.name.startswith(".")
        and not p.name.startswith("internal")
        and not p.name.startswith("misc")
    ):
        yield from gather_example_files(
            parents=[], subdir=subdir, ignored=ignored, recurse=True
        )


def get_examples_json():
    examples = list(ex.dict() for ex in get_examples())
    return json.dumps(examples)


if __name__ == "__main__":
    for example in get_examples():
        print(example.json())


---

## batch inference using huggingface

# ---
# runtimes: ["runc", "gvisor"]
# ---
# # Batch inference using a model from Huggingface
#
# <center>
#   <img src="./batch_inference_huggingface.png" alt="Huggingface company logo" />
# </center>
#
# This example shows how to use a sentiment analysis model from Huggingface to classify
# 25,000 movie reviews in a couple of minutes.
#
# Some Modal features it uses:
# * Container lifecycle hook: this lets us load the model only once in each container
# * CPU requests: the prediction function is very CPU-hungry, so we reserve 8 cores
# * Mapping: we map over 25,000 sentences and Modal manages the pool of containers for us
#
# ## Basic setup
#
# Let's get started writing code.
# For the Modal container image, we need a few Python packages,
# including `transformers`, which is the main Huggingface package.

import io

import modal

app = modal.App(
    "example-batch-inference-using-huggingface",
    image=modal.Image.debian_slim().pip_install(
        "datasets",
        "matplotlib",
        "scikit-learn",
        "torch",
        "transformers",
    ),
)  # Note: prior to April 2024, "app" was called "stub"

# ## Defining the prediction function
#
# Instead of a using `@app.function()` in the global scope,
# we put the method on a class, and define a setup method that we
# decorate with `@modal.enter()`.
#
# Modal reuses containers for successive calls to the same function, so
# we want to take advantage of this and avoid setting up the same model
# for every function call.
#
# Since the transformer model is very CPU-hungry, we allocate 8 CPUs
# to the model. Every container that runs will have 8 CPUs set aside for it.


@app.cls(cpu=8, retries=3)
class SentimentAnalysis:
    @modal.enter()
    def setup_pipeline(self):
        from transformers import pipeline

        self.sentiment_pipeline = pipeline(
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    @modal.method()
    def predict(self, phrase: str):
        pred = self.sentiment_pipeline(
            phrase, truncation=True, max_length=512, top_k=2
        )
        # pred will look like: [{'label': 'NEGATIVE', 'score': 0.99}, {'label': 'POSITIVE', 'score': 0.01}]
        probs = {p["label"]: p["score"] for p in pred}
        return probs["POSITIVE"]


# ## Getting data
#
# We need some data to run the batch inference on.
# We use this [dataset of IMDB reviews](https://ai.stanford.edu/~amaas/data/sentiment/) for this purpose.
# Huggingface actually offers this data [as a preprocessed dataaset](https://huggingface.co/datasets/imdb),
# which we can download using the `datasets` package:


@app.function()
def get_data():
    from datasets import load_dataset

    imdb = load_dataset("imdb")
    data = [(row["text"], row["label"]) for row in imdb["test"]]
    return data


# ## Plotting the ROC curve
#
# In order to evaluate the classifier, let's plot an
# [ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic).
# This is a common way to evaluate classifiers on binary data.


@app.function()
def roc_plot(labels, predictions):
    from matplotlib import pyplot
    from sklearn.metrics import RocCurveDisplay

    pyplot.style.use("ggplot")
    RocCurveDisplay.from_predictions(labels, predictions)
    with io.BytesIO() as buf:
        pyplot.savefig(buf, format="png")
        return buf.getvalue()


# A bit of a spoiler warning, but if you run this script, the ROC curve will look like this:
#
# ![roc](./batch_inference_roc.png)
#
# The AUC of this classifier is 0.96, which means it's very good!

# ## Putting it together
#
# The main flow of the code downloads the data, then runs the batch inference,
# then plots the results.
# Each prediction takes roughly 0.1-1s, so if we ran everything sequentially it would take 2,500-25,000 seconds.
# That's a lot! Luckily because of Modal's `.map` method, we can process everything in a couple of minutes at most.
# Modal will automatically spin up more and more workers until all inputs are processed.


@app.local_entrypoint()
def main():
    print("Downloading data...")
    data = get_data.remote()
    print("Got", len(data), "reviews")
    reviews = [review for review, label in data]
    labels = [label for review, label in data]

    # Let's check that the model works by classifying the first 5 entries
    predictor = SentimentAnalysis()
    for review, label in data[:5]:
        prediction = predictor.predict.remote(review)
        print(
            f"Sample prediction with positivity score {prediction}:\n{review}\n\n"
        )

    # Now, let's run batch inference over it
    print("Running batch prediction...")
    predictions = list(predictor.predict.map(reviews))

    # Generate a ROC plot
    print("Creating ROC plot...")
    png_data = roc_plot.remote(labels, predictions)
    fn = "/tmp/roc.png"
    with open(fn, "wb") as f:
        f.write(png_data)
    print(f"Wrote ROC curve to {fn}")


# ## Running this
#
# When you run this, it will download the dataset and load the model, then output some
# sample predictions:
#
# ```
# Sample prediction with positivity score 0.0003837468393612653:
# I love sci-fi and am willing to put up with a lot. Sci-fi movies/TV are usually underfunded, under-appreciated and misunderstood. I tried to like this, I really did, but it is to good TV sci-fi as Babylon 5 is to Star Trek (the original). Silly prosthetics, cheap cardboard sets, stilted dialogues, CG that doesn't match the background, and painfully one-dimensional characters cannot be overcome with a 'sci-fi' setting. (I'm sure there are those of you out there who think Babylon 5 is good sci-fi TV. It's not. It's clichéd and uninspiring.) While US viewers might like emotion and character development, sci-fi is a genre that does not take itself seriously (cf. Star Trek). It may treat important issues, yet not as a serious philosophy. It's really difficult to care about the characters here as they are not simply foolish, just missing a spark of life. Their actions and reactions are wooden and predictable, often painful to watch. The makers of Earth KNOW it's rubbish as they have to always say "Gene Roddenberry's Earth..." otherwise people would not continue watching. Roddenberry's ashes must be turning in their orbit as this dull, cheap, poorly edited (watching it without advert breaks really brings this home) trudging Trabant of a show lumbers into space. Spoiler. So, kill off a main character. And then bring him back as another actor. Jeeez! Dallas all over again.
#
# Sample prediction with positivity score 0.38294079899787903:
# Worth the entertainment value of a rental, especially if you like action movies. This one features the usual car chases, fights with the great Van Damme kick style, shooting battles with the 40 shell load shotgun, and even terrorist style bombs. All of this is entertaining and competently handled but there is nothing that really blows you away if you've seen your share before.<br /><br />The plot is made interesting by the inclusion of a rabbit, which is clever but hardly profound. Many of the characters are heavily stereotyped -- the angry veterans, the terrified illegal aliens, the crooked cops, the indifferent feds, the bitchy tough lady station head, the crooked politician, the fat federale who looks like he was typecast as the Mexican in a Hollywood movie from the 1940s. All passably acted but again nothing special.<br /><br />I thought the main villains were pretty well done and fairly well acted. By the end of the movie you certainly knew who the good guys were and weren't. There was an emotional lift as the really bad ones got their just deserts. Very simplistic, but then you weren't expecting Hamlet, right? The only thing I found really annoying was the constant cuts to VDs daughter during the last fight scene.<br /><br />Not bad. Not good. Passable 4.
#
# Sample prediction with positivity score 0.0002899310493376106:
# its a totally average film with a few semi-alright action sequences that make the plot seem a little better and remind the viewer of the classic van dam films. parts of the plot don't make sense and seem to be added in to use up time. the end plot is that of a very basic type that doesn't leave the viewer guessing and any twists are obvious from the beginning. the end scene with the flask backs don't make sense as they are added in and seem to have little relevance to the history of van dam's character. not really worth watching again, bit disappointed in the end production, even though it is apparent it was shot on a low budget certain shots and sections in the film are of poor directed quality
#
# Sample prediction with positivity score 0.004243704490363598:
# STAR RATING: ***** Saturday Night **** Friday Night *** Friday Morning ** Sunday Night * Monday Morning <br /><br />Former New Orleans homicide cop Jack Robideaux (Jean Claude Van Damme) is re-assigned to Columbus, a small but violent town in Mexico to help the police there with their efforts to stop a major heroin smuggling operation into their town. The culprits turn out to be ex-military, lead by former commander Benjamin Meyers (Stephen Lord, otherwise known as Jase from East Enders) who is using a special method he learned in Afghanistan to fight off his opponents. But Jack has a more personal reason for taking him down, that draws the two men into an explosive final showdown where only one will walk away alive.<br /><br />After Until Death, Van Damme appeared to be on a high, showing he could make the best straight to video films in the action market. While that was a far more drama oriented film, with The Shepherd he has returned to the high-kicking, no brainer action that first made him famous and has sadly produced his worst film since Derailed. It's nowhere near as bad as that film, but what I said still stands.<br /><br />A dull, predictable film, with very little in the way of any exciting action. What little there is mainly consists of some limp fight scenes, trying to look cool and trendy with some cheap slo-mo/sped up effects added to them that sadly instead make them look more desperate. Being a Mexican set film, director Isaac Florentine has tried to give the film a Robert Rodriguez/Desperado sort of feel, but this only adds to the desperation.<br /><br />VD gives a particularly uninspired performance and given he's never been a Robert De Niro sort of actor, that can't be good. As the villain, Lord shouldn't expect to leave the beeb anytime soon. He gets little dialogue at the beginning as he struggles to muster an American accent but gets mysteriously better towards the end. All the supporting cast are equally bland, and do nothing to raise the films spirits at all.<br /><br />This is one shepherd that's strayed right from the flock. *
#
# Sample prediction with positivity score 0.996307373046875:
# First off let me say, If you haven't enjoyed a Van Damme movie since bloodsport, you probably will not like this movie. Most of these movies may not have the best plots or best actors but I enjoy these kinds of movies for what they are. This movie is much better than any of the movies the other action guys (Segal and Dolph) have thought about putting out the past few years. Van Damme is good in the movie, the movie is only worth watching to Van Damme fans. It is not as good as Wake of Death (which i highly recommend to anyone of likes Van Damme) or In hell but, in my opinion it's worth watching. It has the same type of feel to it as Nowhere to Run. Good fun stuff!
# ```
#
# After that, it kicks off the actual batch inference.
# It should look something like the screenshot below (we are very proud of the progress bar):
#
# ![progress](./batch_inference_progress.png)
#
# The whole thing should take a few minutes to run.
#
# ## Further optimization notes
#
# Every container downloads the model when it starts, which is a bit inefficient.
# In order to improve this, what you could do is store the model in the image that
# backs each container.
# See [`Image.run_function`](/docs/guide/custom-container#run-a-modal-function-during-your-build-with-run_function-beta).
#


---

## falcon bitsandbytes

# ---
# args: ["--prompt", "How do planes work?"]
# ---
# # Run Falcon-40B with bitsandbytes
#
# In this example, we download the full-precision weights of the Falcon-40B LLM but load it in 4-bit using
# Tim Dettmers' [`bitsandbytes`](https://github.com/TimDettmers/bitsandbytes) library. This enables it to fit
# into a single GPU (A100 40GB).
#
# Due to the current limitations of the library, the inference speed is a little over 2 tokens/second and due
# to the sheer size of the model, the cold start time on Modal is around 2 minutes.
#
# For faster cold start at the expense of inference speed, check out
# [Running Falcon-40B with AutoGPTQ](https://modal.com/docs/examples/falcon_gptq).
#
# ## Setup
#
# First we import the components we need from `modal`.

from modal import App, Image, enter, gpu, method, web_endpoint


# Spec for an image where falcon-40b-instruct is cached locally
def download_falcon_40b():
    from huggingface_hub import snapshot_download

    model_name = "tiiuae/falcon-40b-instruct"
    snapshot_download(model_name)


image = (
    Image.micromamba()
    .micromamba_install(
        "cudatoolkit=11.7",
        "cudnn=8.1.0",
        "cuda-nvcc",
        "scipy",
        channels=["conda-forge", "nvidia"],
    )
    .apt_install("git")
    .pip_install(
        "bitsandbytes==0.39.0",
        "bitsandbytes-cuda117==0.26.0.post2",
        "peft==0.6.2",
        "transformers==4.31.0",
        "accelerate==0.26.1",
        "hf-transfer==0.1.5",
        "torch==2.0.0",
        "torchvision==0.15.1",
        "sentencepiece==0.1.97",
        "huggingface_hub==0.14.1",
        "einops==0.6.1",
    )
    # Use huggingface's hi-perf hf-transfer library to download this large model.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(download_falcon_40b)
)

app = App(
    image=image, name="example-falcon-bnb"
)  # Note: prior to April 2024, "app" was called "stub"


# ## The model class
#
# Next, we write the model code. We want Modal to load the model into memory just once every time a container starts up,
# so we use [class syntax](https://modal.com/docs/guide/lifecycle-functions) and the `@enter` decorator.
#
# Within the [@app.cls](https://modal.com/docs/reference/modal.App#cls) decorator, we use the [gpu parameter](/docs/guide/gpu)
# to specify that we want to run our function on an [A100 GPU](https://modal.com/docs/guide/gpu). We also allow each call 10 mintues to complete,
# and request the runner to stay live for 5 minutes after its last request.
#
# We load the model in 4-bit using the `bitsandbytes` library.
#
# The rest is just using the [`pipeline`](https://huggingface.co/docs/transformers/en/main_classes/pipelines)
# abstraction from the `transformers` library. Refer to the documentation for more parameters and tuning.
@app.cls(
    gpu=gpu.A100(),  # Use A100s
    timeout=60 * 10,  # 10 minute timeout on inputs
    container_idle_timeout=60 * 5,  # Keep runner alive for 5 minutes
)
class Falcon40B_4bit:
    @enter()
    def load_model(self):
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        model_name = "tiiuae/falcon-40b-instruct"

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,  # Model is downloaded to cache dir
            device_map="auto",
            quantization_config=nf4_config,
        )
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,
            device_map="auto",
        )
        tokenizer.bos_token_id = 1

        self.model = torch.compile(model)
        self.tokenizer = tokenizer

    @method()
    def generate(self, prompt: str):
        from threading import Thread

        from transformers import GenerationConfig, TextIteratorStreamer

        tokenized = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokenized.input_ids
        input_ids = input_ids.to(self.model.device)

        generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.1,
            max_new_tokens=512,
        )

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_special_tokens=True
        )
        generate_kwargs = dict(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            attention_mask=tokenized.attention_mask,
            output_scores=True,
            streamer=streamer,
        )

        thread = Thread(target=self.model.generate, kwargs=generate_kwargs)
        thread.start()
        for new_text in streamer:
            print(new_text, end="")
            yield new_text

        thread.join()


# ## Run the model
# We define a [`local_entrypoint`](https:modal.com/docs/guide/apps#entrypoints-for-ephemeral-apps) to call our remote function
# sequentially for a list of inputs. You can run this locally with `modal run -q falcon_bitsandbytes.py`. The `-q` flag
# enables streaming to work in the terminal output.
prompt_template = (
    "A chat between a curious human user and an artificial intelligence assistant. The assistant give a helpful, detailed, and accurate answer to the user's question."
    "\n\nUser:\n{}\n\nAssistant:\n"
)


@app.local_entrypoint()
def cli(prompt: str = None):
    question = (
        prompt
        or "What are the main differences between Python and JavaScript programming languages?"
    )
    model = Falcon40B_4bit()
    for text in model.generate.remote_gen(prompt_template.format(question)):
        print(text, end="", flush=True)


# ## Serve the model
# Finally, we can serve the model from a web endpoint with `modal deploy falcon_bitsandbytes.py`. If
# you visit the resulting URL with a question parameter in your URL, you can view the model's
# stream back a response.
# You can try our deployment [here](https://modal-labs--example-falcon-bnb-get.modal.run/?question=How%20do%20planes%20work?).
@app.function(timeout=60 * 10)
@web_endpoint()
def get(question: str):
    from itertools import chain

    from fastapi.responses import StreamingResponse

    model = Falcon40B_4bit()
    return StreamingResponse(
        chain(
            ("Loading model (100GB). This usually takes around 110s ...\n\n"),
            model.generate.remote(prompt_template.format(question)),
        ),
        media_type="text/event-stream",
    )


---

## falcon gptq

# # Run Falcon-40B with AutoGPTQ
#
# In this example, we run a quantized 4-bit version of Falcon-40B, the first open-source large language
# model of its size, using HuggingFace's [transformers](https://huggingface.co/docs/transformers/index)
# library and [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ).
#
# Due to the current limitations of the library, the inference speed is a little under 1 token/second and the
# cold start time on Modal is around 25s.
#
# For faster inference at the expense of a slower cold start, check out
# [Running Falcon-40B with `bitsandbytes` quantization](https://modal.com/docs/examples/falcon_bitsandbytes). You can also
# run a smaller model via the [Gemma 7B example](https://modal.com/docs/examples/vllm_gemma).
#
# ## Setup
#
# First we import the components we need from `modal`.

from modal import App, Image, enter, gpu, method, web_endpoint

# ## Define a container image
#
# To take advantage of Modal's blazing fast cold-start times, we download model weights
# into a folder inside our container image. These weights come from a quantized model
# found on Huggingface.
IMAGE_MODEL_DIR = "/model"


def download_model():
    from huggingface_hub import snapshot_download

    model_name = "TheBloke/falcon-40b-instruct-GPTQ"
    snapshot_download(model_name, local_dir=IMAGE_MODEL_DIR)


# Now, we define our image. We'll use the `debian-slim` base image, and install the dependencies we need
# using [`pip_install`](https://modal.com/docs/reference/modal.Image#pip_install). At the end, we'll use
# [`run_function`](https://modal.com/docs/guide/custom-container#run-a-modal-function-during-your-build-with-run_function-beta) to run the
# function defined above as part of the image build.

image = (
    Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "auto-gptq==0.7.0",
        "einops==0.6.1",
        "hf-transfer==0.1.5",
        "huggingface_hub==0.14.1",
        "transformers==4.31.0",
    )
    # Use huggingface's hi-perf hf-transfer library to download this large model.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(download_model)
)

# Let's instantiate and name our [`App`](https://modal.com/docs/guide/apps).
app = App(
    name="example-falcon-gptq", image=image
)  # Note: prior to April 2024, "app" was called "stub"


# ## The model class
#
# Next, we write the model code. We want Modal to load the model into memory just once every time a container starts up,
# so we use [class syntax](https://modal.com/docs/guide/lifecycle-functions) and the `@enter` decorator.
#
# Within the [`@app.cls`](https://modal.com/docs/reference/modal.App#cls) decorator, we use the [`gpu` parameter](https://modal.com/docs/guide/gpu)
# to specify that we want to run our function on an [A100 GPU](https://modal.com/docs/guide/gpu#a100-gpus). We also allow each call 10 mintues to complete,
# and request the runner to stay live for 5 minutes after its last request.
#
# The rest is just using the `transformers` library to run the model. Refer to the
# [documentation](https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/text_generation#transformers.GenerationMixin.generate)
# for more parameters and tuning.
#
# Note that we need to create a separate thread to call the `generate` function because we need to
# yield the text back from the streamer in the main thread. This is an idiosyncrasy with streaming in `transformers`.
@app.cls(gpu=gpu.A100(), timeout=60 * 10, container_idle_timeout=60 * 5)
class Falcon40BGPTQ:
    @enter()
    def load_model(self):
        from auto_gptq import AutoGPTQForCausalLM
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            IMAGE_MODEL_DIR, use_fast=True
        )
        print("Loaded tokenizer.")

        self.model = AutoGPTQForCausalLM.from_quantized(
            IMAGE_MODEL_DIR,
            trust_remote_code=True,
            use_safetensors=True,
            device_map="auto",
            use_triton=False,
            strict=False,
        )
        print("Loaded model.")

    @method()
    def generate(self, prompt: str):
        from threading import Thread

        from transformers import TextIteratorStreamer

        inputs = self.tokenizer(prompt, return_tensors="pt")
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_special_tokens=True
        )
        generation_kwargs = dict(
            inputs=inputs.input_ids.cuda(),
            attention_mask=inputs.attention_mask,
            temperature=0.1,
            max_new_tokens=512,
            streamer=streamer,
        )

        # Run generation on separate thread to enable response streaming.
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for new_text in streamer:
            yield new_text

        thread.join()


# ## Run the model
# We define a [`local_entrypoint`](https://modal.com/docs/guide/apps#entrypoints-for-ephemeral-apps) to call our remote function
# sequentially for a list of inputs. You can run this locally with `modal run -q falcon_gptq.py`. The `-q` flag
# enables streaming to work in the terminal output.
prompt_template = (
    "A chat between a curious human user and an artificial intelligence assistant. The assistant give a helpful, detailed, and accurate answer to the user's question."
    "\n\nUser:\n{}\n\nAssistant:\n"
)


@app.local_entrypoint()
def cli():
    question = "What are the main differences between Python and JavaScript programming languages?"
    model = Falcon40BGPTQ()
    for text in model.generate.remote_gen(prompt_template.format(question)):
        print(text, end="", flush=True)


# ## Serve the model
# Finally, we can serve the model from a web endpoint with `modal deploy falcon_gptq.py`. If
# you visit the resulting URL with a question parameter in your URL, you can view the model's
# stream back a response.
# You can try our deployment [here](https://modal-labs--example-falcon-gptq-get.modal.run/?question=Why%20are%20manhole%20covers%20round?).
@app.function(timeout=60 * 10)
@web_endpoint()
def get(question: str):
    from itertools import chain

    from fastapi.responses import StreamingResponse

    model = Falcon40BGPTQ()
    return StreamingResponse(
        chain(
            ("Loading model. This usually takes around 20s ...\n\n"),
            model.generate.remote_gen(prompt_template.format(question)),
        ),
        media_type="text/event-stream",
    )


---

## google search generator

# ---
# runtimes: ["runc", "gvisor"]
# ---
#
# # Use a generator to fetch search results
#
# This is a simple example which
#
# 1. Installs a custom Python package.
# 2. Uses a _generator_ to return results back to the launcher process.

import modal

# We build a custom image by adding the `google` package to the base image.
app = modal.App(
    "example-google-search-generator",
    image=modal.Image.debian_slim().pip_install("google"),
)  # Note: prior to April 2024, "app" was called "stub"

# Next, let's define a _generator_ function that uses our custom image.


@app.function()
def scrape(query):
    from googlesearch import search

    for url in search(query.encode(), stop=100):
        yield url


# Finally, let's launch it from the command line with `modal run`:


@app.local_entrypoint()
def main(query: str = "modal"):
    for url in scrape.remote_gen(query):
        print(url)


---

## news summarizer

# # News article summarizer
#
# In this example we scrape news articles from the [New York Times'
# Science section](https://www.nytimes.com/section/science) and summarize them
# using Google's deep learning summarization model [Pegasus](https://ai.googleblog.com/2020/06/pegasus-state-of-art-model-for.html).
# We log the resulting summaries to the terminal, but you can do whatever you want with the
# summaries afterwards: saving to a CSV file, sending to Slack, etc.

import os
import re
from dataclasses import dataclass
from typing import List

import modal

app = modal.App(
    name="example-news-summarizer"
)  # Note: prior to April 2024, "app" was called "stub"

# ## Building Images and Downloading Pre-trained Model
#
# We start by defining our images. In Modal, each function can use a different
# image. This is powerful because you add only the dependencies you need for
# each function.

# The first image contains dependencies for running our model. We also download the
# pre-trained model into the image using the `from_pretrained` method.
# This caches the model so that we don't have to download it on every function call.
# The model will be saved at `/cache` when this function is called at image build time;
# subsequent calls of this function at runtime will then load the model from `/cache`.


def fetch_model(local_files_only: bool = False):
    from transformers import PegasusForConditionalGeneration, PegasusTokenizer

    tokenizer = PegasusTokenizer.from_pretrained(
        "google/pegasus-xsum",
        cache_dir="/cache",
        local_files_only=local_files_only,
    )
    model = PegasusForConditionalGeneration.from_pretrained(
        "google/pegasus-xsum",
        cache_dir="/cache",
        local_files_only=local_files_only,
    )
    return model, tokenizer


deep_learning_image = (
    modal.Image.debian_slim()
    .pip_install("transformers==4.16.2", "torch", "sentencepiece")
    .run_function(fetch_model)
)

# Defining the scraping image is very similar. This image only contains the packages required
# to scrape the New York Times website, though; so it's much smaller.
scraping_image = modal.Image.debian_slim().pip_install(
    "requests", "beautifulsoup4", "lxml"
)


with scraping_image.imports():
    import requests
    from bs4 import BeautifulSoup


# ## Collect Data
#
# Collecting data happens in two stages: first a list of URL articles
# using the NYT API then scrape the NYT web page for each of those articles
# to collect article texts.


@dataclass
class NYArticle:
    title: str
    image_url: str = ""
    url: str = ""
    summary: str = ""
    text: str = ""


# In order to connect to the NYT API, you will need to sign up at [NYT Developer Portal](https://developer.nytimes.com/),
# create an App then grab an API key. Then head to Modal and create a [Secret](https://modal.com/docs/guide/secrets) called `nytimes`.
# Create an environment variable called `NYTIMES_API_KEY` with your API key.


@app.function(
    secrets=[modal.Secret.from_name("nytimes")],
    image=scraping_image,
)
def latest_science_stories(n_stories: int = 5) -> List[NYArticle]:
    # query api for latest science articles
    params = {
        "api-key": os.environ["NYTIMES_API_KEY"],
    }
    nyt_api_url = "https://api.nytimes.com/svc/topstories/v2/science.json"
    response = requests.get(nyt_api_url, params=params)

    # extract data from articles and return list of NYArticle objects
    results = response.json()
    reject_urls = {"null", "", None}
    articles = [
        NYArticle(
            title=u["title"],
            image_url=(
                u.get("multimedia")[0]["url"] if u.get("multimedia") else ""
            ),
            url=u.get("url"),
        )
        for u in results["results"]
        if u.get("url") not in reject_urls
    ]

    # select only a handful of articles; this usually returns 25 articles
    articles = articles[:n_stories]
    print(f"Retrieved {len(articles)} from the NYT Top Stories API")
    return articles


# The NYT API only gives us article URLs but it doesn't include the article text. We'll get the article URLs
# from the API then scrape each URL for the article body. We'll be using
# [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) for that.


@app.function(image=scraping_image)
def scrape_nyc_article(url: str) -> str:
    print(f"Scraping article => {url}")

    # fetch article; simulate desktop browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "lxml")

    # get all text paragraphs & construct single string with article text
    article_text = ""
    article_section = soup.find_all(
        "div", {"class": re.compile(r"\bStoryBodyCompanionColumn\b")}
    )
    if article_section:
        paragraph_tags = article_section[0].find_all("p")
        article_text = " ".join([p.get_text() for p in paragraph_tags])

    # return article with scraped text
    return article_text


# Now the summarization function. We use `huggingface`'s Pegasus tokenizer and model implementation to
# generate a summary of the model. You can learn more about Pegasus does in the [HuggingFace
# documentation](https://huggingface.co/docs/transformers/model_doc/pegasus). Use `gpu="any"` to speed-up inference.


@app.function(
    image=deep_learning_image,
    gpu=False,
    memory=4096,
)
def summarize_article(text: str) -> str:
    print(f"Summarizing text with {len(text)} characters.")

    # `local_files_only` is set to `True` because we expect to read the model
    # files saved in the image.
    model, tokenizer = fetch_model(local_files_only=True)

    # summarize text
    batch = tokenizer(
        [text], truncation=True, padding="longest", return_tensors="pt"
    ).to("cpu")
    translated = model.generate(**batch)
    summary = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

    return summary


# ## Create a Scheduled Function
#
# Put everything together and schedule it to run every day. You can also use `modal.Cron` for a
# more advanced scheduling interface.


@app.function(schedule=modal.Period(days=1))
def trigger():
    articles = latest_science_stories.remote()

    # parallelize article scraping
    for i, text in enumerate(scrape_nyc_article.map([a.url for a in articles])):
        articles[i].text = text

    # parallelize summarization
    for i, summary in enumerate(
        summarize_article.map([a.text for a in articles if len(a.text) > 0])
    ):
        articles[i].summary = summary

    # show all summaries in the terminal
    for article in articles:
        print(f'Summary of "{article.title}" => {article.summary}')


# Create a new Modal scheduled function with:
#
# ```shell
# modal deploy --name news_summarizer news_summarizer.py
# ```

# You can also run this entire Modal app in debugging mode before.
# call it with `modal run news_summarizer.py`


@app.local_entrypoint()
def main():
    trigger.remote()


# And that's it. You will now generate deep learning summaries from the latest
# NYT Science articles every day.


---

## queue simple

# ---
# cmd: ["python", "misc/queue_simple.py"]
# runtimes: ["runc", "gvisor"]
# ---
#
# # Using a queue to send/receive data
#
# This is an example of how to use queues to send/receive data.
# We don't do it here, but you could imagine doing this _between_ two functions.


import asyncio

import modal
import modal.queue


async def run_async(q: modal.Queue) -> None:
    await q.put.aio(42)
    r = await q.get.aio()
    assert r == 42
    await q.put_many.aio([42, 43, 44, 45, 46])
    await q.put_many.aio([47, 48, 49, 50, 51])
    r = await q.get_many.aio(3)
    assert r == [42, 43, 44]
    r = await q.get_many.aio(99)
    assert r == [45, 46, 47, 48, 49, 50, 51]


async def many_consumers(q: modal.Queue) -> None:
    print("Creating getters")
    tasks = [asyncio.create_task(q.get.aio()) for i in range(20)]
    print("Putting values")
    await q.put_many.aio(list(range(10)))
    await asyncio.sleep(1)
    # About 10 tasks should now be done
    n_done_tasks = sum(1 for t in tasks if t.done())
    assert n_done_tasks == 10
    # Finish remaining ones
    await q.put_many.aio(list(range(10)))
    await asyncio.sleep(1)
    assert all(t.done() for t in tasks)


async def main():
    with modal.Queue.ephemeral() as q:
        await run_async(q)
        await many_consumers(q)


if __name__ == "__main__":
    asyncio.run(main())


---

## run fooocus

# # Generate: Fooocus
#
# This example demonstrates how to set up and run a web server using the Modal library with Fooocus as the frontend.
# Fooocus provides a beginner-friendly interface to work with the SDXL 1.0 model for image generation tasks.
# The script includes the setup of a Docker image, initialization of Fooocus, and launching a web server with GPU support.
#
# ## Basic setup

import modal

# To create an image that can run Fooocus, we start from an official NVIDIA base image and then add Python
# and a few system packages.
#
# We then download the Fooocus repository.

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.3.1-base-ubuntu22.04", add_python="3.10"
    )
    .apt_install(
        "software-properties-common",
        "git",
        "git-lfs",
        "coreutils",
        "aria2",
        "libgl1",
        "libglib2.0-0",
        "curl",
        "wget",
        "libsm6",
        "libxrender1",
        "libxext6",
        "ffmpeg",
    )
    .run_commands("git clone https://github.com/lllyasviel/Fooocus.git")
)

# ## Initialize Fooocus
#
# We are not limited to running shell commands and package installers in the image setup.
# We can also run Python functions by defining them in our code and passing them to the `run_function` method.
#
# This function installs Fooocus's dependencies and downloads the SDXL 1.0 model to the container image.
#
# This all happens at the time the container image is defined, so that the image is ready to run Fooocus when it is deployed.


def init_Fooocus():
    import os
    import subprocess

    # change the working directory to the Fooocus directory and install the required Python packages from the requirements file.
    os.chdir("/Fooocus")
    os.system("pip install -r requirements_versions.txt")

    # change the directory to the models' checkpoints and download the SDXL 1.0 model using wget.
    os.chdir("./models/checkpoints")
    subprocess.run(
        "wget -O juggernautXL_v8Rundiffusion.safetensors 'https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/juggernautXL_v8Rundiffusion.safetensors'",
        shell=True,
    )


GPU_CONFIG = modal.gpu.T4()
image = image.run_function(init_Fooocus, gpu=GPU_CONFIG)

# ## Run Fooocus
#
# The `run` function is decorated with `app.function` to define it as a Modal function.
# The `web_server` decorator indicates that this function will serve a web application on the specified port.
# We increase the startup timeout to three minutes to account for the time it takes to load the model and start the server.

app = modal.App("Fooocus", image=image)

PORT = 8000
MINUTES = 60


@app.function(gpu=GPU_CONFIG, timeout=10 * MINUTES)
@modal.web_server(port=PORT, startup_timeout=3 * MINUTES)
def run():
    import os
    import subprocess

    # change the working directory to the Fooocus directory.
    os.chdir("/Fooocus")

    # launch the Fooocus application using a subprocess that listens on the specified port
    subprocess.Popen(
        [
            "python",
            "launch.py",
            "--listen",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--always-high-vram",
        ]
    )


---

## say hello cron

# ---
# lambda-test: false
# ---

import time
from datetime import datetime, timezone

import modal

app = modal.App(
    "example-say-hello-cron"
)  # Note: prior to April 2024, "app" was called "stub"


@app.function(schedule=modal.Period(seconds=10))
def say_hello():
    start_time = datetime.now(timezone.utc)
    for i in range(10):
        print(f"Message #{i} from invocation at {start_time}")
        time.sleep(1.5)


---

## stable lm

import os
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

import modal
from pydantic import BaseModel
from typing_extensions import Annotated, Literal


def build_models():
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = snapshot_download(
        "stabilityai/stablelm-tuned-alpha-7b",
        ignore_patterns=["*.md"],
    )
    m = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    m.save_pretrained(
        model_path, safe_serialization=True, max_shard_size="24GB"
    )
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.save_pretrained(model_path)
    [p.unlink() for p in Path(model_path).rglob("*.bin")]  # type: ignore


image = (
    modal.Image.micromamba()
    .apt_install("git", "software-properties-common", "wget")
    .micromamba_install(
        "cudatoolkit-dev=11.7",
        "pytorch-cuda=11.7",
        "rust=1.69.0",
        channels=["nvidia", "pytorch", "conda-forge"],
    )
    .env(
        {
            "HF_HOME": "/root",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "SAFETENSORS_FAST_GPU": "1",
            "BITSANDBYTES_NOWELCOME": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_CACHE_DIR": "1",
        }
    )
    .pip_install(
        "transformers~=4.28.1",
        "safetensors==0.3.0",
        "accelerate==0.18.0",
        "bitsandbytes==0.38.1",
        "msgspec==0.18.6",
        "sentencepiece==0.1.98",
        "hf-transfer==0.1.3",
        gpu="any",
    )
    .run_function(
        build_models,
        gpu=None,
        timeout=3600,
    )
)

app = modal.App(
    name="example-stability-lm",
    image=image,
    secrets=[
        modal.Secret.from_dict(
            {"REPO_ID": "stabilityai/stablelm-tuned-alpha-7b"}
        )
    ],
)  # Note: prior to April 2024, "app" was called "stub"


class CompletionRequest(BaseModel):
    prompt: Annotated[str, "The prompt for text completion"]
    model: Annotated[
        Literal["stabilityai/stablelm-tuned-alpha-7b"],
        "The model to use for text completion",
    ] = "stabilityai/stablelm-tuned-alpha-7b"
    temperature: Annotated[
        float,
        "Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic.",
    ] = 0.8
    max_tokens: Annotated[
        int, "Maximum number of new tokens to generate for text completion."
    ] = 16
    top_p: Annotated[
        float,
        "Probability threshold for the decoder to use in sampling next most likely token.",
    ] = 0.9
    stream: Annotated[
        bool, "Whether to stream the generated text or return it all at once."
    ] = False
    stop: Annotated[Union[str, List[str]], "Any additional stop words."] = []
    top_k: Annotated[
        int,
        "Limits the set of tokens to consider for next token generation to the top k.",
    ] = 40
    do_sample: Annotated[
        bool, "Whether to use sampling or greedy decoding for text completion."
    ] = True


@app.cls(gpu="A10G")
class StabilityLM:
    def __init__(
        self,
        model_url: str = "stabilityai/stablelm-tuned-alpha-7b",
        decode_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.model_url = model_url
        self.decode_kwargs = decode_kwargs or {}
        self.stop_tokens = [
            "<|USER|>",
            "<|ASSISTANT|>",
            "<|SYSTEM|>",
            "<|padding|>",
            "<|endoftext|>",
        ]
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    @modal.enter()
    def setup_model(self):
        """
        Container-lifeycle method for model setup.
        """
        import torch
        from transformers import AutoTokenizer, TextIteratorStreamer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_url, local_files_only=True
        )
        self.stop_ids = tokenizer.convert_tokens_to_ids(self.stop_tokens)
        self.streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, **self.decode_kwargs
        )
        self.generator = pipeline(
            "text-generation",
            model=self.model_url,
            tokenizer=tokenizer,
            streamer=self.streamer,
            torch_dtype=torch.float16,
            device_map="auto",
            model_kwargs={"local_files_only": True},
        )
        self.generator.model = torch.compile(self.generator.model)

    def get_config(
        self, completion_request: CompletionRequest
    ) -> Dict[str, Any]:
        return dict(
            pad_token_id=self.generator.tokenizer.eos_token_id,
            eos_token_id=list(
                set(
                    self.generator.tokenizer.convert_tokens_to_ids(
                        self.generator.tokenizer.tokenize(
                            "".join(completion_request.stop)
                        )
                    )
                    + self.stop_ids
                )
            ),
            max_new_tokens=completion_request.max_tokens,
            **completion_request.dict(
                exclude={"prompt", "model", "stop", "max_tokens", "stream"}
            ),
        )

    def generate_completion(
        self, completion_request: CompletionRequest
    ) -> Generator[str, None, None]:
        import re
        from threading import Thread

        from transformers import GenerationConfig

        text = format_prompt(completion_request.prompt)
        gen_config = GenerationConfig(**self.get_config(completion_request))
        stop_words = self.generator.tokenizer.convert_ids_to_tokens(
            gen_config.eos_token_id
        )
        stop_words_pattern = re.compile("|".join(map(re.escape, stop_words)))
        thread = Thread(
            target=self.generator.__call__,
            kwargs=dict(text_inputs=text, generation_config=gen_config),
        )
        thread.start()
        for new_text in self.streamer:
            if new_text.strip():
                new_text = stop_words_pattern.sub("", new_text)
                yield new_text
        thread.join()

    @modal.method()
    def generate(self, completion_request: CompletionRequest) -> str:
        return "".join(self.generate_completion(completion_request))

    @modal.method()
    def generate_stream(
        self, completion_request: CompletionRequest
    ) -> Generator:
        for text in self.generate_completion(completion_request):
            yield text


def format_prompt(instruction: str) -> str:
    return f"<|USER|>{instruction}<|ASSISTANT|>"


with app.image.imports():
    import uuid

    import msgspec

    class Choice(msgspec.Struct):
        text: str
        index: Union[int, None] = 0
        logprobs: Union[int, None] = None
        finish_reason: Union[str, None] = None

    class CompletionResponse(msgspec.Struct, kw_only=True):  # type: ignore
        id: Union[str, None] = None
        object: str = "text_completion"
        created: Union[int, None] = None
        model: str
        choices: List[Choice]

        def __post_init__(self):
            if self.id is None:
                self.id = str(uuid.uuid4())
            if self.created is None:
                self.created = int(time.time())


@app.function()
@modal.web_endpoint(method="POST")
async def completions(completion_request: CompletionRequest):
    from fastapi import Response, status
    from fastapi.responses import StreamingResponse

    response_id = str(uuid.uuid4())
    response_utc = int(time.time())

    if not completion_request.stream:
        return Response(
            content=msgspec.json.encode(
                CompletionResponse(
                    id=response_id,
                    created=response_utc,
                    model=completion_request.model,
                    choices=[
                        Choice(
                            index=0,
                            text=StabilityLM().generate.remote(
                                completion_request=completion_request
                            ),
                        )
                    ],
                )
            ),
            status_code=status.HTTP_200_OK,
            media_type="application/json",
        )

    def wrapped_stream():
        for new_text in StabilityLM().generate_stream.remote(
            completion_request=completion_request
        ):
            yield (
                msgspec.json.encode(
                    CompletionResponse(
                        id=response_id,
                        created=response_utc,
                        model=completion_request.model,
                        choices=[Choice(index=0, text=new_text)],
                    )
                )
                + b"\n\n"
            )

    return StreamingResponse(
        content=wrapped_stream(),
        status_code=status.HTTP_200_OK,
        media_type="text/event-stream",
    )


@app.local_entrypoint()
def main():
    q_style, q_end = "\033[1m", "\033[0m"
    instructions = [
        "Generate a list of the 10 most beautiful cities in the world.",
        "How can I tell apart female and male red cardinals?",
    ]
    instruction_requests = [
        CompletionRequest(prompt=q, max_tokens=128) for q in instructions
    ]
    print("Running example non-streaming completions:\n")
    for q, a in zip(
        instructions, list(StabilityLM().generate.map(instruction_requests))
    ):
        print(f"{q_style}{q}{q_end}\n{a}\n\n")

    print("Running example streaming completion:\n")
    for part in StabilityLM().generate_stream.remote_gen(
        CompletionRequest(
            prompt="Generate a list of ten sure-to-be unicorn AI startup names.",
            max_tokens=128,
            stream=True,
        )
    ):
        print(part, end="", flush=True)


# ```bash
# curl $MODEL_APP_ENDPOINT \
#   -H "Content-Type: application/json" \
#   -d '{
#     "prompt": "Generate a list of 20 great names for sentient cheesecakes that teach SQL",
#     "stream": true,
#     "max_tokens": 64
#   }'
# ```


---

## tqdm progress bar

import time

import modal

app = modal.App(
    "example-tqdm",
    image=modal.Image.debian_slim().pip_install("tqdm"),
)  # Note: prior to April 2024, "app" was called "stub"


@app.function()
def f():
    from tqdm import tqdm

    for i in tqdm(range(100)):
        time.sleep(0.1)


if __name__ == "__main__":
    with app.run():
        f.remote()


---

