Introducing Pydantic Logfire¶
From the team behind Pydantic, Logfire is an observability platform built on the same belief as our open source library — that the most powerful tools can be easy to use.

What sets Logfire apart:

Simple and Powerful: Logfire's dashboard is simple relative to the power it provides, ensuring your entire engineering team will actually use it.
Python-centric Insights: From rich display of Python objects, to event-loop telemetry, to profiling Python code and database queries, Logfire gives you unparalleled visibility into your Python application's behavior.
SQL: Query your data using standard SQL — all the control and (for many) nothing new to learn. Using SQL also means you can query your data with existing BI tools and database querying libraries.
OpenTelemetry: Logfire is an opinionated wrapper around OpenTelemetry, allowing you to leverage existing tooling, infrastructure, and instrumentation for many common Python packages, and enabling support for virtually any language.
Pydantic Integration: Understand the data flowing through your Pydantic models and get built-in analytics on validations.
Pydantic Logfire helps you instrument your applications with less code, less time, and better understanding.

Find the needle in a stacktrace¶
Logfire FastAPI screenshot

We understand Python and its peculiarities. Pydantic Logfire was crafted by Python developers, for Python developers, addressing the unique challenges and opportunities of the Python environment. It's not just about having data; it's about having the right data, presented in ways that make sense for Python applications.

In the Spirit of Python¶
Simplicity and Power: Emulating the Pydantic library's philosophy, Pydantic Logfire offers an intuitive start for beginners while providing the depth experts desire. It's the same balance of ease, sophistication, and productivity, reimagined for observability.
Born from Python and Pydantic: As creators immersed in the Python open-source ecosystem, we've designed Pydantic Logfire to deeply integrate with Python and Pydantic's nuances, delivering a more customized experience than generic observability platforms.
Elevating Data to Insights¶
With deep Python integration, Pydantic Logfire automatically instruments your code for minimal manual effort, provides exceptional insights into async code, offers detailed performance analytics, and displays Python objects the same as the interpreter. For existing Pydantic users, it also delivers unparalleled insights into your usage of Pydantic models.
Structured Data and Direct SQL Access means you can use familiar tools like Pandas, SQLAlchemy, or psql for querying, can integrate seamlessly with BI tools, and can even leverage AI for SQL generation, ensuring your Python objects and structured data are query-ready. Using vanilla PostgreSQL as the querying language throughout the platform ensures a consistent, powerful, and flexible querying experience.
By harnessing OpenTelemetry, Pydantic Logfire offers automatic instrumentation for popular Python packages, enables cross-language data integration, and supports data export to any OpenTelemetry-compatible backend or proxy.
Pydantic Logfire: The Observability Platform You Deserve¶
Pydantic Logfire isn't just another tool in the shed; it's the bespoke solution crafted by Python developers, for Python developers, ensuring your development work is as smooth and efficient as Python itself.

From the smallest script to large-scale deployments, Pydantic Logfire is the observability solution you've been waiting for.

Simplicity and Power 🚀¶
Pydantic Logfire should be dead simple to start using, simply run:


pip install logfire 
logfire auth 
Then in your code:


import logfire
from datetime import date

logfire.info('Hello, {name}!', name='world')  

with logfire.span('Asking the user their {question}', question='age'):  
    user_input = input('How old are you [YYYY-mm-dd]? ')
    dob = date.fromisoformat(user_input)  
    logfire.debug('{dob=} {age=!r}', dob=dob, age=date.today() - dob)  
This might look similar to simple logging, but it's much more powerful — you get:

structured data from your logs
nested logs / traces to contextualize what you're viewing
a custom-built platform to view your data, with no configuration required
and more, like pretty display of Python objects — see below
Note

If you have an existing app to instrument, you'll get the most value out of configuring OTel integrations, before you start adding logfire.* calls to your code.

Logfire hello world screenshot

Python and Pydantic insights 🐍¶
From rich display of Python objects to event-loop telemetry and profiling Python code, Pydantic Logfire can give you a clearer view into how your Python is running than any other observability tool.

Logfire also has an out-of-the-box Pydantic integration that lets you understand the data passing through your Pydantic models and get analytics on validations.

We can record Pydantic models directly:


from datetime import date
import logfire
from pydantic import BaseModel

class User(BaseModel):
    name: str
    country_code: str
    dob: date

user = User(name='Anne', country_code='USA', dob='2000-01-01')
logfire.info('user processed: {user!r}', user=user)  
Logfire pydantic manual screenshot

Or we can record information about validations automatically:


from datetime import date
import logfire
from pydantic import BaseModel

logfire.configure(pydantic_plugin=logfire.PydanticPlugin(record='all'))  

class User(BaseModel):
    name: str
    country_code: str
    dob: date

User(name='Anne', country_code='USA', dob='2000-01-01')  
User(name='Ben', country_code='USA', dob='2000-02-02')
User(name='Charlie', country_code='GBR', dob='1990-03-03')
Learn more about the Pydantic Plugin here.

Logfire pydantic plugin screenshot

OpenTelemetry under the hood 🔭¶
Because Pydantic Logfire is built on OpenTelemetry, you can use a wealth of existing tooling and infrastructure, including instrumentation for many common Python packages.

For example, we can instrument a simple FastAPI app with just 2 lines of code:

fastapi_example.py

from datetime import date
import logfire
from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI()

logfire.configure()
logfire.instrument_fastapi(app)  
# next, instrument your database connector, http library etc. and add the logging handler 


class User(BaseModel):
    name: str
    country_code: str
    dob: date


@app.post('/')
async def add_user(user: User):
    # we would store the user here
    return {'message': f'{user.name} added'}
We'll need the FastAPI contrib package, FastAPI itself and uvicorn installed to run this:


pip install 'logfire[fastapi]' fastapi uvicorn  
uvicorn fastapi_example:app 
This will give you information on the HTTP request, but also details of results from successful input validations:

Logfire FastAPI 200 response screenshot

And details of failed input validations:

Logfire FastAPI 422 response screenshot

Structured Data and SQL 🧮¶
Query your data with pure, canonical Postgres SQL — all the control and (for many) nothing new to learn. We even provide direct access to the underlying Postgres database, which means that you can query Logfire using any Postgres-compatible tools you like. This includes dashboard-building platforms like Superset, Grafana, and Google Looker Studio, but also Pandas, SQLAlchemy, or even psql.

One big advantage of using the most widely used SQL databases is that generative AI tools like ChatGPT are excellent at writing SQL for you.

Just include your Python objects in Logfire calls (lists, dict, dataclasses, Pydantic models, dataframes, and more), and it'll end up as structured data in our platform ready to be queried.

For example, using data from the User model above, we could list users from the USA:


SELECT attributes->'result'->>'name' as name, age(attributes->'result'->>'dob') as age
FROM records
WHERE attributes->'result'->>'country_code' = 'USA'
Logfire explore query screenshot

You can also filter to show only traces related to users in the USA in the live view with


attributes->'result'->>'country_code' = 'USA'
Logfire search query screenshot

Intro
Guides
Guides
Here are some tutorials to help you get started using Logfire:

First Steps¶
In this guide, we walk you through installation and authentication in your local environment, sending a log message to Logfire, and viewing it in the Logfire Web UI.

Onboarding Checklist 📋¶
In this guide, we provide a checklist with step-by-step instructions to take an existing application and thoroughly instrument it to send data to Logfire. In particular, we'll show you how to leverage Logfire's various integrations to generate as much useful data with as little development effort as possible.

Following this checklist for your application is critical to getting the most out of Logfire.

Intro to the Web UI¶
In this guide, we introduce the various views and features of the Logfire Web UI, and show you how to use them to investigate your projects' data.

Advanced User Guide¶
We cover additional topics in the Advanced User Guide, including:

Sampling: Down-sample lower-priority traces to reduce costs.
Scrubbing: Remove sensitive data from your logs and traces before sending them to Logfire.
Testing: Test your usage of Logfire.
Direct Database Connections: Connect directly to a read-only postgres database containing your project's data. You can use this for ad-hoc querying, or with third-party business intelligence tools like Grafana, Tableau, Metabase, etc.
... and more.
Integrations and Reference¶
Integrations: In this section of the docs we explain what an OpenTelemetry instrumentation is, and offer detailed guidance about how to get the most out of them in combination with Logfire. We also document here how to send data to Logfire from other logging libraries you might already be using, including loguru, structlog, and the Python standard library's logging module.
Configuration: In this section we document the various ways you can configure which Logfire project your deployment will send data to.
Organization Structure: In this section we document the organization, project, and permissions model in Logfire.
SDK CLI docs: Documentation of the logfire command-line interface.

Intro
Guides
First Steps
First Steps
This guide will walk you through getting started with Logfire. You'll learn how to install Logfire, authenticate your local environment, and use traces and spans to instrument your code for observability.

OpenTelemetry Concepts¶
Before diving in, let's briefly cover two fundamental OpenTelemetry concepts:

Traces: A trace represents the entire journey of a request or operation as it moves through a (possibly distributed) system. It's composed of one or more spans.
Spans: A span represents a unit of work within a trace, and are a way to track the execution of your code. Unlike traditional logs, which contain a message at a single moment in time, spans can be nested to form a tree-like structure, with a root span representing the overall operation, and child spans representing sub-operations. Spans are used to measure timing and record metadata about the work being performed.
In Logfire, we'll frequently visualize traces as a tree of its spans:

Example trace screenshot

Using traces and spans, you can gain valuable insights into your system's behavior and performance.

Installation¶
To install the latest version of Logfire, run:


PIP
Rye
Poetry
Conda

pip install logfire

Authentication¶
Authenticate your local environment with Logfire by running:


logfire auth
This opens a browser window to sign up or log in at logfire.pydantic.dev. Upon successful authentication, credentials are stored in ~/.logfire/default.toml.

Basic Usage¶
The first time you use Logfire in a new environment, you'll need to set up a project. A Logfire project is like a namespace for organizing your data. All data sent to Logfire must be associated with a project.

To use Logfire, simply import it and call the desired logging function:


import logfire

logfire.info('Hello, {name}!', name='world')  
Note

Other log levels are also available to use, including trace, debug, notice, warn, error, and fatal.

If you don't have existing credentials for a Logfire project, you'll see a prompt in your terminal to create a new project:


No Logfire project credentials found.
All data sent to Logfire must be associated with a project.

The project will be created in the organization "dmontagu". Continue? [y/n] (y):
Enter the project name (platform): my-project
Project initialized successfully. You will be able to view it at: https://logfire.pydantic.dev/dmontagu/my-project
Press Enter to continue:
Here's what happens:

Logfire detects that no project credentials exist and prompts you to create a new project.
You're asked to confirm the organization where the project will be created (defaulting to your personal organization).
You enter a name for your new project (defaulting to the name of the folder your script is running from).
Logfire initializes the project and provides the URL where you can view your project's data.
Press Enter to continue, and the script will proceed.
After this one-time setup, Logfire will use the newly created project credentials for subsequent Python runs from the same directory.

Once you've created a project (or if you already had one), you should see:


Logfire project URL: https://logfire.pydantic.dev/dmontagu/my-project
19:52:12.323 Hello, world!
Logfire will always start by displaying the URL of your project, and (with default configuration) will also provide a basic display in the terminal of the logs you are sending to Logfire.

Hello world screenshot

Tracing with Spans¶
Spans let you add context to your logs and measure code execution time. Multiple spans combine to form a trace, providing a complete picture of an operation's journey through your system.


from pathlib import Path
import logfire

cwd = Path.cwd()
total_size = 0

with logfire.span('counting size of {cwd=}', cwd=cwd):
    for path in cwd.iterdir():
        if path.is_file():
            with logfire.span('reading {file}', file=path):
                total_size += len(path.read_bytes())

    logfire.info('total size of {cwd} is {size} bytes', cwd=cwd, size=total_size)
In this example:

The outer span measures the time to count the total size of files in the current directory (cwd).
Inner spans measure the time to read each individual file.
Finally, the total size is logged.
Counting size of loaded files screenshot

By instrumenting your code with traces and spans, you can see how long operations take, identify bottlenecks, and get a high-level view of request flows in your system — all invaluable for maintaining the performance and reliability of your applications.

 Back to top
© Pydantic Services Inc. 2024

Intro
Guides
Onboarding Checklist
Create a Project
Since you are already authenticated, you can now create a new project in Logfire.

To create a new project, you can import logfire, and call logfire.configure().


import logfire

logfire.configure()
You'll then be prompted to select one of your projects, or create a new one:

Terminal with prompt to create a project

If you don't have any projects yet, write "n" and "Enter" to create a new project.

You'll then be asked to select your organization, and to provide a name for your new project:

Terminal with prompt to create a project

You have created a new project in Logfire! 🥳

You can also create a project via Web UI...
To create a new project within the UI, you can follow these steps:

Go to the Logfire Web UI.
Logfire Web UI

Click on the New Project button, fill the form that appears, and click Create Project.
New Project button

Done! You have created a new project in Logfire! 😎

You can also create a project via CLI...
Check the SDK CLI documentation for more information on how to create a project via CLI.


Intro
Guides
Onboarding Checklist
Integrate Logfire
In this section, we'll focus on integrating Logfire with your application.

OpenTelemetry Instrumentation¶
Harnessing the power of OpenTelemetry, Logfire not only offers broad compatibility with any OpenTelemetry instrumentation package, but also includes a user-friendly CLI command that effortlessly highlights any missing components in your project.

To inspect your project, run the following command:


logfire inspect
This will output the projects you need to install to have optimal OpenTelemetry instrumentation:

Logfire inspect command

To install the missing packages, copy the command provided by the inspect command, and run it in your terminal.

Each instrumentation package has its own way to be configured. Check our Integrations page to learn how to configure them.

Logging Integration (Optional)¶
Attention

If you are creating a new application or are not using a logging system, you can skip this step.

You should use Logfire itself to collect logs from your application.

All the standard logging methods are supported e.g. logfire.info().

There are many logging systems within the Python ecosystem, and Logfire provides integrations for the most popular ones: Standard Library Logging, Loguru, and Structlog.

Standard Library¶
To integrate Logfire with the standard library logging module, you can use the LogfireLoggingHandler class.

The minimal configuration would be the following:


from logging import basicConfig

from logfire.integrations.logging import LogfireLoggingHandler

basicConfig(handlers=[LogfireLoggingHandler()])
Now imagine, that you have a logger in your application:

main.py

from logging import basicConfig, getLogger

from logfire.integrations.logging import LogfireLoggingHandler

basicConfig(handlers=[LogfireLoggingHandler()])

logger = getLogger(__name__)
logger.error("Hello %s!", "Fred")
If we run the above code, with python main.py, we will see the following output:

Terminal with Logfire output

If you go to the link, you will see the "Hello Fred!" log in the Web UI:

Logfire Web UI with logs

It is simple as that! Cool, right? 🤘

Loguru¶
To integrate with Loguru, check out the Loguru page.

Structlog¶
To integrate with Structlog, check out the Structlog page.

Intro
Guides
Onboarding Checklist
Add Logfire Manual Tracing
In the previous sections, we focused on how to integrate Logfire with your application and leverage automatic instrumentation. In this section, we'll go into more detail about manual tracing, which allows you to add custom spans and logs to your code for targeted data collection.

Because the specifics of where and how to add manual tracing will depend on your particular application, we'll also spend time discussing the general principles and scenarios where manual tracing can be especially valuable.

How to Add Manual Tracing¶
Using the @logfire.instrument Decorator¶
The @logfire.instrument decorator is a convenient way to create a span around an entire function. To use it, simply add the decorator above the function definition:


@logfire.instrument("Function Name", extract_args=True)
def my_function(arg1, arg2):
    # Function code
The first argument to the decorator is the name of the span, and you can optionally set extract_args=True to automatically log the function arguments as span attributes.

Note

The @logfire.instrument decorator MUST be applied first, i.e., UNDER any other decorators.
The source code of the function MUST be accessible.
Creating Manual Spans¶
To create a manual span, use the logfire.span context manager:


with logfire.span("Span Name", key1=value1, key2=value2):
    # Code block
    logfire.info("Log message", key3=value3)
The first argument is the name of the span, and you can optionally provide key-value pairs to include custom data in the span.

Nesting Spans¶
You can nest spans to create a hierarchical structure:


with logfire.span("Outer Span"):
    # Code block
    with logfire.span("Inner Span"):
        # Code block
        logfire.info("Log message")
When nesting spans, try to keep the hierarchy clean and meaningful, and use clear and concise names for your spans.

Recording Custom Data¶
To record custom data within a span, simply pass key-value pairs when creating the span or when logging messages:


with logfire.span("User Login", user_id=user_id):
    logfire.info("User logged in", user_email=user_email)
Consider recording data that will be useful for debugging, monitoring, or analytics purposes.

Capturing Exceptions¶
Logfire automatically captures exceptions that bubble up through spans. To ensure that exceptions are properly captured and associated with the relevant span, make sure to wrap the code that may raise exceptions in a span:


with logfire.span("Database Query"):
    try:
        result = db.query(query)
    except DatabaseError as e:
        logfire.error(f"Database query failed: {str(e)}")
        raise
When to Use Manual Tracing¶
Now that we've seen how to use manual tracing, let's discuss some scenarios where manual tracing can be particularly useful in enhancing your application's observability:

Scenario 1: Improving Log Organization and Readability¶
When working with complex functions or code blocks, manually nested spans can help organize your logs into a hierarchical structure. This makes it easier to navigate and understand the flow of your application, especially during debugging sessions.


import logfire


@logfire.instrument("Complex Operation")
def complex_operation(data):
    # Step 1
    with logfire.span("Data Preprocessing"):
        preprocessed_data = preprocess(data)
        logfire.info("Data preprocessed successfully")

    # Step 2
    with logfire.span("Data Analysis"):
        analysis_result = analyze(preprocessed_data)
        logfire.info("Analysis completed")

    # Step 3
    with logfire.span("Result Postprocessing"):
        final_result = postprocess(analysis_result)
        logfire.info("Result postprocessed")

    return final_result
In this example, the complex_operation function is decorated with @logfire.instrument, which automatically creates a span for the entire function. Additionally, the function is broken down into three main steps, each wrapped in its own span, and you can imagine that the functions called in each of these sections might each produce various spans as well. This creates a clear hierarchy in the logs, making it easier to identify and focus on relevant sections during debugging.

[TODO: Include a screenshot of the web UI showing the hierarchical structure of spans]

Scenario 2: Measuring Execution Duration¶
Manual spans can be used to measure the duration of specific code sections, helping you identify performance bottlenecks and detect regressions.


import logfire


@logfire.instrument("Process Data Batch", extract_args=True)
def process_data_batch(batch):
    # Process the data batch
    processed_data = []
    for item in batch:
        with logfire.span("Process Item {item}"):
            item = step_1(item)
            item = step_2(item)
            item = step_3(item)
        processed_data.append(item)

    return processed_data
In this example, the process_data_batch function is decorated with @logfire.instrument, which automatically creates a span for the entire function and logs the batch argument as a span attribute.

Additionally, each item in the batch is processed within a separate span created using the logfire.span context manager. The span name includes the item being processed, providing more granular visibility into the processing of individual items.

By using manual spans in this way, you can measure the duration of the overall data batch processing, as well as the duration of processing each individual item. This information can be valuable for identifying performance bottlenecks and optimizing your code.

[Include a screenshot of the web UI showing the duration of the Process Data Batch span and the individual Process Item spans]

Scenario 3: Capturing Exception Information¶
Logfire automatically captures full stack traces when exceptions bubble up through spans. By strategically placing spans around code that may raise exceptions, you can ensure that you have the necessary context and information for debugging and error monitoring.


import logfire


@logfire.instrument("Fetch Data from API", extract_args=True)
def fetch_data_from_api(api_url):
    response = requests.get(api_url)
    response.raise_for_status()
    data = response.json()
    logfire.info("Data fetched successfully")
    return data
If an exception occurs while fetching data from the API, Logfire will capture the stack trace and associate it with the span created by the @logfire.instrument decorator. The api_url argument will also be logged as a span attribute, providing additional context for debugging.

[TODO: Include a screenshot of the web UI showing the exception details, stack trace, and api_url attribute]

Scenario 4: Recording User Actions and Custom Data¶
Manual spans can be used to record user actions, input parameters, or other custom data that may be valuable for analytics and business intelligence purposes.


import logfire


def search_products(user_id, search_query, filters):
    with logfire.span(f"Performing search: {search_query}", search_query=search_query, filters=filters):
        results = perform_search(search_query, filters)

    if not results:
        logfire.info("No results found for search query", search_query=search_query)
        with logfire.span("Suggesting Related Products"):
            related_products = suggest_related_products(search_query)
        return {
            "status": "no_results",
            "related_products": related_products
        }
    else:
        logfire.info(f"Found {len(results)} results for search query", search_query=search_query)
        return {
            "status": "success",
            "results": results
        }
In this example, the search_products function is instrumented with manual spans and logs to capture user actions and custom data related to product searches.

The function starts by creating a span named "Performing search: {search_query}" that measures the duration and captures the details of the actual search operation. The search_query and filters are included as span attributes, allowing for fine-grained analysis of search performance and effectiveness.

After performing the search, the function checks the results:

If no results are found, an info-level log message is recorded, indicating that no results were found for the given search query. Then, a "Suggesting Related Products" span is created, and the suggest_related_products function is called to generate a list of related products. The function returns a response with a status of "no_results" and the list of related_products. This data can be used to identify common search queries that yield no results and help improve the product catalog or search functionality.

If results are found, an info-level log message is recorded, indicating the number of results found for the search query. The function then returns a response with a status of "success" and the results list.

By structuring the spans and logs in this way, you can gain insights into various aspects of the product search functionality:

The "Performing search: {search_query}" span measures the duration of each search operation and includes the specific search query and filters, enabling performance analysis and optimization.
The info-level log messages indicate whether results were found or not, helping to identify successful and unsuccessful searches.
The "Suggesting Related Products" span captures the process of generating related product suggestions when no results are found, providing data for analyzing and improving the suggestion algorithm.
[TODO: Include a screenshot of the web UI showing the spans and custom data logged during a product search]

This example demonstrates how manual spans and logs can be strategically placed to capture valuable data for analytics and business intelligence purposes.

Some specific insights you could gain from this instrumentation include:

Identifying the most common search queries and filters used by users, helping to optimize the search functionality and product catalog.
Measuring the performance of search operations and identifying potential bottlenecks or inefficiencies.
Understanding which search queries frequently result in no results, indicating areas where the product catalog may need to be expanded or the search algorithm improved.
Analyzing the effectiveness of the related product suggestion feature in helping users find relevant products when their initial search yields no results.
By capturing this data through manual spans and logs, you can create a rich dataset for analytics and business intelligence purposes, empowering you to make informed decisions and continuously improve your application's search functionality and user experience.

Best Practices and Tips¶
Use manual tracing judiciously. While it can provide valuable insights, overusing manual spans can lead to cluttered logs and source code, and increased overhead in hot loops.
Focus on critical or complex parts of your application where additional context and visibility will be most beneficial.
Choose clear and concise names for your spans to make it easier to understand the flow and purpose of each span.
Record custom data that will be useful for debugging, monitoring, or analytics purposes, but avoid including sensitive or unnecessary information.
Conclusion¶
Manual tracing is a powerful tool for adding custom spans and logs to your code, providing targeted visibility into your application's behavior. By understanding the principles and best practices of manual tracing, you can adapt this technique to your specific use case and enhance your application's observability.

Remember to balance the benefits of detailed tracing with the overhead of adding manual spans, and focus on the areas where additional context and visibility will be most valuable.

Intro
Guides
Onboarding Checklist
Auto-tracing¶
The logfire.install_auto_tracing will trace all function calls in the specified modules.

This works by changing how those modules are imported, so the function MUST be called before importing the modules you want to trace.

For example, suppose all your code lives in the app package, e.g. app.main, app.server, app.db, etc. Instead of starting your application with python app/main.py, you could create another file outside of the app package, e.g:

main.py

import logfire

logfire.install_auto_tracing(modules=['app'])

from app.main import main

main()
Filtering modules to trace¶
The modules argument can be a list of module names. Any submodule within a given module will also be traced, e.g. app.main and app.server. Other modules whose names start with the same prefix will not be traced, e.g. apples.

If one of the strings in the list isn't a valid module name, it will be treated as a regex, so e.g. modules=['app.*'] will trace apples in addition to app.main etc.

For even more control, the modules argument can be a function which returns True for modules that should be traced. This function will be called with an AutoTraceModule object, which has name and filename attributes. For example, this should trace all modules that aren't part of the standard library or third-party packages in a typical Python installation:


import pathlib

import logfire

PYTHON_LIB_ROOT = str(pathlib.Path(pathlib.__file__).parent)


def should_trace(module: logfire.AutoTraceModule) -> bool:
    return not module.filename.startswith(PYTHON_LIB_ROOT)


logfire.install_auto_tracing(should_trace)
Excluding functions from tracing¶
Once you've selected which modules to trace, you probably don't want to trace every function in those modules. To exclude a function from auto-tracing, add the no_auto_trace decorator to it:


import logfire

@logfire.no_auto_trace
def my_function():
    # Nested functions will also be excluded
    def inner_function():
        ...

    return other_function()


# This function is *not* excluded from auto-tracing.
# It will still be traced even when called from the excluded `my_function` above.
def other_function():
    ...


# All methods of a decorated class will also be excluded
@no_auto_trace
class MyClass:
    def my_method(self):
        ...
The decorator is detected at import time. Only @no_auto_trace or @logfire.no_auto_trace are supported. Renaming/aliasing either the function or module won't work. Neither will calling this indirectly via another function.

This decorator simply returns the argument unchanged, so there is zero runtime overhead.

Only tracing functions above a minimum duration¶
A more convenient way to exclude functions from tracing is to set the min_duration argument, e.g:


# Only trace functions that take longer than 0.1 seconds
logfire.install_auto_tracing(modules=['app'], min_duration=0.1)
This means you automatically get observability for the heavier parts of your application without too much overhead or data. Note that there are some caveats:

A function will only start being traced after it runs longer than min_duration once. This means that:
If it runs faster than min_duration the first few times, you won't get data about those first calls.
The first time that it runs longer than min_duration, you also won't get data about that call.
After a function runs longer than min_duration once, it will be traced every time it's called afterwards, regardless of how long it takes.
Measuring the duration of a function call still adds a small overhead. For tiny functions that are called very frequently, it's best to still use the @no_auto_trace decorator to avoid any overhead. Auto-tracing with min_duration will still work for other undecorated functions.
 Back to top
© Pydantic Services Inc. 2024

Intro
Guides
Onboarding Checklist
Add Logfire Metrics
Pydantic Logfire can be used to collect metrics from your application and send them to a metrics backend.

Let's see how to create, and use metrics in your application.


import logfire

# Create a counter metric
messages_sent = logfire.metric_counter('messages_sent')

# Increment the counter
def send_message():
    messages_sent.add(1)
Metric Types¶
Metrics are a great way to record number values where you want to see an aggregation of the data (e.g. over time), rather than the individual values.

Counter¶
The Counter metric is particularly useful when you want to measure the frequency or occurrence of a certain event or state in your application.

You can use this metric for counting things like:

The number of exceptions caught.
The number of requests received.
The number of items processed.
To create a counter metric, use the logfire.metric_counter function:


import logfire

counter = logfire.metric_counter(
    'exceptions',
    unit='1',  
    description='Number of exceptions caught'
)

try:
    raise Exception('oops')
except Exception:
    counter.add(1)
You can read more about the Counter metric in the OpenTelemetry documentation.

Histogram¶
The Histogram metric is particularly useful when you want to measure the distribution of a set of values.

You can use this metric for measuring things like:

The duration of a request.
The size of a file.
The number of items in a list.
To create a histogram metric, use the logfire.metric_histogram function:


import logfire

histogram = logfire.metric_histogram(
    'request_duration',
    unit='ms',  
    description='Duration of requests'
)

for duration in [10, 20, 30, 40, 50]:
    histogram.record(duration)
You can read more about the Histogram metric in the OpenTelemetry documentation.

Up-Down Counter¶
The "Up-Down Counter" is a type of counter metric that allows both incrementing (up) and decrementing (down) operations. Unlike a regular counter that only allows increments, an up-down counter can be increased or decreased based on the events or states you want to track.

You can use this metric for measuring things like:

The number of active connections.
The number of items in a queue.
The number of users online.
To create an up-down counter metric, use the logfire.metric_up_down_counter function:


import logfire

active_users = logfire.metric_up_down_counter(
    'active_users',
    unit='1',  
    description='Number of active users'
)

def user_logged_in():
    active_users.add(1)

def user_logged_out():
    active_users.add(-1)
You can read more about the Up-Down Counter metric in the OpenTelemetry documentation.

Callback Metrics¶
Callback metrics, or observable metrics, are a way to create metrics that are automatically updated based on a time interval.

Counter Callback¶
To create a counter callback metric, use the logfire.metric_counter_callback function:


import logfire
from opentelemetry.metrics import CallbackOptions, Observable


def cpu_time_callback(options: CallbackOptions) -> Iterable[Observation]:
    observations = []
    with open("/proc/stat") as procstat:
        procstat.readline()  # skip the first line
        for line in procstat:
            if not line.startswith("cpu"):
                break
            cpu, user_time, nice_time, system_time = line.split()
            observations.append(
                Observation(int(user_time) // 100, {"cpu": cpu, "state": "user"})
            )
            observations.append(
                Observation(int(nice_time) // 100, {"cpu": cpu, "state": "nice"})
            )
            observations.append(
                Observation(int(system_time) // 100, {"cpu": cpu, "state": "system"})
            )
    return observations

logfire.metric_counter_callback(
    'system.cpu.time',
    unit='s',
    callbacks=[cpu_time_callback],
    description='CPU time',
)
You can read more about the Counter metric in the OpenTelemetry documentation.

Gauge Callback¶
The gauge metric is particularly useful when you want to measure the current value of a certain state or event in your application. Unlike the counter metric, the gauge metric does not accumulate values over time.

To create a gauge callback metric, use the logfire.metric_gauge_callback function:


import logfire


def get_temperature(room: str) -> float:
    ...


def temperature_callback(options: CallbackOptions) -> Iterable[Observation]:
    for room in ["kitchen", "living_room", "bedroom"]:
        temperature = get_temperature(room)
        yield Observation(temperature, {"room": room})


logfire.metric_gauge_callback(
    'temperature',
    unit='°C',
    callbacks=[temperature_callback],
    description='Temperature',
)
You can read more about the Gauge metric in the OpenTelemetry documentation.

Up-Down Counter Callback¶
This is the callback version of the up-down counter metric.

To create an up-down counter callback metric, use the logfire.metric_up_down_counter_callback function:


import logfire


def get_active_users() -> int:
    ...


def active_users_callback(options: CallbackOptions) -> Iterable[Observation]:
    active_users = get_active_users()
    yield Observation(active_users, {})


logfire.metric_up_down_counter_callback(
    'active_users',
    unit='1',
    callbacks=[active_users_callback],
    description='Number of active users',
)
You can read more about the Up-Down Counter metric in the OpenTelemetry documentation.

System Metrics¶
By default, Logfire does not collect system metrics.

To enable metrics, you need just need install the logfire[system-metrics] extra:


PIP
Rye
Poetry

pip install 'logfire[system-metrics]'

Logfire will automatically collect system metrics if the logfire[system-metrics] extra is installed.

To know more about which system metrics are collected, check the System Metrics documentation.

Intro
Guides
Web UI
Live View¶
The live view is the main view of Logfire, where you can see traces in real-time.

The live view is useful (as the name suggests) for watching what's going on within your application in real-time, but it can also be used to explore historical data.

Details panel closed¶
Logfire OpenAI Image Generation

This is what you'll see when you come to the live view of a project with some data.

Organization and project labels: In this example, the organization is samuelcolvin, and the project is logfire-demo-spider. You can click the organization name to go to the organization overview page; the project name is a link to this page.

Project pages: These are links to the various project-specific pages, including the Live, Dashboards, Alerts, Explore, and Settings pages.

Feedback and Beta buttons: Click the feedback button to provide us feedback. The beta button has more information about Logfire's beta status.

Light/Dark mode toggle: Cycles between light, dark, and system — because everyone seems to have an opinion on this 😄

Link to the current view: Clicking this copies a link to the page you are on, with the same query etc.

Organization selection panel: Opens a drawer with links to the different organizations you are a member of, and also has links to the Terms and Conditions, Support, Documentation, and a Log Out button.

Query text input: Enter a SQL query here to find spans that match the query. The query should be in the form of a Postgres-compatible WHERE clause on the records table (e.g. to find warnings, enter level >= level_num('error')). See the Explore docs for more detail about the schema here.

Search button: You can click here to run the query after you've entered it, or just press cmd+enter (or ctrl+enter on windows/linux).

Extra query menu: Here you can find quick selections for adding filters on various fields to your query. There is also a link to a natural language query entry option, which uses an LLM to generate a query based on a natural language description of what you are looking for.

Toggle timeline position button: Click here to switch the timeline (see the next item for more info) between vertical and horizontal orientation.

Timeline: This shows a histogram of the counts of spans matching your query over time. The blue-highlighted section corresponds to the time range currently visible in the scrollable list of traces below. You can click at points on this line to move to viewing logs from that point in time.

Traces scroll settings: This menu contains some settings related to what is displayed in the traces scroll view.

Status label: This should show "Connected" if your query is successful and you are receiving live data. If you have a syntax error in your query or run into other issues, you should see details about the problem here.

Service, scope, and tags visibility filters: Here you can control whether certain spans are displayed based on their service, scope, or tags.

Level visibility filter: Here you can control which log levels are displayed. By default, 'debug' and 'trace' level spans are hidden from view, but you can change the value here to display them, or you can toggle the visibility of spans of other levels as well.

Time window selection: Here, you can toggle between "Live tail", which shows live logs as they are received, and a historical time range of varying sizes. When a specific time range is selected, the timeline from item 11 will match that range.

Below item 16, we have the "Traces Scroll View", which shows traces matching your current query and visibility filters.

Start timestamp label: This timestamp is the start_timestamp of the span. Hover this to see its age in human-readable format.

Service label: This pill contains the service_name of the span. This is the name of the service that produced the span. You can hover to see version info.

Message: Here you can see the message of this span (which is actually the root span of its trace). You can also click here to see more details. Note that the smaller diamond means that this span has no children

A collapsed trace: The larger diamond to the left of the span message, with a + in it, indicates that this span has child spans, and can be expanded to view them by clicking on the +-diamond.

Scope label: This pill contains the otel_scope_name of the span. This is the name of the OpenTelemetry scope that produced the span. Generally, OpenTelemetry scopes correspond to instrumentations, so this generally gives you a sense of what library's instrumentation produced the span. This will be logfire when producing spans using the logfire APIs, but will be the name of the OpenTelemetry instrumentation package if the span was produced by another instrumentation. You can hover to see version info.

Trace duration line: When the root span of a trace is collapsed, the line on the right will be thicker and rounded, and start at the far left. When this is the case, the length of the line represents the log-scale duration of the trace. See item 25 for contrast.

Trace duration label: Shows the duration of the trace.

An expanded trace: Here we can see what it looks like if you expand a trace down a couple levels. You can click any row within the trace to see more details about the span.

Span duration line: When a trace is expanded, the shape of the lines change, representing a transition to a linear scale where you can see each span's start and end timestamp within the overall trace.

Details panel open¶
Logfire OpenAI Image Generation

When you click on a span in the Traces Scroll, it will open the details panel, which you can see here.

Timeline tooltip: Here you can see the tooltip shown when you hover the timeline. It shows the count of records in the hovered histogram bar, the duration of the bar, the time range that the bar represents, and the exact timestamp you are hovering (and at which you'll retrieve records when you click on the timeline)

Level icon: This icon represents the highest level of this span and any of its descendants.

Span message: Here you can see whether the item is a Span or Log, and its message.

Details panel orientation toggle, and other buttons: The second button copies a link to view this specific span. The X closes the details panel for this span.

Exception warning: This exception indicator is present because an exception bubbled through this span. You can see more details in the Exception Traceback details tab.

Pinned span attributes: This section contains some details about the span. The link icons on the "Trace ID" and "Span ID" pills can be clicked to take you to a view of the trace or span, respectively.

Details tabs: These tabs include more detailed information about the span. Some tabs, such as the Exception Details tab, will only be present for spans with data relevant to that tab.

Arguments panel: If a span was created with one of the logfire span/logging APIs, and some arguments were present, those arguments will be shown here, displayed as a Python dictionary.

Code details panel: When attributes about the source line are present on a span, this panel will be present, and that information displayed here.

Full span attributes panel: When any attributes are present, this panel will show the full list of OpenTelemetry attributes on the span. This panel is collapsed by default, but you can click on its name to show it.

Live view variant¶
Logfire OpenAI Image Generation

This is what the timeline looks like in vertical orientation. You can toggle this orientation at any time.
This is what the details panel looks like in horizontal orientation. You can toggle this orientation whenever the details panel is open.

Intro
Guides
Web UI
Dashboards¶
This guide illustrates how to create and customize dashboards within the Logfire UI, thereby enabling effective monitoring of services and system metrics.

Logfire Dashboard

Get started¶
Logfire provides several pre-built dashboards as a convenient starting point.

Web Service Dashboard¶
This dashboard offers a high-level view of your web services' well-being. It likely displays key metrics like:

Requests: Total number of requests received by your web service.
Exceptions: Number of exceptions encountered during request processing.
Trend Routes: Visualize the most frequently accessed routes or APIs over time.
Percent of 2XX Requests: Percentage of requests that resulted in successful responses (status codes in the 200 range).
Percent of 5XX Requests: Percentage of requests that resulted in server errors (status codes in the 500 range).
Log Type Ratio: Breakdown of the different log types generated by your web service (e.g., info, warning, error).
System Metrics¶
This dashboard focuses on system resource utilization, potentially including:

CPU Usage: Percentage of processing power utilized by the system.
Memory Usage: Amount of memory currently in use by the system.
Number of Processes: Total number of running processes on the system.
Swap Usage: Amount of swap space currently in use by the system.
Custom Dashboards¶
To create a custom dashboard, follow these steps:

From the dashboard page, click on the "Start From Scratch" button.
Once your dashboard is created, you can start rename it and adding charts and blocks to it.
To add a chart, click on the "Add Chart" button.
Choose the type of block you want to add.
Configure the block by providing the necessary data and settings (check the next section).
Repeat steps 4-6 to add more blocks to your dashboard.
To rearrange the blocks, enable the "Edit Mode" in the dashboard setting and simply drag and drop them to the desired position.
Feel free to experiment with different block types and configurations to create a dashboard that suits your monitoring needs.

Choosing and Configuring Dashboard's Charts¶
When creating a custom dashboard or modifying them in Logfire, you can choose from different chart types to visualize your data.

Logfire Dashboard chart types

Define Your Query¶
In the second step of creating a chart, you need to input your SQL query. The Logfire dashboard's charts grab data based on this query. You can see the live result of the query on the table behind your query input. You can use the full power of PostgreSQL to retrieve your records.

Logfire Dashboard chart query

Chart Preview and configuration¶
Based on your need and query, you need to configure the chart to visualize and display your data:

Time Series Chart¶
A time series chart displays data points over a specific time period.

Pie Chart¶
A pie chart represents data as slices of a circle, where each slice represents a category or value.

Table¶
A table displays data in rows and columns, allowing you to present tabular data.

Values¶
A values chart displays a single value or multiple values as a card or panel.

Categories¶
A categories chart represents data as categories or groups, allowing you to compare different groups.

Tips and Tricks¶
Enhanced Viewing with Synchronized Tooltips and Zoom¶
For dashboards containing multiple time-series charts, consider enabling "Sync Tooltip and Zoom." This powerful feature provides a more cohesive viewing experience:

Hover in Sync: When you hover over a data point on any time-series chart, corresponding data points on all synchronized charts will be highlighted simultaneously. This allows you to easily compare values across different metrics at the same time point. Zooming Together: Zooming in or out on a single chart will automatically apply the same zoom level to all synchronized charts. This helps you maintain focus on a specific time range across all metrics, ensuring a consistent analysis. Activating Sync

To enable synchronized tooltips and zoom for your dashboard:

Open your dashboard in Logfire.
Click on Dashboard Setting
activate "Sync Tooltip and Zoom" option.
Customizing Your Charts¶
Logfire empowers you to personalize the appearance and behavior of your charts to better suit your needs. Here's an overview of the available options:

Rename Chart: Assign a clear and descriptive title to your chart for improved readability.
Edit Chart: Change the chart query to better represent your data.
Duplicate Chart: Quickly create a copy of an existing chart for further modifications, saving you time and effort.
Delete Chart: Remove a chart from your dashboard if it's no longer relevant.

Intro
Guides
Web UI
Alerts
With Logfire, you can set up alerts to notify you when certain conditions are met.

Logfire alerts screen

Create an alert¶
Let's see in practice how to create an alert.

Go to the Alerts tab in the left sidebar.
Click the Create alert button.
Then you'll see the following form:

Create alert form

The Query field is where you define the conditions that will trigger the alert. For example, you can set up an alert to notify you when the number of errors in your logs exceeds a certain threshold.

On our example, we're going to set up an alert that will trigger when an exception occurs in the api service and the route is /members/{user_id}.


SELECT * FROM records  
WHERE
    is_exception and  
    service_name = 'api' and  
    attributes->>'http.route' = '/members/{user_id}'  
The Time window field allows you to specify the time range over which the query will be executed.

The Webhook URL field is where you can specify a URL to which the alert will send a POST request when triggered. For now, Logfire alerts only send the requests in Slack format.

Get a Slack webhook URL
After filling in the form, click the Create alert button. And... Alert created! 🎉

Alert History¶
After creating an alert, you'll be redirected to the alerts' list. There you can see the alerts you've created and their status.

If the query was not matched in the last time window, you'll see a 0 in the Matches column, and a green circle next to the alert name.

Alerts list

Otherwise, you'll see the number of matches and a red circle.

Alerts list with error

In this case, you'll also receive a notification in the Webhook URL you've set up.

Edit an alert¶
You can configure an alert by clicking on the Configuration button on the right side of the alert.

Edit alert

You can update the alert, or delete it by clicking the Delete button. If instead of deleting the alert, you want to disable it, you can click on the Active switch.

Intro
Guides
Web UI
SQL Explorer
With Logfire, you can use the Explore page to run arbitrary SQL queries against your trace and metric data to analyze and investigate your system.

Logfire explore screen

Querying Traces¶
The primary table you will query is the records table, which contains all the spans/logs from your traced requests.

To query the records, simply start your query with SELECT ... FROM records and add a WHERE clause to filter the spans you want.

For example, here is a query that returns the message, start_timestamp, duration, and attributes for all spans that have exceptions:


SELECT
  message,
  start_timestamp,
  EXTRACT(EPOCH FROM (end_timestamp - start_timestamp)) * 1000 AS duration_ms,
  attributes
FROM records
WHERE is_exception
You can run more complex queries as well, using subqueries, CTEs, joins, aggregations, custom expressions, and any other standard SQL.

Records Schema¶
The schema of the records table is:


CREATE TABLE records AS (
    start_timestamp timestamp with time zone,
    created_at timestamp with time zone,
    trace_id text,
    span_id text,
    parent_span_id text,
    kind span_kind,
    end_timestamp timestamp with time zone,
    level smallint,
    span_name text,
    message text,
    attributes_json_schema text,
    attributes jsonb,
    tags text[],
    otel_links jsonb,
    otel_events jsonb,
    is_exception boolean,
    otel_status_code status_code,
    otel_status_message text,
    otel_scope_name text,
    otel_scope_version text,
    otel_scope_attributes jsonb,
    service_namespace text,
    service_name text,
    service_version text,
    service_instance_id text,
    process_pid integer
)
Cross-linking with Live View¶
After running a query, you can take any trace_id and/or span_id and use it to look up data shown as traces in the Live View.

Simply go to the Live View and enter a query like:


trace_id = '7bda3ddf6e6d4a0c8386093209eb0bfc' -- replace with a real trace_id of your own
This will show all the spans with that specific trace ID.

Metrics Schema¶
In addition to traces, you can also query your metrics data using the metrics table.

The schema of the metrics table is:


CREATE TABLE metrics AS (
    recorded_timestamp timestamp with time zone,
    metric_name text,
    metric_type text,
    unit text,
    start_timestamp timestamp with time zone,
    aggregation_temporality public.aggregation_temporality,
    is_monotonic boolean,
    metric_description text,
    scalar_value double precision,
    histogram_min double precision,
    histogram_max double precision,
    histogram_count integer,
    histogram_sum double precision,
    exp_histogram_scale integer,
    exp_histogram_zero_count integer,
    exp_histogram_zero_threshold double precision,
    exp_histogram_positive_bucket_counts integer[],
    exp_histogram_positive_bucket_counts_offset integer,
    exp_histogram_negative_bucket_counts integer[],
    exp_histogram_negative_bucket_counts_offset integer,
    histogram_bucket_counts integer[],
    histogram_explicit_bounds double precision[],
    attributes jsonb,
    tags text[],
    otel_scope_name text,
    otel_scope_version text,
    otel_scope_attributes jsonb,
    service_namespace text,
    service_name text,
    service_version text,
    service_instance_id text,
    process_pid integer
)
You can query metrics using standard SQL, just like traces. For example:


SELECT *
FROM metrics
WHERE metric_name = 'system.cpu.time'
  AND recorded_timestamp > now() - interval '1 hour'
Executing Queries¶
To execute a query, type or paste it into the query editor and click the "Run Query" button.

Logfire explore screen

You can modify the time range of the query using the dropdown next to the button. There is also a "Limit" dropdown that controls the maximum number of result rows returned.

The Explore page provides a flexible interface to query your traces and metrics using standard SQL.

Happy querying! 


Intro
Guides
Advanced User Guide
Sampling¶
Sampling is the practice of discarding some traces or spans in order to reduce the amount of data that needs to be stored and analyzed. Sampling is a trade-off between cost and completeness of data.

To configure sampling for the SDK:

Set the trace_sample_rate option of logfire.configure() to a number between 0 and 1, or
Set the LOGFIRE_TRACE_SAMPLE_RATE environment variable, or
Set the trace_sample_rate config file option.
See Configuration for more information.


import logfire

logfire.configure(trace_sample_rate=0.5)

with logfire.span("my_span"):  # This span will be sampled 50% of the time
    pass

Intro
Guides
Advanced User Guide
Scrubbing sensitive data¶
The Logfire SDK scans for and redacts potentially sensitive data from logs and spans before exporting them.

Scrubbing more with custom patterns¶
By default, the SDK looks for some sensitive regular expressions. To add your own patterns, set scrubbing_patterns to a list of regex strings:


import logfire

logfire.configure(scrubbing_patterns=['my_pattern'])

logfire.info('Hello', data={
    'key_matching_my_pattern': 'This string will be redacted because its key matches',
    'other_key': 'This string will also be redacted because it matches MY_PATTERN case-insensitively',
    'password': 'This will be redacted because custom patterns are combined with the default patterns',
})
Here are the default scrubbing patterns:

'password', 'passwd', 'mysql_pwd', 'secret', 'auth', 'credential', 'private[._ -]?key', 'api[._ -]?key', 'session', 'cookie', 'csrf', 'xsrf', 'jwt', 'ssn', 'social[._ -]?security', 'credit[._ -]?card'

Scrubbing less with a callback¶
On the other hand, if the scrubbing is to aggressive, you can pass a function to scrubbing_callback to prevent certain data from being redacted.

The function will be called for each potential match found by the scrubber. If it returns None, the value is redacted. Otherwise, the returned value replaces the matched value. The function accepts a single argument of type logfire.ScrubMatch.

Here's an example:


import logfire

def scrubbing_callback(match: logfire.ScrubMatch):
    # OpenTelemetry database instrumentation libraries conventionally
    # use `db.statement` as the attribute key for SQL queries.
    # Assume that SQL queries are safe even if they contain words like 'password'.
    # Make sure you always use SQL parameters instead of formatting strings directly!
    if match.path == ('attributes', 'db.statement'):
        # Return the original value to prevent redaction.
        return match.value

logfire.configure(scrubbing_callback=scrubbing_callback)
Security tips¶
Use message templates¶
The full span/log message is not scrubbed, only the fields within. For example, this:


logfire.info('User details: {user}', user=User(id=123, password='secret'))
...may log something like:


User details: [Redacted due to 'password']
...but this:


user = User(id=123, password='secret')
logfire.info(f'User details: {user}')
will log:


User details: User(id=123, password='secret')
This is necessary so that safe messages such as 'Password is correct' are not redacted completely.

In short, don't use f-strings or otherwise format the message yourself. This is also a good practice in general for non-security reasons.

Keep sensitive data out URLs¶
The attribute "http.url" which is recorded by OpenTelemetry instrumentation libraries is considered safe so that URLs like "http://example.com/users/123/authenticate" are not redacted.

As a general rule, not just for Logfire, assume that URLs (including query parameters) will be logged, so sensitive data should be put in the request body or headers instead.

Intro
Guides
Advanced User Guide
Testing with Logfire¶
You may want to check that your API is logging the data you expect, that spans correctly track the work they wrap, etc. This can often be difficult, including with Python's built in logging and OpenTelemetry's SDKs.

Logfire makes it very easy to test the emitted logs and spans using the utilities in the logfire.testing module. This is what Logfire uses internally to test itself as well.

capfire fixture¶
This has two attributes exporter and metrics_reader.

exporter¶
This is an instance of TestExporter and is an OpenTelemetry SDK compatible span exporter that keeps exported spans in memory.

The exporter.exported_spans_as_dict() method lets you get a plain dict representation of the exported spans that you can easily assert against and get nice diffs from. This method does some data massaging to make the output more readable and deterministic, e.g. replacing line numbers with 123 and file paths with just the filename.

test.py

import pytest

import logfire
from logfire.testing import  CaptureLogfire


def test_observability(capfire: CaptureLogfire) -> None:
    with pytest.raises(Exception):
        with logfire.span('a span!'):
            logfire.info('a log!')
            raise Exception('an exception!')

    exporter = capfire.exporter

    # insert_assert(exporter.exported_spans_as_dict()) 
    assert exporter.exported_spans_as_dict() == [
        {
            'name': 'a log!',
            'context': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
            'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
            'start_time': 2000000000,
            'end_time': 2000000000,
            'attributes': {
                'logfire.span_type': 'log',
                'logfire.level_num': 9,
                'logfire.msg_template': 'a log!',
                'logfire.msg': 'a log!',
                'code.filepath': 'test.py',
                'code.lineno': 123,
                'code.function': 'test_observability',
            },
        },
        {
            'name': 'a span!',
            'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
            'parent': None,
            'start_time': 1000000000,
            'end_time': 4000000000,
            'attributes': {
                'code.filepath': 'test.py',
                'code.lineno': 123,
                'code.function': 'test_observability',
                'logfire.msg_template': 'a span!',
                'logfire.span_type': 'span',
                'logfire.msg': 'a span!',
            },
            'events': [
                {
                    'name': 'exception',
                    'timestamp': 3000000000,
                    'attributes': {
                        'exception.type': 'Exception',
                        'exception.message': 'an exception!',
                        'exception.stacktrace': 'Exception: an exception!',
                        'exception.escaped': 'True',
                    },
                }
            ],
        },
    ]
You can access exported spans by exporter.exported_spans.


import logfire
from logfire.testing import CaptureLogfire


def test_exported_spans(capfire: CaptureLogfire) -> None:
    with logfire.span('a span!'):
        logfire.info('a log!')

    exporter = capfire.exporter

    expected_span_names = ['a span! (pending)', 'a log!', 'a span!']
    span_names = [span.name for span in exporter.exported_spans]

    assert span_names == expected_span_names
You can call exporter.clear() to reset the captured spans in a test.


import logfire
from logfire.testing import CaptureLogfire


def test_reset_exported_spans(capfire: CaptureLogfire) -> None:
    exporter = capfire.exporter

    assert len(exporter.exported_spans) == 0

    logfire.info('First log!')
    assert len(exporter.exported_spans) == 1
    assert exporter.exported_spans[0].name == 'First log!'

    logfire.info('Second log!')
    assert len(exporter.exported_spans) == 2
    assert exporter.exported_spans[1].name == 'Second log!'

    exporter.clear()
    assert len(exporter.exported_spans) == 0

    logfire.info('Third log!')
    assert len(exporter.exported_spans) == 1
    assert exporter.exported_spans[0].name == 'Third log!'
metrics_reader¶
This is an instance of InMemoryMetricReader which reads metrics into memory.


import json
from typing import cast

from opentelemetry.sdk.metrics.export import MetricsData

from logfire.testing import CaptureLogfire


def test_system_metrics_collection(capfire: CaptureLogfire) -> None:
    exported_metrics = json.loads(cast(MetricsData, capfire.metrics_reader.get_metrics_data()).to_json())  # type: ignore

    metrics_collected = {
        metric['name']
        for resource_metric in exported_metrics['resource_metrics']
        for scope_metric in resource_metric['scope_metrics']
        for metric in scope_metric['metrics']
    }

    # collected metrics vary by platform, etc.
    # assert that we at least collected _some_ of the metrics we expect
    assert metrics_collected.issuperset(
        {
            'system.swap.usage',
            'system.disk.operations',
            'system.memory.usage',
            'system.cpu.utilization',
        }
    ), metrics_collected
Let's walk through the utilities we used.

IncrementalIdGenerator¶
One of the most complicated things about comparing log output to expected results are sources of non-determinism. For OpenTelemetry spans the two biggest ones are the span & trace IDs and timestamps.

The IncrementalIdGenerator generates sequentially increasing span and trace IDs so that test outputs are always the same.

TimeGenerator¶
This class generates nanosecond timestamps that increment by 1s every time a timestamp is generated.

logfire.configure¶
This is the same configuration function you'd use for production and where everything comes together.

Note that we specifically configure:

send_to_logfire=False because we don't want to hit the actual production service
id_generator=IncrementalIdGenerator() to make the span IDs deterministic
ns_timestamp_generator=TimeGenerator() to make the timestamps deterministic
processors=[SimpleSpanProcessor(exporter)] to use our TestExporter to capture spans. We use SimpleSpanProcessor to export spans with no delay.
insert_assert¶
This is a utility function provided by devtools that will automatically insert the output of the code it is called with into the test file when run via pytest. That is, if you comment that line out you'll see that the assert capfire.exported_spans_as_dict() == [...] line is replaced with the current output of capfire.exported_spans_as_dict(), which should be exactly the same given that our test is deterministic!


Intro
Guides
Advanced User Guide
Backfilling data¶
When Logfire fails to send a log to the server, it will dump data to the disk to avoid data loss.

Logfire supports bulk loading data, either to import data from another system or to load data that was dumped to disk.

To backfill data, you can use the logfire backfill command:


$ logfire backfill --help
By default logfire backfill will read from the default fallback file so if you are just trying to upload data after a network failure you can just run:


$ logfire backfill
Bulk loading data¶
This same mechanism can be used to bulk load data, for example if you are importing it from another system.

First create a dump file:


from datetime import datetime

from logfire.backfill import Log, PrepareBackfill, StartSpan

with PrepareBackfill('logfire_spans123.bin') as backfill:
    span = StartSpan(
        start_timestamp=datetime(2023, 1, 1, 0, 0, 0),
        span_name='session',
        msg_template='session {user_id=} {path=}',
        service_name='docs.pydantic.dev',
        log_attributes={'user_id': '123', 'path': '/test'},
    )
    child = StartSpan(
        start_timestamp=datetime(2023, 1, 1, 0, 0, 1),
        span_name='query',
        msg_template='ran db query',
        service_name='docs.pydantic.dev',
        log_attributes={'query': 'SELECT * FROM users'},
        parent=span,
    )
    backfill.write(
        Log(
            timestamp=datetime(2023, 1, 1, 0, 0, 2),
            msg_template='GET {path=}',
            level='info',
            service_name='docs.pydantic.dev',
            attributes={'path': '/test'},
        )
    )
    backfill.write(child.end(end_timestamp=datetime(2023, 1, 1, 0, 0, 3)))
    backfill.write(span.end(end_timestamp=datetime(2023, 1, 1, 0, 0, 4)))
This will create a logfire_spans123.bin file with the data.

Then use the backfill command line tool to load it:


$ logfire backfill --file logfire_spans123.bin


Intro
Guides
Advanced User Guide
Creating Write Tokens
To send data to Logfire, you need to create a write token. A write token is a unique identifier that allows you to send data to a specific Logfire project. If you set up Logfire according to the first steps guide, you already have a write token locally tied to the project you created. But if you want to configure other computers to write to that project, for example in a deployed application, you need to create a new write token.

You can create a write token by following these steps:

Open the Logfire web interface at logfire.pydantic.dev.
Select your project from the Projects section on the left hand side of the page.
Click on the ⚙️ Settings tab on the top right corner of the page.
Select the {} Write tokens tab on the left hand menu.
Click on the Create write token button.
After creating the write token, you'll see a dialog with the token value. Copy this value and store it securely, it will not be shown again.

Now you can use this write token to send data to your Logfire project from any computer or application.

We recommend you inject your write token via environment variables in your deployed application. Set the token as the value for the environment variable LOGFIRE_TOKEN and logfire will automatically use it to send data to your project.

Setting send_to_logfire='if-token-present'¶
You may want to not send data to logfire during local development, but still have the option to send it in production without changing your code. To do this we provide the parameter send_to_logfire='if-token-present' in the logfire.configure() function. If you set it to 'if-token-present', logfire will only send data to logfire if a write token is present in the environment variable LOGFIRE_TOKEN or there is a token saved locally. If you run tests in CI no data will be sent.

You can also set the environmnet variable LOGFIRE_SEND_TO_LOGFIRE to configure this option. For example, you can set it to LOGFIRE_SEND_TO_LOGFIRE=true in your deployed application and LOGFIRE_SEND_TO_LOGFIRE=false in your tests setup.


Intro
Guides
Advanced User Guide
Direct Database Connections¶
The Logfire platform allows you to connect and run SQL queries against your data using PostgreSQL syntax.

By doing this, you can connect your existing tools such as Grafana, Metabase, Superset, or anything else with support for querying PostgreSQL sources.

Generating credentials¶
To connect, you'll first need to generate generate database credentials from your project page at https://logfire.pydantic.dev/<organization>/<project>/settings/database-credentials

Creating database credentials

The credentials generated are a PostgreSQL URI which can be used as a connection string for compatible tools. These will only be shown by the Logfire platform once, so save them to a secure location for future use!

Example: pgcli¶
pgcli is a command-line tool to access PostgreSQL databases.

Using the credentials generated in the previous step as the argument to pgcli, you can connect directly to Logfire:


$ pgcli postgresql://<user>:<password>@db.logfire.dev:5432/proj_david-test  # REDACTED
Version: 4.0.1
Home: http://pgcli.com
proj_david-test> select start_timestamp, message from records limit 10;
+-------------------------------+----------------------------------------+
| start_timestamp               | message                                |
|-------------------------------+----------------------------------------|
| 2024-04-28 10:50:41.681886+00 | action=view-faq size=549 i=0           |
| 2024-04-28 10:50:41.711747+00 | GET /contact/ http send response.body  |
| 2024-04-28 10:50:41.665576+00 | GET /contact/                          |
| 2024-04-28 10:50:41.711119+00 | GET /contact/ http send response.start |
| 2024-04-28 10:50:41.709458+00 | response 500                           |
| 2024-04-28 10:50:38.50534+00  | action=view-cart size=517 i=0          |
| 2024-04-28 10:50:39.446668+00 | action=view-faq size=637 i=2           |
| 2024-04-28 10:50:38.681198+00 | action=view-terms size=216 i=3         |
| 2024-04-28 10:50:39.416706+00 | action=view-product size=380 i=0       |
| 2024-04-28 10:50:38.394237+00 | sub-sub-sub-action=logout              |
+-------------------------------+----------------------------------------+
SELECT 10
Time: 0.218s
With the flexibility of PostgreSQL access available to you, we can't wait to hear what you do with the Logfire platform!