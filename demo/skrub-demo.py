# %% [markdown]
# ## `TableReport`


# %%
from skrub.datasets import fetch_employee_salaries

dataset = fetch_employee_salaries()
employees_df, salaries = dataset.X, dataset.y
employees_df


# %%
from skrub import TableReport

TableReport(employees_df)

# %% [markdown]
# ## `tabular_learner`


# %%
from sklearn.model_selection import cross_val_score
from skrub import tabular_learner

cross_val_score(tabular_learner('regressor'), employees_df, salaries)

# %%
tabular_learner('regressor')


# %%
from sklearn.linear_model import RidgeCV

tabular_learner(RidgeCV())

# %% [markdown]
# ## `TableVectorizer`


# %%
from skrub import TableVectorizer

vectorizer = TableVectorizer()
vectorized_employees = vectorizer.fit_transform(employees_df)
vectorized_employees.dtypes

vectorizer.kind_to_columns_

vectorized_employees[vectorizer.input_to_outputs_["date_first_hired"]]

vectorizer.all_processing_steps_["date_first_hired"]

vectorizer.all_processing_steps_["department"]



# %% [markdown]
# ## Joiners

# %%
import polars as pl
from skrub.datasets import fetch_figshare

airports = pl.from_pandas(
    fetch_figshare("41710257").X
)
TableReport(airports)

# %%
cols = ["ID", "LATITUDE", "LONGITUDE", "ELEVATION", "STATE"]
stations = pl.from_dataframe(
    fetch_figshare("41710524").X[cols]
)
TableReport(stations)

# %%
from skrub import Joiner

joiner = Joiner(
    aux_table=stations,
    aux_key=["LONGITUDE", "LATITUDE"],
    main_key=["long", "lat"],
    suffix="_station",
)
airports_stations = joiner.fit_transform(airports)
TableReport(airports_stations.sort("skrub_Joiner_distance"))

# %% [markdown]

# ![](./worst_distance_1.png)


# %%
from skrub import Joiner

joiner = Joiner(
    aux_table=stations,
    aux_key=["LONGITUDE", "LATITUDE"],
    main_key=["long", "lat"],
    suffix="_station",
    max_dist=0.01,
)
airports_stations = (
    joiner.fit_transform(airports)
    .sort("skrub_Joiner_distance")
)
TableReport(airports_stations)

# %% [markdown]

# ![](./worst_distance_001.png)

# %%

weather = pl.from_pandas(
    fetch_figshare("41771457").X
)
TableReport(weather)


# %%

from skrub import AggJoiner

agg_joiner = AggJoiner(
    aux_table=weather,
    main_key="ID_station",
    aux_key="ID",
    cols=["TMAX", "PRCP"],
    operations=["mean", "std"],
)
airports_stations_weather = agg_joiner.fit_transform(airports_stations)

TableReport(airports_stations_weather)


# %%

from sklearn.pipeline import make_pipeline

airports_stations_weather = make_pipeline(
    joiner,
    agg_joiner,
).fit_transform(airports)

airports_stations_weather.shape
