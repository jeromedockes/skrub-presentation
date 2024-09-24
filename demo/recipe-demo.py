# # %pip install git+https://github.com/jeromedockes/skrub.git@add-recipe
# # %pip install plotly

# %% [markdown]

# ## `Recipe`: a helper to configure a scikit-learn `Pipeline`.
#
# ![](./recipe-graph.svg)

# %%
from skrub import datasets
from skrub import Recipe

dataset = datasets.fetch_employee_salaries()
df = dataset.X
df["salary"] = dataset.y

# %%
recipe = Recipe(df, y_cols="salary", n_jobs=8)
recipe

# %%
recipe.sample()

# %%
recipe.get_report()

# %% [markdown]
# ## Adding transformations to specific columns

# %% [markdown]

# Don't do this:

# %%
# # df["date_first_hired"] = pd.to_datetime(df["date_first_hired"])


# %%
from skrub import ToDatetime

recipe = recipe.add(ToDatetime(), cols="date_first_hired")
recipe


# %%
recipe.get_report()

# %%
from skrub import DatetimeEncoder
from skrub import selectors as s

recipe = recipe.add(DatetimeEncoder(), cols=s.any_date())
recipe


# %%
recipe.get_report()

# %%
from skrub import ToCategorical

recipe = recipe.add(ToCategorical(), cols=s.string() & s.cardinality_below(30))
recipe


# %% [markdown]
# ## Specifying alternative transformers and a hyperparameter grid

# %%
from sklearn.preprocessing import TargetEncoder

from skrub import MinHashEncoder, choose_from

recipe = recipe.add(
    choose_from(
        {"target": TargetEncoder(), "minhash": MinHashEncoder()}, name="encoder"
    ),
    cols=s.string(),
)
recipe

# %%
recipe.get_report()

# %%
from sklearn.ensemble import HistGradientBoostingRegressor
from skrub import choose_float

recipe = recipe.add(
    HistGradientBoostingRegressor(
        categorical_features="from_dtype",
        learning_rate=choose_float(0.001, 1.0, log=True, name="learning rate"),
    )
)
recipe

# %% [markdown]
# ## Getting the model & inspecting search results

# %%
from sklearn.metrics import r2_score

randomized_search = recipe.get_randomized_search(n_iter=16, cv=3, verbose=1)
randomized_search.fit(recipe.get_x_train(), recipe.get_y_train())

predictions = randomized_search.predict(recipe.get_x_test())
score = r2_score(recipe.get_y_test(), predictions)
print(f"R² score: {score:.2f}")

# %%
recipe.get_cv_results_table(randomized_search)

# %%
recipe.plot_parallel_coord(randomized_search)
