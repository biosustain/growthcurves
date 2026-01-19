# %% [markdown]
# # Python Package tutorial
#
# Run when documentation is build. Building documentation therefore can become
# an minimal integration test for the package.

# %%
from python_package import hello_world

# %% [markdown]
# ## Hello World
# mockup module contains a function which returns a string repeating the 'hello world' string
# n-times.

# %%
ret = hello_world(2)
ret

# %% [markdown]
# Print the string.

# %%
print(ret)

# %% [markdown]
# Inspect the signature. In an interactive session you could also use `hello_world?` to
# display the docstring. See the rendered version
# [of `hello_world` under the Reference](python_package.mockup.hello_world)

# %%
help(hello_world)

# %% [markdown]
# Simple example for a recipe showing what the package can do.
